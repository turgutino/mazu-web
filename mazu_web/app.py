import dataclasses
import json
import queue
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import _SECRET_CONFIG_KEYS, config_path, list_config, set_config_value, unset_config_value
from mazu.curator.config import curator_configured, curator_enabled, curator_model
from mazu.curator.orchestrator import KNOWN_AREAS
from mazu.curator.store import CuratorStore
from mazu.diagnostics import check_live_api_key, ensure_gitignore, run_diagnostics
from mazu.llm.capabilities import list_capabilities
from mazu.memory.consolidate import apply_consolidation, find_duplicate_clusters
from mazu.memory.retrieval import explain_retrieval
from mazu.memory.staleness import DEFAULT_STALE_DAYS, apply_archival, find_stale_candidates
from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, MemoryStore
from mazu.runs.router import TASK_TYPES, model_stats_by_task_type
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

from mazu_web.chat_session import ChatSession
from mazu_web.task_session import STDOUT_CAPTURE_LOCK, CouncilSession, CuratorSession, ExploreSession, RunSession

STATIC_DIR = Path(__file__).parent / "static"


def _mask_secret(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def create_app(root: Path, model: str | None, shell_allowlist: list[str] | None):
    """Builds the Flask app for mazu-web. Every store/manager here is opened fresh
    per request (or per session, for chat/run/explore) and closed immediately --
    the same lifetime the CLI commands themselves give these classes, just behind
    HTTP instead of a single terminal invocation.
    """
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)
    chat_sessions: dict[str, ChatSession] = {}
    task_sessions: dict[str, "RunSession | ExploreSession | CouncilSession"] = {}
    app.sessions = chat_sessions  # exposed for tests; not used by any route itself
    app.task_sessions = task_sessions

    def _memory_db_path() -> Path:
        return root / ".mazu" / "memory.db"

    def _runs_db_path() -> Path:
        return root / ".mazu" / "runs.db"

    def _usage_db_path() -> Path:
        return Path.home() / ".mazu" / "usage.db"

    def _action_log_db_path() -> Path:
        return root / ".mazu" / "action_log.db"

    def _curator_db_path() -> Path:
        return root / ".mazu" / "curator.db"

    def _parse_number(raw, field_name: str, cast, default=None):
        """Real bug found via live testing (on the Curator endpoint, but the same
        unguarded pattern existed at every numeric body field across this file):
        a non-numeric value (e.g. a client sending "not-a-number" for max_cost/
        max_steps/max_rounds/etc.) used to reach a bare int()/float() call inside
        a background-task factory lambda, raising deep inside _start_task's
        factory() call and surfacing as a generic Flask 500 HTML page instead of
        a clean error -- confusing for an API caller, even though it never
        crashed the server or left STDOUT_CAPTURE_LOCK held (_start_task already
        releases it on any factory() exception). Every numeric body field across
        /api/run, /api/explore, /api/council, /api/curator/run,
        /api/memory/stale/archive, and /api/memory/consolidate now validates
        through this one helper instead of a bare cast, returning a clean 400
        with a specific field name instead. Returns (value, error_response_or_None)
        -- callers check `if err: return err` before using `value`.
        """
        value = raw if raw not in (None, "") else default
        if value is None:
            return None, None
        try:
            return cast(value), None
        except (TypeError, ValueError):
            return None, (jsonify({"error": f"{field_name} must be a number"}), 400)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # -- status / init / setup ---------------------------------------------

    @app.get("/api/status")
    def status():
        return jsonify({
            "root": str(root),
            "initialized": (root / ".mazu").exists(),
            "is_git_repo": (root / ".git").exists(),
        })

    @app.post("/api/init")
    def init_project():
        # Mirrors mazu/cli.py's `init` command exactly (same three steps, same
        # idempotency -- safe to call again on an already-initialized project).
        already_existed = (root / ".mazu").exists()
        memory_store = MemoryStore(root / ".mazu" / "memory.db")
        memory_store.close()
        ensure_gitignore(root)

        checkpoint_manager = CheckpointManager(root)
        was_git_repo = checkpoint_manager.is_git_repo()
        checkpoint_manager.ensure_git_repo()

        return jsonify({
            "ok": True,
            "already_initialized": already_existed,
            "initialized_git_repo": not was_git_repo,
        })

    @app.post("/api/setup")
    def setup():
        # Non-interactive equivalent of `mazu setup`'s wizard: same three actions
        # (save key, optionally verify live, optionally set as default model), just
        # taking them as one request body instead of a sequence of click.prompt()s.
        from mazu.llm.client import _PROVIDER_DEFAULT_MODELS, _PROVIDERS

        body = request.get_json(silent=True) or {}
        provider_name = body.get("provider")
        api_key = body.get("api_key")
        if provider_name not in _PROVIDERS or not api_key:
            return jsonify({"error": "provider (one of: " + ", ".join(_PROVIDERS) + ") and api_key are required"}), 400

        set_config_value(f"{provider_name}_api_key", api_key)
        result = {"ok": True, "saved_to": str(config_path())}

        if body.get("verify"):
            import os

            env_var = _PROVIDERS[provider_name].api_key_env
            os.environ[env_var] = api_key
            check = check_live_api_key(provider_name, _PROVIDER_DEFAULT_MODELS[provider_name])
            result["verify"] = {"status": check.status, "message": check.message}

        if body.get("set_default"):
            default_model_choice = _PROVIDER_DEFAULT_MODELS[provider_name]
            set_config_value("default_model", default_model_choice)
            result["default_model"] = default_model_choice

        # Unconditional and idempotent (same as /api/init), not gated on whether
        # root/.mazu already "looks" initialized -- that check is unreliable here
        # specifically because set_config_value() above may have just created
        # ~/.mazu itself, which is the SAME directory as root/.mazu whenever HOME
        # and the project root coincide (the common case in this project's own
        # test fixtures, and for a real user running mazu-web from their home
        # directory) -- a false "already initialized" that skipped the real
        # per-project init steps (memory.db, gitignore, git repo) entirely.
        was_already_git_repo = (root / ".git").exists()
        memory_store = MemoryStore(root / ".mazu" / "memory.db")
        memory_store.close()
        ensure_gitignore(root)
        checkpoint_manager = CheckpointManager(root)
        checkpoint_manager.ensure_git_repo()
        result["initialized_project"] = not was_already_git_repo

        return jsonify(result)

    # -- chat -----------------------------------------------------------

    @app.post("/api/chat/start")
    def chat_start():
        session = ChatSession(root=root, model=model, shell_allowlist=shell_allowlist)
        chat_sessions[session.session_id] = session
        return jsonify({"session_id": session.session_id, "model": session.resolved_model})

    @app.post("/api/chat/<session_id>/message")
    def chat_message(session_id: str):
        session = chat_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        text = (request.get_json(silent=True) or {}).get("text", "")
        session.send(text)
        return jsonify({"ok": True})

    @app.post("/api/chat/<session_id>/confirm")
    def chat_confirm(session_id: str):
        session = chat_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        approved = bool((request.get_json(silent=True) or {}).get("approved", False))
        session.answer_confirm(approved)
        return jsonify({"ok": True})

    @app.get("/api/chat/<session_id>/events")
    def chat_events(session_id: str):
        session = chat_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        return Response(_sse_stream(session.outbox, end_type="closed"), mimetype="text/event-stream")

    # -- run (autonomous) / explore -- share one stdout-capture slot ----------

    def _start_task(kind: str, factory) -> tuple[dict, int]:
        if not STDOUT_CAPTURE_LOCK.acquire(blocking=False):
            return {"error": "Another run/explore/council is already in progress on this server -- wait for it to finish."}, 409
        try:
            session = factory()
        except Exception:
            STDOUT_CAPTURE_LOCK.release()
            raise
        task_sessions[session.task_id] = session
        return {"task_id": session.task_id, "kind": kind}, 200

    @app.post("/api/run")
    def start_run():
        # Same validation as mazu/cli.py's `run` command (UsageError -> 400 here).
        body = request.get_json(silent=True) or {}
        task = (body.get("task") or "").strip() or None
        resume_run_id = body.get("resume") or None
        from_checkpoint_id = body.get("from_checkpoint") or None
        branch_name = body.get("branch") or None

        if from_checkpoint_id is not None and resume_run_id is not None:
            return jsonify({"error": "Pass either from_checkpoint or resume, not both."}), 400
        if branch_name is not None and from_checkpoint_id is None:
            return jsonify({"error": "branch requires from_checkpoint."}), 400
        if from_checkpoint_id is not None and not branch_name:
            return jsonify({"error": "from_checkpoint requires branch (a new branch name)."}), 400
        if from_checkpoint_id is not None and task is None:
            return jsonify({"error": "from_checkpoint requires a task to run on the new branch."}), 400
        if resume_run_id is None and from_checkpoint_id is None and task is None:
            return jsonify({"error": "task is required (or pass resume / from_checkpoint+branch)."}), 400
        if resume_run_id is not None and task is not None:
            return jsonify({"error": "Pass either a new task or resume, not both."}), 400

        max_steps, err = _parse_number(body.get("max_steps"), "max_steps", int, default=15)
        if err:
            return err
        checkpoint_every, err = _parse_number(body.get("checkpoint_every"), "checkpoint_every", int, default=1)
        if err:
            return err
        max_cost, err = _parse_number(body.get("max_cost"), "max_cost", float)
        if err:
            return err
        keep_checkpoints, err = _parse_number(body.get("keep_checkpoints"), "keep_checkpoints", int)
        if err:
            return err

        payload, status = _start_task(
            "run",
            lambda: RunSession(
                root=root, task=task, model=body.get("model") or model,
                max_steps=max_steps, checkpoint_every=checkpoint_every,
                allow_shell=bool(body.get("allow_shell", False)),
                keep_checkpoints=keep_checkpoints,
                max_cost=max_cost,
                shell_allowlist=body.get("shell_allowlist") or None,
                dry_run=bool(body.get("dry_run", False)),
                resume_run_id=resume_run_id,
                from_checkpoint_id=from_checkpoint_id,
                branch_name=branch_name,
            ),
        )
        return jsonify(payload), status

    @app.post("/api/explore")
    def start_explore():
        body = request.get_json(silent=True) or {}
        task = (body.get("task") or "").strip()
        if not task:
            return jsonify({"error": "task is required"}), 400

        if body.get("auto_models"):
            # Same picker `mazu explore --auto-models` uses: this project's own
            # router history for this kind of task, topped up with one model per
            # other available provider. Reused directly, not reimplemented.
            from mazu.cli import _auto_pick_models

            approaches, err = _parse_number(body.get("approaches"), "approaches", int, default=2)
            if err:
                return err
            try:
                models = _auto_pick_models(root, task, approaches)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        else:
            models = [m.strip() for m in (body.get("models") or "").split(",") if m.strip()]
            if not models:
                return jsonify({"error": "models is required unless auto_models is set"}), 400

        max_cost, err = _parse_number(body.get("max_cost"), "max_cost", float)
        if err:
            return err
        max_steps, err = _parse_number(body.get("max_steps"), "max_steps", int, default=15)
        if err:
            return err

        payload, status = _start_task(
            "explore",
            lambda: ExploreSession(
                root=root, task=task, models=models,
                test_command=body.get("test_command") or None,
                max_cost=max_cost, max_steps=max_steps,
                from_checkpoint_id=body.get("from_checkpoint") or None,
            ),
        )
        return jsonify(payload), status

    @app.post("/api/council")
    def start_council():
        # Same defaults as `mazu council` itself, not an ad-hoc web-only fallback.
        from mazu.cli import DEFAULT_COUNCIL_LEAD, DEFAULT_COUNCIL_MODELS

        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        models_raw = body.get("models") or DEFAULT_COUNCIL_MODELS
        models = [m.strip() for m in models_raw.split(",") if m.strip()]
        lead_model = (body.get("lead_model") or "").strip() or DEFAULT_COUNCIL_LEAD
        if not question or not models:
            return jsonify({"error": "question is required"}), 400
        max_cost, err = _parse_number(body.get("max_cost"), "max_cost", float)
        if err:
            return err
        payload, status = _start_task(
            "council",
            lambda: CouncilSession(
                root=root, question=question, models=models, lead_model=lead_model,
                max_cost=max_cost,
            ),
        )
        return jsonify(payload), status

    @app.get("/api/tasks/<task_id>/events")
    def task_events(task_id: str):
        session = task_sessions.get(task_id)
        if session is None:
            return jsonify({"error": "unknown task"}), 404
        return Response(_sse_stream(session.outbox, end_type="done"), mimetype="text/event-stream")

    @app.get("/api/tasks/busy")
    def tasks_busy():
        busy = STDOUT_CAPTURE_LOCK.locked()
        return jsonify({"busy": busy})

    # -- checkpoints ------------------------------------------------------

    @app.get("/api/checkpoints")
    def list_checkpoints():
        checkpoint_manager = CheckpointManager(root)
        return jsonify(checkpoint_manager.timeline_entries())

    @app.post("/api/checkpoints")
    def create_checkpoint():
        # Mirrors bare `mazu checkpoint` (no subcommand) -- a manual snapshot with
        # no live conversation to attach, same trigger label ("manual_cli") so it
        # shows up in the timeline indistinguishably from the CLI's own version.
        ensure_gitignore(root)
        checkpoint_manager = CheckpointManager(root)
        entry = checkpoint_manager.snapshot(messages=[], trigger="manual_cli")
        return jsonify(entry)

    @app.get("/api/checkpoints/<checkpoint_id>/diff")
    def checkpoint_diff(checkpoint_id: str):
        checkpoint_manager = CheckpointManager(root)
        try:
            entry, diff_stat = checkpoint_manager.diff_against_current(checkpoint_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"entry": entry, "diff_stat": diff_stat})

    @app.post("/api/checkpoints/<checkpoint_id>/rollback")
    def rollback_checkpoint(checkpoint_id: str):
        checkpoint_manager = CheckpointManager(root)
        try:
            result = checkpoint_manager.restore(checkpoint_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "checkpoint_id": checkpoint_id, "message_count": len(result["messages"])})

    @app.post("/api/checkpoints/<checkpoint_id>/branch-from")
    def branch_from_checkpoint(checkpoint_id: str):
        body = request.get_json(silent=True) or {}
        branch_name = (body.get("branch_name") or "").strip()
        if not branch_name:
            return jsonify({"error": "branch_name is required"}), 400
        checkpoint_manager = CheckpointManager(root)
        try:
            entry = checkpoint_manager.branch_from(checkpoint_id, branch_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "checkpoint_id": entry["id"], "branch_name": branch_name})

    @app.get("/api/checkpoints/compare")
    def compare_checkpoints():
        a, b = request.args.get("a"), request.args.get("b")
        if not a or not b:
            return jsonify({"error": "query params a and b (checkpoint ids) are required"}), 400
        checkpoint_manager = CheckpointManager(root)
        try:
            entry_a, entry_b, diff = checkpoint_manager.compare(a, b)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"entry_a": entry_a, "entry_b": entry_b, "diff": diff})

    @app.get("/api/checkpoints/<checkpoint_id>/inspect")
    def inspect_checkpoint(checkpoint_id: str):
        checkpoint_manager = CheckpointManager(root)
        try:
            entry = checkpoint_manager.show_entry(checkpoint_id)
            memory_snapshot = checkpoint_manager.inspect_memory(checkpoint_id)
            conversation = checkpoint_manager.inspect_conversation(checkpoint_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"entry": entry, "memory": memory_snapshot, "conversation": conversation})

    @app.post("/api/checkpoints/prune")
    def prune_checkpoints():
        body = request.get_json(silent=True) or {}
        keep_last, err = _parse_number(body.get("keep_last"), "keep_last", int)
        if err:
            return err
        checkpoint_manager = CheckpointManager(root)
        deleted = checkpoint_manager.prune(keep_last=keep_last)
        return jsonify({"deleted": deleted})

    # -- runs -- compare-branches -----------------------------------------

    @app.get("/api/runs/compare")
    def compare_runs():
        run_id_a, run_id_b = request.args.get("a"), request.args.get("b")
        if not run_id_a or not run_id_b:
            return jsonify({"error": "query params a and b (run ids) are required"}), 400
        run_store = RunStore(_runs_db_path())
        row_a, row_b = run_store.get(run_id_a), run_store.get(run_id_b)
        run_store.close()
        if row_a is None or row_b is None:
            missing = run_id_a if row_a is None else run_id_b
            return jsonify({"error": f"No run found with id {missing}."}), 404
        usage_store = UsageStore(_usage_db_path())
        cost_a = usage_store.summary(session_id=run_id_a)["total_cost"]
        cost_b = usage_store.summary(session_id=run_id_b)["total_cost"]
        usage_store.close()
        return jsonify({
            "run_a": {**dict(row_a), "estimated_cost": cost_a},
            "run_b": {**dict(row_b), "estimated_cost": cost_b},
        })

    # -- memory -------------------------------------------------------------

    @app.get("/api/memory")
    def list_memory():
        memory_store = MemoryStore(_memory_db_path())
        query = request.args.get("q", "")
        category = request.args.get("category") or None
        rows = memory_store.search(query=query, category=category, limit=200)
        memory_store.close()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/memory/<int:memory_id>/pin")
    def pin_memory(memory_id: int):
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.pin(memory_id)
        memory_store.close()
        return jsonify({"ok": ok})

    @app.post("/api/memory/<int:memory_id>/unpin")
    def unpin_memory(memory_id: int):
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.unpin(memory_id)
        memory_store.close()
        return jsonify({"ok": ok})

    @app.post("/api/memory/<int:memory_id>/edit")
    def edit_memory(memory_id: int):
        body = request.get_json(silent=True) or {}
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.edit(memory_id, title=body.get("title"), body=body.get("body"))
        memory_store.close()
        return jsonify({"ok": ok})

    @app.post("/api/memory/<int:memory_id>/forget")
    def forget_memory(memory_id: int):
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.forget(memory_id)
        memory_store.close()
        return jsonify({"ok": ok})

    @app.post("/api/memory/<int:old_id>/supersede/<int:new_id>")
    def supersede_memory(old_id: int, new_id: int):
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.supersede(old_id, new_id)
        memory_store.close()
        return jsonify({"ok": ok})

    @app.get("/api/memory/stale")
    def stale_memory():
        days = request.args.get("days", default=DEFAULT_STALE_DAYS, type=int)
        memory_store = MemoryStore(_memory_db_path())
        candidates = find_stale_candidates(memory_store, min_days=days)
        memory_store.close()
        return jsonify([dict(r) for r in candidates])

    @app.post("/api/memory/stale/archive")
    def archive_stale_memory():
        body = request.get_json(silent=True) or {}
        days, err = _parse_number(body.get("days"), "days", int, default=DEFAULT_STALE_DAYS)
        if err:
            return err
        memory_store = MemoryStore(_memory_db_path())
        candidates = find_stale_candidates(memory_store, min_days=days)
        summary = apply_archival(memory_store, candidates)
        memory_store.close()
        return jsonify({"archived": summary})

    @app.post("/api/memory/<int:memory_id>/unarchive")
    def unarchive_memory(memory_id: int):
        memory_store = MemoryStore(_memory_db_path())
        ok = memory_store.unarchive(memory_id)
        memory_store.close()
        return jsonify({"ok": ok})

    @app.get("/api/memory/archived")
    def list_archived_memory():
        memory_store = MemoryStore(_memory_db_path())
        rows = memory_store.list_archived()
        memory_store.close()
        return jsonify([dict(r) for r in rows])

    @app.get("/api/memory/stats")
    def memory_stats():
        memory_store = MemoryStore(_memory_db_path())
        stats = memory_store.stats()
        memory_store.close()
        # stats()'s "oldest"/"newest" are raw sqlite3.Row objects, not JSON-serializable.
        stats["oldest"] = dict(stats["oldest"]) if stats["oldest"] is not None else None
        stats["newest"] = dict(stats["newest"]) if stats["newest"] is not None else None
        return jsonify(stats)

    @app.get("/api/memory/why")
    def memory_why():
        query = request.args.get("q", "")
        limit = request.args.get("limit", default=15, type=int)
        memory_store = MemoryStore(_memory_db_path())
        explanations = explain_retrieval(memory_store, query=query, limit=limit)
        memory_store.close()
        return jsonify([
            {**{k: v for k, v in e.items() if k != "row"}, "row": dict(e["row"])}
            for e in explanations
        ])

    @app.post("/api/memory/consolidate")
    def consolidate_memory():
        body = request.get_json(silent=True) or {}
        threshold, err = _parse_number(body.get("threshold"), "threshold", float, default=FUZZY_DUPLICATE_THRESHOLD)
        if err:
            return err
        dry_run = bool(body.get("dry_run", True))
        memory_store = MemoryStore(_memory_db_path())
        clusters = find_duplicate_clusters(memory_store, threshold=threshold)
        if not clusters:
            memory_store.close()
            return jsonify({"clusters": []})
        if dry_run:
            preview = []
            for cluster in clusters:
                newest = max(cluster, key=lambda r: r["created_at"])
                others = [dict(r) for r in cluster if r["id"] != newest["id"]]
                preview.append({"keep": dict(newest), "merges": others})
            memory_store.close()
            return jsonify({"clusters": preview, "applied": False})
        summary = apply_consolidation(memory_store, clusters)
        memory_store.close()
        return jsonify({"clusters": summary, "applied": True})

    # -- runs / router --------------------------------------------------

    @app.get("/api/runs")
    def list_runs():
        run_store = RunStore(_runs_db_path())
        rows = run_store.list_runs(limit=50)
        run_store.close()
        return jsonify([dict(r) for r in rows])

    @app.get("/api/router/stats")
    def router_stats():
        run_store = RunStore(_runs_db_path())
        usage_store = UsageStore(_usage_db_path())
        task_type = request.args.get("task_type") or None
        stats = model_stats_by_task_type(run_store, usage_store, task_type)
        run_store.close()
        usage_store.close()
        return jsonify({
            "task_types": TASK_TYPES,
            "stats": [
                {
                    "model": s.model, "total": s.total, "wins": s.wins, "win_rate": s.win_rate,
                    "tested": s.tested, "passed": s.passed, "pass_rate": s.pass_rate,
                    "total_cost": s.total_cost, "avg_cost": s.avg_cost,
                }
                for s in stats
            ],
        })

    # -- skills ---------------------------------------------------------

    @app.get("/api/skills")
    def list_skills():
        skill_manager = SkillManager(root)
        return jsonify(skill_manager.list())

    @app.post("/api/skills/<name>/forget")
    def forget_skill(name: str):
        skill_manager = SkillManager(root)
        ok = skill_manager.delete(name)
        return jsonify({"ok": ok})

    # -- usage ------------------------------------------------------------

    @app.get("/api/usage")
    def usage_summary():
        usage_store = UsageStore(_usage_db_path())
        since_days = request.args.get("since_days", type=int)
        summary = usage_store.summary(since_days=since_days)
        usage_store.close()
        return jsonify(summary)

    # -- action log -------------------------------------------------------

    @app.get("/api/action-log")
    def action_log_sessions():
        action_log_store = ActionLogStore(_action_log_db_path())
        rows = action_log_store.list_sessions(limit=30)
        action_log_store.close()
        return jsonify([dict(r) for r in rows])

    @app.get("/api/action-log/<session_id>")
    def action_log_session_detail(session_id: str):
        action_log_store = ActionLogStore(_action_log_db_path())
        rows = action_log_store.session_actions(session_id)
        action_log_store.close()
        return jsonify([dict(r) for r in rows])

    # -- curator ------------------------------------------------------------
    # Configuration itself goes through the generic /api/config endpoints above
    # (curator_api_key/curator_model/curator_enabled are plain config keys, same
    # as any other -- curator_api_key is already masked via _SECRET_CONFIG_KEYS).
    # These are the curator-specific read/action endpoints.

    @app.get("/api/curator/status")
    def curator_status():
        if not curator_configured():
            return jsonify({"configured": False})
        result = {"configured": True, "model": curator_model(), "enabled": curator_enabled(), "areas": {}}
        db_path = _curator_db_path()
        if db_path.exists():
            curator_store = CuratorStore(db_path)
            for area_name in KNOWN_AREAS:
                watermark = curator_store.get_watermark(area_name)
                result["areas"][area_name] = (
                    dict(watermark) if watermark is not None and watermark["last_run_at"] is not None else None
                )
            curator_store.close()
        else:
            result["areas"] = {name: None for name in KNOWN_AREAS}
        return jsonify(result)

    @app.post("/api/curator/run")
    def start_curator_run():
        # Same guaranteed-no-op guarantee as `mazu curator run`: if unconfigured/
        # disabled, this returns immediately with no background thread, no
        # STDOUT_CAPTURE_LOCK contention, and no API call -- checked BEFORE
        # acquiring the lock so an unconfigured Curator can never block a real
        # run/explore/council from starting.
        if not curator_configured():
            return jsonify({"ran": False, "reason": "not_configured"})
        if not curator_enabled():
            return jsonify({"ran": False, "reason": "disabled"})

        body = request.get_json(silent=True) or {}
        areas = body.get("areas") or None
        if areas is not None and not isinstance(areas, list):
            return jsonify({"error": "areas must be a list of area names"}), 400

        max_cost, err = _parse_number(body.get("max_cost"), "max_cost", float)
        if err:
            return err
        max_rounds, err = _parse_number(body.get("max_rounds"), "max_rounds", int, default=8)
        if err:
            return err

        payload, status = _start_task(
            "curator",
            lambda: CuratorSession(
                root=root, areas=areas, full=bool(body.get("full", False)),
                dry_run=bool(body.get("dry_run", False)), max_cost=max_cost,
                max_rounds=max_rounds,
            ),
        )
        return jsonify(payload), status

    @app.get("/api/curator/log")
    def curator_log():
        db_path = _curator_db_path()
        if not db_path.exists():
            return jsonify([])
        curator_store = CuratorStore(db_path)
        area = request.args.get("area") or None
        since_days = request.args.get("since_days", type=int)
        limit = request.args.get("limit", default=50, type=int)
        rows = curator_store.log_recent(area=area, since_days=since_days, limit=limit)
        curator_store.close()
        return jsonify([dict(r) for r in rows])

    @app.get("/api/curator/report")
    def curator_report():
        db_path = _curator_db_path()
        if not db_path.exists():
            return jsonify({"error": "No curator runs yet in this project."}), 404
        curator_store = CuratorStore(db_path)
        run_id = request.args.get("run_id") or None
        run = curator_store.last_run(run_id)
        if run is None:
            curator_store.close()
            return jsonify({"error": "No matching curator run."}), 404
        entries = curator_store.log_for_run(run["id"])
        curator_store.close()
        return jsonify({"run": dict(run), "entries": [dict(e) for e in entries]})

    @app.post("/api/curator/undo/<int:log_id>")
    def curator_undo(log_id: int):
        # Same dispatch table as `mazu curator undo` -- reused directly from
        # mazu.cli, not reimplemented, so the two surfaces can never quietly
        # diverge on which actions are safely auto-reversible.
        from mazu.cli import _CURATOR_UNDO_MEMORY_ACTIONS, _CURATOR_UNDO_SKILL_ACTIONS

        db_path = _curator_db_path()
        if not db_path.exists():
            return jsonify({"error": "No curator runs yet in this project."}), 404
        curator_store = CuratorStore(db_path)
        row = curator_store.conn.execute("SELECT * FROM curator_log WHERE id = ?", (log_id,)).fetchone()
        curator_store.close()
        if row is None:
            return jsonify({"error": f"No curator log entry with id {log_id}."}), 404

        action = row["action"]
        if action in _CURATOR_UNDO_MEMORY_ACTIONS:
            memory_store = MemoryStore(_memory_db_path())
            ok = getattr(memory_store, _CURATOR_UNDO_MEMORY_ACTIONS[action])(int(row["target_id"]))
            memory_store.close()
            return jsonify({"ok": ok, "action": action, "target_id": row["target_id"]})
        if action in _CURATOR_UNDO_SKILL_ACTIONS:
            manager = SkillManager(root)
            ok = getattr(manager, _CURATOR_UNDO_SKILL_ACTIONS[action])(row["target_id"])
            return jsonify({"ok": ok, "action": action, "target_id": row["target_id"]})
        return jsonify({
            "ok": False, "action": action,
            "reversal_hint": row["reversal_hint"],
            "message": "No automatic undo for this action." + (
                " See reversal_hint for manual guidance." if row["reversal_hint"] else ""
            ),
        })

    @app.post("/api/curator/enable")
    def curator_enable():
        set_config_value("curator_enabled", "true")
        return jsonify({"ok": True, "enabled": True})

    @app.post("/api/curator/disable")
    def curator_disable():
        set_config_value("curator_enabled", "false")
        return jsonify({"ok": True, "enabled": False})

    # -- config -----------------------------------------------------------

    @app.get("/api/config")
    def get_config():
        # list_config() returns raw values -- masking secrets is the caller's job
        # (mazu/cli.py's own `mazu config list` does the same _mask_secret call).
        # An API key/token must never round-trip through the browser in the clear.
        raw = list_config()
        return jsonify({
            key: (_mask_secret(value) if key in _SECRET_CONFIG_KEYS else value)
            for key, value in raw.items()
        })

    @app.post("/api/config")
    def update_config():
        body = request.get_json(silent=True) or {}
        key, value = body.get("key"), body.get("value")
        if not key or value is None:
            return jsonify({"error": "key and value are required"}), 400
        try:
            set_config_value(key, str(value))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True})

    @app.delete("/api/config/<key>")
    def delete_config(key: str):
        was_set = unset_config_value(key)
        return jsonify({"ok": True, "was_set": was_set})

    # -- models / doctor --------------------------------------------------

    @app.get("/api/models")
    def models_info():
        rows = list_capabilities()
        return jsonify([dataclasses.asdict(r) for r in rows])

    @app.get("/api/doctor")
    def doctor():
        # --live makes one minimal real API call per configured provider (costs a
        # fraction of a cent) -- opt-in via query param, default off, same as the
        # CLI's own --live flag defaulting to False.
        live = request.args.get("live") == "true"
        results = run_diagnostics(root, live=live)
        return jsonify([dataclasses.asdict(r) for r in results])

    return app


def _sse_stream(outbox: "queue.Queue[dict]", end_type: str):
    yield ": connected\n\n"
    while True:
        try:
            event = outbox.get(timeout=15)
        except queue.Empty:
            yield ": keepalive\n\n"
            continue
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") == end_type:
            return
