import dataclasses
import json
import queue
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import _SECRET_CONFIG_KEYS, list_config, set_config_value
from mazu.diagnostics import run_diagnostics
from mazu.llm.capabilities import list_capabilities
from mazu.memory.consolidate import apply_consolidation, find_duplicate_clusters
from mazu.memory.retrieval import explain_retrieval
from mazu.memory.store import FUZZY_DUPLICATE_THRESHOLD, MemoryStore
from mazu.runs.router import TASK_TYPES, model_stats_by_task_type
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

from mazu_web.chat_session import ChatSession
from mazu_web.task_session import STDOUT_CAPTURE_LOCK, CouncilSession, ExploreSession, RunSession

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

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

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
        body = request.get_json(silent=True) or {}
        task = (body.get("task") or "").strip()
        if not task:
            return jsonify({"error": "task is required"}), 400
        payload, status = _start_task(
            "run",
            lambda: RunSession(
                root=root, task=task, model=body.get("model") or model,
                max_steps=int(body.get("max_steps") or 15),
                allow_shell=bool(body.get("allow_shell", False)),
                max_cost=body.get("max_cost"),
            ),
        )
        return jsonify(payload), status

    @app.post("/api/explore")
    def start_explore():
        body = request.get_json(silent=True) or {}
        task = (body.get("task") or "").strip()
        models = [m.strip() for m in (body.get("models") or "").split(",") if m.strip()]
        if not task or not models:
            return jsonify({"error": "task and at least one model are required"}), 400
        payload, status = _start_task(
            "explore",
            lambda: ExploreSession(
                root=root, task=task, models=models,
                test_command=body.get("test_command") or None,
                max_cost=body.get("max_cost"),
                max_steps=int(body.get("max_steps") or 15),
            ),
        )
        return jsonify(payload), status

    @app.post("/api/council")
    def start_council():
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        models = [m.strip() for m in (body.get("models") or "").split(",") if m.strip()]
        lead_model = (body.get("lead_model") or "").strip() or (models[0] if models else None)
        if not question or not models or not lead_model:
            return jsonify({"error": "question and at least one model are required"}), 400
        payload, status = _start_task(
            "council",
            lambda: CouncilSession(
                root=root, question=question, models=models, lead_model=lead_model,
                max_cost=body.get("max_cost"),
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
        keep_last = body.get("keep_last")
        checkpoint_manager = CheckpointManager(root)
        deleted = checkpoint_manager.prune(keep_last=int(keep_last) if keep_last is not None else None)
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
        threshold = float(body.get("threshold") or FUZZY_DUPLICATE_THRESHOLD)
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
