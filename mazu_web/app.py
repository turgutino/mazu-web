import json
import queue
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import _SECRET_CONFIG_KEYS, list_config, set_config_value
from mazu.memory.store import MemoryStore
from mazu.runs.router import TASK_TYPES, model_stats_by_task_type
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

from mazu_web.chat_session import ChatSession
from mazu_web.task_session import STDOUT_CAPTURE_LOCK, ExploreSession, RunSession

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
    task_sessions: dict[str, "RunSession | ExploreSession"] = {}
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
            return {"error": f"Another run/explore is already in progress on this server -- wait for it to finish."}, 409
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
