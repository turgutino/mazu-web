import json
import queue
from pathlib import Path

from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.runs.router import TASK_TYPES, model_stats_by_task_type
from mazu.runs.store import RunStore
from mazu.usage.store import UsageStore

from mazu_web.chat_session import ChatSession

STATIC_DIR = Path(__file__).parent / "static"


def create_app(root: Path, model: str | None, shell_allowlist: list[str] | None):
    """Builds the Flask app for `mazu web`. Every store/manager here is opened once
    per server process and reused across requests/sessions -- the same lifetime
    `mazu ui` (the textual UI) already gives them, just behind HTTP instead of a
    terminal event loop.
    """
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)
    sessions: dict[str, ChatSession] = {}
    app.sessions = sessions  # exposed for tests; not used by any route itself

    def _memory_db_path() -> Path:
        return root / ".mazu" / "memory.db"

    def _runs_db_path() -> Path:
        return root / ".mazu" / "runs.db"

    def _usage_db_path() -> Path:
        return Path.home() / ".mazu" / "usage.db"

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # -- chat -----------------------------------------------------------

    @app.post("/api/chat/start")
    def chat_start():
        session = ChatSession(root=root, model=model, shell_allowlist=shell_allowlist)
        sessions[session.session_id] = session
        return jsonify({"session_id": session.session_id, "model": session.resolved_model})

    @app.post("/api/chat/<session_id>/message")
    def chat_message(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        text = (request.get_json(silent=True) or {}).get("text", "")
        session.send(text)
        return jsonify({"ok": True})

    @app.post("/api/chat/<session_id>/confirm")
    def chat_confirm(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        approved = bool((request.get_json(silent=True) or {}).get("approved", False))
        session.answer_confirm(approved)
        return jsonify({"ok": True})

    @app.get("/api/chat/<session_id>/events")
    def chat_events(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            return jsonify({"error": "unknown session"}), 404

        def stream():
            yield ": connected\n\n"
            while True:
                try:
                    event = session.outbox.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "closed":
                    return

        return Response(stream(), mimetype="text/event-stream")

    # -- checkpoints ------------------------------------------------------

    @app.get("/api/checkpoints")
    def list_checkpoints():
        checkpoint_manager = CheckpointManager(root)
        return jsonify(checkpoint_manager.timeline_entries())

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

    return app
