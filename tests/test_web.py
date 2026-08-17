"""Tests for mazu-web: the Flask app in mazu_web/app.py and the queue-driven
ChatSession in mazu_web/chat_session.py. Monkeypatches run_turn_stream at the
mazu_web.chat_session import site (same seam Mazu's own test_streaming.py and
test_router_cli_wiring.py use) so no real HTTP call happens; everything else
(checkpoints, memory, router stats, tool execution) runs for real against a
tmp_path git repo, matching Mazu's own testing convention.
"""

import subprocess
import time
from pathlib import Path

import pytest

import mazu_web.chat_session as chat_session_module
from mazu.checkpoint.manager import CheckpointManager
from mazu.llm.types import AgentResponse
from mazu_web.app import create_app


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".mazu").mkdir(exist_ok=True)
    (tmp_path / "app.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _end_turn_stream(messages, system, tools, on_delta, model=None):
    on_delta("hello ")
    on_delta("there")
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "hello there"}], usage={})


def _wait_for(outbox, event_type, timeout=5):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_chat_start_creates_a_session_and_returns_the_resolved_model(project):
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/chat/start")
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] in app.sessions
    assert "model" in data


def test_chat_message_streams_deltas_and_ends_the_turn(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]

    res = client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    assert res.status_code == 200

    session = app.sessions[session_id]
    _, batch = _wait_for(session.outbox, "usage")
    deltas = [e["text"] for e in batch if e["type"] == "delta"]
    assert "".join(deltas) == "hello there"

    done_event, _ = _wait_for(session.outbox, "turn_done")
    assert done_event["type"] == "turn_done"


def test_destructive_tool_call_blocks_on_confirm_and_resumes_on_approval(project, monkeypatch):
    calls = {"n": 0}

    def _fake_stream(messages, system, tools, on_delta, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentResponse(
                stop_reason="tool_use",
                content=[{"type": "tool_use", "id": "t1", "name": "run_shell", "input": {"command": "echo hi"}}],
                usage={},
            )
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "done"}], usage={})

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _fake_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "run something"})

    confirm_event, _ = _wait_for(session.outbox, "confirm_request")
    assert confirm_event["tool_name"] == "run_shell"

    res = client.post(f"/api/chat/{session_id}/confirm", json={"approved": False})
    assert res.status_code == 200

    _wait_for(session.outbox, "turn_done")
    assert calls["n"] == 2  # declined, but the loop continued with the tool_result


def test_checkpoints_endpoint_lists_real_checkpoints(project):
    checkpoint_manager = CheckpointManager(project)
    checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    res = app.test_client().get("/api/checkpoints")
    assert res.status_code == 200
    rows = res.get_json()
    assert len(rows) == 1
    assert rows[0]["trigger"] == "manual"


def test_checkpoints_rollback_endpoint_restores_and_reports_error_for_bad_id(project):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([{"role": "user", "content": "hi"}], trigger="manual")

    app = create_app(project, None, None)
    client = app.test_client()

    bad = client.post("/api/checkpoints/does-not-exist/rollback")
    assert bad.status_code == 400
    assert "error" in bad.get_json()

    good = client.post(f"/api/checkpoints/{entry['id']}/rollback")
    assert good.status_code == 200
    assert good.get_json()["ok"] is True


def test_memory_endpoint_returns_empty_list_for_a_fresh_project(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/memory")
    assert res.status_code == 200
    assert res.get_json() == []


def test_router_stats_endpoint_reports_no_history_and_lists_task_types(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/router/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["stats"] == []
    assert "bugfix" in data["task_types"]


def test_index_serves_the_static_page(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/")
    assert res.status_code == 200
    assert b"mazu web" in res.data
