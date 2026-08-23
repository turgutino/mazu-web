"""Verifies mazu-web's chat-history persistence: every message a ChatSession
sends/receives is durably saved to ChatStore as it happens (not just held in
memory), and /api/chat/start can resume a past session two ways -- reconnecting
to a still-live in-process ChatSession, or rebuilding one from ChatStore after
the server restarted (or the session was simply never in this process).
"""

import subprocess
import time
from pathlib import Path

import pytest

import mazu_web.chat_session as chat_session_module
from mazu.chat.store import ChatStore
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
    return tmp_path


def _end_turn_stream(messages, system, tools, on_delta, model=None):
    on_delta("hello ")
    on_delta("there")
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "hello there"}], usage={})


def _wait_for(outbox, event_type, timeout=10):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_a_full_chat_turn_is_persisted_to_chat_store(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    chat_store = ChatStore(project / ".mazu" / "chat_history.db")
    messages = chat_store.get_messages(session_id)
    chat_store.close()
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello there"}]},
    ]


def test_list_chat_sessions_is_empty_before_any_chat_happens(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/chat/sessions")
    assert res.status_code == 200
    assert res.get_json() == []


def test_list_chat_sessions_shows_a_completed_session(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    res = client.get("/api/chat/sessions")
    rows = res.get_json()
    assert len(rows) == 1
    assert rows[0]["session_id"] == session_id
    assert rows[0]["title"] == "hi"
    assert rows[0]["message_count"] == 2


def test_get_session_messages_endpoint_returns_the_saved_transcript(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    res = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert res.status_code == 200
    assert res.get_json() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello there"}]},
    ]


def test_get_session_messages_for_unknown_session_is_an_empty_list(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/chat/sessions/nonexistent/messages")
    assert res.status_code == 200
    assert res.get_json() == []


def test_resume_reconnects_to_a_still_live_session_without_rebuilding_it(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    live_session = app.sessions[session_id]

    res = client.post("/api/chat/start", json={"resume": session_id})
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] == session_id
    assert data["reconnected"] is True
    assert app.sessions[session_id] is live_session  # the exact same object, not a rebuild
    assert len(app.sessions) == 1  # no orphan second session created

    live_session.close()
    live_session.join(timeout=5)


def test_resume_rebuilds_from_chat_store_when_the_session_is_no_longer_live(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    original = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "earlier question"})
    _wait_for(original.outbox, "turn_done")
    original.close()
    original.join(timeout=5)
    del app.sessions[session_id]  # simulates the server having restarted / session evicted

    res = client.post("/api/chat/start", json={"resume": session_id})
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] == session_id  # SAME id, not a freshly minted one
    assert data["reconnected"] is False

    rebuilt = app.sessions[session_id]
    assert rebuilt.messages == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello there"}]},
    ]
    rebuilt.close()
    rebuilt.join(timeout=5)


def test_resume_of_a_session_with_no_saved_messages_is_a_clean_404(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/chat/start", json={"resume": "never-existed"})
    assert res.status_code == 404
    assert "No saved messages found" in res.get_json()["error"]


def test_resumed_session_continues_appending_to_the_same_saved_history(project, monkeypatch):
    # The real end-to-end guarantee: nothing lost across a full stop/resume cycle,
    # AND the resumed conversation keeps growing under the same session_id rather
    # than forking into a second, disconnected transcript.
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    first = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "first question"})
    _wait_for(first.outbox, "turn_done")
    first.close()
    first.join(timeout=5)
    del app.sessions[session_id]

    client.post("/api/chat/start", json={"resume": session_id})
    second = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "second question"})
    _wait_for(second.outbox, "turn_done")
    second.close()
    second.join(timeout=5)

    chat_store = ChatStore(project / ".mazu" / "chat_history.db")
    messages = chat_store.get_messages(session_id)
    chat_store.close()
    assert [m["content"] for m in messages if m["role"] == "user"] == ["first question", "second question"]


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/<id>
# ---------------------------------------------------------------------------


def test_delete_session_removes_it_from_history(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]
    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")

    res = client.delete(f"/api/chat/sessions/{session_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["was_deleted"] is True

    assert client.get("/api/chat/sessions").get_json() == []


def test_delete_session_closes_a_still_live_session(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.delete(f"/api/chat/sessions/{session_id}")

    assert session_id not in app.sessions
    session.join(timeout=5)  # close() was called on it -- its background thread must exit


def test_delete_session_that_never_existed_is_a_clean_no_op(project):
    app = create_app(project, None, None)
    res = app.test_client().delete("/api/chat/sessions/never-existed")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["was_deleted"] is False
