"""Real bug found via live testing: sending an image to a model that doesn't
support vision returned a normal 400 error for that message, but the failed
image-carrying user message stayed stuck in ChatSession.messages forever.
Since every LLM call resends the full history, EVERY later message in that
session -- including plain text ones with no image at all -- kept re-triggering
the exact same "this model does not support image" error, permanently breaking
the conversation until a new chat was started. This verifies chat_session.py's
messages_before_turn rollback actually fixes that.
"""

import subprocess
import time

import pytest

import mazu_web.chat_session as chat_session_module
from mazu.llm.errors import MazuAPIError
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


def _wait_for(outbox, event_type, timeout=10):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_failed_turn_does_not_poison_the_next_turn(project, monkeypatch):
    calls_seen = []

    def _fake_stream(messages, system, tools, on_delta, model=None):
        calls_seen.append([dict(m) for m in messages])
        if len(calls_seen) == 1:
            raise MazuAPIError("this model does not support image")
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "ok"}], usage={})

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _fake_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "", "images": [{"media_type": "image/png", "data": "ZmFrZQ=="}]},
    )
    _wait_for(session.outbox, "error")

    client.post(f"/api/chat/{session_id}/message", json={"text": "plain follow-up"})
    _wait_for(session.outbox, "turn_done")

    session.close()
    session.join(timeout=5)

    assert len(calls_seen) == 2
    second_call_messages = calls_seen[1]
    # The failed image turn must be gone -- otherwise the second (plain-text,
    # image-free) call would still carry the image block and re-fail identically.
    assert not any(
        isinstance(m.get("content"), list) and any(b.get("type") == "image" for b in m["content"])
        for m in second_call_messages
    )
    assert second_call_messages == [{"role": "user", "content": "plain follow-up"}]


def test_chat_store_still_keeps_the_failed_turn(project, monkeypatch):
    # ChatStore is the durable "nothing is lost" record -- unlike the live
    # in-memory messages list, the failed attempt must still be visible in
    # Chat History even though it's excluded from what's resent to the model.
    def _always_fails(messages, system, tools, on_delta, model=None):
        raise MazuAPIError("this model does not support image")

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _always_fails)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "", "images": [{"media_type": "image/png", "data": "ZmFrZQ=="}]},
    )
    _wait_for(session.outbox, "error")
    session.close()
    session.join(timeout=5)

    from mazu.chat.store import ChatStore

    chat_store = ChatStore(project / ".mazu" / "chat_history.db")
    messages = chat_store.get_messages(session_id)
    chat_store.close()
    assert len(messages) == 1
    assert messages[0]["content"][0]["type"] == "image"
