"""Tests for image (vision) input on mazu-web's chat message endpoint. Mirrors
test_web_chat_history.py's fixtures -- this file only adds the `images` field
to that same POST /api/chat/<id>/message flow.
"""

import base64
import subprocess
import time

import pytest

import mazu_web.chat_session as chat_session_module
from mazu.chat.store import ChatStore
from mazu.llm.types import AgentResponse
from mazu_web.app import create_app

_FAKE_PNG_BASE64 = base64.b64encode(b"not a real png, just test bytes").decode("ascii")


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
    on_delta("I see an image")
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "I see an image"}], usage={})


def _wait_for(outbox, event_type, timeout=10):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_image_with_caption_is_persisted_as_canonical_content_blocks(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    res = client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "what is this?", "images": [{"media_type": "image/png", "data": _FAKE_PNG_BASE64}]},
    )
    assert res.status_code == 200
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    chat_store = ChatStore(project / ".mazu" / "chat_history.db")
    messages = chat_store.get_messages(session_id)
    chat_store.close()

    assert messages[0] == {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _FAKE_PNG_BASE64},
            },
            {"type": "text", "text": "what is this?"},
        ],
    }


def test_image_only_no_caption_omits_text_block(project, monkeypatch):
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "", "images": [{"media_type": "image/png", "data": _FAKE_PNG_BASE64}]},
    )
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    chat_store = ChatStore(project / ".mazu" / "chat_history.db")
    messages = chat_store.get_messages(session_id)
    chat_store.close()

    assert messages[0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _FAKE_PNG_BASE64}}
    ]


def test_no_images_field_still_works_as_plain_string(project, monkeypatch):
    # Backward compatibility: existing text-only clients that never send
    # `images` at all must keep getting the old bare-string content shape.
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
    assert messages[0] == {"role": "user", "content": "hi"}


def test_unsupported_media_type_rejected_with_400(project):
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]

    res = client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "hi", "images": [{"media_type": "image/bmp", "data": _FAKE_PNG_BASE64}]},
    )
    assert res.status_code == 400
    assert "unsupported image media_type" in res.get_json()["error"]

    app.sessions[session_id].close()


def test_oversized_image_rejected_with_400(project):
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]

    huge_data = "A" * 7_000_001
    res = client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "hi", "images": [{"media_type": "image/png", "data": huge_data}]},
    )
    assert res.status_code == 400
    assert "too large" in res.get_json()["error"]

    app.sessions[session_id].close()


def test_malformed_image_entry_rejected_with_400(project):
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]

    res = client.post(
        f"/api/chat/{session_id}/message",
        json={"text": "hi", "images": [{"media_type": "image/png"}]},  # missing "data"
    )
    assert res.status_code == 400

    app.sessions[session_id].close()
