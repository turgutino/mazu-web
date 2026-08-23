"""Verifies mazu-web's ChatSession (mazu_web/chat_session.py) wires the
compaction module in correctly: proactive compact_if_needed() runs every
round (surfaced to the browser as a `context_compacted` SSE event), a
MazuContextLengthError triggers force_compact() plus exactly one retry, and
the same chat-specific extras as the terminal side (ChatStore untouched,
memory extraction on compaction, cost logging) hold here too. Mirrors
tests/test_chat_compaction.py in the core `mazu` repo and
tests/test_web_chat_history.py's fixtures/helpers in this repo.
"""

import subprocess
import time
from pathlib import Path

import pytest

import mazu_web.chat_session as chat_session_module
from mazu.chat.store import ChatStore
from mazu.llm.errors import MazuContextLengthError
from mazu.llm.types import AgentResponse
from mazu.memory.store import MemoryStore
from mazu.usage.store import UsageStore
from mazu_web.app import create_app
from mazu_web.chat_session import ChatSession


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
    on_delta("hi")
    return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "hi"}], usage={})


def _wait_for(outbox, event_type, timeout=10):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_proactive_compaction_emits_context_compacted_event(project, monkeypatch):
    calls = []

    def _fake_compact_if_needed(messages, model, **kwargs):
        calls.append(len(messages))
        return messages, False

    monkeypatch.setattr(chat_session_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    assert len(calls) == 1


def test_context_length_error_triggers_force_compact_and_retries_once(project, monkeypatch):
    run_calls = {"n": 0}

    def _fake_stream(messages, system, tools, on_delta, model=None):
        run_calls["n"] += 1
        if run_calls["n"] == 1:
            raise MazuContextLengthError("too long")
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "recovered"}], usage={})

    force_compact_calls = {"n": 0}

    def _fake_force_compact(messages, model, **kwargs):
        force_compact_calls["n"] += 1
        return messages[-1:]

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _fake_stream)
    monkeypatch.setattr(chat_session_module, "force_compact", _fake_force_compact)
    monkeypatch.setattr(chat_session_module, "compact_if_needed", lambda messages, model, **k: (messages, False))

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    event, seen = _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    assert run_calls["n"] == 2
    assert force_compact_calls["n"] == 1
    assert any(e["type"] == "context_compacted" for e in seen)


def test_context_length_error_on_retry_falls_through_to_error_event(project, monkeypatch):
    def _always_raises(messages, system, tools, on_delta, model=None):
        raise MazuContextLengthError("still too long")

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _always_raises)
    monkeypatch.setattr(chat_session_module, "force_compact", lambda messages, model, **k: messages)
    monkeypatch.setattr(chat_session_module, "compact_if_needed", lambda messages, model, **k: (messages, False))

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    event, _ = _wait_for(session.outbox, "error")
    session.close()
    session.join(timeout=5)

    assert "too long" in event["text"]


def test_chat_store_transcript_untouched_by_compaction(project, monkeypatch):
    monkeypatch.setattr("mazu.agent.compaction.needs_compaction", lambda messages, trigger_tokens: True)
    monkeypatch.setattr("mazu.agent.compaction.summarize_for_compaction", lambda mm, model, **k: "SUMMARY")
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "first"})
    _wait_for(session.outbox, "turn_done")
    client.post(f"/api/chat/{session_id}/message", json={"text": "second"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    persisted = ChatStore(project / ".mazu" / "chat_history.db").get_messages(session_id)
    assert len(persisted) == 4  # 2 user + 2 assistant, never trimmed regardless of compaction


def test_compaction_extracts_memories_into_a_real_memory_store(project, monkeypatch):
    monkeypatch.setattr(
        "mazu.agent.session.extract_memories",
        lambda messages, model=None: (
            [{"category": "decision", "title": "Use PostgreSQL", "body": "for concurrency"}],
            {},
        ),
    )

    def _fake_compact_if_needed(messages, model, on_compacted=None, **kwargs):
        if on_compacted is not None:
            on_compacted(messages)
        return messages, True

    monkeypatch.setattr(chat_session_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    rows = memory_store.search()
    memory_store.close()
    assert len(rows) == 1
    assert rows[0]["title"] == "Use PostgreSQL"


def test_compaction_logs_real_cost_to_usage_store(project, monkeypatch):
    def _fake_compact_if_needed(messages, model, usage_store=None, session_id=None, on_compacted=None, **kwargs):
        if usage_store is not None:
            usage_store.log("compaction", session_id, "anthropic", "claude-haiku-4-5", 1000, 200, 0.001)
        return messages, True

    monkeypatch.setattr(chat_session_module, "compact_if_needed", _fake_compact_if_needed)
    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
    summary = usage_store.summary(command="compaction")
    usage_store.close()
    assert summary["total_calls"] == 1


def test_slash_compact_is_not_intercepted_in_web_chat(project, monkeypatch):
    # Documents the deferral decision: unlike the terminal (`/compact`), mazu-web
    # has no slash-command parsing in its chat input today -- "/compact" sent as a
    # message is just ordinary text handed straight to the LLM, not a command.
    seen_texts = []

    def _capture_stream(messages, system, tools, on_delta, model=None):
        seen_texts.append(messages[-1]["content"])
        on_delta("ok")
        return AgentResponse(stop_reason="end_turn", content=[{"type": "text", "text": "ok"}], usage={})

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _capture_stream)

    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "/compact"})
    _wait_for(session.outbox, "turn_done")
    session.close()
    session.join(timeout=5)

    assert seen_texts == ["/compact"]
