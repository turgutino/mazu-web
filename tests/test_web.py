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


def _wait_for(outbox, event_type, timeout=20):
    # 20s, not 5s or even the previously-bumped 10s: Windows CI runners have
    # repeatedly (seen live, not hypothetical) taken long enough on these
    # background-thread queue tests to exceed even a 10s window, even with
    # everything mocked at the run_turn_stream layer -- not specific to any one
    # test, so widened globally here (and in every other test file with its own
    # copy of this helper) rather than bumping timeouts test-by-test as each one
    # happens to flake.
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        event = outbox.get(timeout=deadline - time.time())
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(f"never saw {event_type}, saw {seen}")


def test_reasoning_effort_support_is_model_precise_for_openai(project):
    # Real bug found live: a per-PROVIDER-only check left the Chat page's
    # effort chip "active" for openai:gpt-4o (a non-reasoning OpenAI model that
    # rejects reasoning_effort with a 400) since OpenAI-the-provider does
    # support the parameter for SOME of its own models (o-series, gpt-5).
    app = create_app(project, None, None)
    client = app.test_client()
    assert client.get("/api/providers/reasoning-effort-support?model=openai:o3-mini").get_json() == {"supported": True}
    assert client.get("/api/providers/reasoning-effort-support?model=openai:gpt-4o").get_json() == {"supported": False}
    assert client.get("/api/providers/reasoning-effort-support?model=deepseek:deepseek-chat").get_json() == {"supported": False}
    assert client.get("/api/providers/reasoning-effort-support?model=anthropic:claude-sonnet-5").get_json() == {"supported": True}


def test_reasoning_effort_support_fails_open_for_malformed_or_unknown_input(project):
    app = create_app(project, None, None)
    client = app.test_client()
    assert client.get("/api/providers/reasoning-effort-support").get_json() == {"supported": True}
    assert client.get("/api/providers/reasoning-effort-support?model=not-a-real-model").get_json() == {"supported": True}
    assert client.get("/api/providers/reasoning-effort-support?model=nosuchprovider:x").get_json() == {"supported": True}


def test_vision_support_reflects_deepseek_s_real_model_split(project):
    # Real bug found live: attaching an image to a plain deepseek:deepseek-chat
    # session produced a raw provider error only after sending -- nothing
    # warned upfront that the model is text-only.
    app = create_app(project, None, None)
    client = app.test_client()
    assert client.get("/api/providers/vision-support?model=deepseek:deepseek-chat").get_json() == {"supported": False}
    assert client.get("/api/providers/vision-support?model=deepseek:deepseek-v4-flash-vision-exp").get_json() == {"supported": True}
    assert client.get("/api/providers/vision-support?model=anthropic:claude-sonnet-5").get_json() == {"supported": True}


def test_set_config_rejects_a_malformed_council_lead_value(project):
    # Real bug found live: the generic Config "Set a value" form accepted
    # council_lead/council_models with zero format validation -- a typo
    # silently saved to config.toml and only surfaced later as a confusing,
    # disconnected failure the next time Council actually ran. Validation now
    # lives in mazu.config.set_config_value (see its own tests for full
    # coverage); this just confirms the web route surfaces it as a clean 400,
    # same as any other ValueError from that function.
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/config", json={"key": "council_lead", "value": ":claude-opus-4-8"})
    assert res.status_code == 400
    assert "malformed" in res.get_json()["error"]


def test_cost_trackable_reports_which_models_have_no_pricing_data(project):
    # Real bug found live: --max-cost is silently a no-op for a model
    # estimate_cost() has no pricing entry for (a newly-discovered live model
    # most often), and nothing in the UI said so -- a user could type "$2
    # limit" and it would just never apply. This endpoint is what the Run/
    # Explore/Council pages now poll to warn about that before the user starts
    # a real, unexpectedly-unlimited run.
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post(
        "/api/cost-trackable",
        json={"models": ["anthropic:claude-sonnet-5", "deepseek:deepseek-v4-pro-not-a-real-model"]},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data == {
        "anthropic:claude-sonnet-5": True,
        "deepseek:deepseek-v4-pro-not-a-real-model": False,
    }


def test_cost_trackable_with_no_models_returns_empty(project):
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/cost-trackable", json={"models": []})
    assert res.status_code == 200
    assert res.get_json() == {}


def test_chat_start_creates_a_session_and_returns_the_resolved_model(project):
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/chat/start")
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] in app.sessions
    assert "model" in data

    app.sessions[data["session_id"]].close()
    app.sessions[data["session_id"]].join(timeout=5)


def test_chat_set_model_updates_the_live_session_without_a_new_chat(project):
    # Real bug found live: the Chat page's model chip only ever wrote
    # config.toml's default_model, which a session already on screen never
    # re-reads (resolved_model is resolved once in __init__ and reused for
    # every turn) -- so picking a new model appeared to do nothing until the
    # user started a brand new chat. This is the fix: a dedicated endpoint that
    # changes the *current* session's model immediately.
    app = create_app(project, None, None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]
    original_model = session.resolved_model

    res = client.post(f"/api/chat/{session_id}/model", json={"model": "openai:gpt-4.1"})
    assert res.status_code == 200
    assert res.get_json() == {"ok": True, "model": "openai:gpt-4.1"}
    assert session.model == "openai:gpt-4.1"
    assert session.resolved_model == "openai:gpt-4.1"
    assert session.resolved_model != original_model

    session.close()
    session.join(timeout=5)


def test_chat_set_model_on_unknown_session_returns_404(project):
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/chat/does-not-exist/model", json={"model": "openai:gpt-4.1"})
    assert res.status_code == 404


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

    session.close()
    session.join(timeout=5)


def test_chat_session_picks_up_a_key_saved_via_config_toml_not_just_env_vars(project, monkeypatch):
    """Real bug caught live: mazu-web is a long-running server process, not a
    per-invocation CLI call like `mazu chat` -- nothing had ever called
    load_config() here, so a key saved via `mazu setup`/`mazu config set` (or this
    app's own Config tab) sat in config.toml but never reached this process's
    environment. `mazu chat` in a terminal worked fine with the exact same key;
    mazu-web errored "DEEPSEEK_API_KEY is not set." ChatSession._run() now calls
    load_config() itself, same as every real key-resolution path already does.
    """
    import os

    from mazu.config import set_config_value

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_config_value("deepseek_api_key", "sk-fake-from-config-toml")

    monkeypatch.setattr(chat_session_module, "run_turn_stream", _end_turn_stream)
    app = create_app(project, "deepseek:deepseek-chat", None)
    client = app.test_client()
    session_id = client.post("/api/chat/start").get_json()["session_id"]
    session = app.sessions[session_id]

    client.post(f"/api/chat/{session_id}/message", json={"text": "hi"})
    _wait_for(session.outbox, "turn_done")

    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-fake-from-config-toml"

    session.close()
    session.join(timeout=5)


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

    session.close()
    session.join(timeout=5)


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
