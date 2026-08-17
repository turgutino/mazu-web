"""Tests for the feature-parity endpoints added on top of the original chat/
checkpoints/memory/router MVP: run (autonomous), explore, skills, usage,
action-log, config, checkpoint diff, and memory pin/unpin/edit. Monkeypatches
run_autonomous/run_explore at the mazu_web.task_session import site so no real
agent/API call happens -- same seam pattern as test_web.py's run_turn_stream
monkeypatching.
"""

import subprocess
import time
from pathlib import Path

import pytest

import mazu_web.task_session as task_session_module
from mazu.checkpoint.manager import CheckpointManager
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu_web.app import create_app
from mazu_web.task_session import STDOUT_CAPTURE_LOCK


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


@pytest.fixture(autouse=True)
def _release_stray_lock():
    # Belt-and-suspenders: a failed test elsewhere in the run must never leave
    # the process-wide lock held, or every later run/explore test would 409.
    yield
    if STDOUT_CAPTURE_LOCK.locked():
        STDOUT_CAPTURE_LOCK.release()


def _drain(outbox, timeout=5):
    deadline = time.time() + timeout
    events = []
    while time.time() < deadline:
        event = outbox.get(timeout=max(0.01, deadline - time.time()))
        events.append(event)
        if event["type"] == "done":
            return events
    raise AssertionError(f"never saw done, saw {events}")


# ---------------------------------------------------------------------------
# run / explore -- shared stdout-capture slot
# ---------------------------------------------------------------------------


def test_start_run_streams_captured_print_output_and_releases_the_lock(project, monkeypatch):
    def _fake_run_autonomous(*args, **kwargs):
        print("step 1")
        print("done")

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/run", json={"task": "fix the bug"})
    assert res.status_code == 200
    task_id = res.get_json()["task_id"]

    session = app.task_sessions[task_id]
    events = _drain(session.outbox)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert lines == ["step 1", "done"]
    assert not STDOUT_CAPTURE_LOCK.locked()


def test_start_run_rejects_a_second_concurrent_run(project, monkeypatch):
    started = {"n": 0}

    def _fake_run_autonomous(*args, **kwargs):
        started["n"] += 1
        time.sleep(0.3)

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    first = client.post("/api/run", json={"task": "fix the bug"})
    assert first.status_code == 200

    second = client.post("/api/run", json={"task": "fix another bug"})
    assert second.status_code == 409

    session = app.task_sessions[first.get_json()["task_id"]]
    _drain(session.outbox)


def test_start_run_requires_a_task(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/run", json={})
    assert res.status_code == 400


def test_start_explore_streams_the_formatted_report(project, monkeypatch):
    def _fake_run_explore(task, models, root, checkpoint_manager, **kwargs):
        print("[cost] some progress line")
        return [
            {"model": models[0], "error": None, "test_passed": None, "session_id": "s1", "branch_name": "b1"},
        ]

    def _fake_format_report(results, test_command):
        return "RANKED REPORT\n1. winner"

    monkeypatch.setattr(task_session_module, "run_explore", _fake_run_explore)
    monkeypatch.setattr(task_session_module, "format_explore_report", _fake_format_report)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/explore", json={"task": "fix it", "models": "anthropic:claude-sonnet-5,deepseek:deepseek-chat"})
    assert res.status_code == 200
    task_id = res.get_json()["task_id"]

    session = app.task_sessions[task_id]
    events = _drain(session.outbox)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert "[cost] some progress line" in lines
    assert "RANKED REPORT" in lines


def test_start_explore_requires_task_and_models(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/explore", json={"task": "fix it", "models": ""})
    assert res.status_code == 400


def test_tasks_busy_endpoint_reflects_lock_state(project, monkeypatch):
    def _fake_run_autonomous(*args, **kwargs):
        time.sleep(0.3)

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    assert client.get("/api/tasks/busy").get_json()["busy"] is False
    res = client.post("/api/run", json={"task": "fix the bug"})
    assert client.get("/api/tasks/busy").get_json()["busy"] is True

    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session.outbox)
    assert client.get("/api/tasks/busy").get_json()["busy"] is False


# ---------------------------------------------------------------------------
# checkpoints -- diff
# ---------------------------------------------------------------------------


def test_checkpoint_diff_endpoint(project):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    res = app.test_client().get(f"/api/checkpoints/{entry['id']}/diff")
    assert res.status_code == 200
    data = res.get_json()
    assert data["entry"]["id"] == entry["id"]
    assert "diff_stat" in data


def test_checkpoint_diff_endpoint_reports_error_for_bad_id(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/checkpoints/does-not-exist/diff")
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# memory -- pin / unpin / edit
# ---------------------------------------------------------------------------


def test_memory_pin_unpin_and_edit(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_store.add(category="fact", title="t", body="b", tags="", source="explicit", session_id="s1")
    row = memory_store.search(limit=1)[0]
    memory_store.close()

    app = create_app(project, None, None)
    client = app.test_client()

    assert client.post(f"/api/memory/{row['id']}/pin").get_json()["ok"] is True
    assert client.post(f"/api/memory/{row['id']}/unpin").get_json()["ok"] is True
    assert client.post(f"/api/memory/{row['id']}/edit", json={"title": "new title"}).get_json()["ok"] is True

    memory_store2 = MemoryStore(project / ".mazu" / "memory.db")
    updated = memory_store2.get(row["id"])
    memory_store2.close()
    assert updated["title"] == "new title"


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


def test_skills_list_and_forget(project):
    skill_manager = SkillManager(project)
    skill_manager.save("greet", "says hi", "def run(args: dict) -> str:\n    return 'hi'\n")

    app = create_app(project, None, None)
    client = app.test_client()

    rows = client.get("/api/skills").get_json()
    assert any(r["name"] == "greet" for r in rows)

    res = client.post("/api/skills/greet/forget")
    assert res.get_json()["ok"] is True
    assert not any(r["name"] == "greet" for r in client.get("/api/skills").get_json())


# ---------------------------------------------------------------------------
# council -- shares the same busy lock as run/explore
# ---------------------------------------------------------------------------


def test_start_council_streams_progress_and_the_final_answer(project, monkeypatch):
    def _fake_run_council(question, models, lead_model, full_registry, **kwargs):
        print(f"[{models[0]}] done")
        return "the synthesized answer"

    monkeypatch.setattr(task_session_module, "run_council", _fake_run_council)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/council", json={"question": "should we migrate?", "models": "anthropic:claude-sonnet-5"})
    assert res.status_code == 200
    task_id = res.get_json()["task_id"]

    session = app.task_sessions[task_id]
    events = _drain(session.outbox)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert "the synthesized answer" in lines
    assert not STDOUT_CAPTURE_LOCK.locked()


def test_start_council_requires_question_and_models(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/council", json={"question": "x", "models": ""})
    assert res.status_code == 400


def test_start_council_also_blocked_by_a_concurrent_run(project, monkeypatch):
    def _fake_run_autonomous(*args, **kwargs):
        time.sleep(0.3)

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    first = client.post("/api/run", json={"task": "fix the bug"})
    assert first.status_code == 200

    second = client.post("/api/council", json={"question": "x", "models": "anthropic:claude-sonnet-5"})
    assert second.status_code == 409

    session = app.task_sessions[first.get_json()["task_id"]]
    _drain(session.outbox)


# ---------------------------------------------------------------------------
# checkpoints -- branch-from
# ---------------------------------------------------------------------------


def test_branch_from_checkpoint_endpoint(project):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post(f"/api/checkpoints/{entry['id']}/branch-from", json={"branch_name": "experiment-1"})
    assert res.status_code == 200
    assert res.get_json()["ok"] is True

    bad = client.post(f"/api/checkpoints/does-not-exist/branch-from", json={"branch_name": "x"})
    assert bad.status_code == 400


def test_branch_from_checkpoint_requires_a_branch_name(project):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    res = app.test_client().post(f"/api/checkpoints/{entry['id']}/branch-from", json={})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# models / doctor
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_real_provider_capabilities(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/models")
    assert res.status_code == 200
    rows = res.get_json()
    assert len(rows) > 0
    assert "provider" in rows[0] and "model" in rows[0]


def test_doctor_endpoint_reports_diagnostics_without_live_calls(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/doctor")
    assert res.status_code == 200
    rows = res.get_json()
    assert len(rows) > 0
    assert all("status" in r and "name" in r and "message" in r for r in rows)


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


def test_usage_summary_endpoint(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/usage")
    assert res.status_code == 200
    assert "total_cost" in res.get_json()


# ---------------------------------------------------------------------------
# action log
# ---------------------------------------------------------------------------


def test_action_log_sessions_and_detail_are_empty_for_a_fresh_project(project):
    app = create_app(project, None, None)
    client = app.test_client()
    assert client.get("/api/action-log").get_json() == []
    assert client.get("/api/action-log/does-not-exist").get_json() == []


# ---------------------------------------------------------------------------
# config -- secret masking is the important behavior here
# ---------------------------------------------------------------------------


def test_config_get_masks_secret_values(project):
    app = create_app(project, None, None)
    client = app.test_client()

    set_res = client.post("/api/config", json={"key": "api_key", "value": "sk-supersecretvalue1234"})
    assert set_res.status_code == 200

    data = client.get("/api/config").get_json()
    assert data["api_key"] != "sk-supersecretvalue1234"
    assert data["api_key"].endswith("1234")
    assert "supersecret" not in data["api_key"]


def test_config_set_rejects_unknown_key(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/config", json={"key": "not_a_real_key", "value": "x"})
    assert res.status_code == 400
