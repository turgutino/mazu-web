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
from mazu.runs.store import RunStore
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


def _drain(session, timeout=10):
    """Waits for the session's outbox to emit "done", then joins its background
    thread -- not just draining the queue. An unjoined thread can still be
    mid-exit (closing stores, releasing the lock) when the test returns; see
    ChatSession.join's docstring for why a lingering thread reading process-
    global state (Path.home()) is a real cross-test hazard, not just untidy.
    10s, not 5s: Windows CI runners have repeatedly (seen live) taken long enough
    on these background-thread tests to exceed a 5s window.
    """
    deadline = time.time() + timeout
    events = []
    outbox = session.outbox
    while time.time() < deadline:
        event = outbox.get(timeout=max(0.01, deadline - time.time()))
        events.append(event)
        if event["type"] == "done":
            session.join(timeout=5)
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
    events = _drain(session)
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
    _drain(session)


def test_start_run_requires_a_task(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/run", json={})
    assert res.status_code == 400


def test_start_run_rejects_resume_and_from_checkpoint_together(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/run", json={"resume": "r1", "from_checkpoint": "cp_000001", "branch": "b"})
    assert res.status_code == 400


def test_start_run_rejects_branch_without_from_checkpoint(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/run", json={"task": "x", "branch": "b"})
    assert res.status_code == 400


def test_start_run_rejects_from_checkpoint_without_branch(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/run", json={"task": "x", "from_checkpoint": "cp_000001"})
    assert res.status_code == 400


def test_start_run_from_checkpoint_forks_a_new_branch_and_runs(project, monkeypatch):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([{"role": "user", "content": "hi"}], trigger="manual")

    captured = {}

    def _fake_run_autonomous(registry, task, session_id, cm, **kwargs):
        captured["task"] = task
        captured["origin_checkpoint_id"] = kwargs.get("origin_checkpoint_id")
        captured["branch_name"] = kwargs.get("branch_name")
        captured["resume_messages"] = kwargs.get("resume_messages")
        print("ran on fork")

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/run", json={
        "task": "fix it on a branch", "from_checkpoint": entry["id"], "branch": "web-fork-branch",
    })
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    # checkpoint_manager.fork() below does real git operations (unlike the other
    # tests here, which mock at the run_autonomous layer) -- seen to occasionally
    # exceed the default 5s on Windows CI specifically. A slow git fork here isn't
    # just a slow test: _drain timing out leaves this session's thread still
    # running past the test's end, and it releases STDOUT_CAPTURE_LOCK whenever it
    # finally finishes -- racing whatever later test happens to be mid-flight by
    # then. Caught for real via a Windows-CI-only failure, not reasoned about.
    events = _drain(session, timeout=20)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert any("Forked from" in line for line in lines)
    assert "ran on fork" in lines
    assert captured["task"] == "fix it on a branch"
    assert captured["origin_checkpoint_id"] == entry["id"]
    assert captured["branch_name"] == "web-fork-branch"
    assert captured["resume_messages"] == [{"role": "user", "content": "hi"}]


def test_start_run_resume_reuses_the_original_runs_config(project, monkeypatch):
    checkpoint_manager = CheckpointManager(project)
    run_store = RunStore(project / ".mazu" / "runs.db")
    run_store.start("run-1", "the original task", "deepseek:deepseek-chat", 7, 2, True, ["git", "npm"], 0.5, False)
    run_store.close()
    checkpoint_manager.snapshot([{"role": "assistant", "content": "ok"}], trigger="manual", session_id="run-1")

    captured = {}

    def _fake_run_autonomous(registry, task, session_id, cm, **kwargs):
        captured["task"] = task
        captured["session_id"] = session_id
        captured["max_steps"] = kwargs.get("max_steps")
        captured["checkpoint_every"] = kwargs.get("checkpoint_every")
        captured["allow_shell"] = kwargs.get("allow_shell")
        captured["shell_allowlist"] = kwargs.get("shell_allowlist")
        captured["max_cost"] = kwargs.get("max_cost")
        print("resumed")

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/run", json={"resume": "run-1"})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    events = _drain(session)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert any("Resuming run run-1" in line for line in lines)
    assert captured["task"] == "the original task"
    assert captured["session_id"] == "run-1"
    assert captured["max_steps"] == 7
    assert captured["checkpoint_every"] == 2
    assert captured["allow_shell"] is True
    assert captured["shell_allowlist"] == ["git", "npm"]
    assert captured["max_cost"] == 0.5


def test_start_run_resume_reports_missing_run(project):
    app = create_app(project, None, None)
    client = app.test_client()
    res = client.post("/api/run", json={"resume": "does-not-exist"})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    events = _drain(session)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert any("No run found with id does-not-exist" in line for line in lines)


def test_start_run_passes_dry_run_and_keep_checkpoints_through(project, monkeypatch):
    captured = {}

    def _fake_run_autonomous(registry, task, session_id, cm, **kwargs):
        captured["dry_run"] = kwargs.get("dry_run")
        captured["retention"] = cm.retention

    monkeypatch.setattr(task_session_module, "run_autonomous", _fake_run_autonomous)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/run", json={"task": "x", "dry_run": True, "keep_checkpoints": 5})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session)
    assert captured["dry_run"] is True
    assert captured["retention"] == 5


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
    events = _drain(session)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert "[cost] some progress line" in lines
    assert "RANKED REPORT" in lines


def test_start_explore_requires_task_and_models(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/explore", json={"task": "fix it", "models": ""})
    assert res.status_code == 400


def test_start_explore_auto_models_reuses_the_cli_picker(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused-not-real")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    captured = {}

    def _fake_run_explore(task, models, root, checkpoint_manager, **kwargs):
        captured["models"] = models
        return []

    monkeypatch.setattr(task_session_module, "run_explore", _fake_run_explore)
    monkeypatch.setattr(task_session_module, "format_explore_report", lambda results, tc: "")
    app = create_app(project, None, None)

    res = app.test_client().post("/api/explore", json={"task": "fix the bug", "auto_models": True, "approaches": 1})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session)
    assert captured["models"] == ["deepseek:deepseek-chat"]


def test_start_explore_auto_models_reports_not_enough_distinct_models(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = create_app(project, None, None)
    res = app.test_client().post("/api/explore", json={"task": "fix the bug", "auto_models": True, "approaches": 2})
    assert res.status_code == 400
    assert "could only find" in res.get_json()["error"]


def test_start_explore_passes_from_checkpoint_through(project, monkeypatch):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([], trigger="manual")

    captured = {}

    def _fake_run_explore(task, models, root, checkpoint_manager, from_checkpoint_id=None, **kwargs):
        captured["from_checkpoint_id"] = from_checkpoint_id
        return []

    monkeypatch.setattr(task_session_module, "run_explore", _fake_run_explore)
    monkeypatch.setattr(task_session_module, "format_explore_report", lambda results, tc: "")
    app = create_app(project, None, None)

    res = app.test_client().post("/api/explore", json={
        "task": "fix it", "models": "anthropic:claude-sonnet-5", "from_checkpoint": entry["id"],
    })
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session)
    assert captured["from_checkpoint_id"] == entry["id"]


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
    _drain(session)
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
    events = _drain(session)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert "the synthesized answer" in lines
    assert not STDOUT_CAPTURE_LOCK.locked()


def test_start_council_requires_a_question(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/council", json={"models": "anthropic:claude-sonnet-5"})
    assert res.status_code == 400


def test_start_council_falls_back_to_the_cli_defaults_when_models_omitted(project, monkeypatch):
    from mazu.cli import DEFAULT_COUNCIL_LEAD, DEFAULT_COUNCIL_MODELS

    captured = {}

    def _fake_run_council(question, models, lead_model, full_registry, **kwargs):
        captured["models"] = models
        captured["lead_model"] = lead_model
        return "answer"

    monkeypatch.setattr(task_session_module, "run_council", _fake_run_council)
    app = create_app(project, None, None)
    res = app.test_client().post("/api/council", json={"question": "should we migrate?"})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session)
    assert captured["models"] == DEFAULT_COUNCIL_MODELS.split(",")
    assert captured["lead_model"] == DEFAULT_COUNCIL_LEAD


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
    _drain(session)


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
# checkpoints -- compare / inspect / prune
# ---------------------------------------------------------------------------


def test_compare_checkpoints_endpoint(project):
    checkpoint_manager = CheckpointManager(project)
    entry_a = checkpoint_manager.snapshot([], trigger="manual")
    entry_b = checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.get(f"/api/checkpoints/compare?a={entry_a['id']}&b={entry_b['id']}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["entry_a"]["id"] == entry_a["id"]
    assert data["entry_b"]["id"] == entry_b["id"]

    missing = client.get("/api/checkpoints/compare?a=does-not-exist&b=" + entry_b["id"])
    assert missing.status_code == 400

    incomplete = client.get(f"/api/checkpoints/compare?a={entry_a['id']}")
    assert incomplete.status_code == 400


def test_inspect_checkpoint_endpoint(project):
    checkpoint_manager = CheckpointManager(project)
    entry = checkpoint_manager.snapshot([{"role": "user", "content": "hi"}], trigger="manual")

    app = create_app(project, None, None)
    res = app.test_client().get(f"/api/checkpoints/{entry['id']}/inspect")
    assert res.status_code == 200
    data = res.get_json()
    assert data["entry"]["id"] == entry["id"]
    assert "memory" in data and "conversation" in data


def test_prune_checkpoints_endpoint(project):
    checkpoint_manager = CheckpointManager(project)
    for _ in range(3):
        checkpoint_manager.snapshot([], trigger="manual")

    app = create_app(project, None, None)
    res = app.test_client().post("/api/checkpoints/prune", json={"keep_last": 1})
    assert res.status_code == 200
    assert res.get_json()["deleted"] >= 1


# ---------------------------------------------------------------------------
# runs -- compare-branches
# ---------------------------------------------------------------------------


def test_compare_runs_endpoint(project):
    run_store = RunStore(project / ".mazu" / "runs.db")
    run_store.start("run-a", "task a", "anthropic:claude-sonnet-5", 15, 1, True, None, None, False)
    run_store.finish("run-a", status="completed", stop_reason="end_turn")
    run_store.start("run-b", "task b", "deepseek:deepseek-chat", 15, 1, True, None, None, False)
    run_store.finish("run-b", status="completed", stop_reason="end_turn")
    run_store.close()

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.get("/api/runs/compare?a=run-a&b=run-b")
    assert res.status_code == 200
    data = res.get_json()
    assert data["run_a"]["id"] == "run-a"
    assert data["run_b"]["id"] == "run-b"

    missing = client.get("/api/runs/compare?a=run-a&b=does-not-exist")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# memory -- forget / supersede / stats / why / consolidate
# ---------------------------------------------------------------------------


def test_memory_forget_and_supersede(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_store.add(category="fact", title="a", body="a body", tags="", source="explicit", session_id="s1")
    memory_store.add(category="fact", title="b", body="b body", tags="", source="explicit", session_id="s1")
    rows = memory_store.search(limit=2)
    memory_store.close()

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post(f"/api/memory/{rows[0]['id']}/supersede/{rows[1]['id']}")
    assert res.get_json()["ok"] is True

    res = client.post(f"/api/memory/{rows[1]['id']}/forget")
    assert res.get_json()["ok"] is True


def test_memory_stats_endpoint(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_store.add(category="fact", title="a", body="a body", tags="", source="explicit", session_id="s1")
    memory_store.close()

    app = create_app(project, None, None)
    res = app.test_client().get("/api/memory/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 1


def test_memory_why_endpoint(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_store.add(category="fact", title="parser bug", body="the parser has a bug", tags="", source="explicit", session_id="s1")
    memory_store.close()

    app = create_app(project, None, None)
    res = app.test_client().get("/api/memory/why?q=parser")
    assert res.status_code == 200
    rows = res.get_json()
    assert len(rows) >= 1
    assert "row" in rows[0] and "included" in rows[0]
    assert rows[0]["row"]["title"] == "parser bug"


def test_memory_consolidate_dry_run_does_not_change_anything(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_store.add(category="fact", title="the sky is blue", body="the sky is blue", tags="", source="explicit", session_id="s1")
    memory_store.add(category="fact", title="the sky is blue", body="the sky is blue", tags="", source="explicit", session_id="s1")
    memory_store.close()

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/memory/consolidate", json={"dry_run": True})
    assert res.status_code == 200
    data = res.get_json()
    assert data["applied"] is False
    assert len(data["clusters"]) == 1

    memory_store2 = MemoryStore(project / ".mazu" / "memory.db")
    remaining = memory_store2.search(limit=10)
    memory_store2.close()
    assert len(remaining) == 2  # dry run -- nothing actually merged


# ---------------------------------------------------------------------------
# memory -- stale / archive / unarchive
# ---------------------------------------------------------------------------


def _backdate(store, memory_id, days_ago):
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    store.conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (ts, memory_id))
    store.conn.commit()


def test_memory_stale_endpoint_lists_candidates_without_changing_anything(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    stale_id = memory_store.add(category="fact", title="old fact", body="body", tags="", source="explicit", session_id="s1")
    _backdate(memory_store, stale_id, days_ago=40)
    fresh_id = memory_store.add(category="fact", title="fresh fact", body="body", tags="", source="explicit", session_id="s1")
    memory_store.close()

    app = create_app(project, None, None)
    res = app.test_client().get("/api/memory/stale")
    assert res.status_code == 200
    rows = res.get_json()
    assert [r["id"] for r in rows] == [stale_id]

    memory_store2 = MemoryStore(project / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in memory_store2.all_active()}
    memory_store2.close()
    assert stale_id in active_ids
    assert fresh_id in active_ids


def test_memory_stale_archive_endpoint_archives_candidates_reversibly(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    stale_id = memory_store.add(category="fact", title="old fact", body="body", tags="", source="explicit", session_id="s1")
    _backdate(memory_store, stale_id, days_ago=40)
    memory_store.close()

    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/memory/stale/archive", json={})
    assert res.status_code == 200
    archived = res.get_json()["archived"]
    assert [a["id"] for a in archived] == [stale_id]

    memory_store2 = MemoryStore(project / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in memory_store2.all_active()}
    archived_ids = {r["id"] for r in memory_store2.list_archived()}
    memory_store2.close()
    assert stale_id not in active_ids
    assert stale_id in archived_ids

    res = client.post(f"/api/memory/{stale_id}/unarchive")
    assert res.get_json()["ok"] is True

    memory_store3 = MemoryStore(project / ".mazu" / "memory.db")
    active_ids_after = {r["id"] for r in memory_store3.all_active()}
    memory_store3.close()
    assert stale_id in active_ids_after


def test_memory_archived_endpoint_lists_archived_only(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    active_id = memory_store.add(category="fact", title="active", body="body", tags="", source="explicit", session_id="s1")
    archived_id = memory_store.add(category="fact", title="archived", body="body", tags="", source="explicit", session_id="s1")
    memory_store.archive(archived_id)
    memory_store.close()

    app = create_app(project, None, None)
    res = app.test_client().get("/api/memory/archived")
    assert res.status_code == 200
    rows = res.get_json()
    assert [r["id"] for r in rows] == [archived_id]
    assert active_id not in {r["id"] for r in rows}


def test_memory_unarchive_unknown_id_returns_not_ok(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/memory/9999/unarchive")
    assert res.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# config -- unset (the one config subcommand missed in the first pass)
# ---------------------------------------------------------------------------


def test_config_unset_endpoint(project):
    app = create_app(project, None, None)
    client = app.test_client()

    client.post("/api/config", json={"key": "router_suggestions", "value": "false"})
    assert client.get("/api/config").get_json()["router_suggestions"] == "false"

    res = client.delete("/api/config/router_suggestions")
    assert res.status_code == 200
    assert res.get_json() == {"ok": True, "was_set": True}
    assert "router_suggestions" not in client.get("/api/config").get_json()


def test_config_unset_endpoint_is_a_no_op_for_an_unset_key(project):
    app = create_app(project, None, None)
    res = app.test_client().delete("/api/config/router_suggestions")
    assert res.status_code == 200
    assert res.get_json() == {"ok": True, "was_set": False}


# ---------------------------------------------------------------------------
# checkpoints -- bare "take a checkpoint now" (mirrors bare `mazu checkpoint`)
# ---------------------------------------------------------------------------


def test_create_checkpoint_endpoint(project):
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/checkpoints")
    assert res.status_code == 200
    entry = res.get_json()
    assert entry["trigger"] == "manual_cli"

    rows = client.get("/api/checkpoints").get_json()
    assert any(r["id"] == entry["id"] for r in rows)


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
