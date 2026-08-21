"""Tests for the mazu-web Curator endpoints: status, run (background, shares the
same busy lock as run/explore/council), log, report, undo, enable/disable.
Monkeypatches mazu.curator.orchestrator.run_curator at the mazu_web.task_session
import site so no real Curator LLM call happens -- same seam pattern
test_web_extended.py already uses for run_autonomous/run_explore/run_council.
"""

import subprocess
import time
from pathlib import Path

import pytest

import mazu_web.task_session as task_session_module
from mazu.config import set_config_value
from mazu.curator.orchestrator import AreaResult, CuratorRunSummary
from mazu.curator.store import CuratorStore, new_curator_run_id
from mazu.memory.store import MemoryStore
from mazu.skills.manager import SkillManager
from mazu_web.app import create_app
from mazu_web.task_session import STDOUT_CAPTURE_LOCK


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".mazu").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _release_stray_lock():
    yield
    if STDOUT_CAPTURE_LOCK.locked():
        STDOUT_CAPTURE_LOCK.release()


def _drain(session, timeout=10):
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


def _configure_curator():
    set_config_value("curator_api_key", "sk-curator-secret")
    set_config_value("curator_model", "anthropic:claude-haiku-4-5")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_reports_unconfigured(project):
    app = create_app(project, None, None)
    res = app.test_client().get("/api/curator/status")
    assert res.get_json() == {"configured": False}


def test_status_reports_configured(project):
    _configure_curator()
    app = create_app(project, None, None)
    res = app.test_client().get("/api/curator/status")
    data = res.get_json()
    assert data["configured"] is True
    assert data["model"] == "anthropic:claude-haiku-4-5"
    assert data["enabled"] is True
    assert data["areas"]["memory"] is None  # never curated yet


# ---------------------------------------------------------------------------
# run -- shares the same busy lock as run/explore/council
# ---------------------------------------------------------------------------


def test_run_is_a_noop_when_unconfigured(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run")
    assert res.get_json() == {"ran": False, "reason": "not_configured"}
    assert not STDOUT_CAPTURE_LOCK.locked()
    assert not (project / ".mazu" / "curator.db").exists()


def test_run_is_a_noop_when_disabled(project):
    _configure_curator()
    set_config_value("curator_enabled", "false")
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run")
    assert res.get_json() == {"ran": False, "reason": "disabled"}


def test_run_streams_progress_and_releases_the_lock(project, monkeypatch):
    _configure_curator()

    def _fake_run_curator(root, areas=None, full=False, dry_run=False, max_cost=None, max_rounds=8, verbose=False):
        print("fake curator pass ran")
        return CuratorRunSummary(
            ran=True, run_id="cur_fake123",
            areas=[AreaResult(area="memory", ran=True, log_entries=2, cost=0.01)],
            total_cost=0.01,
        )

    monkeypatch.setattr(task_session_module, "run_curator", _fake_run_curator)
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/curator/run", json={"areas": ["memory"]})
    assert res.status_code == 200
    task_id = res.get_json()["task_id"]

    session = app.task_sessions[task_id]
    events = _drain(session)
    lines = [e["text"] for e in events if e["type"] == "log"]
    assert "fake curator pass ran" in lines
    assert any("cur_fake123" in line for line in lines)
    assert not STDOUT_CAPTURE_LOCK.locked()


def test_run_rejects_non_list_areas(project):
    _configure_curator()
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run", json={"areas": "memory"})
    assert res.status_code == 400


def test_run_conflicts_with_a_concurrent_run(project, monkeypatch):
    _configure_curator()

    def _slow_fake_run_curator(*args, **kwargs):
        time.sleep(0.3)
        return CuratorRunSummary(ran=True, run_id="cur_x", areas=[], total_cost=0.0)

    monkeypatch.setattr(task_session_module, "run_curator", _slow_fake_run_curator)
    app = create_app(project, None, None)
    client = app.test_client()

    first = client.post("/api/curator/run")
    assert first.status_code == 200
    second = client.post("/api/run", json={"task": "x"})
    assert second.status_code == 409

    session = app.task_sessions[first.get_json()["task_id"]]
    _drain(session)


# ---------------------------------------------------------------------------
# log / report
# ---------------------------------------------------------------------------


def test_log_and_report_empty_before_any_run(project):
    app = create_app(project, None, None)
    client = app.test_client()
    assert client.get("/api/curator/log").get_json() == []
    res = client.get("/api/curator/report")
    assert res.status_code == 404


def _seed_curator_log(project) -> tuple[str, int, int]:
    curator_store = CuratorStore(project / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    curator_store.start_run(run_id, "anthropic:claude-haiku-4-5", ["memory"], dry_run=False)
    log_id = curator_store.log_entry(
        run_id=run_id, area="memory", action="archive_memory", target_type="memory",
        target_id="1", rationale="stale", reversal_hint="mazu memory unarchive 1",
    )
    curator_store.finish_run(run_id, "completed", None, 0.01)
    curator_store.close()
    return run_id, log_id, 1


def test_log_and_report_after_a_seeded_run(project):
    run_id, log_id, _ = _seed_curator_log(project)
    app = create_app(project, None, None)
    client = app.test_client()

    log_rows = client.get("/api/curator/log").get_json()
    assert len(log_rows) == 1
    assert log_rows[0]["action"] == "archive_memory"

    report = client.get("/api/curator/report").get_json()
    assert report["run"]["id"] == run_id
    assert len(report["entries"]) == 1


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


def test_undo_unarchives_a_memory(project):
    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    memory_id = memory_store.add(category="fact", title="Old fact", body="body")
    memory_store.archive(memory_id)
    memory_store.close()

    curator_store = CuratorStore(project / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="memory", action="archive_memory", target_type="memory",
        target_id=str(memory_id), rationale="stale", reversal_hint=f"mazu memory unarchive {memory_id}",
    )
    curator_store.close()

    app = create_app(project, None, None)
    res = app.test_client().post(f"/api/curator/undo/{log_id}")
    assert res.get_json()["ok"] is True

    memory_store = MemoryStore(project / ".mazu" / "memory.db")
    active_ids = {r["id"] for r in memory_store.all_active()}
    memory_store.close()
    assert memory_id in active_ids


def test_undo_unarchives_a_skill(project):
    manager = SkillManager(project)
    manager.save("s1", "desc", "def run(args):\n    return 'ok'\n")
    manager.archive("s1")

    curator_store = CuratorStore(project / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="skills", action="archive_skill", target_type="skill",
        target_id="s1", rationale="failing", reversal_hint="mazu skills unarchive s1",
    )
    curator_store.close()

    app = create_app(project, None, None)
    res = app.test_client().post(f"/api/curator/undo/{log_id}")
    assert res.get_json()["ok"] is True

    manager2 = SkillManager(project)
    assert "s1" in {m["name"] for m in manager2.list()}


def test_undo_returns_guidance_for_non_reversible_action(project):
    curator_store = CuratorStore(project / ".mazu" / "curator.db")
    run_id = new_curator_run_id()
    log_id = curator_store.log_entry(
        run_id=run_id, area="memory", action="edit_memory", target_type="memory",
        target_id="5", rationale="fixed typo", reversal_hint='mazu memory edit 5 --title "Old"',
    )
    curator_store.close()

    app = create_app(project, None, None)
    res = app.test_client().post(f"/api/curator/undo/{log_id}")
    data = res.get_json()
    assert data["ok"] is False
    assert data["reversal_hint"] == 'mazu memory edit 5 --title "Old"'


def test_undo_unknown_log_id_returns_404(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/undo/9999")
    assert res.status_code == 404


def test_undo_with_no_curator_db_returns_404(project):
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/undo/1")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


def test_enable_disable_round_trip(project):
    app = create_app(project, None, None)
    client = app.test_client()

    res = client.post("/api/curator/disable")
    assert res.get_json() == {"ok": True, "enabled": False}

    res = client.post("/api/curator/enable")
    assert res.get_json() == {"ok": True, "enabled": True}


# ---------------------------------------------------------------------------
# config endpoints already generically support curator_api_key/curator_model
# ---------------------------------------------------------------------------


def test_curator_api_key_is_masked_by_the_generic_config_endpoint(project):
    set_config_value("curator_api_key", "sk-curator-secret")
    app = create_app(project, None, None)
    res = app.test_client().get("/api/config")
    data = res.get_json()
    assert data["curator_api_key"] != "sk-curator-secret"
    assert data["curator_api_key"].endswith("cret")


# ---------------------------------------------------------------------------
# Numeric body-field validation -- regression tests for a real bug found via
# live testing: a non-numeric max_cost/max_rounds used to reach a bare
# int()/float() call inside the _start_task factory lambda, raising deep
# inside a background-task factory and surfacing as a generic Flask 500 HTML
# page instead of a clean 400 JSON error.
# ---------------------------------------------------------------------------


def test_run_rejects_non_numeric_max_cost_with_clean_400(project):
    _configure_curator()  # unrelated to curator, but keeps fixture setup consistent
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run", json={"max_cost": "not-a-number"})
    assert res.status_code == 400
    assert "max_cost" in res.get_json()["error"]


def test_curator_run_rejects_non_numeric_max_rounds_with_clean_400(project):
    _configure_curator()
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run", json={"max_rounds": "not-a-number"})
    assert res.status_code == 400
    assert "max_rounds" in res.get_json()["error"]
    # Must never reach the busy lock -- rejected before _start_task is called.
    assert not STDOUT_CAPTURE_LOCK.locked()


def test_curator_run_accepts_valid_numeric_strings(project, monkeypatch):
    """JSON numbers arrive as Python int/float already, but a client that sends
    them as strings (e.g. a form field) must still work -- only genuinely
    non-numeric values should be rejected."""
    _configure_curator()

    def _fake_run_curator(root, areas=None, full=False, dry_run=False, max_cost=None, max_rounds=8, verbose=False):
        assert max_cost == 0.05
        assert max_rounds == 3
        return CuratorRunSummary(ran=True, run_id="cur_ok", areas=[], total_cost=0.0)

    monkeypatch.setattr(task_session_module, "run_curator", _fake_run_curator)
    app = create_app(project, None, None)
    res = app.test_client().post("/api/curator/run", json={"max_cost": "0.05", "max_rounds": "3"})
    assert res.status_code == 200
    session = app.task_sessions[res.get_json()["task_id"]]
    _drain(session)
