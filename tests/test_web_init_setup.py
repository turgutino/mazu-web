"""Tests for the last two CLI commands mazu-web didn't cover: `mazu init` and
`mazu setup`. Unlike every other test file here, these deliberately do NOT run
`mazu init` themselves in the fixture -- the whole point is confirming the server
starts and works against a directory that isn't a Mazu project yet, and that
/api/init and /api/setup bring it up to the same state `mazu init`/`mazu setup`
would from the terminal.
"""

import subprocess

import pytest

import mazu_web.app as app_module
from mazu.config import list_config
from mazu.diagnostics import CheckResult
from mazu_web.app import create_app


@pytest.fixture()
def bare_dir(tmp_path, monkeypatch):
    """A directory that is not a git repo and has no .mazu/ -- deliberately not
    calling `mazu init` here, unlike every other test file's `project` fixture.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_status_reports_uninitialized_for_a_bare_directory(bare_dir):
    app = create_app(bare_dir, None, None)
    res = app.test_client().get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["initialized"] is False
    assert data["is_git_repo"] is False


def test_read_only_endpoints_work_against_an_uninitialized_project(bare_dir):
    # Every store class self-creates its own directory/files lazily -- this is
    # the actual behavior that makes it safe for `mazu-web` to start without
    # requiring `mazu init` first, not just an assumption.
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    assert client.get("/api/checkpoints").get_json() == []
    assert client.get("/api/memory").get_json() == []
    assert client.get("/api/skills").get_json() == []
    assert client.get("/api/runs").get_json() == []


def test_init_endpoint_creates_mazu_dir_and_git_repo(bare_dir):
    app = create_app(bare_dir, None, None)
    client = app.test_client()

    res = client.post("/api/init")
    assert res.status_code == 200
    data = res.get_json()
    assert data["already_initialized"] is False
    assert data["initialized_git_repo"] is True

    assert (bare_dir / ".mazu").exists()
    assert (bare_dir / ".git").exists()

    status = client.get("/api/status").get_json()
    assert status["initialized"] is True
    assert status["is_git_repo"] is True


def test_init_endpoint_is_idempotent(bare_dir):
    subprocess.run(["git", "init"], cwd=bare_dir, check=True, capture_output=True)
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    client.post("/api/init")

    res = client.post("/api/init")
    data = res.get_json()
    assert data["already_initialized"] is True
    assert data["initialized_git_repo"] is False  # git repo already existed both times


def test_setup_endpoint_saves_key_and_initializes_project(bare_dir):
    app = create_app(bare_dir, None, None)
    client = app.test_client()

    res = client.post("/api/setup", json={
        "provider": "deepseek", "api_key": "sk-not-real-1234",
        "verify": False, "set_default": True,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["default_model"] == "deepseek:deepseek-chat"
    assert data["initialized_project"] is True
    assert (bare_dir / ".mazu").exists()

    cfg = list_config()
    assert cfg["deepseek_api_key"] == "sk-not-real-1234"
    assert cfg["default_model"] == "deepseek:deepseek-chat"


def test_setup_endpoint_verify_flag_calls_check_live_api_key(bare_dir, monkeypatch):
    captured = {}

    def _fake_check(provider_name, model):
        captured["provider"] = provider_name
        captured["model"] = model
        return CheckResult(name="live key check", status="ok", message="looks good")

    monkeypatch.setattr(app_module, "check_live_api_key", _fake_check)
    app = create_app(bare_dir, None, None)
    client = app.test_client()

    res = client.post("/api/setup", json={
        "provider": "anthropic", "api_key": "sk-not-real", "verify": True, "set_default": False,
    })
    data = res.get_json()
    assert data["verify"] == {"status": "ok", "message": "looks good"}
    assert captured["provider"] == "anthropic"
    assert "anthropic" in captured["model"]


# ---------------------------------------------------------------------------
# /api/project/switch, /api/project/browse -- retarget the running server at a
# different directory without restarting the process. Added alongside the
# `mazu/config.py` concurrency fix found via live testing: the user wanted a way
# to point mazu-web at a different project from the browser instead of having to
# stop the process, `cd`, and restart it from a terminal.
# ---------------------------------------------------------------------------


def test_switch_project_retargets_every_route(tmp_path, bare_dir):
    # bare_dir is the server's starting root; other_dir is a second, separate
    # project it switches to -- distinguished by which one has a memory.db entry.
    other_dir = tmp_path / "other-project"
    other_dir.mkdir()

    app = create_app(bare_dir, None, None)
    client = app.test_client()
    assert client.get("/api/status").get_json()["root"] == str(bare_dir)

    res = client.post("/api/project/switch", json={"path": str(other_dir)})
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["root"] == str(other_dir)
    assert data["initialized"] is False

    # Every other route must now see the new root too, not just /api/status --
    # this is the actual bug a naive fix (only updating one closure) would leave.
    status = client.get("/api/status").get_json()
    assert status["root"] == str(other_dir)

    client.post("/api/init")
    assert (other_dir / ".mazu").exists()
    assert not (bare_dir / ".mazu").exists()  # the original root was never touched


def test_switch_project_rejects_a_nonexistent_path(bare_dir):
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.post("/api/project/switch", json={"path": str(bare_dir / "does-not-exist")})
    assert res.status_code == 400
    # The switch must be rejected atomically -- root stays exactly what it was.
    assert client.get("/api/status").get_json()["root"] == str(bare_dir)


def test_switch_project_rejects_a_file_path(bare_dir):
    file_path = bare_dir / "not-a-directory.txt"
    file_path.write_text("hello")
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.post("/api/project/switch", json={"path": str(file_path)})
    assert res.status_code == 400


def test_switch_project_requires_a_path(bare_dir):
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.post("/api/project/switch", json={})
    assert res.status_code == 400


def test_browse_project_lists_subdirectories_not_files(tmp_path, bare_dir):
    (bare_dir / "sub_a").mkdir()
    (bare_dir / "sub_b").mkdir()
    (bare_dir / ".hidden_dir").mkdir()
    (bare_dir / "a_file.txt").write_text("x")

    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.get("/api/project/browse")
    assert res.status_code == 200
    data = res.get_json()
    assert data["path"] == str(bare_dir)
    assert data["dirs"] == ["sub_a", "sub_b"]  # no files, no dotdirs, sorted


def test_browse_project_reports_parent_for_navigating_up(bare_dir):
    child = bare_dir / "child"
    child.mkdir()
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.get("/api/project/browse", query_string={"path": str(child)})
    data = res.get_json()
    assert data["parent"] == str(bare_dir)


def test_browse_project_rejects_a_file_path(bare_dir):
    file_path = bare_dir / "a_file.txt"
    file_path.write_text("x")
    app = create_app(bare_dir, None, None)
    client = app.test_client()
    res = client.get("/api/project/browse", query_string={"path": str(file_path)})
    assert res.status_code == 400


def test_setup_endpoint_requires_provider_and_key(bare_dir):
    app = create_app(bare_dir, None, None)
    res = app.test_client().post("/api/setup", json={"provider": "not-a-provider", "api_key": "x"})
    assert res.status_code == 400

    res2 = app.test_client().post("/api/setup", json={"provider": "anthropic"})
    assert res2.status_code == 400
