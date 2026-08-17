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


def test_setup_endpoint_requires_provider_and_key(bare_dir):
    app = create_app(bare_dir, None, None)
    res = app.test_client().post("/api/setup", json={"provider": "not-a-provider", "api_key": "x"})
    assert res.status_code == 400

    res2 = app.test_client().post("/api/setup", json={"provider": "anthropic"})
    assert res2.status_code == 400
