"""mazu-web has no authentication, so it must refuse the two ways a hostile web page
can reach a local server through the user's own browser: DNS rebinding (a foreign
Host header) and cross-site POSTs (a foreign Origin). Both were reproduced against
the unprotected app before this protection existed.
"""

import subprocess

import pytest

from mazu_web.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    return create_app(tmp_path, None, None).test_client()


def test_normal_local_requests_still_work(client):
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/status", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "localhost:8765"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "[::1]:8765"}).status_code == 200


def test_foreign_host_header_is_rejected_for_every_method(client):
    evil = {"Host": "evil.example:8765"}
    assert client.get("/api/status", headers=evil).status_code == 403
    assert client.get("/", headers=evil).status_code == 403
    assert client.post("/api/init", headers=evil).status_code == 403
    assert client.get("/api/project/browse?path=/", headers=evil).status_code == 403


def test_lookalike_hosts_are_rejected(client):
    for host in ("127.0.0.1.evil.example:8765", "localhost.evil.example", "evil.example:127.0.0.1"):
        assert client.get("/api/status", headers={"Host": host}).status_code == 403, host


def test_cross_site_post_is_rejected_and_has_no_effect(client, tmp_path):
    response = client.post(
        "/api/init", headers={"Origin": "http://evil.example", "Content-Type": "text/plain"}
    )
    assert response.status_code == 403
    assert not (tmp_path / ".mazu").exists()


def test_other_localhost_port_is_cross_origin(client, tmp_path):
    # Another local app (say a dev server on :3000) is still a different origin.
    response = client.post(
        "/api/init", headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:3000"}
    )
    assert response.status_code == 403


def test_null_origin_is_rejected(client):
    assert client.post("/api/init", headers={"Origin": "null"}).status_code == 403


def test_same_origin_post_works(client, tmp_path):
    response = client.post(
        "/api/init", headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
    )
    assert response.status_code == 200
    assert (tmp_path / ".mazu").exists()


def test_browser_fetch_metadata_cross_site_is_rejected_when_origin_missing(client):
    assert client.post("/api/init", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/api/init", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_non_browser_clients_without_origin_headers_work(client):
    # curl, scripts, and the project's own tests send neither Origin nor Sec-Fetch-Site.
    assert client.post("/api/init").status_code == 200


def test_explicitly_allowed_host_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    client = create_app(tmp_path, None, None, allowed_hosts=["mybox.lan"]).test_client()
    assert client.get("/api/status", headers={"Host": "mybox.lan:8765"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "other.lan:8765"}).status_code == 403
