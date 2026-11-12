"""Synthetic auth contracts: no real SMS, no real accounts, test phone numbers
and an isolated tmp_path database only.

Covers the operator-console login design: the public (ngrok-reachable) main
app never returns a verification code; only the loopback-only operator app
(api/operator.py) does, via AuthService.request_code_for_operator -- the
SAME account/code database as the main app, not a second auth system.
"""


import pytest
from fastapi.testclient import TestClient

from accident_risk.api.db import normalise_phone
from accident_risk.api.main import create_app
from accident_risk.api.operator import create_operator_app

PHONE = "+12025550101"


@pytest.fixture
def backend(tmp_path):
    app = create_app(database_path=tmp_path / "test.sqlite3")
    with TestClient(app) as client:
        yield client, app.state.db, app.state.auth


@pytest.fixture
def operator(backend):
    _, _, auth = backend
    return TestClient(create_operator_app(auth))


def test_public_request_code_never_returns_a_code(backend, capsys):
    client, db, _ = backend
    result = client.post("/app/api/auth/request-code", json={"phone": PHONE})
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    payload = result.json()
    assert "code" not in payload
    assert payload["sent"] is False  # no real SMS provider configured
    assert payload["delivery"] == "operator_console"
    # Not in stdout/logs either.
    assert db.get_user_by_phone(PHONE) is not None  # still auto-enrolled
    out = capsys.readouterr().out
    with db.connect() as conn:
        code_hash = conn.execute(
            "SELECT code_hash FROM auth_codes WHERE phone = ?", (normalise_phone(PHONE),)
        ).fetchone()["code_hash"]
    assert code_hash not in out


def test_main_app_has_no_route_that_reveals_a_code(backend):
    client, _, _ = backend
    # The only way the main app's OpenAPI schema could leak this is if a
    # route literally existed for it -- confirm none does.
    paths = client.get("/openapi.json").json()["paths"]
    assert "/app/api/auth/request-code" in paths
    for path in paths:
        assert "operator" not in path and "generate" not in path


@pytest.mark.parametrize("existing", [False, True])
def test_operator_generates_a_usable_code_for_new_and_existing_numbers(backend, operator, existing):
    client, db, _ = backend
    if existing:
        db.create_user(PHONE, "Synthetic Driver", "driver")

    generated = operator.post("/generate", json={"phone": PHONE})
    assert generated.status_code == 200
    body = generated.json()
    assert len(body["code"]) == 6 and body["code"].isdigit()
    assert body["expires_in"] > 0

    # The iPhone verifies the SAME code the operator just displayed,
    # without a second request-code call replacing it.
    verified = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": body["code"]})
    assert verified.status_code == 200
    headers = {"Authorization": f"Bearer {verified.json()['token']}"}
    profile = client.get("/app/api/me", headers=headers)
    assert profile.status_code == 200 and profile.json()["phone"] == normalise_phone(PHONE)
    session = client.post("/app/api/driver/session/start", json={}, headers=headers)
    assert session.status_code == 200 and session.json()["stream_id"]


def test_phone_normalization_matches_between_operator_and_app(backend, operator):
    client, _, _ = backend
    messy = "971-50 000 0000"
    generated = operator.post("/generate", json={"phone": messy})
    assert generated.status_code == 200
    code = generated.json()["code"]
    # The app side submits a differently-formatted but equivalent number.
    verified = client.post(
        "/app/api/auth/verify", json={"phone": "+971500000000", "code": code}
    )
    assert verified.status_code == 200


def test_incorrect_code_is_rejected(backend, operator):
    client, _, _ = backend
    operator.post("/generate", json={"phone": PHONE})
    wrong = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": "000000"})
    assert wrong.status_code == 401


def test_expired_code_is_rejected(backend, operator, monkeypatch):
    client, _, _ = backend
    clock = [1_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    code = operator.post("/generate", json={"phone": PHONE}).json()["code"]
    clock[0] += 301.0  # past CODE_TTL_SECONDS
    expired = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": code})
    assert expired.status_code == 401


def test_code_is_single_use(backend, operator):
    client, _, _ = backend
    code = operator.post("/generate", json={"phone": PHONE}).json()["code"]
    first = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": code})
    assert first.status_code == 200
    reused = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": code})
    assert reused.status_code == 401


def test_public_request_code_is_rate_limited(backend, monkeypatch):
    client, _, _ = backend
    first = client.post("/app/api/auth/request-code", json={"phone": PHONE})
    assert first.status_code == 200
    throttled = client.post("/app/api/auth/request-code", json={"phone": PHONE})
    assert throttled.status_code == 429


def test_verify_lockout_after_repeated_failures(backend, operator):
    client, _, _ = backend
    operator.post("/generate", json={"phone": PHONE})
    for _ in range(5):
        client.post("/app/api/auth/verify", json={"phone": PHONE, "code": "000000"})
    locked = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": "000000"})
    assert locked.status_code == 429


def test_operator_console_is_a_separate_app_not_mounted_on_the_main_one(backend):
    """The main app (what ngrok tunnels) must not expose /generate at all --
    it exists only on the separate operator FastAPI instance, which
    scripts/serve.py binds to a different, loopback-only port."""
    client, _, _ = backend
    assert client.post("/generate", json={"phone": PHONE}).status_code == 404
    assert client.get("/").status_code != 404  # the main app's own SPA shell, unrelated


def test_demo_launcher_is_removed():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "scripts" / "serve_demo.py").exists()
