"""Synthetic auth contracts: no real SMS, accounts, tunnels or participant databases."""
import pytest
from fastapi.testclient import TestClient

from accident_risk.api.auth import AuthService
from accident_risk.api.db import Database
from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine

KEY = "synthetic-demo-pairing-key-0000000000"
PHONE = "+12025550101"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """An isolated demo with no possibility of contacting an SMS provider."""
    def no_sms(*args):
        pytest.fail("Demo must never send SMS")
    monkeypatch.setattr("accident_risk.api.auth._send_real_sms", no_sms)
    app = create_app(InferenceEngine(), database_path=tmp_path / "demo.sqlite3",
                     dev_mock_sms_delivery=True, demo_access_key=KEY, demo_phone_numbers=[PHONE])
    with TestClient(app) as client:
        yield client, app.state.db


@pytest.mark.parametrize("existing", [False, True])
def test_demo_new_and_existing_accounts_sign_in_and_start(demo, existing, capsys):
    client, db = demo
    if existing:
        db.create_user(PHONE, "Synthetic Driver", "driver")
    result = client.post("/app/api/auth/request-code", json={"phone": PHONE},
                         headers={"X-RoadSense-Demo-Key": KEY})
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    payload = result.json()
    assert payload["delivery"] == "demo" and len(payload["code"]) == 6
    assert payload["code"] not in capsys.readouterr().out
    verified = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": payload["code"]})
    assert verified.status_code == 200
    headers = {"Authorization": f"Bearer {verified.json()['token']}"}
    profile = client.get("/app/api/me", headers=headers)
    assert profile.status_code == 200 and profile.json()["phone"] == PHONE
    session = client.post("/app/api/driver/session/start", json={}, headers=headers)
    assert session.status_code == 200 and session.json()["stream_id"]
    assert client.post("/app/api/driver/session/start", json={}, headers=headers).json()["id"] == session.json()["id"]


@pytest.mark.parametrize("headers", [{}, {"X-RoadSense-Demo-Key": "wrong"},
    {"X-Forwarded-For": "127.0.0.1", "X-Forwarded-Host": "localhost"}])
def test_public_or_spoofed_local_requests_never_get_demo_codes(demo, headers):
    client, db = demo
    result = client.post("/app/api/auth/request-code", json={"phone": PHONE}, headers=headers)
    assert result.status_code == 403 and "code" not in result.json()
    assert db.get_user_by_phone(PHONE) is None
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_codes").fetchone()[0] == 0


def test_demo_allowlist_and_rejected_request_preserves_existing_code(demo):
    client, db = demo
    result = client.post("/app/api/auth/request-code", json={"phone": PHONE}, headers={"X-RoadSense-Demo-Key": KEY}).json()
    assert client.post("/app/api/auth/request-code", json={"phone": PHONE}).status_code == 403
    rejected = client.post("/app/api/auth/request-code", json={"phone": "+12025550102"}, headers={"X-RoadSense-Demo-Key": KEY})
    assert rejected.status_code == 403
    assert db.get_user_by_phone("+12025550102") is None
    assert client.post("/app/api/auth/verify", json={"phone": PHONE, "code": result["code"]}).status_code == 200


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("delivered", [False, True])
def test_normal_delivery_never_exposes_code(tmp_path, monkeypatch, existing, delivered, capsys):
    sent = []
    def sms(phone: str, code: str) -> bool:
        sent.append((phone, code))
        return delivered
    monkeypatch.setattr("accident_risk.api.auth._send_real_sms", sms)
    app = create_app(InferenceEngine(), database_path=tmp_path / "normal.sqlite3", dev_mock_sms_delivery=False)
    if existing:
        app.state.db.create_user(PHONE, "Synthetic", "driver")
    with TestClient(app) as client:
        result = client.post("/app/api/auth/request-code", json={"phone": PHONE})
        assert result.status_code == (200 if delivered else 503)
        assert "code" not in result.json()
        assert sent[0][1] not in capsys.readouterr().out
        verify = client.post("/app/api/auth/verify", json={"phone": PHONE, "code": sent[0][1]})
        assert verify.status_code == (200 if delivered else 401)
        if delivered:
            assert result.json()["delivery"] == "sms"
        assert client.post("/app/api/auth/request-code", json={"phone": PHONE}, headers={"X-RoadSense-Demo-Key": KEY}).status_code == 400


def test_demo_requires_complete_configuration(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    for key, phones in [(None, [PHONE]), ("short", [PHONE]), (KEY, [])]:
        with pytest.raises(ValueError, match="pairing key"):
            AuthService(db, True, key, phones)


def test_probe_and_stale_token_contract(demo):
    client, _ = demo
    assert client.get("/app/api/connection").json() == {"service": "roadsense", "api_version": 1, "demo_mode": True}
    headers = {"Authorization": "Bearer expired-synthetic-token"}
    for method, path in [("GET", "/app/api/me"), ("POST", "/app/api/driver/session/start")]:
        response = client.request(method, path, headers=headers, **({"json": {}} if method == "POST" else {}))
        assert response.status_code == 401 and response.headers["content-type"].startswith("application/json")


def test_demo_cannot_implicitly_open_real_database(monkeypatch):
    monkeypatch.setenv("ACCIDENT_RISK_DEV_MOCK_SMS_DELIVERY", "true")
    with pytest.raises(ValueError, match="isolated database"):
        create_app(InferenceEngine())


def test_unknown_api_path_returns_json_not_spa_html(demo):
    client, _ = demo
    response = client.get("/app/api/not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_demo_launcher_uses_private_key_and_isolated_database(tmp_path, monkeypatch):
    from scripts.serve_demo import prepare_demo, DEMO_PHONE
    monkeypatch.setenv("ACCIDENT_RISK_BOOTSTRAP_ADMIN_PHONE", "+12025550199")
    app = prepare_demo(tmp_path)
    key_file = tmp_path / "pairing-key.txt"
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert app.state.db.path == tmp_path / "demo.sqlite3"
    assert app.state.db.get_user_by_phone("+12025550199") is None
    with TestClient(app) as client:
        result = client.post("/app/api/auth/request-code", json={"phone": DEMO_PHONE},
                             headers={"X-RoadSense-Demo-Key": key_file.read_text().strip()})
        assert result.status_code == 200 and result.json()["delivery"] == "demo"
