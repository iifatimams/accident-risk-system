"""Mac-local operator console for verification codes.

Security model: this FastAPI app is started on a SEPARATE port, bound to
127.0.0.1 ONLY, by scripts/serve.py -- not mounted on the main app, not
behind any shared router. ngrok is only ever pointed at the main app's
port (see scripts/serve.py's own printed instructions), so this app's port
is simply never reachable from the public internet: there is no tunnel
configuration, header check, IP check, or client-side gate standing between
an attacker and these codes, because there is no route at all from outside
this Mac to this port. That is the actual isolation boundary, not anything
enforced in this file's own code.

Reuses AuthService.request_code_for_operator -- the SAME account/code
database as the main app, not a second authentication system. Generating a
code here creates/replaces the SAME pending auth_codes row the main app's
verify endpoint reads, so the iPhone can verify it with zero extra steps.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from accident_risk.api.auth import AuthError, AuthService

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RoadSense Operator Console (Mac-only)</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 420px; margin: 48px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  p.note { color: #666; font-size: 13px; }
  input { font-size: 18px; padding: 10px; width: 100%; box-sizing: border-box; margin-bottom: 12px; }
  button { font-size: 16px; padding: 10px 16px; cursor: pointer; }
  .code { font-size: 40px; font-weight: 700; letter-spacing: 4px; margin: 16px 0; }
  .error { color: #b00020; }
  .muted { color: #666; font-size: 13px; }
</style>
</head>
<body>
<h1>RoadSense Operator Console</h1>
<p class="note">Local to this Mac only -- never reachable through ngrok. No real SMS is sent.</p>
<input id="phone" placeholder="Phone number, e.g. +971500000000" autofocus>
<button onclick="generate()">Generate Verification Code</button>
<div id="result"></div>
<script>
async function generate() {
  const phone = document.getElementById('phone').value.trim();
  const result = document.getElementById('result');
  result.innerHTML = '';
  if (!phone) { result.innerHTML = '<p class="error">Enter a phone number.</p>'; return; }
  try {
    const res = await fetch('/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({phone: phone}),
    });
    const data = await res.json();
    if (!res.ok) { result.innerHTML = '<p class="error">' + (data.detail || 'Request failed') + '</p>'; return; }
    result.innerHTML = '<div class="code">' + data.code + '</div>'
      + '<p class="muted">For ' + data.phone + '. Enter this on the iPhone within '
      + Math.round(data.expires_in / 60) + ' minutes. Single-use.</p>';
  } catch (e) {
    result.innerHTML = '<p class="error">Could not reach the local backend.</p>';
  }
}
</script>
</body>
</html>"""


class OperatorCodeRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)


def create_operator_app(auth: AuthService) -> FastAPI:
    """Reuses the SAME AuthService instance as the main app (shared account
    and auth_codes database) -- not a second, independent auth system."""
    app = FastAPI(title="RoadSense Operator Console", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return _PAGE

    @app.post("/generate")
    def generate(payload: OperatorCodeRequest) -> dict[str, Any]:
        try:
            return auth.request_code_for_operator(payload.phone)
        except AuthError as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
