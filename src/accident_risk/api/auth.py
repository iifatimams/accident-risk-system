"""Phone-number authentication with single-use codes and bearer tokens.

No real SMS provider is configured for this capstone. A verification code is
never returned by request_code (the public method, reachable via ngrok) --
only request_code_for_operator, called exclusively by the loopback-only
operator console (api/operator.py), reveals it. See that module for how
network-level isolation (a separate port, never tunneled) enforces this,
not a header, IP check, or client-side gate.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from accident_risk.api.db import Database, normalise_phone

LOGGER = logging.getLogger("accident_risk.auth")
CODE_TTL_SECONDS = 300.0
TOKEN_TTL_SECONDS = 12 * 3600.0
MAX_CODE_ATTEMPTS = 5


def _digest(value: str) -> str:
    """Hash a secret for storage; never persist the clear-text value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _send_real_sms(phone: str, code: str) -> bool:
    """Sends SMS via Twilio API if environment variables are set."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not (account_sid and auth_token and from_number):
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = urllib.parse.urlencode({
        "To": phone,
        "From": from_number,
        "Body": f"Your RoadSense verification code is: {code}. It expires in 5 minutes.",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    auth_header = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("utf-8")
    req.add_header("Authorization", f"Basic {auth_header}")

    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status in (200, 201)
    except Exception as exc:
        LOGGER.error("SMS delivery failed (%s)", type(exc).__name__)
        return False


@dataclass(frozen=True)
class Principal:
    user_id: int
    phone: str
    name: str
    role: str


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class AuthService:
    def __init__(self, database: Database) -> None:
        self.db = database
        # phone -> last request_code() timestamp, for the minimal per-phone
        # request throttle below. In-memory is sufficient: this is abuse
        # protection on one process, not a durable audit trail.
        self._last_code_request: dict[str, float] = {}

    def _check_request_throttle(self, phone: str, min_interval_seconds: float) -> None:
        now = time.time()
        last = self._last_code_request.get(phone)
        if last is not None and now - last < min_interval_seconds:
            raise AuthError("Please wait a few seconds before requesting another code", 429)
        self._last_code_request[phone] = now

    def request_code(self, raw_phone: str, *, min_interval_seconds: float = 5.0) -> dict[str, Any]:
        """Public path (reachable via ngrok). NEVER returns the code --
        only whether it was actually sent by real SMS. The driver learns
        the code from the Mac-local operator console instead."""
        phone = normalise_phone(raw_phone)
        self._check_request_throttle(phone, min_interval_seconds)
        delivered = {"ok": False}

        def sms(p: str, c: str) -> None:
            # Best-effort real SMS if Twilio env vars happen to be
            # configured; never required to succeed -- there is no real
            # SMS provider for this capstone, and the operator console is
            # how a code is actually read during testing/demonstration.
            delivered["ok"] = _send_real_sms(p, c)

        self._issue_code_with_sms(phone, sms)
        return {
            "sent": delivered["ok"],
            "delivery": "sms" if delivered["ok"] else "operator_console",
            "expires_in": int(CODE_TTL_SECONDS),
        }

    def request_code_for_operator(self, raw_phone: str) -> dict[str, Any]:
        """Loopback-only path. Reveals the code. Callable only from
        api/operator.py, which network-isolation (a separate port never
        tunneled by ngrok) keeps unreachable from the public internet --
        this method itself has no header/flag gate, by design, since that
        would be a client-side-style check on the wrong side of the real
        boundary (which port you can reach)."""
        phone = normalise_phone(raw_phone)
        # No SMS attempt at all from this path: the operator console is
        # the intended, deliberate way to read the code, not a fallback.
        code, _ = self._issue_code_with_sms(phone, lambda p, c: None)
        return {"phone": phone, "code": code, "expires_in": int(CODE_TTL_SECONDS)}

    def _issue_code_with_sms(
        self, phone: str, sms_fn: Callable[[str, str], None]
    ) -> tuple[str, bool]:
        """Shared core used by both request_code and
        request_code_for_operator: create/replace the pending code for this
        phone (auto-enrolling a new number), then let the caller decide
        whether/how to attempt real delivery via sms_fn. Returns (code,
        user_already_existed) -- callers decide what, if anything, to
        expose to their own caller."""
        user = self.db.get_user_by_phone(phone)
        if user is not None and not user["active"]:
            raise AuthError("This account has been deactivated. Please contact an admin", 403)

        code = f"{secrets.randbelow(1_000_000):06d}"
        sms_fn(phone, code)

        if user is None:
            try:
                self.db.create_user(phone=phone, name=f"Driver {phone[-4:]}", role="driver")
            except ValueError:
                if self.db.get_user_by_phone(phone) is None:
                    raise

        with self.db.connect() as connection:
            connection.execute(
                "INSERT INTO auth_codes(phone, code_hash, expires_at, attempts) "
                "VALUES (?, ?, ?, 0) ON CONFLICT(phone) DO UPDATE SET "
                "code_hash = excluded.code_hash, expires_at = excluded.expires_at, attempts = 0",
                (phone, _digest(code), time.time() + CODE_TTL_SECONDS),
            )
        return code, user is not None

    def verify_code(self, raw_phone: str, code: str) -> dict[str, Any]:
        phone = normalise_phone(raw_phone)
        now = time.time()

        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_codes WHERE phone = ?", (phone,)
            ).fetchone()
            state = dict(row) if row else None

        if state is None:
            raise AuthError("Request a login code first")

        failure: AuthError | None = None
        if state["expires_at"] < now:
            failure = AuthError("That code has expired; request a new one")
        elif state["attempts"] >= MAX_CODE_ATTEMPTS:
            failure = AuthError("Too many attempts; request a new code", status=429)
        elif not hmac.compare_digest(state["code_hash"], _digest(code.strip())):
            failure = AuthError("That code is not correct")

        with self.db.connect() as connection:
            if failure is None or failure.status == 429 or state["expires_at"] < now:
                connection.execute("DELETE FROM auth_codes WHERE phone = ?", (phone,))
            else:
                connection.execute(
                    "UPDATE auth_codes SET attempts = attempts + 1 WHERE phone = ?", (phone,)
                )

        if failure is not None:
            raise failure

        user = self.db.get_user_by_phone(phone)
        if user is None or not user["active"]:
            raise AuthError("This account is not active", status=403)

        token = secrets.token_urlsafe(32)
        with self.db.connect() as connection:
            connection.execute(
                "INSERT INTO tokens(token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (_digest(token), user["id"], now, now + TOKEN_TTL_SECONDS),
            )

        return {
            "token": token,
            "expires_in": int(TOKEN_TTL_SECONDS),
            "user": {
                "id": user["id"],
                "name": user["name"],
                "phone": user["phone"],
                "role": user["role"],
            },
        }

    def resolve(self, token: str | None) -> Principal:
        if not token:
            raise AuthError("Sign in to continue")
        now = time.time()
        with self.db.connect() as connection:
            connection.execute("DELETE FROM tokens WHERE expires_at < ?", (now,))
            row = connection.execute(
                "SELECT t.user_id, u.phone, u.name, u.role, u.active FROM tokens t "
                "JOIN users u ON u.id = t.user_id WHERE t.token_hash = ? AND t.expires_at >= ?",
                (_digest(token), now),
            ).fetchone()
        if row is None:
            raise AuthError("Your session has expired; sign in again")
        if not row["active"]:
            raise AuthError("This account is not active", status=403)
        return Principal(
            user_id=int(row["user_id"]), phone=row["phone"], name=row["name"], role=row["role"]
        )

    def revoke(self, token: str) -> None:
        with self.db.connect() as connection:
            connection.execute("DELETE FROM tokens WHERE token_hash = ?", (_digest(token),))


def bearer_from_header(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()