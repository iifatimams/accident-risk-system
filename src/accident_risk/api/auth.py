"""Phone-number authentication with single-use codes and bearer tokens.
Supports real SMS delivery via Twilio environment variables, with fallback to dev mode.
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
        LOGGER.error("Failed to send real SMS via Twilio to %s: %s", phone, exc)
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
    def __init__(self, database: Database, expose_codes: bool = True) -> None:
        self.db = database
        self.expose_codes = expose_codes

    def request_code(self, raw_phone: str) -> dict[str, Any]:
        """Create a single-use login code.
        Auto-enrols unknown pilot numbers as 'driver' to prevent silent code drop.
        """
        phone = normalise_phone(raw_phone)
        user = self.db.get_user_by_phone(phone)

        # Auto-create driver account if not existing yet (pilot testing usability)
        if user is None:
            try:
                user = self.db.create_user(phone=phone, name=f"Driver {phone[-4:]}", role="driver")
            except Exception:
                # If race condition or user created concurrently, query again
                user = self.db.get_user_by_phone(phone)
        
        if user is not None and not user.get("active", 1):
            raise AuthError("This account has been deactivated. Please contact an admin.", status=403)

        code = f"{secrets.randbelow(1_000_000):06d}"
        with self.db.connect() as connection:
            connection.execute(
                "INSERT INTO auth_codes(phone, code_hash, expires_at, attempts) "
                "VALUES (?, ?, ?, 0) ON CONFLICT(phone) DO UPDATE SET "
                "code_hash = excluded.code_hash, expires_at = excluded.expires_at, attempts = 0",
                (phone, _digest(code), time.time() + CODE_TTL_SECONDS),
            )

        sms_dispatched = _send_real_sms(phone, code)
        print(f"\n=======================================================")
        print(f"[AUTH] >>> SMS VERIFICATION CODE FOR {phone}: {code} <<<")
        print(f"[AUTH] Twilio Sent: {sms_dispatched}")
        print(f"=======================================================\n")

        result: dict[str, Any] = {
            "sent": True,
            "delivery": "sms" if sms_dispatched else "development",
            "expires_in": int(CODE_TTL_SECONDS),
        }

        # Expose code directly in API response if real SMS is not configured or in dev mode
        if self.expose_codes or not sms_dispatched:
            result["code"] = code
            result["warning"] = "Development login: code provided directly in response."

        return result

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