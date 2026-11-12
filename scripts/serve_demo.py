"""Start an isolated university demo. No real database, SMS or participant identities.

Forward an HTTPS tunnel to this server for an iPhone on a different network.
The random pairing key is saved in a mode-0600 file, never printed in server logs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import tempfile

import uvicorn
from fastapi import FastAPI

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine

DEMO_PHONE = "+12025550101"


def prepare_demo(directory: Path) -> FastAPI:
    """Create a new synthetic database and a private pairing key in a fresh directory."""
    key = secrets.token_urlsafe(32)
    key_path = directory / "pairing-key.txt"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write(key + "\n")
    return create_app(
        InferenceEngine(), database_path=directory / "demo.sqlite3",
        dev_mock_sms_delivery=True, demo_access_key=key, demo_phone_numbers=[DEMO_PHONE],
    )


def main() -> None:
    """Bind loopback by default; retain the demo directory on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="roadsense-demo-"))
    app = prepare_demo(directory)
    print(f"Synthetic demo account: {DEMO_PHONE}", flush=True)
    print(f"Private pairing key file: {directory / 'pairing-key.txt'}", flush=True)
    print(f"Isolated demo database: {directory / 'demo.sqlite3'}", flush=True)
    print("In iPhone Settings: enter the HTTPS tunnel URL, enable demo login, and paste the private key.", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
