"""Run the RoadSense backend AND the Mac-local operator console together.

One worker only: multiple processes would have separate in-memory session
state (see main.create_app). The main app is what you tunnel with ngrok;
the operator console is bound to 127.0.0.1 and is never reachable through
that tunnel -- see api/operator.py for why that is a real isolation
boundary, not a client-side check.
"""

import argparse
import asyncio

import uvicorn

from accident_risk.api.main import create_app
from accident_risk.api.operator import create_operator_app
from accident_risk.settings import EnvironmentSettings


async def _run(host: str, port: int, operator_port: int) -> None:
    app = create_app()
    operator_app = create_operator_app(app.state.auth)

    main_server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    operator_server = uvicorn.Server(
        uvicorn.Config(operator_app, host="127.0.0.1", port=operator_port, log_level="warning")
    )

    print(f"RoadSense backend:      http://{host}:{port}  (point ngrok at this)")
    print(f"Operator console (Mac): http://127.0.0.1:{operator_port}  (never exposed via ngrok)")
    print()
    print("iPhone sign-in: enter a phone number in the app, then generate/read its code")
    print("from the operator console above, and enter that code in the app.")
    print()

    await asyncio.gather(main_server.serve(), operator_server.serve())


if __name__ == "__main__":
    env = EnvironmentSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--operator-port", type=int, default=env.operator_port)
    args = parser.parse_args()
    asyncio.run(_run(args.host, args.port, args.operator_port))
