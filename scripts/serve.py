"""Run one local worker; multiple processes would have separate session state."""

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("accident_risk.api.main:app", host=args.host, port=args.port, workers=1)
