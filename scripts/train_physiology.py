"""Train independent driver-state baselines."""

from accident_risk.cli import train_cli

if __name__ == "__main__":
    train_cli(physiology=True)
