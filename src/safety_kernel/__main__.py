"""Entry point for the failure-first safety kernel package."""

from safety_kernel.telemetry.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
