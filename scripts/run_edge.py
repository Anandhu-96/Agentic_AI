#!/usr/bin/env python3
"""Run the ISIP edge orchestrator (inference + telemetry + FastAPI control plane).

Usage:
    python scripts/run_edge.py [--config config/config.yaml] [--no-api]
"""

from __future__ import annotations

import argparse
import logging

from isip.orchestrator import from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ISIP edge orchestrator")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-api", action="store_true", help="skip REST control plane")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    orchestrator = from_yaml(args.config)
    try:
        import asyncio
        asyncio.run(orchestrator.run(with_api=not args.no_api))
    except KeyboardInterrupt:
        logging.info("edge orchestrator stopped")


if __name__ == "__main__":
    main()
