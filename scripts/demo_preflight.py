#!/usr/bin/env python3
"""Check local demonstration artifacts and compatibility without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only local demo preflight")
    parser.add_argument(
        "--config",
        default="configs/demo_artifacts.example.json",
        help="artifact-path JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--database",
        help="SQLite rehearsal queue override; checked without creating or changing it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(args.config, database_override=args.database)
    print("Local demo preflight (read-only; no training)")
    print(f"Config: {report.config}")
    if report.database is not None:
        print(f"SQLite: {report.database}")
    for check in report.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        if check.fix:
            print(f"       Fix: {check.fix}")
    print("PREFLIGHT PASS" if report.ok else "PREFLIGHT FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
