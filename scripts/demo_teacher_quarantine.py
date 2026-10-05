#!/usr/bin/env python3
"""Run the saved Task 1 teacher/gate once and populate the local review queue.

This is CPU inference only. It never trains or changes a checkpoint/replay file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.artifacts import file_sha256, load_demo_config
from src.demo.inference import load_verified_model
from src.demo.quarantine import (
    QuarantineStore,
    evaluate_targeted_gate,
    load_replay_buffer,
)
from src.demo.results import load_task2_playback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic Task 1 replay-label corruption and quarantine demo"
    )
    parser.add_argument(
        "--config",
        default="configs/demo_artifacts.example.json",
        help="artifact-path JSON (default: %(default)s)",
    )
    parser.add_argument("--database", help="optional SQLite queue path override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_demo_config(args.config)
    teacher_spec = config.models[config.task1_teacher]
    teacher, _ = load_verified_model(teacher_spec)
    replay = load_replay_buffer(config.replay_buffer)
    playback = load_task2_playback(
        config.task2_results_zip, config.task2_results_sha256
    )
    threshold = float(playback.calibration["threshold"])
    probabilities = teacher.predict_proba(replay.features)
    result = evaluate_targeted_gate(
        replay,
        probabilities,
        threshold=threshold,
        seed=42,
        rate=0.2,
    )
    identity = f"{teacher_spec.identity}:{file_sha256(teacher_spec.checkpoint)[:12]}"
    database = Path(args.database).expanduser() if args.database else config.quarantine_db
    store = QuarantineStore(database)
    inserted = store.enqueue(result.queue_records(replay, identity))
    print(
        json.dumps(
            {
                "mode": "CPU inference and review-queue demo; no training",
                "database": str(database.resolve()),
                "teacher": identity,
                "threshold": threshold,
                "new_queue_rows": inserted,
                "pending_queue_rows": len(store.items(status="pending")),
                "experiment_audit_only": result.audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
