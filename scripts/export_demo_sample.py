#!/usr/bin/env python3
"""Export a bounded, class-stratified held-out CSV for the local demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.artifacts import ArtifactConfigurationError, load_demo_config
from src.demo.sample_export import SampleExportError, export_rehearsal_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw held-out flows using saved test-row indices; no prediction or training"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/demo_artifacts.example.json",
        help="artifact-path JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="uploads/rehearsal",
        help="ignored output directory (default: %(default)s)",
    )
    parser.add_argument("--rows-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = export_rehearsal_sample(
            load_demo_config(args.config),
            args.output_dir,
            rows_per_class=args.rows_per_class,
            seed=args.seed,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (ArtifactConfigurationError, SampleExportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "mode": "held-out sample export; no prediction or training",
                "flow_csv": str(result.flow_csv),
                "provenance_csv": str(result.provenance_csv),
                "manifest_json": str(result.manifest_json),
                "rows": result.row_count,
                "class_counts": result.class_counts,
                "warning": "sample metrics are not full-test performance",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
