#!/usr/bin/env python3
"""Validate H2O source data and build manifest plus fixed motion splits."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_dataset.indexing import build_dataset_index, build_motion_splits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="h2o_data")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--non-strict", action="store_true", help="Keep incomplete rows instead of failing.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    manifest_path = project_root / "manifests" / "dataset_index.jsonl"
    rows = build_dataset_index(Path(args.data_root), manifest_path, strict=not args.non_strict)
    splits = build_motion_splits(
        [str(row["motion_id"]) for row in rows],
        project_root / "splits",
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "motions": len(rows),
                "splits": {key: len(value) for key, value in splits.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
