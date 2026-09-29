#!/usr/bin/env python3
"""Load representative samples and verify default DataLoader collation."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from h2o_dataset import H2ODataset, ROBOT_IDS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="h2o_data")
    parser.add_argument("--robot-id", choices=("all",) + ROBOT_IDS, default="all")
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(128, 128))
    return parser.parse_args()


def main():
    args = parse_args()
    robot_ids = ROBOT_IDS if args.robot_id == "all" else (args.robot_id,)
    report = {}
    for robot_id in robot_ids:
        dataset = H2ODataset(
            data_root=args.data_root,
            robot_id=robot_id,
            split="test",
            clip_length=args.clip_length,
            window_stride=1000000,
            view_mode="fixed_single",
            image_size=tuple(args.image_size),
            use_background=False,
        )
        sample = dataset[0]
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        report[robot_id] = {
            "motions": len(dataset.rows),
            "clips": len(dataset),
            "rgb": list(sample["visual"]["rgb"].shape),
            "depth": list(sample["visual"]["depth"].shape),
            "mask": list(sample["visual"]["mask"].shape),
            "dof_pos": list(sample["robot"]["dof_pos"].shape),
            "batch_rgb": list(batch["visual"]["rgb"].shape),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
