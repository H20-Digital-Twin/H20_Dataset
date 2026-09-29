#!/usr/bin/env python3
"""Save the first RGB, mask and colored depth frame for each selected view."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_dataset import H2ODataset


def depth_color(depth, mask):
    valid = mask & np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        normalized[valid] = np.clip((depth[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    color = np.stack((normalized, 1.0 - np.abs(normalized * 2.0 - 1.0), 1.0 - normalized), axis=-1)
    color[~valid] = 0
    return (color * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sample_preview.png")
    parser.add_argument("--robot-id", default="unitree_g1")
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    dataset = H2ODataset(
        robot_id=args.robot_id,
        split="test",
        clip_length=8,
        view_mode="all" if args.all_views else "fixed_single",
        use_background=args.background,
        background_probability=1.0,
    )
    sample = dataset[0]
    rows = []
    for view_index in range(len(sample["view_ids"])):
        rgb = sample["visual"]["rgb"][view_index, 0].permute(1, 2, 0).numpy()
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        mask = sample["visual"]["mask"][view_index, 0, 0].numpy()
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
        depth = sample["visual"]["depth"][view_index, 0, 0].numpy()
        rows.append(np.concatenate((rgb, mask_rgb, depth_color(depth, mask)), axis=1))
    preview = np.concatenate(rows, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(output)
    print("saved:", output.resolve())


if __name__ == "__main__":
    main()
