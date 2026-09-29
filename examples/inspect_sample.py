#!/usr/bin/env python3
"""Print the structure and tensor shapes of one H2ODataset sample."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_dataset import H2ODataset


dataset = H2ODataset(
    data_root="h2o_data",
    robot_id="unitree_g1",
    split="train",
    clip_length=16,
    view_mode="random_single",
    use_background=True,
    background_probability=0.8,
)
sample = dataset[0]

print(dataset)
print("motion:", sample["motion_id"], "views:", sample["view_ids"])
for name, tensor in sample["visual"].items():
    print("visual/{:<16} shape={} dtype={}".format(name, tuple(tensor.shape), tensor.dtype))
for name in ("dof_pos", "root_pos", "root_quat", "visual_frame_id"):
    tensor = sample["robot"][name]
    print("robot/{:<17} shape={} dtype={}".format(name, tuple(tensor.shape), tensor.dtype))
print("background:", sample["background"])
