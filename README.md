# H2O Dataset

The H2O Dataset provides synchronized multi-view human visual data paired with humanoid robot motion trajectories, alongside out-of-the-box PyTorch utilities for data loading, indexing, and integrity verification.

## Dataset Overview

| Metric | Value / Description |
|---|---|
| Motion Sequences | 515 |
| Total Timeline Frames | 51,031 |
| Frame Rate | 15 FPS |
| Cameras | 4 fixed viewpoints |
| Humanoid Robots | 6 models |
| Training Set | 464 motions |
| Test Set | 51 motions |

Supported Robots:

| `robot_id` | Degrees of Freedom (DoF) |
|---|---:|
| `unitree_g1` | 29 |
| `booster_t1` | 21 |
| `stanford_toddy` | 22 |
| `fourier_n1` | 23 |
| `hightorque_hi` | 25 |
| `engineai_pm01` | 24 |

The dataset is partitioned at the motion level; no motion sequence appears in both the training and test sets. A fixed validation set is not provided. If validation is required, we recommend holding out sequences from the training set on a per-motion basis.

## Data Download

Download and extract the following three packages from [Google Drive](https://drive.google.com/drive/folders/1XWSuOfgn5w1D1ZYNVAZhrdFN6LdZkmF6?usp=sharing):

- `h2o_human_multiview`: Multi-view RGB images, depth maps, foreground masks, and camera parameters.
- `h2o_humanoid_motion`: Retargeted motion trajectories for the 6 humanoid robots.
- `background`: Background images used for background composition / data augmentation.

Organize the extracted directories under a unified data root:

```text
h2o_data/
├── h2o_human_multiview/
│   └── <motion_id>/
│       ├── render_metadata.json
│       ├── cameras.npz
│       ├── cam_000/
│       │   ├── rgb/
│       │   │   └── 000000.png
│       │   ├── depth/
│       │   │   └── 000000.tiff
│       │   └── mask/
│       │       └── 000000.png
│       ├── cam_001/
│       ├── cam_002/
│       └── cam_003/
├── h2o_humanoid_motion/
│   └── motions/
│       └── <motion_id>/
│           └── robots/
│               └── <robot_id>/
│                   └── trajectory_15fps.npz
└── background/
    └── <image>.{jpg,jpeg,png,webp}
```

The `<motion_id>` corresponds one-to-one between the visual data and robot trajectories. Visual frames and robot states are aligned explicitly via `visual_frame_id`, independent of directory traversal order.

## Installation

Requirements: Python >= 3.8.

```bash
conda activate <your_env>
cd h2o_dataset
pip install -e .
```

Core dependencies include PyTorch, NumPy, Pillow, and tifffile.

## Quick Start

```python
from torch.utils.data import DataLoader
from h2o_dataset import H2ODataset

dataset = H2ODataset(
    data_root="h2o_data",
    robot_id="unitree_g1",
    split="train",
    clip_length=16,
    frame_stride=1,
    window_stride=1,
    view_mode="random_single",
    modalities=("rgb", "depth", "mask"),
    use_background=True,
    background_probability=0.8,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

batch = next(iter(loader))
```

## Sample Structure

Each sample corresponds to a temporal clip paired with a specific robot:

```text
sample
├── motion_id
├── robot_id
├── view_ids
├── clip_start
├── frame_indices
├── visual_frame_ids
├── visual
│   ├── rgb
│   ├── depth
│   └── mask
├── robot
│   ├── dof_pos
│   ├── dof_vel
│   ├── root_pos
│   ├── root_quat
│   ├── root_lin_vel
│   ├── root_ang_vel
│   └── ...
├── camera
│   ├── K
│   ├── world_from_camera
│   ├── camera_from_world
│   └── ...
└── background
    ├── applied
    ├── ids
    └── crop_boxes_xyxy
```

Key Tensor Formats:

| Field | Shape | Type / Range |
|---|---|---|
| `visual/rgb` | `[V, T, 3, H, W]` | `float32`, `[0, 1]` |
| `visual/depth` | `[V, T, 1, H, W]` | `float32`, raw metric depth from TIFF |
| `visual/mask` | `[V, T, 1, H, W]` | `bool` |
| `robot/*` | `[T, ...]` | Matches selected robot trajectory format |
| `camera/*` | `[V, ...]` | Camera intrinsics, extrinsics, poses, etc. |

Here, `T` denotes `clip_length` and `V` denotes the number of viewpoints (`V=1` in single-view modes, `V=4` in multi-view mode).

## View Modes

| `view_mode` | Behavior |
|---|---|
| `random_single` | Randomly samples one camera viewpoint per clip; held consistent across time. |
| `fixed_single` | Fixes the viewpoint to the camera specified by `fixed_view`. |
| `all` | Returns all 4 camera viewpoints simultaneously. |

The test set defaults to deterministic sampling, whereas the training set uses stochastic view and background sampling. For exact reproducibility, set `deterministic=True` and specify a random `seed`.

## Background Composition

When enabled, human foregrounds are composited onto background images from `background/` using the binary foreground masks:

```python
dataset = H2ODataset(
    data_root="h2o_data",
    robot_id="unitree_g1",
    use_background=True,
    background_probability=1.0,
    background_per_view=True,
    return_foreground_rgb=True,
)
```

- A temporal clip within a given viewpoint shares the identical background image and crop region across all frames.
- If `background_per_view=True`, each camera viewpoint samples a background independently.
- If `background_per_view=False`, all camera views share the same background and crop region.
- Composition only alters the RGB modality; depth and mask remain untouched.
- When `return_foreground_rgb=True`, the unaugmented foreground RGB is preserved at `visual/foreground_rgb`.

## Key Arguments

| Argument | Description |
|---|---|
| `data_root` | Root directory of the uncompressed dataset. |
| `robot_id` | Target humanoid robot identifier for the dataset instance. |
| `split` | Dataset split: `train`, `test`, or `all`. |
| `clip_length` | Number of frames per temporal sample. |
| `frame_stride` | Frame sampling interval within a clip. |
| `window_stride` | Temporal stride between adjacent sampled clips. |
| `view_mode` | View selection policy: `random_single`, `fixed_single`, or `all`. |
| `fixed_view` | Fixed camera viewpoint ID (defaults to `cam_000`). |
| `modalities` | Combination of visual modalities: `rgb`, `depth`, and/or `mask`. |
| `image_size` | Target resolution `(H, W)`; camera intrinsics are scaled accordingly. |
| `use_background` | Whether to enable dynamic background composition. |
| `background_probability` | Probability of applying background composition per sample. |
| `background_per_view` | Whether to sample backgrounds independently across different camera views. |
| `robot_keys` | Specific state keys to retrieve from the robot trajectory files. |

> **Note:** Different humanoid models have different DoFs. Samples with different `robot_id`s cannot be collated directly into a standard batch with the default PyTorch `DataLoader`.

## Dataset Manifests & Splits

The repository includes pre-built index manifests and canonical motion splits:

```text
manifests/dataset_index.jsonl
splits/train.json
splits/test.json
```

Rebuilding the index is only required if the underlying data changes:

```bash
python scripts/build_index.py --data-root h2o_data
```

By default, strict mode verifies the frame counts of RGB, depth, and masks across all 4 views, the trajectory lengths across all 6 robots, and validates `visual_frame_id` and `frame_valid` flags. Pass `--non-strict` to index incomplete subsets.

## Verification & Testing

Verify dataset integrity across all 6 humanoid robots, check tensor dimensions, and test PyTorch `DataLoader` batch collation:

```bash
python scripts/validate_dataset.py --data-root h2o_data
```

Run test suite:

```bash
python -m unittest discover -s tests -v
```

## Repository Structure

```text
.
├── h2o_dataset/        # Core dataset loader, indexing, and composition implementation
├── manifests/          # Precomputed dataset index files
├── splits/             # Canonical motion-level train/test splits
├── scripts/            # Scripts for manifest generation and dataset validation
├── examples/           # Sample inspection and visualization scripts
├── tests/              # Unit and integration tests
├── pyproject.toml
└── requirements.txt
```
