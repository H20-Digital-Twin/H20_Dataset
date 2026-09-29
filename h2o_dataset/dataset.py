"""PyTorch Dataset for synchronized H2O human images and humanoid motion."""

import hashlib
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .backgrounds import (
    choose_background_paths,
    composite_clip,
    list_backgrounds,
    load_background,
)
from .indexing import ROBOT_IDS, load_dataset_index, load_split


VIEW_MODES = ("random_single", "fixed_single", "all")
SUPPORTED_MODALITIES = ("rgb", "depth", "mask")
DEFAULT_ROBOT_KEYS = (
    "canonical_frame_id",
    "dof_pos",
    "dof_vel",
    "frame_quality_flags",
    "frame_valid",
    "qpos_mujoco",
    "qvel_mujoco",
    "robot_frame_id",
    "root_ang_vel",
    "root_lin_vel",
    "root_pos",
    "root_quat",
    "source_frame_float",
    "timestamp_sec",
    "visual_frame_id",
)


def _stable_seed(seed: int, index: int) -> int:
    text = "{}:{}".format(seed, index).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _pil_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _pil_mask_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return torch.from_numpy(array).unsqueeze(0).ne(0)


def _depth_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Depth TIFF must be HxW, got {} from {}".format(array.shape, path))
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _resize_clip(clip: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if tuple(clip.shape[-2:]) == tuple(size):
        return clip
    if mode == "bilinear":
        return F.interpolate(clip, size=size, mode=mode, align_corners=False)
    return F.interpolate(clip.to(torch.float32), size=size, mode=mode)


@lru_cache(maxsize=32)
def _load_npz_cached(path_string: str) -> Dict[str, np.ndarray]:
    with np.load(path_string, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    value = np.ascontiguousarray(array)
    if value.dtype.kind == "b":
        return torch.from_numpy(value.astype(np.bool_, copy=False))
    if value.dtype.kind in ("i", "u"):
        return torch.from_numpy(value.astype(np.int64, copy=False))
    return torch.from_numpy(value.astype(np.float32, copy=False))


class H2ODataset(Dataset):
    """Return aligned visual clips and one selected humanoid trajectory.

    All visual outputs keep a view dimension: RGB is [V,T,3,H,W], while
    depth and mask are [V,T,1,H,W]. Robot arrays are [T,...].
    """

    def __init__(
        self,
        data_root: str = "h2o_data",
        robot_id: str = "unitree_g1",
        split: str = "train",
        manifest_path: Optional[str] = None,
        split_path: Optional[str] = None,
        clip_length: int = 16,
        frame_stride: int = 1,
        window_stride: int = 1,
        view_mode: str = "random_single",
        fixed_view: str = "cam_000",
        modalities: Sequence[str] = ("rgb", "depth", "mask"),
        image_size: Optional[Tuple[int, int]] = None,
        use_background: bool = False,
        background_probability: float = 1.0,
        background_per_view: bool = True,
        background_root: Optional[str] = None,
        deterministic: Optional[bool] = None,
        seed: int = 2026,
        robot_keys: Sequence[str] = DEFAULT_ROBOT_KEYS,
        return_foreground_rgb: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.robot_id = str(robot_id)
        self.split = str(split)
        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.window_stride = int(window_stride)
        self.view_mode = str(view_mode)
        self.fixed_view = str(fixed_view)
        self.modalities = tuple(str(value) for value in modalities)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.use_background = bool(use_background)
        self.background_probability = float(background_probability)
        self.background_per_view = bool(background_per_view)
        self.deterministic = self.split != "train" if deterministic is None else bool(deterministic)
        self.seed = int(seed)
        self.robot_keys = tuple(str(value) for value in robot_keys)
        self.return_foreground_rgb = bool(return_foreground_rgb)

        if self.robot_id not in ROBOT_IDS:
            raise ValueError("Unknown robot_id {!r}; choose from {}".format(self.robot_id, ROBOT_IDS))
        if self.split not in ("train", "test", "all"):
            raise ValueError("split must be one of ('train', 'test', 'all')")
        if self.view_mode not in VIEW_MODES:
            raise ValueError("view_mode must be one of {}".format(VIEW_MODES))
        unknown_modalities = sorted(set(self.modalities) - set(SUPPORTED_MODALITIES))
        if unknown_modalities:
            raise ValueError("Unsupported modalities: {}".format(unknown_modalities))
        if self.clip_length <= 0 or self.frame_stride <= 0 or self.window_stride <= 0:
            raise ValueError("clip_length, frame_stride and window_stride must be positive")
        if not 0.0 <= self.background_probability <= 1.0:
            raise ValueError("background_probability must lie in [0, 1]")
        if self.image_size is not None and (
            len(self.image_size) != 2 or min(int(value) for value in self.image_size) <= 0
        ):
            raise ValueError("image_size must be a positive (height, width) pair")
        if self.use_background and "rgb" not in self.modalities:
            raise ValueError("Background compositing requires the rgb modality")

        project_root = Path(__file__).resolve().parent.parent
        manifest = Path(manifest_path) if manifest_path else project_root / "manifests" / "dataset_index.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                "Dataset manifest not found: {}. Run scripts/build_index.py first.".format(manifest)
            )
        rows = load_dataset_index(manifest)
        if self.split != "all":
            selected_split = Path(split_path) if split_path else project_root / "splits" / (self.split + ".json")
            if not selected_split.is_file():
                raise FileNotFoundError("Split file not found: {}".format(selected_split))
            allowed = set(load_split(selected_split))
            rows = [row for row in rows if str(row["motion_id"]) in allowed]

        self.rows: List[Dict[str, object]] = []
        for row in rows:
            robots = row.get("robots", {})
            if self.robot_id not in robots:
                continue
            camera_ids = tuple(str(value) for value in row.get("camera_ids", []))
            if self.view_mode == "fixed_single" and self.fixed_view not in camera_ids:
                continue
            self.rows.append(row)
        if not self.rows:
            raise RuntimeError("No indexed motions remain for robot={} split={}".format(self.robot_id, self.split))

        self.windows: List[Tuple[int, int]] = []
        required_span = (self.clip_length - 1) * self.frame_stride + 1
        for row_index, row in enumerate(self.rows):
            num_frames = int(row["num_frames"])
            for start in range(0, num_frames - required_span + 1, self.window_stride):
                self.windows.append((row_index, start))
        if not self.windows:
            raise RuntimeError("No clips fit clip_length={} frame_stride={}".format(self.clip_length, self.frame_stride))

        selected_background_root = Path(background_root) if background_root else self.data_root / "background"
        self.background_paths = list_backgrounds(selected_background_root) if self.use_background else []
        if self.use_background and not self.background_paths:
            raise RuntimeError("No background images found in {}".format(selected_background_root))

    def __len__(self) -> int:
        return len(self.windows)

    def _rng(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random(_stable_seed(self.seed, index))
        return random

    def _select_views(self, camera_ids: Sequence[str], rng: random.Random) -> List[str]:
        if self.view_mode == "all":
            return list(camera_ids)
        if self.view_mode == "fixed_single":
            return [self.fixed_view]
        return [camera_ids[rng.randrange(len(camera_ids))]]

    def _load_camera(self, row: Dict[str, object], view_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
        path = self.data_root / str(row["cameras_relative_path"])
        camera = _load_npz_cached(str(path))
        all_view_ids = list(row["camera_ids"])
        indices = [all_view_ids.index(view_id) for view_id in view_ids]
        output: Dict[str, torch.Tensor] = {}
        for key in ("K", "world_from_camera", "camera_from_world", "positions", "targets", "azimuth_deg"):
            if key in camera:
                output[key] = _numpy_to_tensor(camera[key][indices])
        original_height = int(camera["image_height"])
        original_width = int(camera["image_width"])
        output_height, output_width = self.image_size or (original_height, original_width)
        if "K" in output and (output_height != original_height or output_width != original_width):
            output["K"] = output["K"].clone()
            output["K"][:, 0, :] *= float(output_width) / float(original_width)
            output["K"][:, 1, :] *= float(output_height) / float(original_height)
        output["image_size_hw"] = torch.tensor([output_height, output_width], dtype=torch.int64)
        if "znear" in camera:
            output["znear"] = _numpy_to_tensor(np.asarray(camera["znear"]))
        if "zfar" in camera:
            output["zfar"] = _numpy_to_tensor(np.asarray(camera["zfar"]))
        return output

    def _load_visual_clip(
        self,
        row: Dict[str, object],
        view_ids: Sequence[str],
        visual_ids: np.ndarray,
        need_mask: bool,
    ) -> Dict[str, torch.Tensor]:
        motion_root = self.data_root / str(row["visual_relative_root"])
        output: Dict[str, List[torch.Tensor]] = {key: [] for key in self.modalities}
        if need_mask and "mask" not in output:
            output["mask"] = []
        for view_id in view_ids:
            view_root = motion_root / view_id
            if "rgb" in output:
                rgb = torch.stack(
                    [_pil_rgb_tensor(view_root / "rgb" / ("{:06d}.png".format(int(frame)))) for frame in visual_ids]
                )
                output["rgb"].append(_resize_clip(rgb, self.image_size, "bilinear") if self.image_size else rgb)
            if "depth" in output:
                depth = torch.stack(
                    [_depth_tensor(view_root / "depth" / ("{:06d}.tiff".format(int(frame)))) for frame in visual_ids]
                )
                output["depth"].append(_resize_clip(depth, self.image_size, "nearest") if self.image_size else depth)
            if "mask" in output:
                mask = torch.stack(
                    [_pil_mask_tensor(view_root / "mask" / ("{:06d}.png".format(int(frame)))) for frame in visual_ids]
                )
                if self.image_size:
                    mask = _resize_clip(mask, self.image_size, "nearest").ne(0)
                output["mask"].append(mask)
        return {key: torch.stack(value) for key, value in output.items()}

    def __getitem__(self, index: int) -> Dict[str, object]:
        row_index, start = self.windows[index]
        row = self.rows[row_index]
        rng = self._rng(index)
        view_ids = self._select_views(tuple(row["camera_ids"]), rng)

        robot_path = self.data_root / str(row["robots"][self.robot_id])
        trajectory = _load_npz_cached(str(robot_path))
        positions = start + np.arange(self.clip_length, dtype=np.int64) * self.frame_stride
        visual_ids = trajectory["visual_frame_id"][positions]
        need_mask = self.use_background and "rgb" in self.modalities
        visual = self._load_visual_clip(row, view_ids, visual_ids, need_mask=need_mask)

        background_applied = bool(
            self.use_background and "rgb" in visual and rng.random() < self.background_probability
        )
        background_ids = [""] * len(view_ids)
        crop_boxes = torch.full((len(view_ids), 4), -1, dtype=torch.int64)
        if background_applied:
            paths = choose_background_paths(
                self.background_paths,
                len(view_ids),
                self.background_per_view,
                rng.randrange,
            )
            foreground_rgb = visual["rgb"]
            if self.return_foreground_rgb:
                visual["foreground_rgb"] = foreground_rgb.clone()
            loaded = []
            if self.background_per_view:
                loaded = [load_background(path, tuple(foreground_rgb.shape[-2:]), rng.randrange) for path in paths]
            else:
                one = load_background(paths[0], tuple(foreground_rgb.shape[-2:]), rng.randrange)
                loaded = [one] * len(view_ids)
            composited = []
            for view_index, (background, metadata) in enumerate(loaded):
                composited.append(composite_clip(foreground_rgb[view_index], visual["mask"][view_index], background))
                background_ids[view_index] = str(metadata["background_id"])
                crop_boxes[view_index] = torch.tensor(metadata["crop_box_xyxy"], dtype=torch.int64)
            visual["rgb"] = torch.stack(composited)
        elif self.return_foreground_rgb and "rgb" in visual:
            visual["foreground_rgb"] = visual["rgb"].clone()

        if "mask" not in self.modalities:
            visual.pop("mask", None)
        robot: Dict[str, torch.Tensor] = {}
        for key in self.robot_keys:
            if key not in trajectory:
                raise KeyError("Robot trajectory does not contain requested key {!r}: {}".format(key, robot_path))
            array = trajectory[key]
            robot[key] = _numpy_to_tensor(array[positions] if array.ndim > 0 else array)

        return {
            "motion_id": str(row["motion_id"]),
            "robot_id": self.robot_id,
            "view_ids": view_ids,
            "clip_start": torch.tensor(start, dtype=torch.int64),
            "frame_indices": torch.from_numpy(positions.copy()),
            "visual_frame_ids": torch.from_numpy(np.asarray(visual_ids, dtype=np.int64).copy()),
            "visual": visual,
            "robot": robot,
            "camera": self._load_camera(row, view_ids),
            "background": {
                "applied": torch.tensor(background_applied, dtype=torch.bool),
                "ids": background_ids,
                "crop_boxes_xyxy": crop_boxes,
            },
        }

    def __repr__(self) -> str:
        return (
            "H2ODataset(robot_id={!r}, split={!r}, motions={}, clips={}, clip_length={}, "
            "view_mode={!r}, modalities={!r}, use_background={})"
        ).format(
            self.robot_id,
            self.split,
            len(self.rows),
            len(self),
            self.clip_length,
            self.view_mode,
            self.modalities,
            self.use_background,
        )
