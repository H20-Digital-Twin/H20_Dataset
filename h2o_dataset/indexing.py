"""Manifest and motion-level split construction for H2O."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROBOT_IDS: Tuple[str, ...] = (
    "unitree_g1",
    "booster_t1",
    "stanford_toddy",
    "fourier_n1",
    "hightorque_hi",
    "engineai_pm01",
)
CAMERA_IDS: Tuple[str, ...] = ("cam_000", "cam_001", "cam_002", "cam_003")


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _count_modality_frames(motion_root: Path, camera_id: str, modality: str, suffix: str) -> int:
    return sum(1 for _ in (motion_root / camera_id / modality).glob("*" + suffix))


def inspect_motion(data_root: Path, motion_id: str, strict: bool = True) -> Dict[str, object]:
    data_root = Path(data_root)
    visual_root = data_root / "h2o_human_multiview" / motion_id
    robot_root = data_root / "h2o_humanoid_motion" / "motions" / motion_id / "robots"
    metadata_path = visual_root / "render_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(str(metadata_path))
    metadata = _read_json(metadata_path)
    identity = metadata.get("identity", {})
    if not isinstance(identity, Mapping):
        identity = {}
    num_frames = int(identity.get("num_frames", 0))
    if num_frames <= 0:
        num_frames = _count_modality_frames(visual_root, "cam_000", "rgb", ".png")

    cameras: List[str] = []
    for camera_id in CAMERA_IDS:
        rgb_count = _count_modality_frames(visual_root, camera_id, "rgb", ".png")
        depth_count = _count_modality_frames(visual_root, camera_id, "depth", ".tiff")
        mask_count = _count_modality_frames(visual_root, camera_id, "mask", ".png")
        counts = (rgb_count, depth_count, mask_count)
        if counts == (num_frames, num_frames, num_frames):
            cameras.append(camera_id)
        elif strict:
            raise RuntimeError(
                "Modality count mismatch for {} {}: expected {}, got {}".format(
                    motion_id, camera_id, num_frames, counts
                )
            )

    robots: Dict[str, str] = {}
    robot_num_dofs: Dict[str, int] = {}
    for robot_id in ROBOT_IDS:
        trajectory = robot_root / robot_id / "trajectory_15fps.npz"
        if not trajectory.is_file():
            if strict:
                raise FileNotFoundError(str(trajectory))
            continue
        with np.load(str(trajectory), allow_pickle=False) as archive:
            visual_ids = archive["visual_frame_id"]
            if strict:
                if len(visual_ids) != num_frames:
                    raise RuntimeError(
                        "Frame mismatch for {} {}: visual={}, robot={}".format(
                            motion_id, robot_id, num_frames, len(visual_ids)
                        )
                    )
                if not np.array_equal(visual_ids, np.arange(num_frames, dtype=visual_ids.dtype)):
                    raise RuntimeError("Non-contiguous visual_frame_id for {} {}".format(motion_id, robot_id))
                if not archive["frame_valid"].all():
                    raise RuntimeError("Invalid robot frame for {} {}".format(motion_id, robot_id))
            robot_num_dofs[robot_id] = int(archive["dof_pos"].shape[1])
        robots[robot_id] = _relative(trajectory, data_root)

    return {
        "schema_version": "h2o_dataset_index_v1",
        "motion_id": motion_id,
        "num_frames": num_frames,
        "fps": 15.0,
        "camera_ids": cameras,
        "visual_relative_root": _relative(visual_root, data_root),
        "cameras_relative_path": _relative(visual_root / "cameras.npz", data_root),
        "robots": robots,
        "robot_num_dofs": robot_num_dofs,
        "source_dataset": str(metadata.get("source", {}).get("candidate_relative_path", "")).split("/")[0]
        if isinstance(metadata.get("source"), Mapping)
        else "",
    }


def build_dataset_index(data_root: Path, output_path: Path, strict: bool = True) -> List[Dict[str, object]]:
    """Build one JSONL row per motion and validate visual/robot alignment."""
    data_root = Path(data_root)
    visual_root = data_root / "h2o_human_multiview"
    motion_ids = sorted(
        path.name
        for path in visual_root.iterdir()
        if path.is_dir() and (path / "render_metadata.json").is_file()
    )
    rows = [inspect_motion(data_root, motion_id, strict=strict) for motion_id in motion_ids]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def load_dataset_index(path: Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_rank(motion_id: str, seed: int) -> str:
    return hashlib.sha256((str(seed) + ":" + motion_id).encode("utf-8")).hexdigest()


def build_motion_splits(
    motion_ids: Sequence[str],
    output_dir: Path,
    seed: int = 2026,
    train_ratio: float = 0.9,
) -> Dict[str, List[str]]:
    """Create exact, deterministic motion-level train/test partitions."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie strictly between 0 and 1")
    unique = sorted(set(motion_ids), key=lambda value: _stable_rank(value, seed))
    total = len(unique)
    train_count = int(round(total * train_ratio))
    train_count = min(train_count, total)
    splits = {
        "train": sorted(unique[:train_count]),
        "test": sorted(unique[train_count:]),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_val_path = output_dir / "val.json"
    if legacy_val_path.is_file():
        legacy_val_path.unlink()
    for split_name, ids in splits.items():
        payload = {
            "schema_version": "h2o_motion_split_v2",
            "split": split_name,
            "seed": seed,
            "split_policy": "motion_level_sha256_rank_90_10",
            "motion_count": len(ids),
            "motion_ids": ids,
        }
        with (output_dir / (split_name + ".json")).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return splits


def load_split(path: Path) -> List[str]:
    payload = _read_json(Path(path))
    motion_ids = payload.get("motion_ids")
    if not isinstance(motion_ids, list):
        raise ValueError("Split file does not contain a motion_ids list: {}".format(path))
    return [str(value) for value in motion_ids]
