"""Temporally consistent 2D background compositing."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


RandomInt = Callable[[int], int]


def list_backgrounds(background_root: Path) -> List[Path]:
    """Return the sorted supported background images."""
    root = Path(background_root)
    paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.glob(pattern))
        paths.extend(root.glob(pattern.upper()))
    return sorted(set(paths))


def _sample_crop_box(width: int, height: int, randbelow: RandomInt) -> Tuple[int, int, int, int]:
    side = min(width, height)
    max_left = width - side
    max_top = height - side
    left = randbelow(max_left + 1) if max_left > 0 else 0
    top = randbelow(max_top + 1) if max_top > 0 else 0
    return left, top, left + side, top + side


def load_background(
    path: Path,
    output_size: Tuple[int, int],
    randbelow: RandomInt,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Load one background using one crop that can be reused for a whole clip."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        crop_box = _sample_crop_box(image.width, image.height, randbelow)
        image = image.crop(crop_box)
        output_height, output_width = output_size
        image = image.resize((output_width, output_height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor, {
        "background_id": path.name,
        "crop_box_xyxy": tuple(int(value) for value in crop_box),
    }


def composite_clip(
    foreground_rgb: torch.Tensor,
    mask: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Composite [T,3,H,W] RGB with one [3,H,W] background."""
    if foreground_rgb.ndim != 4 or mask.ndim != 4 or background.ndim != 3:
        raise ValueError("Expected RGB [T,3,H,W], mask [T,1,H,W], background [3,H,W].")
    alpha = mask.to(dtype=foreground_rgb.dtype)
    return foreground_rgb * alpha + background.unsqueeze(0) * (1.0 - alpha)


def choose_background_paths(
    paths: Sequence[Path],
    num_views: int,
    per_view: bool,
    randbelow: RandomInt,
) -> List[Path]:
    if not paths:
        raise RuntimeError("Background compositing was requested but no images were found.")
    if per_view:
        return [paths[randbelow(len(paths))] for _ in range(num_views)]
    selected = paths[randbelow(len(paths))]
    return [selected] * num_views
