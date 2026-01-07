"""Common helper utilities used by scripts in this repository.
This module centralizes small image IO, masking, padding and metric helpers
so other scripts don't carry duplicate implementations.
"""
from pathlib import Path
from typing import Tuple, Optional
import math
import cv2
import numpy as np
import torch


def load_img_rgb(path: Path, target_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
    """Read an image with OpenCV and return RGB uint8 HxWx3.
    If target_size is provided as (H, W) the image will be resized.
    Returns None if image cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None and (img.shape[0] != target_size[0] or img.shape[1] != target_size[1]):
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return img


def save_result(out_arr: np.ndarray, out_path: Path) -> None:
    """Save an RGB image represented as float 0..1 or uint8 0..255 to disk as PNG/JPG.
    Converts to BGR for OpenCV.
    """
    if out_arr.dtype in (np.float32, np.float64):
        out = np.clip(out_arr, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
    else:
        out = out_arr
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_bgr)


def pad_to_multiple(img: np.ndarray, factor: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad an image or single-channel mask so H and W are multiples of `factor`.
    Returns (padded, (top, bottom, left, right)).
    """
    h, w = img.shape[:2]
    new_h = int(math.ceil(h / factor) * factor)
    new_w = int(math.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    else:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)


def unpad(img: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    h, w = img.shape[:2]
    bottom_idx = h - bottom if bottom != 0 else h
    right_idx = w - right if right != 0 else w
    return img[top:bottom_idx, left:right_idx]


def prepare_tensor_for_lpips(img_rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB to a torch tensor scaled to [-1, 1] as expected by LPIPS."""
    img_f = img_rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t


def psnr_uint8(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 RGB images (HxWx3)."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for PSNR")
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)


def ensure_gray_mask(mask: Optional[np.ndarray], size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Ensure mask is single-channel, binary, and matches `size` (H,W)."""
    if mask is None:
        return None
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask
