import cv2
import numpy as np
from pathlib import Path
import random


def _central_block_mask(h, w, scale_min=0.2, scale_max=0.5):
    mask = np.zeros((h, w), dtype=np.uint8)
    min_dim = min(h, w)
    scale = random.uniform(scale_min, scale_max)
    box_h = int(min_dim * scale)
    box_w = int(min_dim * scale)
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - box_h // 2)
    y2 = min(h, cy + box_h // 2)
    x1 = max(0, cx - box_w // 2)
    x2 = min(w, cx + box_w // 2)
    mask[y1:y2, x1:x2] = 255
    return mask


def _random_holes_mask(h, w, holes_min=5, holes_max=25):
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.randint(holes_min, holes_max)
    for _ in range(n):
        # random circle or rectangle
        if random.random() < 0.6:
            # circle/ellipse
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            ry = random.randint(max(1, h // 50), h // 6)
            rx = random.randint(max(1, w // 50), w // 6)
            angle = random.randint(0, 360)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
        else:
            # rectangle
            y1 = random.randint(0, h - 1)
            x1 = random.randint(0, w - 1)
            y2 = min(h - 1, y1 + random.randint(1, h // 5))
            x2 = min(w - 1, x1 + random.randint(1, w // 5))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    # optionally dilate/erode to make shapes more organic
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _irregular_mask(h, w, strokes_min=3, strokes_max=10):
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.randint(strokes_min, strokes_max)
    for _ in range(n):
        # draw a thick random polyline
        points = []
        length = random.randint(3, 10)
        for _p in range(length):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            points.append((x, y))
        pts = np.array(points, dtype=np.int32)
        thickness = random.randint(max(3, min(h, w) // 60), max(10, min(h, w) // 20))
        cv2.polylines(mask, [pts], False, 255, thickness=thickness)
        cv2.fillPoly(mask, [pts], 255)
    # blur and threshold to form blobs
    ksize = max(3, min(h, w) // 50 * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask


def generate_mask_for_image(img_path, out_path, mask_type=None, seed=None):
    """Generate a single mask for an image and save as 8-bit PNG.

    Args:
        img_path (str or Path): path to source image (used for size)
        out_path (str or Path): path to write mask png
        mask_type (str or None): 'central', 'holes', 'irregular' or None to choose randomly
        seed (int or None): seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    if mask_type is None:
        mask_type = random.choice(["central", "holes", "irregular"])
    if mask_type == "central":
        mask = _central_block_mask(h, w)
    elif mask_type == "holes":
        mask = _random_holes_mask(h, w)
    elif mask_type == "irregular":
        mask = _irregular_mask(h, w)
    else:
        raise ValueError("Unknown mask_type: " + str(mask_type))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    return str(out_path)


def generate_masks_for_directory(raw_dir, mask_dir, mask_type=None, seed=0):
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for img_p in raw_dir.glob("*.jpg"):
        stem = img_p.stem
        out_p = mask_dir / f"{stem}_mask.png"
        if out_p.exists():
            stems.append(stem)
            continue
        s = seed + hash(stem) % 100000
        try:
            generate_mask_for_image(img_p, out_p, mask_type=mask_type, seed=s)
            stems.append(stem)
        except Exception as e:
            print(f"Skipping {img_p}: {e}")
    return stems


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--mask-dir", default="data/masks")
    p.add_argument("--type", choices=["central", "holes", "irregular"], default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate_masks_for_directory(args.raw_dir, args.mask_dir, mask_type=args.type, seed=args.seed)
