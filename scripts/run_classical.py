import cv2
import numpy as np
import random
from pathlib import Path
import time
import csv

# Force absolute paths
RAW_DIR = Path("/workspace/data/raw")
MASK_DIR = Path("/workspace/data/masks")
OUT_DIR = Path("/workspace/results/classical")

for p in [MASK_DIR, OUT_DIR]: p.mkdir(parents=True, exist_ok=True)

image_files = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.JPEG"))

def _ensure_gray_mask(mask, size):
    # mask: ndarray, may be 3-channel or single
    if mask is None:
        return None
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # binary
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask

def run_classical(raw_dir="data/raw", mask_dir="data/masks", out_dir="results/classical",
                  methods=("telea", "ns"), radius=3, limit=None, force=False, write_csv=False):
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(raw_dir.glob("*.jpg"))
    if limit is not None:
        img_paths = img_paths[:int(limit)]

    results = []

    for img_p in img_paths:
        stem = img_p.stem
        mask_p = mask_dir / f"{stem}_mask.png"
        if not mask_p.exists():
            print(f"⚠️  Mask missing for {stem}, skipping.")
            continue

        out_telea = out_dir / f"{stem}_telea.png"
        out_ns = out_dir / f"{stem}_ns.png"

        if (not force) and any(p.exists() for p in [out_telea, out_ns]):
            print(f"ℹ️  Outputs already exist for {stem}, use --force to overwrite. Skipping.")
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            print(f"Failed to read image {img_p}, skipping.")
            continue
        mask = cv2.imread(str(mask_p))
        mask = _ensure_gray_mask(mask, img.shape[:2])
        if mask is None:
            print(f"Failed to read mask {mask_p}, skipping.")
            continue

        record = {"image": stem, "mask": str(mask_p), "methods": [], "times": {}}

        if "telea" in methods:
            t0 = time.time()
            try:
                out_img = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
                cv2.imwrite(str(out_telea), out_img)
                dt = time.time() - t0
                record["methods"].append("telea")
                record["times"]["telea"] = dt
                print(f"✓ {stem}: Telea saved ({dt:.2f}s)")
            except Exception as e:
                print(f"Error running Telea on {stem}: {e}")

        if "ns" in methods:
            t0 = time.time()
            try:
                out_img = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
                cv2.imwrite(str(out_ns), out_img)
                dt = time.time() - t0
                record["methods"].append("ns")
                record["times"]["ns"] = dt
                print(f"✓ {stem}: Navier–Stokes saved ({dt:.2f}s)")
            except Exception as e:
                print(f"Error running Navier–Stokes on {stem}: {e}")

        results.append(record)

    if write_csv:
        csv_p = out_dir / "run_classical_results.csv"
        with open(csv_p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask", "method", "time_s"])
            for r in results:
                for m in r["methods"]:
                    writer.writerow([r["image"], r["mask"], m, f"{r['times'].get(m, '')}"])
        print(f"Results written to {csv_p}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run classical inpainting methods (Telea, Navier–Stokes)")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--mask-dir", default="data/masks")
    p.add_argument("--out-dir", default="results/classical")
    p.add_argument("--methods", nargs="*", choices=["telea", "ns"], default=["telea", "ns"],
                   help="Which methods to run")
    p.add_argument("--radius", type=float, default=3.0, help="inpainting radius")
    p.add_argument("--limit", type=int, default=None, help="Process only first N images")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--write-csv", action="store_true", help="Write a CSV summary to out-dir")
    args = p.parse_args()

    run_classical(args.raw_dir, args.mask_dir, args.out_dir, methods=args.methods,
                  radius=args.radius, limit=args.limit, force=args.force, write_csv=args.write_csv)