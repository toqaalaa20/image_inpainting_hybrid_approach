#!/usr/bin/env python3
"""
Select best variant per image based on LPIPS (if available) otherwise PSNR.
Considers all methods (classical, lama, hybrid variants) and copies the chosen
files into results/best_variant/ with a summary CSV.
"""
import csv
from pathlib import Path
import shutil

METRICS_CSV = Path('results/final_metrics.csv')
OUT_DIR = Path('results/best_variant')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'best_variants.csv'

if not METRICS_CSV.exists():
    print('Metrics CSV not found:', METRICS_CSV)
    raise SystemExit(1)

# Read metrics CSV
rows = []
with open(METRICS_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# organize by image
from collections import defaultdict
by_img = defaultdict(list)
for r in rows:
    by_img[r['image']].append(r)

summary = []
for img, items in sorted(by_img.items()):
    # consider all candidate items for this image
    candidates = items

    # helper to convert to float or None
    def tofloat(x):
        try:
            return float(x)
        except:
            return None

    # Prefer LPIPS if any candidate has valid lpips; choose lowest (better)
    best = None
    lpips_vals = [(tofloat(it.get('lpips')), it) for it in candidates]
    lpips_valid = [(v, it) for v, it in lpips_vals if v is not None]
    if lpips_valid:
        best = min(lpips_valid, key=lambda x: x[0])[1]
        score_type = 'lpips'
        score = tofloat(best.get('lpips'))
    else:
        # choose by max PSNR
        psnr_vals = [(tofloat(it.get('psnr')), it) for it in candidates]
        psnr_valid = [(v, it) for v, it in psnr_vals if v is not None]
        if psnr_valid:
            best = max(psnr_valid, key=lambda x: x[0])[1]
            score_type = 'psnr'
            score = tofloat(best.get('psnr'))
        else:
            # none available, pick the first
            best = candidates[0]
            score_type = 'none'
            score = None

    src = Path(best.get('path'))
    method = best.get('method')
    # create a safe filename: image + method
    ext = src.suffix if src.exists() else '.png'
    dst_name = f"{img}_best_{method}{ext}"
    dst = OUT_DIR / dst_name
    copied = ''
    if src.exists():
        try:
            shutil.copy(src, dst)
            copied = str(dst)
        except Exception as e:
            print(f"Failed to copy {src} -> {dst}: {e}")
            copied = ''

    summary.append({'image': img, 'method': method, 'score_type': score_type, 'score': score, 'src': best.get('path'), 'copied': copied})

# write summary CSV
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'method', 'score_type', 'score', 'src', 'copied'])
    for s in summary:
        writer.writerow([s['image'], s['method'], s['score_type'], '' if s['score'] is None else s['score'], s['src'], s['copied']])

print('Best variants copied to', OUT_DIR)
print('Summary CSV:', OUT_CSV)
