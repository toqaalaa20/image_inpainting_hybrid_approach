import csv
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

# Reuse shared helpers to avoid duplication. When this script is executed directly
# the interpreter's sys.path may not include the repository root, causing
# `ModuleNotFoundError: No module named 'scripts'`. Attempt a normal import and
# fall back to prepending the project root to sys.path.
try:
    from scripts.utils import load_img_rgb, prepare_tensor_for_lpips, psnr_uint8
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts.utils import load_img_rgb, prepare_tensor_for_lpips, psnr_uint8


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Compute PSNR and LPIPS between ground truth and inpainted outputs; also aggregate column-wise')
    p.add_argument('--raw-dir', default='data/raw')
    p.add_argument('--classical-dir', default='results/classical')
    p.add_argument('--lama-dir', default='results/lama')
    p.add_argument('--hybrid-dir', default=None, help='Optional hybrid outputs dir')
    p.add_argument('--out-csv', default='results/final_metrics.csv')
    p.add_argument('--out-agg-csv', default='results/metrics_comparison.csv')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--skip-lpips', action='store_true', help='Skip LPIPS even if lpips package is available')
    p.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    classical_dir = Path(args.classical_dir)
    lama_dir = Path(args.lama_dir)
    # Auto-detect hybrid directory if not provided on the CLI.
    candidates = [Path('results/lama_hybrid'), Path('results/lama-hybrid'), Path('results/lama_hybrid_outputs')]
    hybrid_dir = None
    if args.hybrid_dir:
        hybrid_dir = Path(args.hybrid_dir)
        if not hybrid_dir.exists():
            print(f"Provided hybrid-dir does not exist: {hybrid_dir} — hybrid metrics will be skipped.")
            hybrid_dir = None
    else:
        for c in candidates:
            if c.exists():
                hybrid_dir = c
                print(f"Auto-detected hybrid-dir: {hybrid_dir}")
                break

    out_csv = Path(args.out_csv)
    out_agg_csv = Path(args.out_agg_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_agg_csv.parent.mkdir(parents=True, exist_ok=True)

    # Try to import lpips
    lpips_model = None
    if not args.skip_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(args.device)
            print('Using LPIPS model on', args.device)
        except Exception as e:
            print('LPIPS not available or failed to initialize - LPIPS will be skipped. Error:', e)
            lpips_model = None

    img_paths = sorted(raw_dir.glob('*.jpg'))
    if args.limit is not None:
        img_paths = img_paths[:args.limit]

    rows = []

    for img_path in img_paths:
        stem = img_path.stem
        gt = load_img_rgb(img_path)
        if gt is None:
            print(f"Skipping {stem}: cannot read ground truth")
            continue
        H, W = gt.shape[:2]

        # Collect candidate outputs
        candidates = []
        seen_paths = set()
        # classical telea/ns
        telea_p = classical_dir / f"{stem}_telea.png"
        ns_p = classical_dir / f"{stem}_ns.png"
        if telea_p.exists():
            candidates.append(('telea', telea_p))
            seen_paths.add(str(telea_p))
        if ns_p.exists():
            candidates.append(('ns', ns_p))
            seen_paths.add(str(ns_p))
        # lama
        lama_p = lama_dir / f"{stem}_lama.png"
        if lama_p.exists():
            candidates.append(('lama', lama_p))
            seen_paths.add(str(lama_p))
        # hybrid: find all hybrid variants for this stem (support multiple variant suffixes)
        if hybrid_dir is not None:
            for hp in hybrid_dir.glob(f"{stem}*_lama_hybrid*.png"):
                if not hp.exists():
                    continue
                pstr = str(hp)
                if pstr in seen_paths:
                    continue
                # derive method name by removing the "<stem>_" prefix
                st = hp.stem
                if st.startswith(stem + '_'):
                    method = st[len(stem) + 1:]
                else:
                    method = st
                method = method.replace('.', '_')
                candidates.append((method, hp))
                seen_paths.add(pstr)
            # fallback to legacy name
            hp = hybrid_dir / f"{stem}_lama_hybrid.png"
            if hp.exists() and str(hp) not in seen_paths:
                candidates.append(('lama_hybrid', hp))
                seen_paths.add(str(hp))

        if len(candidates) == 0:
            print(f"No outputs found for {stem}, skipping")
            continue

        for method, out_p in candidates:
            out_img = load_img_rgb(out_p, target_size=(H, W))
            if out_img is None:
                print(f"Failed to read result {out_p} for {stem}, method {method}")
                continue
            try:
                pval = psnr_uint8(gt, out_img)
            except Exception as e:
                print(f"PSNR error for {stem} {method}: {e}")
                pval = None

            lval = None
            if lpips_model is not None:
                try:
                    gt_t = prepare_tensor_for_lpips(gt).to(args.device)
                    out_t = prepare_tensor_for_lpips(out_img).to(args.device)
                    with torch.no_grad():
                        l = lpips_model(gt_t, out_t)
                    # lpips returns a 1x1x1x1 tensor
                    lval = float(l.squeeze().cpu().numpy())
                except Exception as e:
                    print(f"LPIPS error for {stem} {method}: {e}")
                    lval = None

            rows.append({
                'image': stem,
                'method': method,
                'psnr': pval,
                'lpips': lval,
                'path': str(out_p)
            })
            print(f"{stem} | {method} | PSNR={pval} | LPIPS={lval}")

    # Write row-wise CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'method', 'psnr', 'lpips', 'path'])
        for r in rows:
            writer.writerow([r['image'], r['method'], '' if r['psnr'] is None else f"{r['psnr']:.4f}", '' if r['lpips'] is None else f"{r['lpips']:.6f}", r['path']])

    print(f"Row-wise metrics written to {out_csv}")

    # Aggregate / pivot to column-wise comparison
    data = {}
    methods_seen = set()
    for r in rows:
        img = r['image']
        m = r['method']
        methods_seen.add(m)
        if img not in data:
            data[img] = {}
        data[img][m] = {'psnr': '' if r['psnr'] is None else f"{r['psnr']:.4f}",
                        'lpips': '' if r['lpips'] is None else f"{r['lpips']:.6f}",
                        'path': r['path']}

    preferred_order = ['telea', 'ns', 'lama', 'lama_hybrid']
    methods = [m for m in preferred_order if m in methods_seen]
    for m in sorted(methods_seen):
        if m not in methods:
            methods.append(m)

    header = ['image']
    for m in methods:
        header.append(f"{m}_psnr")
    for m in methods:
        header.append(f"{m}_lpips")
    for m in methods:
        header.append(f"{m}_path")

    with open(out_agg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for img in sorted(data.keys()):
            row = [img]
            for m in methods:
                row.append(data[img].get(m, {}).get('psnr', ''))
            for m in methods:
                row.append(data[img].get(m, {}).get('lpips', ''))
            for m in methods:
                row.append(data[img].get(m, {}).get('path', ''))
            writer.writerow(row)

    print(f"Aggregated comparison written to {out_agg_csv}\n")

    # Print brief summary: per-method means for PSNR and LPIPS where available
    psnr_acc = defaultdict(list)
    lpips_acc = defaultdict(list)
    for img, md in data.items():
        for m in methods:
            entry = md.get(m)
            if not entry:
                continue
            p = entry.get('psnr')
            l = entry.get('lpips')
            try:
                if p is not None and p != '':
                    psnr_acc[m].append(float(p))
            except Exception:
                pass
            try:
                if l is not None and l != '':
                    lpips_acc[m].append(float(l))
            except Exception:
                pass

    print('Summary:')
    with open("results/metrics_output.txt", "w") as f:
        for m in methods:
            ps = psnr_acc.get(m, [])
            ls = lpips_acc.get(m, [])
            ps_mean = sum(ps)/len(ps) if ps else None
            ls_mean = sum(ls)/len(ls) if ls else None

            # Prepare the line
            line = f"{m}: PSNR mean = {ps_mean if ps_mean is None else f'{ps_mean:.4f}'} | LPIPS mean = {ls_mean if ls_mean is None else f'{ls_mean:.6f}'}\n"

            # Write to file
            f.write(line)

            # Also print
            print(line, end='')

    print('\nDone. Data saved to metrics_output.txt')