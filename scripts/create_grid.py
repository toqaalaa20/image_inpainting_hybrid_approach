import cv2
import numpy as np
from pathlib import Path

def create_grid(raw_dir='data/raw', mask_dir='data/masks', class_dir='results/classical', lama_dir='results/lama', out_file='results/final_comparison_grid.jpg', n=6, target_h=256, randomize=False, hybrid_dir=None):
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    class_dir = Path(class_dir)
    lama_dir = Path(lama_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect hybrid directory if not provided on the CLI.
    # If a `results/lama_hybrid` folder exists, use it automatically so the user
    # doesn't have to pass --hybrid-dir.
    if hybrid_dir is None:
        candidate = Path('results/lama_hybrid')
        hybrid_dir = candidate if candidate.exists() else None
    else:
        hybrid_dir = Path(hybrid_dir)

    # collect candidate stems that have at least raw + lama (classical optional)
    stems = []
    for p in sorted(raw_dir.glob('*.jpg')):
        stem = p.stem
        # require raw and LaMa output; classical outputs are optional now
        required = [lama_dir / f"{stem}_lama.png",]
        # if classical directory exists, prefer to include entries that have classical outputs too
        # but do not strictly require them
        if all(r.exists() for r in required):
            stems.append(stem)

    if not stems:
        print('❌ No completed images found. Check your results folders.')
        return

    if randomize:
        import random
        random.shuffle(stems)

    stems = stems[:n]

    # Gather hybrid variant suffixes (e.g. "lama_hybrid_blend_cut02") across the selected stems
    hybrid_variants = []
    if hybrid_dir is not None:
        variant_set = []
        for stem in stems:
            # find any files that look like <stem>_*lama_hybrid*.png or <stem>*hybrid*.png
            for hp in hybrid_dir.glob(f"{stem}*hybrid*.png"):
                st = hp.stem
                # remove the leading '<stem>_' if present
                if st.startswith(stem + '_'):
                    var = st[len(stem) + 1:]
                else:
                    var = st
                if var not in variant_set:
                    variant_set.append(var)
        # include legacy generic name if present
        for s in stems:
            if (hybrid_dir / f"{s}_lama_hybrid.png").exists():
                if 'lama_hybrid' not in variant_set:
                    variant_set.append('lama_hybrid')
                break
        hybrid_variants = sorted(variant_set)

    rows = []
    for stem in stems:
        try:
            gt = cv2.imread(str(raw_dir / f"{stem}.jpg"))
            # mask might not exist; allow missing mask
            mask_p = mask_dir / f"{stem}_mask.png"
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None
            telea = cv2.imread(str(class_dir / f"{stem}_telea.png")) if (class_dir / f"{stem}_telea.png").exists() else None
            ns = cv2.imread(str(class_dir / f"{stem}_ns.png")) if (class_dir / f"{stem}_ns.png").exists() else None
            lama = cv2.imread(str(lama_dir / f"{stem}_lama.png"))

            # collect hybrid images matching each variant; keep order consistent with hybrid_variants
            hybrid_imgs = []
            if hybrid_dir is not None and hybrid_variants:
                for var in hybrid_variants:
                    # prefer exact '<stem>_<var>.png'
                    candidate = hybrid_dir / f"{stem}_{var}.png"
                    img = None
                    if candidate.exists():
                        img = cv2.imread(str(candidate))
                    else:
                        # fallback: find any filename that contains the variant token
                        for hp in hybrid_dir.glob(f"{stem}*{var}*.png"):
                            img = cv2.imread(str(hp))
                            if img is not None:
                                break
                    hybrid_imgs.append(img)

            # If essential images missing skip
            if gt is None or lama is None:
                print(f"⚠️  Skipping {stem}: some essential files unreadable")
                continue

            # Create masked view: if mask exists, show white where mask > 0; else show original again
            masked_view = gt.copy()
            if mask is not None:
                gray_mask = mask
                masked_view[gray_mask > 0] = [255, 255, 255]

            # Resize each to target_h (preserve aspect ratio)
            def resize_keep_h(im, h):
                H, W = im.shape[:2]
                if H == h:
                    return im
                new_w = int(W * (h / H))
                return cv2.resize(im, (new_w, h), interpolation=cv2.INTER_AREA)

            gt_r = resize_keep_h(gt, target_h)
            mv_r = resize_keep_h(masked_view, target_h)
            telea_r = resize_keep_h(telea, target_h) if telea is not None else None
            ns_r = resize_keep_h(ns, target_h) if ns is not None else None
            lama_r = resize_keep_h(lama, target_h)

            hybrid_rs = []
            if hybrid_imgs:
                for hi in hybrid_imgs:
                    if hi is not None:
                        hybrid_rs.append(resize_keep_h(hi, target_h))
                    else:
                        placeholder_w = telea_r.shape[1] if telea_r is not None else lama_r.shape[1]
                        hybrid_rs.append(np.full((target_h, placeholder_w, 3), 255, dtype=np.uint8))

            # Ensure same number of channels
            def ensure_3c(im):
                if len(im.shape) == 2:
                    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                return im

            gt_r = ensure_3c(gt_r)
            mv_r = ensure_3c(mv_r)
            telea_r = ensure_3c(telea_r) if telea_r is not None else None
            ns_r = ensure_3c(ns_r) if ns_r is not None else None
            lama_r = ensure_3c(lama_r)
            hybrid_rs = [ensure_3c(x) for x in hybrid_rs]

            # Put small labels on images
            font = cv2.FONT_HERSHEY_SIMPLEX
            def put_label(im, text):
                out = im.copy()
                # Draw semi-transparent rectangle for better visibility
                h, w = out.shape[:2]
                cv2.rectangle(out, (0,0), (w, 26), (0,0,0), -1)
                cv2.putText(out, text, (8, 18), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                return out

            gt_r = put_label(gt_r, 'Original')
            mv_r = put_label(mv_r, 'Masked')
            telea_r = put_label(telea_r, 'Classical (Telea)') if telea_r is not None else None
            ns_r = put_label(ns_r, 'Classical (Navier–Stokes)') if ns_r is not None else None
            lama_r = put_label(lama_r, 'LaMa')
            # label hybrid columns with their variant token
            hybrid_rs = [put_label(x, f'LaMa Hybrid ({v})') for x, v in zip(hybrid_rs, hybrid_variants)]

            # Build row horizontally
            parts = [gt_r, mv_r]
            if telea_r is not None:
                parts.append(telea_r)
            if ns_r is not None:
                parts.append(ns_r)
            parts.append(lama_r)
            parts += hybrid_rs
            row = np.hstack(parts)

            rows.append(row)
        except Exception as e:
            print(f"Error processing {stem}: {e}")

    if not rows:
        print('❌ No rows to save after processing. Aborting.')
        return

    # Stack rows vertically, add small spacing between rows
    spacer = np.full((8, rows[0].shape[1], 3), 255, dtype=np.uint8)
    final = rows[0]
    for r in rows[1:]:
        final = np.vstack((final, spacer, r))

    cv2.imwrite(str(out_file), final)
    print(f"📸 Success! Grid saved to {out_file}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Create a visual comparison grid for inpainting results')
    p.add_argument('--raw-dir', default='data/raw')
    p.add_argument('--mask-dir', default='data/masks')
    p.add_argument('--class-dir', default='results/classical')
    p.add_argument('--lama-dir', default='results/lama')
    p.add_argument('--out-file', default='results/final_comparison_grid.jpg')
    p.add_argument('-n', type=int, default=6, help='Number of samples (rows)')
    p.add_argument('--height', type=int, default=256, help='Target height for each sample')
    p.add_argument('--randomize', action='store_true')
    p.add_argument('--hybrid-dir', default=None, help='Optional LaMa hybrid outputs dir')
    args = p.parse_args()

    create_grid(args.raw_dir, args.mask_dir, args.class_dir, args.lama_dir, args.out_file, n=args.n, target_h=args.height, randomize=args.randomize, hybrid_dir=args.hybrid_dir)