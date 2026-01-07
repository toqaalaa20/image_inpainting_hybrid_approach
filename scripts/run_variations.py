#!/usr/bin/env python3
"""
Run multiple hybrid postprocessing variations (different modes and cutoffs) and then
run the metrics and grid creators without requiring CLI arguments.

Usage: python3 scripts/run_variations.py

This script writes outputs into the auto-detected folder `results/lama_hybrid` so
existing evaluation scripts (`run_metrics.py`, `create_grid.py`) will pick them up
without passing extra arguments.
"""
from pathlib import Path
import subprocess
import time

# Import the process_image function from the hybrid script
# We import to call the function directly rather than shelling out.
try:
    from scripts.hybrid_postprocess import process_image
except Exception:
    # If importing as module fails due to package path, try relative import via sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.hybrid_postprocess import process_image


RAW_DIR = Path('data/raw')
LAMA_DIR = Path('results/lama')
OUT_DIR = Path('results/lama_hybrid')  # chosen so run_metrics/create_grid auto-detect
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ['blend', 'lowpass']
CUTOFFS = [0.01, 0.02, 0.05, 0.10]
OVERWRITE = True
LIMIT = None  # set to an int to limit number of images


def main():
    files = sorted(LAMA_DIR.glob('*_lama.png'))
    if LIMIT is not None:
        files = files[:LIMIT]

    if not files:
        print('No LaMa outputs found in', LAMA_DIR)
        return

    for pth in files:
        stem = pth.stem.replace('_lama', '')
        gt_p = RAW_DIR / f"{stem}.jpg"
        if not gt_p.exists():
            print(f"Skipping {stem}: ground truth not found ({gt_p})")
            continue

        for mode in MODES:
            for cutoff in CUTOFFS:
                out_name = f"{stem}_lama_hybrid_{mode}_cut{int(cutoff*100):02d}.png"
                out_p = OUT_DIR / out_name
                if out_p.exists() and not OVERWRITE:
                    print(f"Skipping existing {out_p}")
                    continue
                try:
                    print(f"Processing {stem} | mode={mode} cutoff={cutoff} -> {out_p.name}")
                    process_image(pth, gt_p, out_p, mode=mode, cutoff=cutoff)
                except Exception as e:
                    print(f"Failed {pth.name} mode={mode} cutoff={cutoff}: {e}")

    # Small pause to ensure files are flushed
    time.sleep(0.5)

    # Run metrics and create grid without any CLI args (they auto-detect results/lama_hybrid)
    print('\nRunning metrics...')
    subprocess.run(['python3', 'scripts/run_metrics.py'], check=False)

    print('\nCreating final grid...')
    subprocess.run(['python3', 'scripts/create_grid.py'], check=False)

    print('\nAll done.')


if __name__ == '__main__':
    main()
