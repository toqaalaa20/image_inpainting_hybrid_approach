# Image Inpainting — Hybrid Evaluation Suite

A benchmarking and evaluation suite that compares classical inpainting methods (e.g. Navier‑Stokes, Telea) with a modern deep learning model (LaMa) and a hybrid postprocessing pipeline built on top of LaMa outputs.

This repository runs multiple inpainting variants on a dataset of images and masks, computes quantitative metrics (PSNR, LPIPS), and generates visual comparison grids for easy inspection.

Key goals
- Reproduce and compare results from classical algorithms and a state-of-the-art deep model (LaMa).
- Apply hybrid postprocessing to LaMa outputs (blend / lowpass variants) and sweep parameters.
- Produce metrics and visual grids used to select the best method per image.

Contents
- data/: input images and masks used for experiments
  - data/raw/: original ground-truth images (expected .jpg files)
  - data/masks/: binary masks used to remove image regions for inpainting
  - data/test_256/: optional preprocessed test images
- scripts/: utility scripts to run experiments and evaluations
  - run_lama.py — run the LaMa model on inputs
  - run_classical.py — run classical inpainting (OpenCV Telea / Navier‑Stokes)
  - hybrid_postprocess.py — hybrid postprocessing functions (process_image)
  - run_hybrid_lama_variations.py — driver that applies hybrid variants to LaMa outputs
  - run_metrics.py — compute PSNR / LPIPS and write CSV outputs
  - create_grid.py — create final comparison image grids
  - select_best_method.py — pick best variant per image using metrics
  - utils.py, sample_data.py, generate_masks.py, etc.
- results/: generated outputs grouped by method (lama, lama_hybrid, classical, best_variant)
- weights/: pretrained model weights required by LaMa (e.g. big-lama.pt)
- scripts/*.py produce artifacts under results/ so evaluation scripts can auto-detect them.

Prerequisites
- A machine with an NVIDIA GPU (recommended) or a CPU-only setup (will be slower / may require different torch build).
- Docker (recommended) or a compatible Python environment with the packages listed below.

Recommended environment (Docker)
To avoid dependency issues and ensure CUDA/PyTorch compatibility, a Docker image is provided via the included Dockerfile.

Build the Docker image

```bash
docker build -t inpaint-gpu .
```

Run an interactive container with GPU access

```bash
docker run --gpus all -it --rm \
  -u $(id -u):$(id -g) \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e XDG_CACHE_HOME=/workspace/.cache \
  --shm-size=8g inpaint-gpu
```


Quick start — run the full evaluation
1. Ensure `data/raw/` contains ground-truth JPG images named like `XXXX.jpg` and `data/masks/` contains matching masks named like `XXXX_mask.png`.
2. Place LaMa weights in `weights/` (for example `big-lama.pt`) if required by the LaMa runner.
3. From inside the Docker container (or a configured Python env), run:

```bash
# Run LaMa model on the inputs (produces results/lama)
python3 scripts/run_lama.py

# Optionally run classical methods (produces results/classical)
python3 scripts/run_classical.py

# Run hybrid postprocessing variations (produces results/lama_hybrid)
python3 scripts/run_hybrid_lama_variations.py

# Compute metrics (PSNR, LPIPS) and write CSV outputs
python3 scripts/run_metrics.py

# Create visual comparison grid(s)
python3 scripts/create_grid.py
```

Notes on `run_hybrid_lama_variations.py`
- This driver imports `process_image` from `scripts/hybrid_postprocess.py` and applies multiple combinations of modes and cutoff thresholds to every LaMa output.
- Default output directory is `results/lama_hybrid` so the evaluation helpers auto-detect them.
- Configuration constants at the top of the script let you change: MODES, CUTOFFS, OVERWRITE, LIMIT.

Results and outputs
- `results/lama/` — LaMa outputs
- `results/classical/` — classical inpainting outputs
- `results/lama_hybrid/` — hybrid postprocessed variants produced by the driver
- `results/best_variant/` — chosen best variant per image (if selection is run)
- `results/metrics*.csv` and `final_comparison_grid.jpg` — aggregated metrics and final visualization

How metrics are computed
- PSNR: pixel-wise reconstruction quality (higher is better).
- LPIPS: learned perceptual metric (lower is better for similarity) using the standard LPIPS model.
- The script `run_metrics.py` iterates over results and compares them to the ground truth images in `data/raw/`.

Customizing experiments
- To change the dataset, replace `data/raw/` and `data/masks/` contents with your own images/masks (keep consistent naming).
- To change hybrid sweep parameters, edit `scripts/run_hybrid_lama_variations.py` or call `scripts.hybrid_postprocess.process_image` directly from your own driver.
- To add another inpainting method, implement it in `scripts/` and write outputs to `results/<method>/` following the existing filename conventions (stem + suffix).
