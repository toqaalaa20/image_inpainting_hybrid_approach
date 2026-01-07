# Project Overview

This project compares classical mathematical inpainting (Navier-Stokes, Telea) against modern deep learning models (LaMa) using the Inpaint32K/Places2 datasets. It evaluates performance using PSNR and LPIPS metrics.

## Step 1: Environment Setup (Docker + GPU)

To ensure the RTX 3060 is utilized and to avoid dependency conflicts, the environment is containerized using Docker.
### 1.1 Dockerfile Configuration

The environment uses cuda11.8 and pytorch 2.1.0 with a specific fix for numpy compatibility.
1.2 Build and Run
Bash

#### Build the image
```
docker build -t inpaint-gpu .
```
#### Launch the container with GPU access and folder mirroring
```
# recommended: mount project, set working dir, and provide a writable cache for torchvision/LPIPS
docker run --gpus all -it --rm \
  -u $(id -u):$(id -g) \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e XDG_CACHE_HOME=/workspace/.cache \
  --shm-size=8g inpaint-gpu
```