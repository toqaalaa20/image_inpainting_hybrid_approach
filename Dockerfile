# Use the official NVIDIA PyTorch image (Optimized for RTX 30-series)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables to prevent prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install project-specific Python libraries
RUN pip install --upgrade pip && \
    pip install opencv-python "numpy<2.0" matplotlib scipy tqdm lpips torchmetrics pandas

# Set the working directory inside the container
WORKDIR /workspace