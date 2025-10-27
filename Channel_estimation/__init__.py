"""
Diffusion-based Channel State Information (CSI) Estimation
==========================================================

A high-performance, GPU-accelerated implementation of denoising diffusion
models for MIMO channel estimation in 5G/6G wireless systems.

Directory Structure:
-------------------
Channel_estimation/     # This directory (code files)
    __init__.py
    dataset.py         # Dataset loader
    channel_losses.py  # Loss functions
    models.py          # Diffusion models (U-Net, AMM)
    diffusion.py       # DDIM implementation
    train.py          # MAIN TRAINING SCRIPT ← RUN THIS
    inference.py       # Prediction script
    utils.py          # Utilities

Dataset/               # Data directory (sibling folder)
    umi_channel_data.mat    # Your 10GB MATLAB file

Usage:
------
1. Place your .mat file in: ../Dataset/umi_channel_data.mat
2. Run training: python train.py
3. Run inference: python inference.py --checkpoint best_model.pth

Author: Wireless CSI Research
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "Wireless AI Research"

import os
from pathlib import Path

# Get absolute paths
CHANNEL_ESTIMATION_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CHANNEL_ESTIMATION_DIR.parent
DATASET_DIR = PROJECT_ROOT / 'Dataset'
CHECKPOINTS_DIR = CHANNEL_ESTIMATION_DIR / 'checkpoints'
LOGS_DIR = CHANNEL_ESTIMATION_DIR / 'logs'
RESULTS_DIR = CHANNEL_ESTIMATION_DIR / 'results'

# Create directories if they don't exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# CUDA availability check
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

print("="*70)
print("🚀 Diffusion-Based CSI Estimation Package")
print("="*70)

if CUDA_AVAILABLE:
    print(f"✅ CUDA Available:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available. Using CPU (training will be very slow!)")

print(f"\n📁 Directory Structure:")
print(f"   Code: {CHANNEL_ESTIMATION_DIR}")
print(f"   Data: {DATASET_DIR}")
print(f"   Checkpoints: {CHECKPOINTS_DIR}")
print(f"   Logs: {LOGS_DIR}")
print(f"   Results: {RESULTS_DIR}")
print("="*70)

# Configuration defaults
CONFIG = {
    # Paths
    'dataset_dir': str(DATASET_DIR),
    'checkpoint_dir': str(CHECKPOINTS_DIR),
    'log_dir': str(LOGS_DIR),
    'results_dir': str(RESULTS_DIR),
    
    # Hardware
    'device': DEVICE,
    'mixed_precision': True if CUDA_AVAILABLE else False,
    'num_workers': 4,
    'pin_memory': True if CUDA_AVAILABLE else False,
    
    # Dataset file (default name)
    'mat_file': 'umi_channel_data.mat',
}