"""
Utility Functions
=================

Metrics, checkpointing, and helper functions.
"""

import torch
import numpy as np
from pathlib import Path


def compute_nmse_db(pred, target, eps=1e-10):
    """
    Compute NMSE in dB.
    
    NMSE = 10*log10(||pred - target||² / ||target||²)
    
    Args:
        pred: [B, ...] predicted (complex)
        target: [B, ...] target (complex)
    Returns:
        NMSE in dB (scalar)
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    mse = torch.mean(torch.abs(pred_flat - target_flat) ** 2, dim=1)
    signal_power = torch.mean(torch.abs(target_flat) ** 2, dim=1) + eps
    
    nmse = mse / signal_power
    nmse_db = 10 * torch.log10(nmse.mean() + eps)
    
    return nmse_db


def compute_evm_percent(pred, target, eps=1e-10):
    """
    Compute Error Vector Magnitude in percent.
    
    EVM = 100 * sqrt(E[||pred-target||²]) / sqrt(E[||target||²])
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    rms_error = torch.sqrt(torch.mean(torch.abs(pred_flat - target_flat) ** 2, dim=1))
    rms_signal = torch.sqrt(torch.mean(torch.abs(target_flat) ** 2, dim=1) + eps)
    
    evm = 100.0 * (rms_error / rms_signal)
    
    return evm.mean()


def compute_correlation(pred, target, eps=1e-10):
    """
    Compute complex correlation coefficient.
    
    corr = |<pred, target>| / (||pred|| · ||target||)
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    inner_prod = torch.sum(pred_flat * torch.conj(target_flat), dim=1)
    pred_norm = torch.sqrt(torch.sum(torch.abs(pred_flat) ** 2, dim=1) + eps)
    target_norm = torch.sqrt(torch.sum(torch.abs(target_flat) ** 2, dim=1) + eps)
    
    corr = torch.abs(inner_prod) / (pred_norm * target_norm + eps)
    
    return corr.mean()


def save_checkpoint(model, optimizer, epoch, best_metric, filepath):
    """Save model checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_nmse': best_metric,
    }
    
    torch.save(checkpoint, filepath)
    print(f"   💾 Checkpoint saved: {filepath}")


def load_checkpoint(filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)