"""
Custom Loss Functions for Wireless Channel Estimation
=====================================================

Implements advanced loss functions designed specifically for MIMO CSI:
- Scale/Phase Invariant NMSE
- Correlation Loss
- Composite losses with regularization
- EVM-based losses

All losses handle complex-valued tensors properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NMSELoss(nn.Module):
    """
    Normalized Mean Square Error Loss for channel estimation.
    
    NMSE = ||H_est - H_true||^2 / ||H_true||^2
    
    More robust than MSE across different SNR regimes.
    """
    
    def __init__(self, eps: float = 1e-10, reduction: str = 'mean', in_db: bool = False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.in_db = in_db
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted channel [B, K, L, nRx, nTx, slots] (complex)
            target: Target channel [B, K, L, nRx, nTx, slots] (complex)
        
        Returns:
            NMSE loss (scalar or per-batch depending on reduction)
        """
        # Flatten all dimensions except batch
        pred_flat = pred.reshape(pred.shape[0], -1)  # [B, K*L*nRx*nTx*slots]
        target_flat = target.reshape(target.shape[0], -1)
        
        # Compute MSE
        mse = torch.sum(torch.abs(pred_flat - target_flat) ** 2, dim=1)  # [B]
        
        # Compute signal power
        signal_power = torch.sum(torch.abs(target_flat) ** 2, dim=1) + self.eps  # [B]
        
        # NMSE
        nmse = mse / signal_power  # [B]
        
        # Convert to dB if requested
        if self.in_db:
            nmse = 10 * torch.log10(nmse + self.eps)
        
        # Reduction
        if self.reduction == 'mean':
            return nmse.mean()
        elif self.reduction == 'sum':
            return nmse.sum()
        elif self.reduction == 'none':
            return nmse
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class ScalePhaseInvariantNMSE(nn.Module):
    """
    Scale and Phase Invariant NMSE Loss.
    
    Accounts for global complex scaling/phase ambiguity:
    α* = <H_est, H_true> / ||H_true||^2
    H_aligned = α* · H_true
    Loss = ||H_est - H_aligned||^2 / ||H_true||^2
    
    This handles timing/frequency offset effects.
    """
    
    def __init__(self, eps: float = 1e-10, reduction: str = 'mean', in_db: bool = False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.in_db = in_db
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted channel [B, ...] (complex)
            target: Target channel [B, ...] (complex)
        """
        # Flatten
        pred_flat = pred.reshape(pred.shape[0], -1)  # [B, N]
        target_flat = target.reshape(target.shape[0], -1)  # [B, N]
        
        # Compute optimal complex scale factor
        # α* = <pred, target> / ||target||^2
        inner_prod = torch.sum(pred_flat * torch.conj(target_flat), dim=1)  # [B]
        target_power = torch.sum(torch.abs(target_flat) ** 2, dim=1) + self.eps  # [B]
        
        alpha = inner_prod / target_power  # [B] complex scalar
        
        # Align target: target_aligned = α* · target
        target_aligned = alpha.unsqueeze(1) * target_flat  # [B, N]
        
        # Compute NMSE with aligned target
        mse = torch.sum(torch.abs(pred_flat - target_aligned) ** 2, dim=1)  # [B]
        nmse = mse / target_power  # [B]
        
        # Convert to dB if requested
        if self.in_db:
            nmse = 10 * torch.log10(nmse + self.eps)
        
        # Reduction
        if self.reduction == 'mean':
            return nmse.mean()
        elif self.reduction == 'sum':
            return nmse.sum()
        elif self.reduction == 'none':
            return nmse
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class CorrelationLoss(nn.Module):
    """
    Complex Correlation Loss.
    
    Loss = 1 - |<H_est, H_true>| / (||H_est|| · ||H_true||)
    
    Measures complex correlation (0 = perfect, 2 = worst).
    Invariant to global complex scaling.
    """
    
    def __init__(self, eps: float = 1e-10, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted channel [B, ...] (complex)
            target: Target channel [B, ...] (complex)
        """
        # Flatten
        pred_flat = pred.reshape(pred.shape[0], -1)  # [B, N]
        target_flat = target.reshape(target.shape[0], -1)  # [B, N]
        
        # Inner product
        inner_prod = torch.sum(pred_flat * torch.conj(target_flat), dim=1)  # [B] complex
        
        # Norms
        pred_norm = torch.sqrt(torch.sum(torch.abs(pred_flat) ** 2, dim=1) + self.eps)  # [B]
        target_norm = torch.sqrt(torch.sum(torch.abs(target_flat) ** 2, dim=1) + self.eps)  # [B]
        
        # Correlation
        corr = torch.abs(inner_prod) / (pred_norm * target_norm + self.eps)  # [B]
        
        # Loss = 1 - correlation
        loss = 1.0 - corr  # [B]
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class EVMLoss(nn.Module):
    """
    Error Vector Magnitude (EVM) Loss.
    
    EVM = sqrt(E[||H_est - H_true||^2]) / sqrt(E[||H_true||^2])
    
    Commonly used in RF measurements.
    """
    
    def __init__(self, eps: float = 1e-10, reduction: str = 'mean', as_percentage: bool = False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.as_percentage = as_percentage
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        # RMS error
        rms_error = torch.sqrt(torch.mean(torch.abs(pred_flat - target_flat) ** 2, dim=1))  # [B]
        
        # RMS signal
        rms_signal = torch.sqrt(torch.mean(torch.abs(target_flat) ** 2, dim=1) + self.eps)  # [B]
        
        # EVM
        evm = rms_error / rms_signal  # [B]
        
        if self.as_percentage:
            evm = evm * 100.0
        
        # Reduction
        if self.reduction == 'mean':
            return evm.mean()
        elif self.reduction == 'sum':
            return evm.sum()
        elif self.reduction == 'none':
            return evm
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class TimeFrequencySmoothnessLoss(nn.Module):
    """
    Encourages time-frequency smoothness.
    
    L_smooth = λ_t * ||∇_t H||^2 + λ_f * ||∇_f H||^2
    
    Helps with slight motion and promotes physically plausible channels.
    """
    
    def __init__(self, lambda_time: float = 0.1, lambda_freq: float = 0.1):
        super().__init__()
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted channel [B, K, L, nRx, nTx, slots] (complex)
        """
        loss = 0.0
        
        # Time smoothness (across OFDM symbols L)
        if self.lambda_time > 0:
            # Gradient along dimension 2 (L - OFDM symbols)
            diff_time = pred[:, :, 1:, :, :, :] - pred[:, :, :-1, :, :, :]
            loss_time = torch.mean(torch.abs(diff_time) ** 2)
            loss = loss + self.lambda_time * loss_time
        
        # Frequency smoothness (across subcarriers K)
        if self.lambda_freq > 0:
            # Gradient along dimension 1 (K - subcarriers)
            diff_freq = pred[:, 1:, :, :, :, :] - pred[:, :-1, :, :, :, :]
            loss_freq = torch.mean(torch.abs(diff_freq) ** 2)
            loss = loss + self.lambda_freq * loss_freq
        
        return loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple objectives.
    
    L = L_primary + β·L_smooth + γ·L_corr
    
    Recommended configuration:
    - Primary: ScalePhaseInvariantNMSE or CorrelationLoss
    - Smoothness: Small weight for stability
    - Additional correlation term for alignment
    """
    
    def __init__(
        self,
        primary_loss: str = 'sp_nmse',  # 'nmse', 'sp_nmse', 'corr', 'evm'
        use_smoothness: bool = True,
        use_correlation: bool = True,
        weight_primary: float = 1.0,
        weight_smoothness: float = 0.01,
        weight_correlation: float = 0.1,
        in_db: bool = False
    ):
        super().__init__()
        
        self.weight_primary = weight_primary
        self.weight_smoothness = weight_smoothness
        self.weight_correlation = weight_correlation
        
        # Primary loss
        if primary_loss == 'nmse':
            self.primary = NMSELoss(in_db=in_db)
        elif primary_loss == 'sp_nmse':
            self.primary = ScalePhaseInvariantNMSE(in_db=in_db)
        elif primary_loss == 'corr':
            self.primary = CorrelationLoss()
        elif primary_loss == 'evm':
            self.primary = EVMLoss()
        else:
            raise ValueError(f"Unknown primary loss: {primary_loss}")
        
        # Smoothness loss
        if use_smoothness:
            self.smoothness = TimeFrequencySmoothnessLoss()
        else:
            self.smoothness = None
        
        # Correlation loss
        if use_correlation and primary_loss != 'corr':
            self.correlation = CorrelationLoss()
        else:
            self.correlation = None
        
        print(f"📊 Composite Loss initialized:")
        print(f"   Primary: {primary_loss} (weight={weight_primary})")
        print(f"   Smoothness: {use_smoothness} (weight={weight_smoothness})")
        print(f"   Correlation: {use_correlation} (weight={weight_correlation})")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        # Primary loss
        loss_primary = self.primary(pred, target)
        total_loss = self.weight_primary * loss_primary
        
        loss_dict = {'primary': loss_primary.item()}
        
        # Smoothness loss
        if self.smoothness is not None:
            loss_smooth = self.smoothness(pred)
            total_loss = total_loss + self.weight_smoothness * loss_smooth
            loss_dict['smoothness'] = loss_smooth.item()
        
        # Correlation loss
        if self.correlation is not None:
            loss_corr = self.correlation(pred, target)
            total_loss = total_loss + self.weight_correlation * loss_corr
            loss_dict['correlation'] = loss_corr.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Quick test
def test_losses():
    """Test all loss functions."""
    print("\n🧪 Testing Loss Functions...")
    
    # Create dummy data
    B, K, L, nRx, nTx, slots = 4, 768, 14, 4, 2, 10
    pred = torch.randn(B, K, L, nRx, nTx, slots, dtype=torch.cfloat)
    target = torch.randn(B, K, L, nRx, nTx, slots, dtype=torch.cfloat)
    
    # Test each loss
    losses = {
        'NMSE': NMSELoss(),
        'SP-NMSE': ScalePhaseInvariantNMSE(),
        'Correlation': CorrelationLoss(),
        'EVM': EVMLoss(),
        'Smoothness': TimeFrequencySmoothnessLoss(),
    }
    
    for name, loss_fn in losses.items():
        if name == 'Smoothness':
            loss = loss_fn(pred)
        else:
            loss = loss_fn(pred, target)
        print(f"  {name:15s}: {loss.item():.6f}")
    
    # Test composite
    print("\n  Testing Composite Loss:")
    composite = CompositeLoss()
    total_loss, loss_dict = composite(pred, target)
    for k, v in loss_dict.items():
        print(f"    {k:12s}: {v:.6f}")
    
    print("\n✅ All losses tested successfully!")


if __name__ == "__main__":
    test_losses()