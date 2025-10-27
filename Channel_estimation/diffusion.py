"""
DDIM (Denoising Diffusion Implicit Model) Implementation
========================================================

Fast 4-step sampling for real-time channel estimation.
"""

import torch
import torch.nn as nn
import numpy as np


class DDIM:
    """
    Denoising Diffusion Implicit Model.
    
    Features:
    - Linear noise schedule
    - 4-step fast sampling
    - Posterior sampling with residual correction (from paper)
    """
    
    def __init__(
        self,
        timesteps=300,
        beta_start=1e-4,
        beta_end=0.035,
        beta_schedule='linear',
        device='cuda'
    ):
        self.timesteps = timesteps
        self.device = device
        
        # Create noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise NotImplementedError(f"Schedule {beta_schedule} not implemented")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        print(f"✅ DDIM initialized:")
        print(f"   Timesteps: {timesteps}")
        print(f"   Beta range: [{beta_start:.6f}, {beta_end:.6f}]")
        print(f"   Device: {device}")
    
    def to(self, device):
        """Move all tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
    
    def forward_diffusion(self, x0, t):
        """
        Add noise to clean data.
        
        q(x_t | x_0) = N(√ᾱₜ·x₀, (1-ᾱₜ)I)
        
        Args:
            x0: [B, C, H, W] clean data
            t: [B] timestep indices
        Returns:
            x_t: [B, C, H, W] noisy data
            noise: [B, C, H, W] added noise
        """
        # Get device from input
        device = x0.device
        
        # Ensure schedules are on the same device
        if self.sqrt_alphas_cumprod.device != device:
            self.to(device)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[:, None, None, None]
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward diffusion
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def denoise(self, model, x_t, num_steps=4):
        """
        Denoise noisy input using DDIM sampling.
        
        Uses 4-step accelerated sampling with posterior correction (from paper).
        
        Args:
            model: Denoising model
            x_t: [B, C, H, W] noisy input
            num_steps: Number of sampling steps (default 4)
        Returns:
            x_0: [B, C, H, W] denoised output
        """
        device = x_t.device
        B = x_t.shape[0]
        
        # Ensure schedules are on correct device
        if self.alphas_cumprod.device != device:
            self.to(device)
        
        # Create sampling timestep schedule (e.g., [300, 200, 100, 0])
        timestep_seq = np.linspace(self.timesteps-1, 0, num_steps, dtype=int)
        
        x = x_t
        
        for i, t in enumerate(timestep_seq[:-1]):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_next = timestep_seq[i + 1]
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # DDIM update with posterior correction (from paper)
            # x_{t-1} = √α_{t-1}·x̂_0 + √(1-α_{t-1})·ε_θ(x_t,t) + temporal_weight·residual
            
            # Standard DDIM step
            x_next = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
            
            # Posterior correction with temporal weighting (from paper Eq. 18)
            temporal_weight = 1 - t / self.timesteps
            residual = x - (torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * noise_pred)
            x_next = x_next + temporal_weight * residual
            
            x = x_next
        
        # Final step to t=0
        t_final = torch.zeros((B,), device=device, dtype=torch.long)
        noise_pred = model(x, t_final)
        x0_pred = (x - torch.sqrt(1 - self.alphas_cumprod[0]) * noise_pred) / torch.sqrt(self.alphas_cumprod[0])
        
        return x0_pred