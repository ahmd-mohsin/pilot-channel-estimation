"""
MAIN TRAINING SCRIPT - Run this to train the diffusion model!
==============================================================

Usage:
    python train.py [options]

Example:
    python train.py --epochs 500 --batch_size 16 --lr 5e-4

This script will:
1. Load your dataset from ../Dataset/umi_channel_data.mat
2. Create diffusion model with U-Net + AMM
3. Train using custom CSI losses (SP-NMSE + correlation)
4. Save checkpoints to ./checkpoints/
5. Log metrics to TensorBoard in ./logs/
6. Save final model as best_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from dataset import CSIDataset, create_dataloaders
from channel_losses import CompositeLoss, NMSELoss
from models import EnhancedDiffusionUNet
from diffusion import DDIM
from utils import compute_nmse_db, save_checkpoint, load_checkpoint

# Get paths
CHANNEL_ESTIMATION_DIR = Path(__file__).parent
DATASET_DIR = CHANNEL_ESTIMATION_DIR.parent / 'Dataset'
CHECKPOINT_DIR = CHANNEL_ESTIMATION_DIR / 'checkpoints'
LOG_DIR = CHANNEL_ESTIMATION_DIR / 'logs'

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Diffusion Model for CSI Estimation')
    
    # Data
    parser.add_argument('--data_file', type=str, default='umi_channel_data.mat',
                       help='Name of .mat file in Dataset/ folder')
    parser.add_argument('--normalize', type=str, default='per_sample',
                       choices=['global', 'per_sample', 'per_slot'],
                       help='Normalization method')
    
    # Model
    parser.add_argument('--model_channels', type=int, default=64,
                       help='Base channel width (32=small, 64=medium, 128=large)')
    parser.add_argument('--use_amm', action='store_true', default=True,
                       help='Use Antenna Modifier Module')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use attention layers')
    
    # Diffusion
    parser.add_argument('--timesteps', type=int, default=300,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                       help='Start of noise schedule')
    parser.add_argument('--beta_end', type=float, default=0.035,
                       help='End of noise schedule')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (reduce if OOM)')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Loss weights
    parser.add_argument('--loss_primary', type=str, default='sp_nmse',
                       choices=['nmse', 'sp_nmse', 'corr', 'evm'],
                       help='Primary loss function')
    parser.add_argument('--weight_smooth', type=float, default=0.01,
                       help='Smoothness loss weight')
    parser.add_argument('--weight_corr', type=float, default=0.1,
                       help='Correlation loss weight')
    
    # Optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate on test set every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    return parser.parse_args()


def train_epoch(model, diffusion, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_sum = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (noisy_csi, clean_csi) in enumerate(pbar):
        # Move to device
        noisy_csi = noisy_csi.to(device)  # [B, 768, 14, 4, 2, 10] complex
        clean_csi = clean_csi.to(device)  # [B, 768, 14, 4, 2, 10] complex
        
        # Convert complex to real (2 channels: real, imag)
        # Shape: [B, 768, 14, 4, 2, 10] -> [B, 2, 768, 14, 4, 2, 10]
        noisy_real = torch.stack([noisy_csi.real, noisy_csi.imag], dim=1)
        clean_real = torch.stack([clean_csi.real, clean_csi.imag], dim=1)
        
        # Flatten spatial dimensions for model
        # [B, 2, 768, 14, 4, 2, 10] -> [B, 2, 768*14, 4*2*10]
        B = noisy_real.shape[0]
        noisy_input = noisy_real.reshape(B, 2, 768*14, 4*2*10)  # [B, 2, 10752, 80]
        clean_input = clean_real.reshape(B, 2, 768*14, 4*2*10)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=scaler is not None):
            # Sample timestep
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            
            # Forward diffusion: add noise to clean CSI
            noisy_diffused, noise_gt = diffusion.forward_diffusion(clean_input, t)
            
            # Predict noise with model
            noise_pred = model(noisy_diffused, t)
            
            # Compute loss
            # Convert back to complex for loss computation
            noise_pred_complex = torch.complex(
                noise_pred[:, 0:1, :, :],
                noise_pred[:, 1:2, :, :]
            ).reshape(B, 768, 14, 4, 2, 10)
            
            noise_gt_complex = torch.complex(
                noise_gt[:, 0:1, :, :],
                noise_gt[:, 1:2, :, :]
            ).reshape(B, 768, 14, 4, 2, 10)
            
            loss, loss_dict = criterion(noise_pred_complex, noise_gt_complex)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


@torch.no_grad()
def evaluate(model, diffusion, test_loader, criterion, device, num_sampling_steps=4):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    total_nmse = 0
    
    pbar = tqdm(test_loader, desc='Evaluating')
    
    for noisy_csi, clean_csi in pbar:
        noisy_csi = noisy_csi.to(device)
        clean_csi = clean_csi.to(device)
        
        # Convert to real
        noisy_real = torch.stack([noisy_csi.real, noisy_csi.imag], dim=1)
        clean_real = torch.stack([clean_csi.real, clean_csi.imag], dim=1)
        
        # Flatten
        B = noisy_real.shape[0]
        noisy_input = noisy_real.reshape(B, 2, 768*14, 4*2*10)
        clean_input = clean_real.reshape(B, 2, 768*14, 4*2*10)
        
        # Denoise with model (fast sampling)
        denoised = diffusion.denoise(
            model, noisy_input, num_steps=num_sampling_steps
        )
        
        # Convert back to complex
        denoised_complex = torch.complex(
            denoised[:, 0:1, :, :],
            denoised[:, 1:2, :, :]
        ).reshape(B, 768, 14, 4, 2, 10)
        
        # Compute NMSE
        nmse = compute_nmse_db(denoised_complex, clean_csi)
        total_nmse += nmse.item()
        
        # Compute loss
        loss, _ = criterion(denoised_complex, clean_csi)
        total_loss += loss.item()
        
        pbar.set_postfix({'nmse_db': nmse.item()})
    
    avg_loss = total_loss / len(test_loader)
    avg_nmse = total_nmse / len(test_loader)
    
    return avg_loss, avg_nmse


def main(args):
    """Main training function."""
    
    # Print configuration
    print("\n" + "="*70)
    print("🚀 TRAINING DIFFUSION MODEL FOR CSI ESTIMATION")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*70 + "\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'diffusion_csi_{timestamp}'
    
    exp_dir = CHECKPOINT_DIR / args.exp_name
    exp_dir.mkdir(exist_ok=True)
    
    # Save args
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # TensorBoard
    writer = SummaryWriter(LOG_DIR / args.exp_name)
    
    # Create dataloaders
    print("📦 Loading dataset...")
    data_path = DATASET_DIR / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    train_loader, test_loader = create_dataloaders(
        mat_file_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        normalize_method=args.normalize
    )
    
    # Create model
    print("\n🏗️  Building model...")
    model = EnhancedDiffusionUNet(
        in_channels=2,  # real + imaginary
        model_channels=args.model_channels,
        out_channels=2,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=[16, 8] if args.use_attention else [],
        use_amm=args.use_amm,
        spatial_size=(768*14, 4*2*10)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Create diffusion
    diffusion = DDIM(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    
    # Create loss
    criterion = CompositeLoss(
        primary_loss=args.loss_primary,
        use_smoothness=True,
        use_correlation=True,
        weight_smoothness=args.weight_smooth,
        weight_correlation=args.weight_corr
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr/100
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_nmse = float('inf')
    
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_nmse = checkpoint['best_nmse']
    
    # Training loop
    print("\n" + "="*70)
    print("🎯 STARTING TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, diffusion, train_loader, criterion, 
            optimizer, scaler, device, epoch
        )
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        for k, v in train_loss_dict.items():
            writer.add_scalar(f'Loss/train_{k}', v, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            test_loss, test_nmse = evaluate(
                model, diffusion, test_loader, criterion, device
            )
            
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('NMSE/test_db', test_nmse, epoch)
            
            print(f"   Test Loss: {test_loss:.6f}, NMSE: {test_nmse:.2f} dB")
            
            # Save best model
            if test_nmse < best_nmse:
                best_nmse = test_nmse
                save_checkpoint(
                    model, optimizer, epoch, best_nmse,
                    exp_dir / 'best_model.pth'
                )
                print(f"   ✅ New best model saved! NMSE: {best_nmse:.2f} dB")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, best_nmse,
                exp_dir / f'checkpoint_epoch_{epoch}.pth'
            )
        
        # Step scheduler
        scheduler.step()
    
    # Final save
    save_checkpoint(
        model, optimizer, args.epochs-1, best_nmse,
        exp_dir / 'final_model.pth'
    )
    
    writer.close()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best NMSE: {best_nmse:.2f} dB")
    print(f"Models saved to: {exp_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {LOG_DIR}")
    print("="*70 + "\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)