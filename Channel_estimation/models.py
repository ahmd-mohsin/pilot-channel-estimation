"""
Diffusion Model Architectures
==============================

Enhanced U-Net with:
- Antenna Modifier Module (AMM) from paper
- Timestep embedding
- Attention mechanisms
- Complex-valued CSI support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] tensor of timestep indices
        Returns:
            [B, dim] embedded timesteps
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class AntennaModifierModule(nn.Module):
    """
    Antenna Modifier Module (AMM) from paper.
    
    Learns antenna spacing-dependent feature modulation.
    For simplicity, we'll make it learn from data implicitly.
    """
    
    def __init__(self, channels, embedding_dim=256):
        super().__init__()
        self.channels = channels
        
        # MLP to generate scale and shift parameters
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, channels * 2)  # scale + shift
        )
        
        # Initialize to identity
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data[:channels] = 1.0  # scale = 1
        self.mlp[-1].bias.data[channels:] = 0.0  # shift = 0
    
    def forward(self, x, embedding):
        """
        Args:
            x: [B, C, H, W] feature map
            embedding: [B, D] antenna/timestep embedding
        Returns:
            [B, C, H, W] modulated features
        """
        # Generate scale and shift
        params = self.mlp(embedding)  # [B, 2*C]
        scale, shift = params.chunk(2, dim=1)  # [B, C], [B, C]
        
        # Reshape for broadcasting
        scale = scale[:, :, None, None]  # [B, C, 1, 1]
        shift = shift[:, :, None, None]  # [B, C, 1, 1]
        
        # Modulate
        return x * scale + shift


class ResidualBlock(nn.Module):
    """Residual block with timestep embedding."""
    
    def __init__(self, in_channels, out_channels, time_dim, use_amm=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        # AMM
        self.use_amm = use_amm
        if use_amm:
            self.amm = AntennaModifierModule(out_channels, time_dim)
    
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, time_dim] timestep embedding
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add timestep embedding
        t_proj = self.time_mlp(t_emb)
        h = h + t_proj[:, :, None, None]
        
        # AMM modulation
        if self.use_amm:
            h = self.amm(h, t_emb)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        
        # Reshape for attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [B, heads, HW, C//heads]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        out = self.proj(out)
        
        return x + out


class EnhancedDiffusionUNet(nn.Module):
    """
    Enhanced U-Net for diffusion-based CSI estimation.
    
    Features:
    - Timestep embedding
    - Residual blocks with AMM
    - Multi-head attention
    - Skip connections
    """
    
    def __init__(
        self,
        in_channels=2,
        model_channels=64,
        out_channels=2,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        use_amm=True,
        spatial_size=(10752, 80),
        timestep_dim=256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.use_amm = use_amm
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_dim * 4, timestep_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        input_ch = ch
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                block = ResidualBlock(input_ch, out_ch, timestep_dim, use_amm)
                self.down_blocks.append(block)
                input_ch = out_ch
                
                # Add attention at specified resolutions
                if level in attention_resolutions:
                    attn = AttentionBlock(out_ch)
                    self.down_blocks.append(attn)
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                self.down_samples.append(downsample)
            else:
                self.down_samples.append(nn.Identity())
        
        # Bottleneck
        self.mid_block1 = ResidualBlock(input_ch, input_ch, timestep_dim, use_amm)
        self.mid_attn = AttentionBlock(input_ch)
        self.mid_block2 = ResidualBlock(input_ch, input_ch, timestep_dim, use_amm)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                # Skip connection from encoder
                skip_ch = model_channels * mult if i > 0 else (model_channels * channel_mult[level-1] if level > 0 else model_channels)
                block = ResidualBlock(input_ch + skip_ch, out_ch, timestep_dim, use_amm)
                self.up_blocks.append(block)
                input_ch = out_ch
                
                # Add attention
                if level in attention_resolutions:
                    attn = AttentionBlock(out_ch)
                    self.up_blocks.append(attn)
            
            # Upsample (except first level)
            if level > 0:
                upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                self.up_samples.append(upsample)
            else:
                self.up_samples.append(nn.Identity())
        
        # Output
        self.norm_out = nn.GroupNorm(8, model_channels)
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        print(f"✅ Enhanced Diffusion U-Net created:")
        print(f"   Channels: {model_channels}, Multipliers: {channel_mult}")
        print(f"   AMM: {use_amm}, Attention: {len(attention_resolutions)} levels")
    
    def forward(self, x, timesteps):
        """
        Args:
            x: [B, 2, H, W] noisy input (real, imag)
            timesteps: [B] timestep indices
        Returns:
            [B, 2, H, W] predicted noise
        """
        # Timestep embedding
        t_emb = self.time_embed(timesteps)  # [B, time_dim]
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skips = []
        for block, downsample in zip(self.down_blocks, self.down_samples):
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:  # Attention
                h = block(h)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for block, upsample in zip(self.up_blocks, self.up_samples):
            h = upsample(h)
            if len(skips) > 0:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:  # Attention
                h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h