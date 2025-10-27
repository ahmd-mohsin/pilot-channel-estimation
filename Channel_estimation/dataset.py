"""
Dataset Loader for MIMO Channel Estimation
==========================================

Handles loading and preprocessing of channel state information (CSI) data
from MATLAB .mat files for training diffusion models.

Data Shape: [768 × 14 × 4 × 2 × 10 × 1000]
- 768: Subcarriers (K)
- 14: OFDM symbols (L)
- 4: Receive antennas (nRx)
- 2: Transmit antennas (nTx)
- 10: Time slots
- 1000: Monte Carlo realizations
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class CSIDataset(Dataset):
    """
    Dataset for MIMO channel estimation using diffusion models.
    
    Loads H_perfect (clean) and H_estimated (noisy) from .mat files.
    Applies normalization and converts to PyTorch tensors.
    
    Args:
        mat_file_path: Path to .mat file containing channel data
        train: If True, use first 800 realizations; else use last 200
        normalize_method: 'global', 'per_sample', or 'per_slot'
        return_metadata: If True, also return metadata dict
    """
    
    def __init__(
        self,
        mat_file_path: str,
        train: bool = True,
        normalize_method: str = 'per_sample',
        return_metadata: bool = False,
        transform: Optional[callable] = None
    ):
        super().__init__()
        
        self.mat_file_path = Path(mat_file_path)
        self.train = train
        self.normalize_method = normalize_method
        self.return_metadata = return_metadata
        self.transform = transform
        
        # Load data
        print(f"Loading CSI data from: {self.mat_file_path}")
        self._load_data()
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
        print(f"Dataset loaded: {'Train' if train else 'Test'}")
        print(f"  Samples: {len(self)}")
        print(f"  Perfect CSI shape: {self.H_perfect.shape}")
        print(f"  Noisy CSI shape: {self.H_noisy.shape}")
        print(f"  Normalization: {self.normalize_method}")
        
    def _load_data(self):
        """Load .mat file and extract channel matrices."""
        try:
            mat_data = loadmat(str(self.mat_file_path))
            
            # Extract channel matrices
            H_perfect_all = mat_data['H_perfect_all']  # [768,14,4,2,10,1000]
            H_estimated_all = mat_data['H_estimated_all']  # [768,14,4,2,10,1000]
            
            # Extract metadata if available
            if 'metadata' in mat_data:
                self.metadata = mat_data['metadata']
            else:
                self.metadata = None
                
            # Split train/test (800 train, 200 test)
            if self.train:
                self.H_perfect = H_perfect_all[..., :800]
                self.H_noisy = H_estimated_all[..., :800]
                if self.metadata is not None:
                    self.metadata_subset = {
                        k: v[:800] if hasattr(v, '__len__') else v 
                        for k, v in self.metadata.items()
                    }
            else:
                self.H_perfect = H_perfect_all[..., 800:]
                self.H_noisy = H_estimated_all[..., 800:]
                if self.metadata is not None:
                    self.metadata_subset = {
                        k: v[800:] if hasattr(v, '__len__') else v 
                        for k, v in self.metadata.items()
                    }
            
            # Verify shapes
            expected_shape_prefix = (768, 14, 4, 2, 10)
            assert self.H_perfect.shape[:5] == expected_shape_prefix, \
                f"Unexpected shape: {self.H_perfect.shape}"
            assert self.H_noisy.shape[:5] == expected_shape_prefix, \
                f"Unexpected shape: {self.H_noisy.shape}"
            
            print(f"✅ Data loaded successfully")
            print(f"   Perfect CSI: {H_perfect_all.shape} -> {self.H_perfect.shape}")
            print(f"   Noisy CSI: {H_estimated_all.shape} -> {self.H_noisy.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load .mat file: {e}")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics based on method."""
        if self.normalize_method == 'global':
            # Global statistics across all data
            self.mean_perfect = np.mean(self.H_perfect)
            self.std_perfect = np.std(self.H_perfect)
            self.mean_noisy = np.mean(self.H_noisy)
            self.std_noisy = np.std(self.H_noisy)
            
            # Power normalization (more stable for complex values)
            self.power_perfect = np.sqrt(np.mean(np.abs(self.H_perfect) ** 2))
            self.power_noisy = np.sqrt(np.mean(np.abs(self.H_noisy) ** 2))
            
            print(f"Global normalization stats computed:")
            print(f"  Perfect - Mean: {self.mean_perfect:.6f}, Std: {self.std_perfect:.6f}")
            print(f"  Perfect - Power: {self.power_perfect:.6f}")
            print(f"  Noisy - Mean: {self.mean_noisy:.6f}, Std: {self.std_noisy:.6f}")
            print(f"  Noisy - Power: {self.power_noisy:.6f}")
            
        elif self.normalize_method in ['per_sample', 'per_slot']:
            # Will normalize per sample during __getitem__
            self.power_perfect = 1.0
            self.power_noisy = 1.0
            print(f"Per-sample normalization will be applied during data loading")
        
    def normalize_channel(self, H: np.ndarray, is_perfect: bool = True) -> np.ndarray:
        """
        Normalize channel matrix.
        
        For complex channels, power normalization is typically better:
        H_norm = H / sqrt(mean(|H|^2))
        """
        if self.normalize_method == 'global':
            power = self.power_perfect if is_perfect else self.power_noisy
            return H / (power + 1e-10)
            
        elif self.normalize_method == 'per_sample':
            # Normalize by power of this sample
            power = np.sqrt(np.mean(np.abs(H) ** 2))
            return H / (power + 1e-10)
            
        elif self.normalize_method == 'per_slot':
            # Normalize each slot independently
            H_norm = np.zeros_like(H)
            for slot in range(H.shape[4]):
                H_slot = H[..., slot]
                power = np.sqrt(np.mean(np.abs(H_slot) ** 2))
                H_norm[..., slot] = H_slot / (power + 1e-10)
            return H_norm
            
        else:
            return H
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.H_perfect.shape[-1]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            noisy: Noisy CSI [768, 14, 4, 2, 10] complex tensor
            clean: Perfect CSI [768, 14, 4, 2, 10] complex tensor
            (metadata): Optional metadata dict
        """
        # Extract single realization
        H_perfect = self.H_perfect[..., idx]  # [768,14,4,2,10]
        H_noisy = self.H_noisy[..., idx]      # [768,14,4,2,10]
        
        # Normalize
        H_perfect_norm = self.normalize_channel(H_perfect, is_perfect=True)
        H_noisy_norm = self.normalize_channel(H_noisy, is_perfect=False)
        
        # Convert to PyTorch tensors (keep as complex)
        H_perfect_tensor = torch.from_numpy(H_perfect_norm).cfloat()
        H_noisy_tensor = torch.from_numpy(H_noisy_norm).cfloat()
        
        # Apply transforms if any
        if self.transform is not None:
            H_perfect_tensor = self.transform(H_perfect_tensor)
            H_noisy_tensor = self.transform(H_noisy_tensor)
        
        if self.return_metadata and self.metadata_subset is not None:
            metadata = {
                k: v[idx] if hasattr(v, '__len__') else v 
                for k, v in self.metadata_subset.items()
            }
            return H_noisy_tensor, H_perfect_tensor, metadata
        
        return H_noisy_tensor, H_perfect_tensor
    
    def get_normalization_stats(self) -> Dict:
        """Return normalization statistics for denormalization."""
        return {
            'method': self.normalize_method,
            'power_perfect': self.power_perfect,
            'power_noisy': self.power_noisy,
        }


def create_dataloaders(
    mat_file_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_method: str = 'per_sample',
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        mat_file_path: Path to .mat file
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        normalize_method: Normalization method
        
    Returns:
        train_loader, test_loader
    """
    print("\n" + "="*60)
    print("Creating DataLoaders")
    print("="*60)
    
    # Create datasets
    train_dataset = CSIDataset(
        mat_file_path=mat_file_path,
        train=True,
        normalize_method=normalize_method,
        **kwargs
    )
    
    test_dataset = CSIDataset(
        mat_file_path=mat_file_path,
        train=False,
        normalize_method=normalize_method,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\n✅ DataLoaders created:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"   Batch size: {batch_size}")
    print("="*60 + "\n")
    
    return train_loader, test_loader


# Helper function for quick testing
def test_dataset(mat_file_path: str):
    """Quick test of dataset loading."""
    print("\n🧪 Testing Dataset Loader...")
    
    train_dataset = CSIDataset(mat_file_path, train=True)
    test_dataset = CSIDataset(mat_file_path, train=False)
    
    # Get one sample
    noisy, clean = train_dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  Noisy: {noisy.shape}, dtype: {noisy.dtype}")
    print(f"  Clean: {clean.shape}, dtype: {clean.dtype}")
    print(f"  Complex: {noisy.is_complex()}")
    
    # Check statistics
    print(f"\nSample statistics:")
    print(f"  Noisy - Mean: {noisy.abs().mean():.6f}, Std: {noisy.abs().std():.6f}")
    print(f"  Clean - Mean: {clean.abs().mean():.6f}, Std: {clean.abs().std():.6f}")
    
    print("\n✅ Dataset test passed!")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
    else:
        mat_file = "/mnt/user-data/outputs/umi_montecarlo_data/umi_channel_data.mat"
    
    test_dataset(mat_file)