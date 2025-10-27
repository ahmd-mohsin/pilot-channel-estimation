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
import h5py
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
            print(f"Attempting to load: {self.mat_file_path}")
            
            # Try h5py first (for MATLAB v7.3 files)
            try:
                with h5py.File(str(self.mat_file_path), 'r') as f:
                    print(f"✅ Using h5py (MATLAB v7.3 format)")
                    
                    # List available variables
                    print(f"Available variables: {list(f.keys())}")
                    
                    # Load channel matrices
                    # Note: h5py loads in REVERSED dimension order compared to MATLAB!
                    # MATLAB saves as [768,14,4,2,10,1000]
                    # h5py loads as [1000,10,2,4,14,768]
                    # So we need to transpose/permute to get correct order
                    
                    if 'H_perfect_all' in f:
                        H_perfect_h5 = f['H_perfect_all']
                        print(f"   H_perfect_all shape in file: {H_perfect_h5.shape}")
                        print(f"   H_perfect_all dtype: {H_perfect_h5.dtype}")
                        
                        # Load the data
                        H_perfect_data = H_perfect_h5[()]
                        
                        # Check if complex
                        if np.iscomplexobj(H_perfect_data):
                            print(f"   H_perfect_all is complex: True")
                            H_perfect_all = H_perfect_data
                        else:
                            print(f"   H_perfect_all is complex: False (will convert)")
                            # Handle real/imag storage: last dim might be [real,imag]
                            if H_perfect_data.ndim == 7 and H_perfect_data.shape[-1] == 2:
                                H_perfect_all = H_perfect_data[..., 0] + 1j * H_perfect_data[..., 1]
                                print(f"   Converted from real/imag to complex")
                            else:
                                # Try to interpret as complex
                                # HDF5 stores complex as compound dtype
                                if H_perfect_data.dtype.names:
                                    # Compound type - try different field name combinations
                                    field_names = H_perfect_data.dtype.names
                                    print(f"   Compound dtype with fields: {field_names}")
                                    
                                    if 'real' in field_names and 'imag' in field_names:
                                        H_perfect_all = H_perfect_data['real'] + 1j * H_perfect_data['imag']
                                        print(f"   Converted from compound dtype (real, imag)")
                                    elif 'r' in field_names and 'i' in field_names:
                                        H_perfect_all = H_perfect_data['r'] + 1j * H_perfect_data['i']
                                        print(f"   Converted from compound dtype (r, i)")
                                    else:
                                        raise ValueError(f"Unknown compound dtype fields: {field_names}")
                                else:
                                    H_perfect_all = H_perfect_data
                        
                        print(f"   H_perfect_all loaded shape: {H_perfect_all.shape}")
                        
                        # Transpose to correct order if needed
                        # Expected: [768, 14, 4, 2, 10, 1000]
                        # h5py gives: [1000, 10, 2, 4, 14, 768]
                        if H_perfect_all.shape[-1] == 768 and H_perfect_all.shape[0] == 1000:
                            print(f"   Detected reversed dimensions, transposing...")
                            H_perfect_all = np.transpose(H_perfect_all, (5, 4, 3, 2, 1, 0))
                            print(f"   H_perfect_all after transpose: {H_perfect_all.shape}")
                        
                    else:
                        raise KeyError("H_perfect_all not found in file")
                    
                    if 'H_estimated_all' in f:
                        H_estimated_h5 = f['H_estimated_all']
                        print(f"   H_estimated_all shape in file: {H_estimated_h5.shape}")
                        print(f"   H_estimated_all dtype: {H_estimated_h5.dtype}")
                        
                        # Load the data
                        H_estimated_data = H_estimated_h5[()]
                        
                        # Check if complex
                        if np.iscomplexobj(H_estimated_data):
                            print(f"   H_estimated_all is complex: True")
                            H_estimated_all = H_estimated_data
                        else:
                            print(f"   H_estimated_all is complex: False (will convert)")
                            # Handle real/imag storage
                            if H_estimated_data.ndim == 7 and H_estimated_data.shape[-1] == 2:
                                H_estimated_all = H_estimated_data[..., 0] + 1j * H_estimated_data[..., 1]
                                print(f"   Converted from real/imag to complex")
                            else:
                                # Try compound dtype
                                if H_estimated_data.dtype.names:
                                    field_names = H_estimated_data.dtype.names
                                    print(f"   Compound dtype with fields: {field_names}")
                                    
                                    if 'real' in field_names and 'imag' in field_names:
                                        H_estimated_all = H_estimated_data['real'] + 1j * H_estimated_data['imag']
                                        print(f"   Converted from compound dtype (real, imag)")
                                    elif 'r' in field_names and 'i' in field_names:
                                        H_estimated_all = H_estimated_data['r'] + 1j * H_estimated_data['i']
                                        print(f"   Converted from compound dtype (r, i)")
                                    else:
                                        raise ValueError(f"Unknown compound dtype fields: {field_names}")
                                else:
                                    H_estimated_all = H_estimated_data
                        
                        print(f"   H_estimated_all loaded shape: {H_estimated_all.shape}")
                        
                        # Transpose to correct order if needed
                        if H_estimated_all.shape[-1] == 768 and H_estimated_all.shape[0] == 1000:
                            print(f"   Detected reversed dimensions, transposing...")
                            H_estimated_all = np.transpose(H_estimated_all, (5, 4, 3, 2, 1, 0))
                            print(f"   H_estimated_all after transpose: {H_estimated_all.shape}")
                    else:
                        raise KeyError("H_estimated_all not found in file")
                    
                    # Load metadata if available
                    self.metadata = None
                    if 'metadata' in f:
                        print(f"   Loading metadata...")
                        self.metadata = {}
                        try:
                            for key in f['metadata'].keys():
                                self.metadata[key] = f['metadata'][key][()]
                        except:
                            pass
            
            except (OSError, KeyError) as e_h5:
                # Fall back to scipy.io.loadmat (for older MATLAB files)
                print(f"h5py failed ({e_h5}), trying scipy.io.loadmat...")
                mat_data = loadmat(str(self.mat_file_path))
                
                H_perfect_all = mat_data['H_perfect_all']
                H_estimated_all = mat_data['H_estimated_all']
                
                print(f"✅ Using scipy.io.loadmat (MATLAB v7 format)")
                print(f"   H_perfect_all: {H_perfect_all.shape}")
                print(f"   H_estimated_all: {H_estimated_all.shape}")
                
                # Extract metadata if available
                if 'metadata' in mat_data:
                    self.metadata = mat_data['metadata']
                else:
                    self.metadata = None
            
            # Verify shapes
            expected_shape_prefix = (768, 14, 4, 2, 10)
            print(f"\nVerifying shapes...")
            print(f"   Expected: {expected_shape_prefix + (1000,)}")
            print(f"   H_perfect_all: {H_perfect_all.shape}")
            print(f"   H_estimated_all: {H_estimated_all.shape}")
            
            assert H_perfect_all.shape[:5] == expected_shape_prefix, \
                f"Unexpected H_perfect shape: {H_perfect_all.shape}, expected {expected_shape_prefix + (1000,)}"
            assert H_estimated_all.shape[:5] == expected_shape_prefix, \
                f"Unexpected H_estimated shape: {H_estimated_all.shape}, expected {expected_shape_prefix + (1000,)}"
            
            # Split train/test (800 train, 200 test)
            if self.train:
                self.H_perfect = H_perfect_all[..., :800]
                self.H_noisy = H_estimated_all[..., :800]
                if self.metadata is not None:
                    self.metadata_subset = {
                        k: v[:800] if hasattr(v, '__len__') and len(v) == 1000 else v 
                        for k, v in self.metadata.items()
                    }
            else:
                self.H_perfect = H_perfect_all[..., 800:]
                self.H_noisy = H_estimated_all[..., 800:]
                if self.metadata is not None:
                    self.metadata_subset = {
                        k: v[800:] if hasattr(v, '__len__') and len(v) == 1000 else v 
                        for k, v in self.metadata.items()
                    }
            
            print(f"\n✅ Data loaded successfully")
            print(f"   Dataset split: {'Train (first 800)' if self.train else 'Test (last 200)'}")
            print(f"   Perfect CSI: {H_perfect_all.shape} -> {self.H_perfect.shape}")
            print(f"   Noisy CSI: {H_estimated_all.shape} -> {self.H_noisy.shape}")
            print(f"   Data type: {self.H_perfect.dtype}")
            print(f"   Complex: {np.iscomplexobj(self.H_perfect)}")
            print(f"   Sample value check: max={np.max(np.abs(self.H_perfect)):.4f}, mean={np.mean(np.abs(self.H_perfect)):.4f}")
            
        except Exception as e:
            print(f"\n❌ Error loading data!")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
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