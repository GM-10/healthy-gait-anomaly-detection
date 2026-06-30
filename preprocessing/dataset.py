import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

from .loader import (
    load_subject_metadata,
    load_trial_data,
    normalize_kinetics_by_weight,
    SIGNAL_COLUMNS
)
from .cleaner import drop_invalid_frames, interpolate_dropouts
from .conditioner import apply_lowpass_filter, downsample_signals, apply_minmax_scaling
from .windower import create_sliding_windows

class SIATGaitDataset(Dataset):
    """
    A PyTorch Dataset for SIAT-LLMD kinematics and kinetics data.
    Loads and runs the preprocessing pipeline for a cohort of subjects and movements.
    """
    def __init__(
        self,
        base_dir: str,
        subjects: List[str],
        movements: List[str],
        window_size: int = 180,
        overlap_size: int = 90,
        target_fs: float = 120.0,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        scaling_params: Optional[Dict[str, Tuple[float, float]]] = None,
        fit_scaling: bool = True
    ):
        """
        Parameters
        ----------
        base_dir : str
            Path to the SIAT_LLMD20230404 directory.
        subjects : List[str]
            List of subject ID strings (e.g. ['Sub01', 'Sub02']).
        movements : List[str]
            List of movement codes to load (e.g. ['WAK', 'UPS']).
        window_size : int, default 180
            Size of the sliding windows in frames.
        overlap_size : int, default 90
            Overlap size between consecutive windows.
        target_fs : float, default 120.0
            Target downsampled frequency in Hz.
        feature_range : Tuple[float, float], default (-1.0, 1.0)
            Target scale range for Min-Max scaling.
        scaling_params : Dict[str, Tuple[float, float]], optional
            Precomputed Min-Max scaling parameters.
        fit_scaling : bool, default True
            If True, Min-Max parameters will be computed from the loaded cohort.
            If False, scaling_params must be supplied.
        """
        self.base_dir = base_dir
        self.subjects = subjects
        self.movements = movements
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.target_fs = target_fs
        self.feature_range = feature_range
        self.scaling_params = scaling_params
        
        self.windows_data = []
        self.windows_metadata = []
        
        # Load subject metadata for torque weight-normalization
        metadata_path = os.path.join(base_dir, 'SubjectInformation.xlsx')
        self.metadata_df = load_subject_metadata(metadata_path)
        
        # Load and preprocess all trials
        self._load_and_preprocess_cohort(fit_scaling)
        
    def _load_and_preprocess_cohort(self, fit_scaling: bool):
        raw_trial_dfs = []
        trial_identifiers = []
        
        # 1. Load raw data and apply initial cleaning/filtering per trial
        for sub in self.subjects:
            subject_weight = float(self.metadata_df.loc[sub, 'weight'])
            
            for mov in self.movements:
                data_pattern = os.path.join(self.base_dir, sub, 'Data', f'*_{mov}_Data.csv')
                label_pattern = os.path.join(self.base_dir, sub, 'Labels', f'*_{mov}_Label.csv')
                
                data_files = glob.glob(data_pattern)
                label_files = glob.glob(label_pattern)
                
                if not data_files or not label_files:
                    continue
                    
                data_path = data_files[0]
                label_path = label_files[0]
                
                # Load
                df = load_trial_data(data_path, label_path)
                
                # Clean boundary frames
                df = drop_invalid_frames(df)
                
                # Interpolate minor dropouts
                df = interpolate_dropouts(df)
                
                # Low-pass filter (Butterworth)
                df = apply_lowpass_filter(df, cutoff_kinematic=6.0, cutoff_kinetic=10.0, fs=1920.0)
                
                # Downsample
                df = downsample_signals(df, original_fs=1920.0, target_fs=self.target_fs)
                
                # Weight normalization for kinetic torques
                df = normalize_kinetics_by_weight(df, subject_weight)
                
                raw_trial_dfs.append(df)
                trial_identifiers.append((sub, mov))
                
        if not raw_trial_dfs:
            return
            
        # 2. Compute/Fit global scaling parameters across the cohort if fit_scaling is True
        if fit_scaling or self.scaling_params is None:
            combined_df = pd.concat(raw_trial_dfs, ignore_index=True)
            self.scaling_params = {}
            for col in SIGNAL_COLUMNS:
                self.scaling_params[col] = (combined_df[col].min(), combined_df[col].max())
                
        # 3. Apply scaling and windowing
        all_windows = []
        for df, (sub, mov) in zip(raw_trial_dfs, trial_identifiers):
            # Scale
            scaled_df, _ = apply_minmax_scaling(
                df, 
                feature_range=self.feature_range, 
                scaling_params=self.scaling_params
            )
            
            # Segment into sliding windows
            win_arr, win_meta = create_sliding_windows(
                scaled_df, 
                window_size=self.window_size, 
                overlap_size=self.overlap_size
            )
            
            if len(win_arr) > 0:
                all_windows.append(win_arr)
                # Inject subject and movement details into window metadata
                for meta in win_meta:
                    meta['subject'] = sub
                    meta['movement'] = mov
                    self.windows_metadata.append(meta)
                    
        if all_windows:
            self.windows_data = np.concatenate(all_windows, axis=0)
        else:
            self.windows_data = np.empty((0, self.window_size, len(SIGNAL_COLUMNS)))
            
    def __len__(self) -> int:
        return len(self.windows_data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Sensor sequence tensor of shape (window_size, num_features)
            - Metadata dictionary containing subject, movement, cycle number, and gait phase.
        """
        x = torch.tensor(self.windows_data[idx], dtype=torch.float32)
        meta = self.windows_metadata[idx]
        return x, meta

def get_subject_split_loaders(
    base_dir: str,
    movements: List[str],
    window_size: int = 180,
    overlap_size: int = 90,
    batch_size: int = 32,
    target_fs: float = 120.0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Tuple[float, float]]]:
    """
    Generates PyTorch DataLoaders with a Leave-Group-Out (subject-wise) split:
      - Train: Subjects 1 to 30
      - Validation: Subjects 31 to 35
      - Test: Subjects 36 to 40
      
    Parameters
    ----------
    base_dir : str
        Path to the SIAT_LLMD20230404 directory.
    movements : List[str]
        Locomotion movements to load.
    window_size : int
        Window frame count.
    overlap_size : int
        Overlap frame count.
    batch_size : int
        Batch size.
    target_fs : float
        Target downsampled rate.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Tuple[float, float]]]
        - Train DataLoader
        - Validation DataLoader
        - Test DataLoader
        - Dict of fitted scaling parameters.
    """
    train_subs = [f'Sub{i:02d}' for i in range(1, 31)]
    val_subs = [f'Sub{i:02d}' for i in range(31, 36)]
    test_subs = [f'Sub{i:02d}' for i in range(36, 41)]
    
    # 1. Fit scaling on Train cohort
    train_dataset = SIATGaitDataset(
        base_dir=base_dir,
        subjects=train_subs,
        movements=movements,
        window_size=window_size,
        overlap_size=overlap_size,
        target_fs=target_fs,
        fit_scaling=True
    )
    
    # 2. Reuse train scaling parameters for Validation and Test to avoid data leakage
    val_dataset = SIATGaitDataset(
        base_dir=base_dir,
        subjects=val_subs,
        movements=movements,
        window_size=window_size,
        overlap_size=overlap_size,
        target_fs=target_fs,
        scaling_params=train_dataset.scaling_params,
        fit_scaling=False
    )
    
    test_dataset = SIATGaitDataset(
        base_dir=base_dir,
        subjects=test_subs,
        movements=movements,
        window_size=window_size,
        overlap_size=overlap_size,
        target_fs=target_fs,
        scaling_params=train_dataset.scaling_params,
        fit_scaling=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.scaling_params
