import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Dict
from .loader import KINEMATIC_COLUMNS, KINETIC_COLUMNS

SIGNAL_COLUMNS = KINEMATIC_COLUMNS + KINETIC_COLUMNS

def apply_lowpass_filter(
    df: pd.DataFrame, 
    cutoff_kinematic: float = 6.0, 
    cutoff_kinetic: float = 10.0, 
    fs: float = 1920.0
) -> pd.DataFrame:
    """
    Applies a zero-phase 4th order low-pass Butterworth filter to kinematics and kinetics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the trial signals.
    cutoff_kinematic : float, default 6.0
        Cutoff frequency for kinematics (joint angles) in Hz.
    cutoff_kinetic : float, default 10.0
        Cutoff frequency for kinetics (joint torques) in Hz.
    fs : float, default 1920.0
        Sampling frequency of the signals in Hz.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with filtered signals.
    """
    filtered_df = df.copy()
    nyquist = 0.5 * fs
    
    # 4th order low-pass Butterworth filters
    b_kin, a_kin = butter(4, cutoff_kinematic / nyquist, btype='low')
    b_kinet, a_kinet = butter(4, cutoff_kinetic / nyquist, btype='low')
    
    # Apply zero-phase filter to kinematics
    for col in KINEMATIC_COLUMNS:
        if col in filtered_df.columns and len(filtered_df) > 15: # filtfilt needs sufficient samples
            filtered_df[col] = filtfilt(b_kin, a_kin, filtered_df[col])
            
    # Apply zero-phase filter to kinetics
    for col in KINETIC_COLUMNS:
        if col in filtered_df.columns and len(filtered_df) > 15:
            filtered_df[col] = filtfilt(b_kinet, a_kinet, filtered_df[col])
            
    return filtered_df

def downsample_signals(
    df: pd.DataFrame, 
    original_fs: float = 1920.0, 
    target_fs: float = 120.0
) -> pd.DataFrame:
    """
    Resamples the signals from original_fs to target_fs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
    original_fs : float, default 1920.0
        Original sampling rate of the signals in Hz.
    target_fs : float, default 120.0
        Target sampling rate in Hz.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with downsampled signals and updated timestamps.
    """
    if original_fs == target_fs or df.empty:
        return df.copy()
        
    ratio = target_fs / original_fs
    num_samples = int(round(len(df) * ratio))
    if num_samples < 2:
        return df.copy()
        
    # Build resampled DataFrame
    resampled_df = pd.DataFrame()
    resampled_df['Time'] = np.linspace(df['Time'].iloc[0], df['Time'].iloc[-1], num_samples)
    
    # Resample continuous signal columns
    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            resampled_df[col] = resample(df[col].values, num_samples)
            
    # Resample categorical/discrete label columns (Status and Group) via nearest-neighbor
    orig_indices = np.linspace(0, len(df) - 1, num_samples)
    nearest_indices = np.round(orig_indices).astype(int)
    
    if 'Status' in df.columns:
        resampled_df['Status'] = df['Status'].values[nearest_indices]
    if 'Group' in df.columns:
        resampled_df['Group'] = df['Group'].values[nearest_indices]
        
    return resampled_df

def apply_minmax_scaling(
    df: pd.DataFrame, 
    feature_range: Tuple[float, float] = (-1.0, 1.0),
    scaling_params: Dict[str, Tuple[float, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Scales kinematics and kinetics columns using Min-Max scaling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
    feature_range : Tuple[float, float], default (-1.0, 1.0)
        Scale target range.
    scaling_params : Dict[str, Tuple[float, float]], optional
        Precomputed min/max parameters per column, formatted as:
        {column_name: (min_val, max_val)}.
        If None, parameters will be computed from the input DataFrame.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]
        - Scaled DataFrame
        - Dictionary of computed/used min/max parameters per column.
    """
    scaled_df = df.copy()
    params = {} if scaling_params is None else scaling_params.copy()
    
    lower, upper = feature_range
    
    for col in SIGNAL_COLUMNS:
        if col not in scaled_df.columns:
            continue
            
        series = scaled_df[col]
        
        if scaling_params is None:
            min_val = series.min()
            max_val = series.max()
            params[col] = (min_val, max_val)
        else:
            min_val, max_val = scaling_params[col]
            
        # Avoid division by zero
        diff = max_val - min_val
        if np.isclose(diff, 0.0):
            scaled_df[col] = 0.0
        else:
            scaled_df[col] = lower + ((series - min_val) / diff) * (upper - lower)
            
    return scaled_df, params
