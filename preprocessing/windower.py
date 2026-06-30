import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from .loader import KINEMATIC_COLUMNS, KINETIC_COLUMNS

SIGNAL_COLUMNS = KINEMATIC_COLUMNS + KINETIC_COLUMNS

def create_sliding_windows(
    df: pd.DataFrame, 
    window_size: int = 180, 
    overlap_size: int = 90
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Segments the continuous joint signals into fixed-size overlapping windows.
    Ensures that a window does not cross cyclic boundary boundaries (i.e. all rows in the window
    must belong to the same Group index).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
    window_size : int, default 180
        Number of frames in a window. (1.5 seconds at 120 Hz)
    overlap_size : int, default 90
        Number of overlapping frames between consecutive windows.
        
    Returns
    -------
    Tuple[np.ndarray, List[Dict]]
        - A numpy array of shape (num_windows, window_size, num_features)
        - A list of metadata dicts corresponding to each window (e.g. status, group, subject, trial).
    """
    if df.empty or len(df) < window_size:
        return np.empty((0, window_size, len(SIGNAL_COLUMNS))), []
        
    step = window_size - overlap_size
    if step <= 0:
        raise ValueError("overlap_size must be strictly less than window_size.")
        
    windows = []
    metadata = []
    
    # Extract coordinates as numpy array for fast slicing
    signal_data = df[SIGNAL_COLUMNS].values
    groups = df['Group'].values
    status_vals = df['Status'].values
    times = df['Time'].values
    
    for start_idx in range(0, len(df) - window_size + 1, step):
        end_idx = start_idx + window_size
        
        # Check boundary constraint: all samples in window must have the SAME group index
        window_groups = groups[start_idx:end_idx]
        if not np.all(window_groups == window_groups[0]):
            # Skip window if it crosses cycles
            continue
            
        # Check if the group is valid (non-NaN, non-zero)
        if np.isnan(window_groups[0]) or window_groups[0] <= 0:
            continue
            
        # Get the prevailing gait phase / status for this window
        # For classification/label mapping, we take the mode or prevailing status in the window
        unique_status, counts = np.unique(status_vals[start_idx:end_idx], return_counts=True)
        # Drop NaNs or select primary phase
        prevailing_status = unique_status[np.argmax(counts)]
        
        windows.append(signal_data[start_idx:end_idx])
        metadata.append({
            'start_time': times[start_idx],
            'end_time': times[end_idx - 1],
            'group_cycle': int(window_groups[0]),
            'gait_phase_status': prevailing_status,
            'start_index': start_idx,
            'end_index': end_idx
        })
        
    if len(windows) == 0:
        return np.empty((0, window_size, len(SIGNAL_COLUMNS))), []
        
    return np.array(windows), metadata
