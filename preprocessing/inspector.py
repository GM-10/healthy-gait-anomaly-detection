import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .loader import KINEMATIC_COLUMNS, KINETIC_COLUMNS

SIGNAL_COLUMNS = KINEMATIC_COLUMNS + KINETIC_COLUMNS

def check_signal_sanity(df: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
    """
    Checks signals for NaNs, Infs, and constant (flat) values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
        
    Returns
    -------
    Dict[str, Dict[str, bool]]
        A dictionary mapping columns to sanity flags:
        {column_name: {'has_nan': bool, 'has_inf': bool, 'is_constant': bool}}
    """
    results = {}
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            continue
            
        series = df[col]
        has_nan = series.isna().any()
        has_inf = np.isinf(series).any()
        
        # Check if the signal is constant/flat (no variance)
        is_constant = False
        if not series.empty:
            is_constant = np.isclose(series.std(), 0.0) or series.nunique() <= 1
            
        results[col] = {
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'is_constant': bool(is_constant)
        }
        
    return results

def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, List[int]]:
    """
    Detects outlier timestamps using the Interquartile Range (IQR) method.
    Outliers can indicate motion capture tracking glitches or force plate impact noise.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
    factor : float, default 1.5
        The IQR scale multiplier (1.5 for standard outliers, 3.0 for extreme outliers).
        
    Returns
    -------
    Dict[str, List[int]]
        A dictionary mapping column names to the row indices that are flagged as outliers.
    """
    outliers = {}
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            continue
            
        series = df[col]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - factor * iqr
        upper_bound = q75 + factor * iqr
        
        # Get indices of elements outside the bounds
        flagged_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        outliers[col] = flagged_indices
        
    return outliers
