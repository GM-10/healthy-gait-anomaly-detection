import pandas as pd
import numpy as np
from .loader import KINEMATIC_COLUMNS, KINETIC_COLUMNS

SIGNAL_COLUMNS = KINEMATIC_COLUMNS + KINETIC_COLUMNS

def drop_invalid_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes boundary transition phases by filtering out rows where Status is NaN.
    Also ensures that the cycle index (Group) is valid.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing only valid cyclic movement cycles.
    """
    # Drop rows where 'Status' is missing (which maps to non-cyclic phases)
    cleaned_df = df.dropna(subset=['Status']).copy()
    
    # Also drop where Group is NaN or <= 0
    cleaned_df = cleaned_df[cleaned_df['Group'] > 0]
    
    # Reset index to maintain contiguous indexing after row drop
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def interpolate_dropouts(df: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """
    Interpolates minor dropouts or extreme values (which should be set to NaN prior to calling).
    Uses cubic spline interpolation for smooth joint trajectories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the trial.
    max_gap : int, default 10
        The maximum number of consecutive NaNs to interpolate. Gaps larger than this
        will remain as NaN to avoid inventing large chunks of signal.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with filled dropouts.
    """
    df_filled = df.copy()
    for col in SIGNAL_COLUMNS:
        if col not in df_filled.columns:
            continue
            
        # Check if there are any NaNs to interpolate
        if df_filled[col].isna().any():
            # Interpolate limit parameter specifies maximum consecutive NaNs to fill
            df_filled[col] = df_filled[col].interpolate(
                method='cubic', 
                limit=max_gap, 
                limit_direction='both'
            ).bfill().ffill() # Fallback edge hold
            
    return df_filled
