import os
import pandas as pd
import numpy as np

# Column lists for Kinematics and Kinetics
KINEMATIC_COLUMNS = [
    'Kinematic: left hip adduction angle',
    'Kinematic: left hip flexion angle',
    'Kinematic: left knee flexion angle',
    'Kinematic: left ankle flexion angle',
    'Kinematic: right hip adduction angle',
    'Kinematic: right hip flexion angle',
    'Kinematic: right knee flexion angle',
    'Kinematic: right ankle flexion angle'
]

KINETIC_COLUMNS = [
    'Kinetic: left hip adduction torque',
    'Kinetic: left hip flexion torque',
    'Kinetic: left knee flexion torque',
    'Kinetic: left ankle flexion torque',
    'Kinetic: right hip adduction torque',
    'Kinetic: right hip flexion torque',
    'Kinetic: right knee flexion torque',
    'Kinetic: right ankle flexion torque'
]

SIGNAL_COLUMNS = KINEMATIC_COLUMNS + KINETIC_COLUMNS

def load_subject_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Loads subject information from the metadata Excel file.
    
    Parameters
    ----------
    metadata_path : str
        Absolute or relative path to SubjectInformation.xlsx.
        
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by subject ID string (e.g., 'Sub01').
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    
    df = pd.read_excel(metadata_path)
    # Ensure standard subject format (e.g. Sub01) matches exactly
    df['Subject'] = df['Subject'].astype(str).str.strip()
    df.set_index('Subject', inplace=True)
    return df

def load_trial_data(data_path: str, label_path: str) -> pd.DataFrame:
    """
    Loads and aligns synchronized kinematics, kinetics data and labels for a single trial.
    
    Parameters
    ----------
    data_path : str
        Path to the trial's Data CSV file.
    label_path : str
        Path to the trial's Label CSV file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing aligned Time, Kinematics, Kinetics, Status, and Group.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
        
    # Read Data and Label CSVs
    data_df = pd.read_csv(data_path)
    label_df = pd.read_csv(label_path)
    
    # Check shape compatibility
    if len(data_df) != len(label_df):
        # Fallback/Truncate alignment if slight length discrepancy exists
        min_len = min(len(data_df), len(label_df))
        data_df = data_df.iloc[:min_len].copy()
        label_df = label_df.iloc[:min_len].copy()
        
    # Build aligned DataFrame
    combined = pd.DataFrame()
    combined['Time'] = data_df['Time']
    
    # Kinematics
    for col in KINEMATIC_COLUMNS:
        if col in data_df.columns:
            combined[col] = data_df[col]
        else:
            raise KeyError(f"Kinematic column '{col}' missing from data file: {data_path}")
            
    # Kinetics
    for col in KINETIC_COLUMNS:
        if col in data_df.columns:
            combined[col] = data_df[col]
        else:
            raise KeyError(f"Kinetic column '{col}' missing from data file: {data_path}")
            
    # Labels
    combined['Status'] = label_df['Status']
    combined['Group'] = label_df['Group']
    
    return combined

def normalize_kinetics_by_weight(df: pd.DataFrame, subject_weight: float) -> pd.DataFrame:
    """
    Normalizes joint torques (kinetics) by the subject's body weight (in kg)
    to compute biological joint moments (N*m/kg).
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame containing kinetic columns.
    subject_weight : float
        Subject weight in kg.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with weight-normalized kinetic columns.
    """
    if subject_weight <= 0:
        raise ValueError("Subject weight must be strictly positive.")
        
    df_normalized = df.copy()
    for col in KINETIC_COLUMNS:
        df_normalized[col] = df_normalized[col] / subject_weight
    return df_normalized
