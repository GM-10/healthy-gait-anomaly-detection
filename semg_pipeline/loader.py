"""
semg_pipeline/loader.py

Loads sEMG signal columns and gait-phase label files from the SIAT-LLMD
dataset CSVs for a single trial (subject × movement).

Column layout of Data CSVs (0-indexed):
    Col  0     : Time
    Cols  1– 8 : Kinematics (8 channels)  — not loaded here
    Cols  9–16 : Kinetics   (8 channels)  — not loaded here
    Cols 17–25 : sEMG       (9 channels)  ← loaded here

Label CSV columns: Time, Status, Group
    Status : NaN = transition / boundary, 0 = rest, 1 = active gait
    Group  : integer gait-cycle index (1-based)
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Canonical short channel names used in output CSVs and scaler dicts.
SEMG_CHANNELS = [
    "tensor_fascia_lata",
    "rectus_femoris",
    "vastus_medialis",
    "semimembranosus",
    "upper_tibialis_anterior",
    "lower_tibialis_anterior",
    "lateral_gastrocnemius",
    "medial_gastrocnemius",
    "soleus",
]

# Mapping: exact CSV column name → canonical short name
SEMG_COL_MAP = {
    "sEMG: tensor fascia lata":        "tensor_fascia_lata",
    "sEMG: rectus femoris":            "rectus_femoris",
    "sEMG: vastus medialis":           "vastus_medialis",
    "sEMG: semimembranosus":           "semimembranosus",
    "sEMG: upper tibialis anterior":   "upper_tibialis_anterior",
    "sEMG: lower tibialis anterior":   "lower_tibialis_anterior",
    "sEMG: lateral gastrocnemius":     "lateral_gastrocnemius",
    "sEMG: medial gastrocnemius":      "medial_gastrocnemius",
    "sEMG: soleus":                    "soleus",
}

# All 16 movement codes present in the SIAT-LLMD dataset
MOVEMENTS = [
    "DNS", "HS",   "KLCL", "KLFT", "LLB",  "LLF",
    "LLS", "LUGB", "LUGF", "SITDN","STC",  "STDUP",
    "TO",  "TPTO", "UPS",  "WAK",
]

# Subject splits — must match teammate's kinematics/kinetics pipeline exactly
TRAIN_SUBS = [f"Sub{i:02d}" for i in range(1, 31)]   # Sub01–Sub30
VAL_SUBS   = [f"Sub{i:02d}" for i in range(31, 36)]  # Sub31–Sub35
TEST_SUBS  = [f"Sub{i:02d}" for i in range(36, 41)]  # Sub36–Sub40


# ─────────────────────────────────────────────────────────────────────────────
# Core loading function
# ─────────────────────────────────────────────────────────────────────────────

def load_semg_trial(
    data_path: str,
    label_path: str,
    active_only: bool = True,
) -> pd.DataFrame:
    """
    Load sEMG channels and labels for a single trial.

    Parameters
    ----------
    data_path : str
        Path to Sub##_[MOV]_Data.csv
    label_path : str
        Path to Sub##_[MOV]_Label.csv
    active_only : bool, default True
        If True, keep only rows corresponding to active gait/movement.
        The filtering logic adapts to the three label conventions used
        in SIAT-LLMD:

        Convention A — numeric gait phases (WAK, UPS, DNS):
            Status ∈ {1.0, 2.0, 3.0, 4.0, 5.0} (or 1–3 for UPS/DNS)
            Keep: all non-NaN rows (every phase is active gait)
            Discard: NaN rows (boundary/initialisation frames)

        Convention B — string Active/Rest (LLB, LLF, LLS, LUGB, LUGF,
                        KLCL, KLFT, HS, SITDN, STDUP, TPTO, TO):
            Status ∈ {'A', 'R'}
            Keep: rows where Status == 'A'
            Discard: 'R' (rest) and NaN rows

        Convention C — static trial (STC):
            Status = 'R' only — no active movement phase exists.
            If active_only=True, returns empty DataFrame (no active rows).

    Returns
    -------
    pd.DataFrame
        Columns: Time, <9 sEMG short-named channels>, Status, Group
        Index is reset (integer).

    Raises
    ------
    FileNotFoundError
        If either CSV path does not exist.
    KeyError
        If expected sEMG columns are missing from the data file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # ------------------------------------------------------------------
    # 1. Read CSVs
    # ------------------------------------------------------------------
    data_df  = pd.read_csv(data_path)
    label_df = pd.read_csv(label_path)

    # ------------------------------------------------------------------
    # 2. Length alignment (fallback for minor sync discrepancies)
    # ------------------------------------------------------------------
    if len(data_df) != len(label_df):
        min_len  = min(len(data_df), len(label_df))
        data_df  = data_df.iloc[:min_len].copy()
        label_df = label_df.iloc[:min_len].copy()

    # ------------------------------------------------------------------
    # 3. Extract sEMG columns and rename to canonical short names
    # ------------------------------------------------------------------
    missing = [c for c in SEMG_COL_MAP if c not in data_df.columns]
    if missing:
        raise KeyError(
            f"Missing sEMG columns in {data_path}:\n  {missing}"
        )

    combined = pd.DataFrame()
    combined["Time"] = data_df["Time"].values

    for csv_col, short_name in SEMG_COL_MAP.items():
        combined[short_name] = data_df[csv_col].values

    # ------------------------------------------------------------------
    # 4. Attach label columns
    # ------------------------------------------------------------------
    combined["Status"] = label_df["Status"].values
    combined["Group"]  = label_df["Group"].values

    # ------------------------------------------------------------------
    # 5. Keep only active gait frames
    #
    # Label convention detection:
    #   - If non-NaN Status values are numeric  → Convention A (gait phases)
    #     Keep all non-NaN rows.
    #   - If non-NaN Status values are strings  → Convention B (A / R labels)
    #     Keep only rows where Status == 'A'.
    # ------------------------------------------------------------------
    if active_only:
        status_col   = combined["Status"]
        non_nan_vals = status_col.dropna()

        if len(non_nan_vals) == 0:
            # No usable labels — return empty
            return combined.iloc[0:0].copy()

        # Try numeric coercion to distinguish convention
        coerced = pd.to_numeric(non_nan_vals, errors="coerce")
        if coerced.notna().all():
            # Convention A: all non-NaN are numeric gait-phase codes
            # → keep every non-NaN row (NaN = boundary transition)
            mask = status_col.notna()
        else:
            # Convention B: string 'A' / 'R' labels
            mask = status_col == "A"

        combined = combined[mask].copy()
        combined.reset_index(drop=True, inplace=True)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Path helper
# ─────────────────────────────────────────────────────────────────────────────

def build_trial_paths(
    base_dir: str,
    subject: str,
    movement: str,
) -> Tuple[str, str]:
    """
    Construct the canonical data and label CSV paths for a given trial.

    Parameters
    ----------
    base_dir : str
        Root of the dataset, e.g. 'SIAT_LLMD20230404/SIAT_LLMD20230404'.
    subject : str
        Subject ID, e.g. 'Sub01'.
    movement : str
        Movement code, e.g. 'WAK'.

    Returns
    -------
    (data_path, label_path) : Tuple[str, str]
    """
    data_path  = os.path.join(
        base_dir, subject, "Data",   f"{subject}_{movement}_Data.csv"
    )
    label_path = os.path.join(
        base_dir, subject, "Labels", f"{subject}_{movement}_Label.csv"
    )
    return data_path, label_path
