# Kinematics and Kinetics Preprocessing Pipeline Implementation Plan

This implementation plan details the step-by-step roadmap for building a scientifically rigorous, modular, and reproducible preprocessing pipeline for the kinematics and kinetics modalities of the SIAT-LLMD dataset.

---

## Stage 1: Dataset Loading

### Objective
Build a robust, memory-efficient data loader to read aligned sensor CSV files and metadata across subjects.

### Proposed Changes
- **Module:** `preprocessing/loader.py`
- **Features:**
  - Read subject metadata (`SubjectInformation.xlsx`) to retrieve physical parameters (age, weight, height) for normalization.
  - Scan the directory structure and load synchronized kinematics and kinetics columns from CSV trials.
  - Parse corresponding label CSV files to extract `Status` (gait phase / movement state) and `Group` (cycle number).
- **Reasoning:** Inter-subject comparison requires torque normalization by body weight, which requires metadata alignment at loading time.

---

## Stage 2: Signal Inspection

### Objective
Analyze signals programmatically to detect signal dropouts, extreme values, or hardware synchronization issues.

### Proposed Changes
- **Module:** `preprocessing/inspector.py`
- **Features:**
  - Check for NaNs, infs, and constant signals (indicating sensor failure).
  - Compute statistics (mean, std, min, max) for each channel per subject and movement.
  - Identify outliers using a standard boxplot threshold ($1.5 \times \text{IQR}$) to detect motion capture tracking errors or transient spikes.
- **Reasoning:** Programmatic anomaly detection requires clean training data; any existing noise or sensor drift in the healthy baseline must be caught and logged early.

---

## Stage 3: Data Cleaning

### Objective
Handle missing values, segment boundaries, and artifacts identified in Stage 2.

### Proposed Changes
- **Module:** `preprocessing/cleaner.py`
- **Features:**
  - Filter out boundary transition frames where `Status` is marked as `NaN` (non-cyclic startup/shutdown phases).
  - Correct localized sensor dropouts or tracking errors via cubic spline interpolation (only for short windows, e.g. $< 10$ consecutive frames).
- **Reasoning:** Excluding non-cyclic transition states ensures that the neural networks model stable gait patterns, preventing boundaries from inflating the reconstruction error.

---

## Stage 4: Preprocessing

### Objective
Apply noise filtering, downsampling, and scaling to condition the signals for deep learning models.

### Proposed Changes
- **Module:** `preprocessing/conditioner.py`
- **Features:**
  - **Low-pass filtering:** Apply a bidirectional zero-phase 4th order Butterworth low-pass filter (cutoff at $6\text{ Hz}$ for kinematics; $10\text{ Hz}$ for kinetics).
  - **Downsampling:** Resample data from $1920\text{ Hz}$ to a computationally manageable rate (e.g. $120\text{ Hz}$ or $200\text{ Hz}$) using decimation.
  - **Torque Normalization:** Divide joint torques by subject weight ($N \cdot m / kg$).
  - **Scaling:** Apply subject-specific Min-Max scaling to range $[-1, 1]$.
- **Reasoning:** Zero-phase filtering avoids shifting the temporal features (crucial for alignment), and downsampling makes sequence lengths manageable for LSTMs and Transformers without losing essential gait dynamics.

---

## Stage 5: Window Generation

### Objective
Generate overlapping sliding windows representing coherent movement periods.

### Proposed Changes
- **Module:** `preprocessing/windower.py`
- **Features:**
  - Segment the continuous time series into sliding windows of fixed size (e.g., $1.5\text{ seconds}$, which is $180\text{ frames}$ at $120\text{ Hz}$).
  - Implement configurable overlap (default $50\%$ to $75\%$).
  - Keep windows only if they belong to the same trial cycle (`Group`) and have a valid, uniform phase label.
- **Reasoning:** Baseline models (LSTM, Transformer) process fixed-size input tensors; segmenting by cycle prevents cross-trial leakage.

---

## Stage 6: Dataset Preparation for Models

### Objective
Produce standardized PyTorch `Dataset` and `DataLoader` instances with rigorous splitting.

### Proposed Changes
- **Module:** `preprocessing/dataset.py`
- **Features:**
  - Implement a `SIATGaitDataset` subclass of `torch.utils.data.Dataset`.
  - Partition the data using a **Leave-Group-Out (Subject-wise)** cross-validation strategy:
    - **Train:** Subjects 1–30.
    - **Validation:** Subjects 31–35.
    - **Test:** Subjects 36–40.
  - Ensure test subjects are completely unseen during training and parameter tuning.
- **Reasoning:** Splitting by subject is mandatory in biomedical signal processing to validate the model's generalizability to new, unseen individuals.

---

## Stage 7: Model Integration (Future)

### Objective
Integrate preprocessed datasets with reconstruction models.

### Proposed Changes
- **Module:** `models/base.py`
- **Features:**
  - Define interfaces for SARIMA, LSTM, and Transformer autoencoders.
  - Ensure the output shape matches the input shape for reconstruction loss calculation.
  - Establish MSE (Mean Squared Error) thresholding logic for anomaly scoring.
- **Reasoning:** A standardized pipeline interface ensures that the same preprocessed windows can be routed into any of the candidate architectures for benchmarking.

---

## Verification Plan

### Automated Tests
- Verification scripts will check:
  - Resampling accuracy (verify target sampling rate).
  - Zero-lag alignment (plot original vs filtered to confirm no shift).
  - Shape compliance of sliding windows.

### Manual Verification
- Visualize filtered kinematic trajectories to ensure high-frequency jitter is eliminated without attenuating peak angles.
