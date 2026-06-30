# SIAT-LLMD Preprocessing Recommendations (Kinematics and Kinetics Focus)

This document outlines the recommendations and specifications for preprocessing kinematic and kinetic modalities, drawing from the data acquisition details, original dataset scripts, and biomechanical modeling principles.

## 1. Data Acquisition and Sensor Setup

- **sEMG System (Left Limb):** 9-channel wireless sEMG system (Delsys, USA) placed on major lower-limb muscle groups: tensor fascia lata, rectus femoris, vastus medialis, semimembranosus, upper tibialis anterior, lower tibialis anterior, lateral gastrocnemius, medial gastrocnemius, and soleus.
- **Motion Capture System (Kinematics):** 3D optical motion capture system (Vicon, UK) with marker trajectories mapped onto anatomical landmarks.
- **Force Plate System (Kinetics):** Six-dimensional force platforms (force plates) embedded in the walking path to measure 3D ground reaction forces (GRF) and moments.
- **Musculoskeletal Modeling:** Kinematics (joint angles) and kinetics (joint torques) were computed by processing raw marker trajectories and force plate data in **OpenSim** using the standard **Gait2392** model via Inverse Kinematics (IK) and Inverse Dynamics (ID) solvers.
- **Coordinate System:** Follows the right-handed Cartesian coordinate system native to OpenSim/Vicon (typically $X$: anterior-posterior, $Y$: vertical, $Z$: mediolateral).
- **Synchronization:** The Delsys acquisition system, Vicon motion capture system, and force plates were hardware-synchronized. The joint angles and torques derived from OpenSim were resampled to match the high-frequency sEMG sampling rate of **1920 Hz**.

---

## 2. Existing Preprocessing Recommendations in Dataset Literature

### sEMG Preprocessing (Official code baseline)
For muscle signals, the official code implements:
- **Baseline removal:** Detrending.
- **Notch Filter:** Removal of 50 Hz powerline interference.
- **Butterworth Bandpass Filter:** 7th order bandpass filter between 15 Hz and 400 Hz.
- **Wavelet Packet Denoising:** 8/9 level decomposition using `db7` with soft thresholding (threshold of 0.08).

### Kinematics and Kinetics Preprocessing
The official scripts read joint angles and torques raw from the Excel/CSV outputs because OpenSim IK/ID processes already involve some degree of marker filtering. However, for deep-learning-based gait anomaly detection, the following recommendations are highly relevant:

- **Filter Specification:** Apply a zero-phase (bidirectional) 4th order low-pass Butterworth filter.
  - **Kinematics cutoff frequency:** $6 \text{ Hz}$ to $10 \text{ Hz}$ is standard in gait analysis to remove marker jitter and soft-tissue movement artifacts.
  - **Kinetics cutoff frequency:** $10 \text{ Hz}$ to $15 \text{ Hz}$ is standard to remove impact transients and force plate noise.
- **Zero-lag Filtering:** Bidirectional filtering (`scipy.signal.filtfilt`) is critical to ensure that no phase shift is introduced into the signals, keeping the time-alignment of heel-strikes and toe-offs accurate.

---

## 3. Windowing Recommendations

- **Classification vs. Anomaly Detection:**
  - **Classification (Official Code):** Uses small overlapping sliding windows ($100$ or $150$ samples, corresponding to $52 \text{ ms}$ or $78 \text{ ms}$ at 1920 Hz) with an overlap of $50$ samples. This is optimal for low-latency real-time control but insufficient for learning structural patterns of healthy gait.
  - **Reconstruction-based Anomaly Detection (Our Objective):** Since our baseline models (SARIMA, LSTM, Transformer) need to model the underlying distribution of healthy gait cycles, windows should capture **at least one complete gait cycle** (gait cycle period for healthy walking is typically $1.0$ to $1.2$ seconds).
- **Proposed Windowing Scheme:**
  - **Window Size:** $2.0$ seconds to capture a complete stride including boundaries (equivalent to $3840$ samples at 1920 Hz).
  - **Overlap:** $50\%$ to $75\%$ overlap during training to generate sufficient sample density.
  - **Boundary Handling:** Filter out windows that contain transition states or start/stop boundaries where label `Status` is `NaN`.

---

## 4. Normalization and Scaling Recommendations

- **Subject-Specific vs. Global Scaling:**
  - Joint angles and torques exhibit significant inter-subject variability due to differences in heights, weights, and walking speeds.
  - **Recommendation:** Perform **subject-specific Min-Max scaling** to range $[-1, 1]$ or $[0, 1]$ for kinematics, and **Z-score standardization** for kinetics.
  - Optionally, normalize torques by the subject's body weight (using `weight` from `SubjectInformation.xlsx`) to compute biological joint moments ($N \cdot m / kg$), making kinetic profiles comparable across subjects.
- **Channel-Wise Scaling:** Apply scaling independently per channel (column) to prevent channels with larger absolute ranges (e.g. knee flexion angles $0^\circ$ to $60^\circ$) from dominating channels with smaller ranges (e.g. hip adduction angles $-5^\circ$ to $5^\circ$).
