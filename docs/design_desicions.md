# Design & Engineering Decisions

This document records every major design decision made during the development of the preprocessing and modeling pipeline.

The objective is to ensure that every engineering choice is either:

- Supported by the SIAT-LLMD paper,
- Supported by existing biomechanics or machine learning literature,
- Or explicitly documented as an engineering decision requiring future validation.

---

# Decision Tracking Legend

| Status | Meaning |
|---------|---------|
| ✅ | Justified by literature or strong engineering rationale |
| 🟡 | Engineering decision (reasonable but requires justification) |
| ❌ | Not yet justified |

---

# 1. Subject-wise Train / Validation / Test Split

## Current Decision

Train:
- Subjects 01–30

Validation:
- Subjects 31–35

Test:
- Subjects 36–40

Status:
✅ Accepted

Reason

Random train/test splitting causes data leakage because windows from the same individual may appear in both training and testing.

Subject-wise splitting ensures the model is evaluated on completely unseen individuals, which better reflects real-world deployment.

Action

Keep this split.

Mention data leakage prevention explicitly in the paper.

---

# 2. Missing Values

Question

Does the dataset contain missing values?

Current Finding

- 1280 files inspected
- 0 NaN values
- 0 corrupted files

Status

✅ Verified

Action

Mention dataset quality in the paper.

---

# 3. Corrupted Files

Question

Are there corrupted recordings?

Current Finding

None detected.

Status

✅ Verified

---

# 4. Kinematic and Kinetic Channels

Question

Have all channels been mapped?

Current Finding

Yes.

- 8 Kinematic channels
- 8 Kinetic channels

Status

✅ Verified

---

# 5. Sampling Frequency

Question

What sampling frequency is used?

Current Finding

1920 Hz synchronized recordings.

Status

✅ Verified

Reference

SIAT-LLMD dataset paper.

---

# 6. Butterworth Filtering

Current Decision

Kinematics:
- 6 Hz low-pass Butterworth

Kinetics:
- 10 Hz low-pass Butterworth

Status

🟡 Engineering Decision

Current Justification

Butterworth filtering is standard practice in gait biomechanics to suppress high-frequency measurement noise while preserving physiologically meaningful motion.

Issue

The SIAT-LLMD paper does NOT explicitly prescribe these cutoff frequencies.

Required

Find literature supporting:

- 6 Hz cutoff for kinematics
- 10 Hz cutoff for kinetics

---

# 7. Downsampling

Current Decision

1920 Hz

↓

120 Hz

Status

🟡 Engineering Decision

Reason

Reduce computational cost while retaining biomechanical information.

Issue

No explicit recommendation exists in the SIAT-LLMD paper.

Need

Literature support OR engineering justification.

Potential future experiment

Compare:

- 100 Hz
- 120 Hz
- 200 Hz

---

# 8. Window Size

Current Decision

1 second

120 samples

50% overlap

Status

🟡 Engineering Decision

Reason

Approximately captures one complete gait cycle.

Issue

Not specified by SIAT-LLMD.

Need

Literature showing:

Typical healthy gait cycle duration.

---

# 9. Window Overlap

Current Decision

50%

Status

🟡 Engineering Decision

Need

Literature justification.

---

# 10. Min-Max Scaling

Current Decision

Normalize features to

[-1,1]

Status

✅ Reasonable Engineering Decision

Reason

Different joints and torques have different numerical ranges.

Scaling improves optimization stability for neural networks.

Need

General ML citation.

---

# 11. Torque Normalization

Current Decision

Normalize by body weight.

Status

✅ Supported

Reason

Joint torques naturally scale with subject size.

Allows comparison across individuals.

Need

Biomechanics citation.

---

# 12. Gait Phase Labels

Current Decision

Keep metadata.

Do NOT train using them.

Status

✅ Accepted

Reason

Useful later for error analysis.

Possible paper figure:

Reconstruction Error vs Gait Phase

---

# 13. Metadata Preservation

Current Decision

Every window stores:

- Subject
- Movement
- Start time
- End time
- Group cycle
- Gait phase

Status

✅ Accepted

Reason

Allows later statistical analysis and interpretability.

---

# 14. Dataset Output

Current Decision

Keep processed windows in memory.

Output:

windows_data

windows_metadata

scaling_params

Status

✅ Accepted

Future Improvement

Add versioned disk caching.

---

# 15. Disk Caching

Current Decision

Not implemented.

Recommended

cache/

    v1/

        dataset.pt

        metadata.json

Status

🟡 Future Improvement

---

# 16. Reconstruction Target

Question

Exactly what will the models reconstruct?

Options

- Entire 16-channel window
- Separate modality reconstruction
- Channel-wise reconstruction

Status

❌ Open

Needs discussion before modeling.

---

# 17. Synthetic Anomaly Injection

Current Decision

Not yet implemented.

Status

❌ Open

Need literature supporting:

- Amplitude scaling
- Time warping
- Time shifting

---

# 18. Reconstruction Error

Question

How should anomaly score be computed?

Possible metrics

- MSE
- MAE
- RMSE
- Dynamic Time Warping

Status

❌ Open

---

# 19. Evaluation Metrics

Current Decision

Primary:

Recall

Secondary:

F1

Status

✅ Accepted

Reason

False negatives are clinically more serious than false positives.

Need

Clinical rehabilitation citation.

---

# 20. Multimodal Fusion

Current Decision

Late Fusion

Status

🟡 Planned

Need

Literature comparing:

- Early Fusion
- Intermediate Fusion
- Late Fusion

---

# 21. Models

Planned

1. SARIMA

2. LSTM Autoencoder

3. Transformer Autoencoder

Status

🟡 Planned

Need

Final architecture discussion before implementation.

---

# Open Questions Before Modeling

- Why exactly 120 Hz?
- Why exactly 1 second?
- Why 50% overlap?
- Why 6 Hz and 10 Hz cutoffs?
- What reconstruction loss should be used?
- What exactly is reconstructed?
- How should anomaly thresholds be chosen?
- How should synthetic anomalies be validated?
- Should preprocessing outputs be cached to disk?

---

# Overall Project Status

Dataset Exploration

✅ Complete

Documentation Review

✅ Complete

Preprocessing Pipeline

✅ Complete

Testing

✅ Complete

Engineering Justifications

🟡 In Progress

Model Development

❌ Not Started

Paper Writing

❌ Not Started