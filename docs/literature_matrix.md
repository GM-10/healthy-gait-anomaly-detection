# Literature Matrix

This document tracks every engineering and methodological decision used in the project and the research papers supporting each decision.

The objective is to ensure every statement made in the final paper is backed by published literature whenever possible.

---

# Legend

| Status | Meaning |
|---------|---------|
| ✅ | Strong literature support found |
| 🟡 | Partially supported; engineering judgement still required |
| ❌ | Literature still needed |

---

# Dataset

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Use SIAT-LLMD | ✅ | Wei et al., Scientific Data (2023) | Official dataset paper |
| Healthy subjects only | ✅ | Wei et al. (2023) | 40 healthy participants |
| 16 lower-limb movements | ✅ | Wei et al. (2023) | Dataset description |
| 9 sEMG + 8 Kinematics + 8 Kinetics | ✅ | Wei et al. (2023) | Multimodal dataset |
| 1920 Hz synchronized sampling | ✅ | Wei et al. (2023) | Dataset specification |

---

# Preprocessing

## Butterworth Filtering

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Low-pass Butterworth filter | 🟡 | Need biomechanics paper | Widely used but cutoff needs justification |
| 6 Hz cutoff (Kinematics) | ❌ | Need citation | Verify with gait biomechanics literature |
| 10 Hz cutoff (Kinetics) | ❌ | Need citation | Verify with gait biomechanics literature |
| Zero-phase filtering | 🟡 | Need citation | Standard signal processing practice |

---

## Downsampling

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Downsample to 120 Hz | ❌ | None yet | Current engineering decision |
| Preserve gait dynamics after downsampling | 🟡 | Need citation | Search biomechanics literature |

---

## Windowing

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| 1 second window | ❌ | Need citation | Hypothesis: approximates one gait cycle |
| 50% overlap | ❌ | Need citation | Common in HAR literature; verify |
| Sliding windows | 🟡 | Need citation | Standard time-series preprocessing |

---

## Normalization

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Body-weight normalization for torques | 🟡 | Need biomechanics citation | Common biomechanics practice |
| Min-Max scaling [-1,1] | 🟡 | Need ML citation | Neural network preprocessing |
| Scale using training subjects only | ✅ | ML best practice | Prevents data leakage |

---

# Dataset Split

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Subject-wise split | ✅ | HAR literature | Prevents identity leakage |
| Train: 1–30 | 🟡 | Engineering decision | Exact ratio is project-specific |
| Validation: 31–35 | 🟡 | Engineering decision | |
| Test: 36–40 | 🟡 | Engineering decision | |

---

# Data Quality

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Remove NaNs | ✅ | General preprocessing | Dataset contains none |
| Corruption detection | ✅ | Engineering verification | 1280 files verified |
| IQR outlier detection | 🟡 | Need citation | Robust statistics |

---

# Modeling

## SARIMA

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Classical baseline | 🟡 | Need citation | Baseline time-series model |

---

## LSTM

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| LSTM Autoencoder | 🟡 | Need anomaly detection paper | Reconstruction-based |

---

## Transformer

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Transformer Autoencoder | 🟡 | Need citation | Sequential reconstruction |

---

# Anomaly Detection

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| One-class learning | 🟡 | Need survey paper | Healthy-only training |
| Reconstruction error | 🟡 | Need anomaly detection paper | Common approach |
| MSE loss | ❌ | Need decision | Compare alternatives |
| Threshold selection | ❌ | Need methodology | |

---

# Synthetic Anomalies

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Amplitude scaling | ❌ | Need rehabilitation literature | |
| Time warping | ❌ | Need gait literature | |
| Time shifting | ❌ | Need biomechanics literature | |

---

# Fusion

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Late fusion | ❌ | Need multimodal paper | |
| Early fusion comparison | ❌ | Need survey | |

---

# Evaluation

| Decision | Status | Supporting Paper | Notes |
|----------|--------|-----------------|------|
| Recall as primary metric | 🟡 | Need clinical citation | False negatives are clinically costly |
| F1-score secondary | ✅ | Standard ML | |
| Reconstruction error distributions | 🟡 | Need anomaly detection paper | |

---

# Open Literature Tasks

## Highest Priority

- [ ] Justify Butterworth cutoff frequencies.
- [ ] Justify 120 Hz downsampling.
- [ ] Justify 1-second windows.
- [ ] Justify 50% overlap.
- [ ] Justify body-weight normalization.
- [ ] Justify reconstruction loss.
- [ ] Justify reconstruction threshold.

---

## Medium Priority

- [ ] Survey reconstruction-based anomaly detection.
- [ ] Survey gait anomaly detection.
- [ ] Survey multimodal fusion.
- [ ] Survey Transformer-based gait modeling.

---

## Low Priority

- [ ] Compare Min-Max vs Z-score normalization.
- [ ] Compare reconstruction losses.
- [ ] Compare anomaly thresholding methods.

---

# Key Papers

## Dataset

Wei et al.

"Shenzhen Institute of Advanced Technology Lower Limb Motion Dataset (SIAT-LLMD)."

Scientific Data, 2023.

Status

Read

---

## Gait Biomechanics

Pending

---

## Signal Processing

Pending

---

## Human Activity Recognition

Pending

---

## Anomaly Detection

Pending

---

## Autoencoders

Pending

---

## Transformer Models

Pending

---

# Notes

Every engineering decision should eventually move from

❌

↓

🟡

↓

✅

before the paper is finalized.

The methodology section should reference this document to ensure every claim is supported by published literature or clearly identified as an engineering choice.