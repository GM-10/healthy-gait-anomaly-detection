# SIAT-LLMD Dataset Summary (Kinematics and Kinetics Focus)

This document provides a comprehensive summary of the SIAT-LLMD dataset based on direct file inspections, official loading scripts, and published literature.

## 1. Folder Structure and File Naming Conventions

### Folder Structure
```
SIAT_LLMD20230404/
├── Code/
│   ├── codeV4.2/
│   │   ├── environments/
│   │   │   └── sEMG_IR.yaml          # Conda environment configuration
│   │   ├── EMGbox_WWH.py             # Feature extraction and utility box
│   │   ├── SIAT_LLMD_Load_ForClassification.py
│   │   └── SIAT_LLMD_Load_ForGait.py
│   ├── LICENSE
│   └── ReadMe.txt
├── Sub01/ ... Sub40/                 # 40 Healthy Subjects
│   ├── Data/
│   │   └── Subxx_[Movement]_Data.csv # Aligned sensor recordings (Time, Kinematics, Kinetics, sEMG)
│   ├── Figures/                      # Visualizations (empty or unreferenced in pipeline)
│   │   └── ...
│   └── Labels/
│       └── Subxx_[Movement]_Label.csv # Temporal annotation files (Time, Status, Group)
└── SubjectInformation.xlsx            # Physical metadata for all 40 subjects
```

### File Naming Convention
- **Data files:** `SIAT_LLMD20230404/Sub{Subject_ID}/Data/Sub{Subject_ID}_{Movement_Code}_Data.csv`
- **Label files:** `SIAT_LLMD20230404/Sub{Subject_ID}/Labels/Sub{Subject_ID}_{Movement_Code}_Label.csv`
  - `Subject_ID` is formatted as two digits: `01` to `40`.
  - `Movement_Code` is one of the 16 movements described below.

---

## 2. Subjects and Movements

### Number of Subjects
- **Total:** 40 healthy subjects.
- **Physical Metadata:** Recorded in `SubjectInformation.xlsx` including `Subject`, `age`, `weight`, `sex`, `height`, and anatomical segment measurements (e.g., segment lengths and joint widths like knee and ankle width).

### Number of Movements
There are exactly **16 movements** recorded per subject (16 data CSV files and 16 label CSV files per subject):
1. **WAK:** Walking (level walking)
2. **UPS:** Upstairs (stair ascent)
3. **DNS:** Downstairs (stair descent)
4. **HS:** High Step
5. **KLCL:** Knee flexion/extension in a closed-loop setup
6. **KLFT:** Knee flexion/extension in a floating (open-loop) setup
7. **LLB:** Left leg backward swing
8. **LLF:** Left leg forward swing
9. **LLS:** Left leg sideways swing
10. **LUGB:** Lunge backward
11. **LUGF:** Lunge forward
12. **SITDN:** Sit down
13. **STC:** Static trial (standing still)
14. **STDUP:** Stand up
15. **TO:** Toe-off / Tiptoe stance
16. **TPTO:** Triple tiptoe stance

---

## 3. Channels and Signal Specifications

The project focus is strictly on **Kinematics** and **Kinetics** modalities.

### Kinematic Channel Names (Columns 1–8 in Data CSVs)
- `Kinematic: left hip adduction angle`
- `Kinematic: left hip flexion angle`
- `Kinematic: left knee flexion angle`
- `Kinematic: left ankle flexion angle`
- `Kinematic: right hip adduction angle`
- `Kinematic: right hip flexion angle`
- `Kinematic: right knee flexion angle`
- `Kinematic: right ankle flexion angle`

### Kinetic Channel Names (Columns 9–16 in Data CSVs)
- `Kinetic: left hip adduction torque`
- `Kinetic: left hip flexion torque`
- `Kinetic: left knee flexion torque`
- `Kinetic: left ankle flexion torque`
- `Kinetic: right hip adduction torque`
- `Kinetic: right hip flexion torque`
- `Kinetic: right knee flexion torque`
- `Kinetic: right ankle flexion torque`

### Units of Measurement
- **Kinematics (Joint Angles):** Degrees (standard OpenSim joint angles, confirmed by typical ranges e.g. $-10^\circ$ to $40^\circ$).
- **Kinetics (Joint Torques):** $N \cdot m$ (Newton-meters) or normalized $N \cdot m / kg$ (standard OpenSim inverse dynamics output).

### Sampling Frequency
- **Raw Frequency:** sEMG was acquired at 1920 Hz.
- **Resampling/Synchronization:** Kinematic and kinetic data were resampled and synchronized to match sEMG rows. Therefore, the files are aligned row-by-row at a unified frequency of **1920 Hz**.
- **Timestamp Format:** Column 0 (`Time`) is a floating-point representation starting from 0.000000000000000 with a uniform delta of approximately $0.000520833$ seconds ($1 / 1920$ Hz).

---

## 4. Label Conventions and Gait Phases

The label files contain three columns: `Time`, `Status`, and `Group`.
- **Status:** Annotates the active gait phase or action state:
  - **Discrete gait phases** (for walking/stair climbing):
    - `WAK_Label.csv`: Uses numeric values `1.0` through `5.0` corresponding to the 5 gait phases:
      - `1.0`: Heel Strike to Mid-Stance Flexion (`HS-MSF`)
      - `2.0`: Mid-Stance Flexion to Mid-Stance Extension (`MSF-MSE`)
      - `3.0`: Mid-Stance Extension to Toe-Off (`MSE-TO`)
      - `4.0`: Toe-Off to Mid-Swing Flexion (`TO-MWF`)
      - `5.0`: Mid-Swing Flexion to Heel Strike (`MWF-HS`)
    - `UPS_Label.csv` & `DNS_Label.csv`: Uses numeric values `1.0` to `3.0` corresponding to:
      - `1.0`: Heel Strike to Toe-Off (`HS-TO`)
      - `2.0`: Toe-Off to Mid-Swing Flexion (`TO-MWF`)
      - `3.0`: Mid-Swing Flexion to Heel Strike (`MWF-HS`)
  - **Active vs. Rest states** (for discrete movements like lunges or knee flexions):
    - Values are `'A'` (Active movement) and `'R'` (Rest/Static stance).
  - **Static trials** (`STC`):
    - Annotated strictly with `'R'` (Rest).
- **Group:** Integer denoting the cycle or repetition index of the trial (e.g. 1, 2, 3, etc.).
- **NaNs in Labels:** The initialization/termination phases of trials are marked as `NaN` (Status) to filter out non-cyclic boundary movements.

---

## 5. Data Quality, Missing Values, and Anomalies

A comprehensive programmatic validation was performed across all $1,280$ files ($40 \text{ subjects} \times 16 \text{ trials} \times 2 \text{ files (Data + Labels)}$):
- **Missing Values:** Exactly **0 missing values (NaNs)** exist in the kinematic and kinetic channels across the entire dataset.
- **Corrupted Files:** **0 files are corrupted**. All CSV files are readable and well-formed.
- **Recording Durations:** Durations vary by subject and trial. For example, a typical walking trial (`Sub01_WAK_Data.csv`) contains $27,166$ samples, representing $14.15$ seconds of movement.
