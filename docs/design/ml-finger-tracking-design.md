# SIMCAP ML Design Document

**Document Version:** 2.0
**Date:** 2026-01-02
**Status:** Living Document
**Authors:** SIMCAP Development Team

---

## Executive Summary

This document tracks the evolution of machine learning approaches for **finger state inference** from magnetometer data in SIMCAP. The goal is to classify 5-finger binary states (extended/flexed = 32 combinations) using magnets attached to each finger and a magnetometer sensor on the palm.

**Current Best Results:**
| Approach | Accuracy | Use Case |
|----------|----------|----------|
| Real Data Only (single-sample) | **99.8%** | Observed combos (10/32) |
| Calibrated Hybrid | **95.9%** | All 32 combos coverage |
| Deployed Model (finger_aligned_v2) | **99.6%** | Windowed CNN-LSTM (50×9) |

**Key Challenge:** Only 10 of 32 finger combinations have real training data. Physics-based synthetic data generation is needed for full coverage.

---

## 1. Project Context

### 1.1 Physical Setup

- **Sensor:** Puck.js v2 magnetometer (MMC5603NJ) on palm of hand
- **Magnets:** Small neodymium magnets on mid-finger (middle phalanx), palmar side
- **Mechanics:** Fingers curl TOWARD sensor when flexed (closer to palm)
- **Data Rate:** 50 Hz, 9-DoF IMU (accel, gyro, mag)

### 1.2 Data Source

Primary training data from Dec 31, 2025 session:
- **File:** `data/GAMBIT/2025-12-31T14_06_18.270Z.json`
- **Observed combos:** 10 of 32 possible combinations
- **Samples:** ~2,165 labeled residual samples
- **Baseline:** [46.0, -45.8, 31.3] μT (open palm = eeeee)

### 1.3 Related Documentation

| Document | Location | Content |
|----------|----------|---------|
| ML Pipeline README | [`ml/README.md`](ml/README.md) | Build pipeline, commands |
| Model Deployment | [`public/models/CLAUDE.md`](public/models/CLAUDE.md) | TF.js conversion, registry |
| Magnetic Simulation | [`docs/design/magnetic-field-simulation-exploration.md`](docs/design/magnetic-field-simulation-exploration.md) | Physics-based synthetic data |
| FFO$$ Research | [`docs/design/ffo-dollar-research-analysis.md`](docs/design/ffo-dollar-research-analysis.md) | Template matching alternative |
| Residual Analysis | [`ml/RESIDUAL_ANALYSIS_SUMMARY.md`](ml/RESIDUAL_ANALYSIS_SUMMARY.md) | Magnetic residual findings |
| Clustering Guide | [`ml/CLUSTERING.md`](ml/CLUSTERING.md) | Unsupervised labeling |

---

## 2. Architecture Evolution

### 2.1 Model Versions (Deployed)

| Model ID | Date | Type | Accuracy | Status |
|----------|------|------|----------|--------|
| `gesture_v1` | 2025-12-09 | 10-class gesture CNN | ~55% | Superseded |
| `finger_contrastive_v1` | 2025-12-26 | Contrastive pre-training | Experimental | Inactive |
| `finger_aligned_v1` | 2025-12-31 | Single-sample magnetic | 97.3% | Baseline |
| **`finger_aligned_v2`** | 2026-01-02 | CNN-LSTM hybrid (50×9) | **99.6%** | **Active** |

### 2.2 Current Deployed Model

**`finger_aligned_v2`** configuration (from `public/models/finger_aligned_v2/config.json`):

```json
{
  "inputShape": [null, 50, 9],
  "windowSize": 50,
  "accuracy": {
    "overall": 0.996,
    "per_finger": {
      "thumb": 0.992, "index": 0.997, "middle": 0.995,
      "ring": 1.0, "pinky": 0.997
    }
  }
}
```

Architecture: CNN-LSTM Hybrid
```
Input(50, 9) → Conv1D(32) → BN → MaxPool → Conv1D(64) → BN → MaxPool
            → LSTM(32) → Dropout(0.3) → Dense(32) → Dense(5, sigmoid)
```

---

## 3. Approaches Explored

### 3.1 Supervised Learning (Primary Path)

**Timeline:** Dec 2025 - Present

| Approach | Script | Result | Notes |
|----------|--------|--------|-------|
| Windowed CNN | `ml/train.py` | ~90% | Standard gesture classification |
| Per-finger binary | `ml/train_finger_classifier.py` | 97%+ | Multi-output sigmoid |
| CNN-LSTM hybrid | `ml/deploy_finger_model_v2.py` | **99.6%** | Best architecture |
| Ground truth aligned | `ml/train_aligned_classifier.py` | 99%+ | Uses measured baseline |

### 3.2 Magnetic Residual Analysis

**Timeline:** Jan 2026

**Key Finding:** Using magnetic residual (raw_mag - baseline) instead of raw magnetometer values.

| Feature Set | Accuracy | Features |
|-------------|----------|----------|
| 9-feature raw (windowed) | 97.2% | accel, gyro, mag |
| 3-feature residual (windowed) | **97.8%** | mag_residual only |
| 3-feature residual (single-sample) | **98.6%** | mag_residual only |

**Conclusion:** 3-feature residual achieves **66% fewer features** with same accuracy.

See: [`ml/RESIDUAL_ANALYSIS_SUMMARY.md`](ml/RESIDUAL_ANALYSIS_SUMMARY.md)

### 3.3 Physics-Based Synthetic Data

**Timeline:** Jan 2026

**Goal:** Generate training data for the 22 missing finger combinations using magnetic dipole physics.

#### 3.3.1 Additive Model (Failed)

Assumption: `residual(combo) = Σ residual(individual fingers)`

**Result:** 265% mean error, 36.8% training accuracy

```
thumb + index → predicted [+925, -225, +641] vs actual [+352, -185, +636] (76% error)
all 5 fingers → predicted [+1463, -1244, +4355] vs actual [+589, -245, +599] (455% error)
```

**Conclusion:** Finger effects are NOT additive due to physical magnet interactions.

See: [`ml/analyze_finger_interactions.py`](ml/analyze_finger_interactions.py)

#### 3.3.2 Interaction Model (Improved)

**Discovery:** Each additional flexed finger SUPPRESSES the total field.

```python
interaction_strength = -0.702  # From fitted model
scaling = 1.0 + interaction * (n_flexed - 1) / 4  # Per additional finger
# Result: 0.30x scaling per additional finger
```

**Per-Finger Fitted Effects (μT):**
| Finger | X | Y | Z | Magnitude |
|--------|-----|-----|------|-----------|
| Thumb | +249 | -79 | -315 | 410 |
| Index | +398 | -169 | +1046 | 1130 |
| Middle | +423 | -332 | +602 | 810 |
| Ring | -613 | +527 | +11 | 808 |
| Pinky | +699 | -460 | +1456 | 1670 |

**Training with Interaction Model:**
| Approach | Accuracy | Notes |
|----------|----------|-------|
| Additive Synthetic | 36.8% | Baseline (fails) |
| Interaction Synthetic | 34.8% | Domain gap |
| Hybrid (Real + Interaction) | 89.0% | Prior approach |
| **Calibrated Hybrid** | **95.9%** | +6.9% improvement |

See: [`ml/per_finger_fit_results.json`](ml/per_finger_fit_results.json)

#### 3.3.3 Calibrated Synthetic (Current Best)

**Key Improvements:**
1. Real data ONLY for observed combos (no mixing)
2. Synthetic ONLY for missing combos
3. Noise calibrated from nearest observed combo
4. Confidence weighting by Hamming distance

See: [`ml/train_improved_hybrid.py`](ml/train_improved_hybrid.py)

### 3.4 Magpylib Physics Simulation

**Timeline:** Jan 2026

Used `magpylib` for accurate magnetic dipole field calculation:

```python
import magpylib as magpy

magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1400),  # mT (NdFeB)
    dimension=(6, 3),  # 6mm diameter, 3mm height
    position=(x, y, z),
)
B = collection.getB([0, 0, 0]) * 1000  # Sensor at origin, convert to μT
```

**Anatomical Hand Model:**
- Sensor on palm (origin)
- Magnets on mid-finger (middle phalanx)
- Extended: fingers far (~35mm Z)
- Flexed: fingers close (~12mm Z)

**Limitation:** Physics model has ~39% prediction error → synthetic quality varies by combo.

See: [`ml/physics_sim_constrained.py`](ml/physics_sim_constrained.py)

### 3.5 Leave-One-Out Generalization

**Question:** Can models trained with synthetic data predict UNSEEN combos?

**Results:**
| Approach | Mean Generalization |
|----------|---------------------|
| Real Data Only | **0%** (cannot generalize) |
| With Synthetic | **16.1%** (+16.1%) |

**Per-Combo Breakdown:**
| Held-Out | Real Only | With Synthetic |
|----------|-----------|----------------|
| eeefe (ring) | 0% | **100%** |
| feeee (thumb) | 0% | **44.7%** |
| Others | 0% | 0% |

**Conclusion:** Synthetic helps for ring and thumb (well-fitted), fails for poorly-fitted combos.

See: [`ml/generalization_analysis.json`](ml/generalization_analysis.json)

---

## 4. Current State Summary

### 4.1 What Works

1. **Single-sample residual classification:** 98.6-99.8% on observed combos
2. **CNN-LSTM windowed model:** 99.6% accuracy (deployed as v2)
3. **Interaction model prediction:** 39% mean error (vs 265% additive)
4. **Calibrated hybrid training:** 95.9% with synthetic augmentation

### 4.2 What Doesn't Work

1. **Additive synthetic:** Fails completely (36.8%)
2. **Generalization to multi-finger combos:** 0% without real data
3. **Physics simulation accuracy:** 39% error is too high for reliable synthetic

### 4.3 Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Observed combos | 10/32 | Dec 31 session |
| Missing combos | 22/32 | Need synthetic or collection |
| Best observed accuracy | 99.8% | Real data only |
| Best full coverage | 95.9% | Calibrated hybrid |
| Generalization gap | 4.1% | (99.8 - 95.9) |
| Interaction strength | -0.702 | Fitted model |

---

## 5. Implementation Files

### 5.1 Core Pipeline

| File | Purpose |
|------|---------|
| `ml/build.py` | Unified train → convert → deploy |
| `ml/train.py` | Main supervised training |
| `ml/model.py` | Model architectures |
| `ml/data_loader.py` | Session data loading |

### 5.2 Finger State Models

| File | Purpose |
|------|---------|
| `ml/deploy_finger_model_v2.py` | CNN-LSTM deployment |
| `ml/train_finger_classifier.py` | Multi-output training |
| `ml/train_aligned_classifier.py` | Baseline-aligned training |

### 5.3 Synthetic Data Generation

| File | Purpose |
|------|---------|
| `ml/per_finger_residual_fit.py` | Interaction model fitting |
| `ml/physics_sim_constrained.py` | Anatomical magpylib simulation |
| `ml/improved_synthetic_generator.py` | NN interpolation |
| `ml/train_improved_hybrid.py` | Calibrated hybrid training |

### 5.4 Analysis

| File | Purpose |
|------|---------|
| `ml/analyze_finger_interactions.py` | Additivity testing |
| `ml/analyze_generalization.py` | Leave-one-out validation |
| `ml/analyze_incremental_improvements.py` | Progress tracking |

### 5.5 Results Files

| File | Content |
|------|---------|
| `ml/per_finger_fit_results.json` | Fitted effects + interaction |
| `ml/physics_synthetic_training_results.json` | Training comparisons |
| `ml/generalization_analysis.json` | Leave-one-out results |
| `ml/improved_hybrid_results.json` | Calibrated hybrid results |

---

## 6. Future Directions

### 6.1 Near-Term (Priority)

1. **Collect more real data**
   - Priority: Multi-finger combos (ffeee, eeeff, eefff already have data)
   - Target: Adjacent finger pairs (fefee, effee, eefef)
   - Impact: Each new combo should improve generalization significantly

2. **Per-finger physics calibration**
   - Current: Global interaction strength (-0.702)
   - Better: Per-pair interaction terms (thumb-index, index-middle, etc.)
   - Impact: Could reduce 39% prediction error

3. **Noise model improvement**
   - Current: Magnitude-proportional Gaussian
   - Better: Per-combo covariance from observed data
   - Impact: Better synthetic distribution matching

### 6.2 Medium-Term

1. **Orientation compensation**
   - Issue: Hand orientation affects magnetic field measurements
   - Approach: Use quaternion data to rotate residuals to common frame
   - Reference: [`apps/gambit/shared/orientation-model.ts`](apps/gambit/shared/orientation-model.ts)

2. **Earth field subtraction**
   - Issue: Earth's magnetic field (~50 μT) adds to measurements
   - Approach: Use IGRF model based on location
   - Reference: [`apps/gambit/shared/geomagnetic-field.ts`](apps/gambit/shared/geomagnetic-field.ts)

3. **Transfer learning across sessions**
   - Issue: Calibration varies between sessions
   - Approach: Few-shot adaptation with baseline recalibration
   - Impact: Better cross-session generalization

### 6.3 Long-Term Research

1. **FFO$$ Template Matching**
   - Alternative to neural networks for embedded inference
   - Reference: [`docs/design/ffo-dollar-research-analysis.md`](docs/design/ffo-dollar-research-analysis.md)

2. **Continuous finger angle estimation**
   - Beyond binary: Estimate 0-1 flexion per finger
   - Requires more granular training data

3. **Smaller magnets / Weaker fields**
   - Goal: Less intrusive finger attachments
   - Challenge: Lower SNR requires better models

---

## 7. Lessons Learned

### 7.1 What Worked

1. **Magnetic residual over raw:** 3 features ≈ 9 features
2. **CNN-LSTM hybrid:** Best architecture for windowed inference
3. **Interaction model:** Captures non-linear finger physics
4. **Calibrated noise:** Use observed combo stats for synthetic

### 7.2 What Didn't Work

1. **Additive superposition:** Physics doesn't work that way
2. **Magnitude-based noise:** Doesn't match real distribution
3. **Too much synthetic:** Dilutes real data signal (300/combo < 150/combo)
4. **Global interaction:** Need per-pair terms for accuracy

### 7.3 Design Principles

1. **Real data is king:** Always prefer real over synthetic when available
2. **Incremental improvement:** Build on existing results, don't rewrite
3. **Validate on real:** Synthetic accuracy means nothing without real validation
4. **Track experiments:** JSON results files enable comparison

---

## 8. References

### 8.1 Internal Documentation

- [`ml/README.md`](ml/README.md) - ML pipeline documentation
- [`ml/CLUSTERING.md`](ml/CLUSTERING.md) - Unsupervised clustering guide
- [`public/models/CLAUDE.md`](public/models/CLAUDE.md) - Model deployment guide
- [`docs/design/magnetic-field-simulation-exploration.md`](docs/design/magnetic-field-simulation-exploration.md) - Physics simulation design

### 8.2 External Libraries

- [TensorFlow/Keras](https://www.tensorflow.org/) - Model training
- [TensorFlow.js](https://www.tensorflow.org/js) - Browser inference
- [Magpylib](https://magpylib.readthedocs.io/) - Magnetic field simulation
- [NumPy/SciPy](https://numpy.org/) - Numerical optimization

### 8.3 Commit History (Recent)

```
7696acd Improve hybrid training with calibrated synthetic (+6.9% accuracy)
aedc02f Add physics-based synthetic training with generalization analysis
7aaf358 Add physics-based magnetic field simulation for synthetic data
5389f66 Add single-sample residual training with non-additive synthetic
5f4009f Add magnetic residual exploration scripts
```

---

## Appendix A: Quick Reference

### A.1 Run Training

```bash
# Real data only (best for observed combos)
python ml/train_improved_hybrid.py

# Full pipeline with deployment
python -m ml.build all --data-dir data/GAMBIT --version v3
```

### A.2 Key Constants

```python
BASELINE = [46.0, -45.8, 31.3]  # μT (open palm)
INTERACTION_STRENGTH = -0.702   # Per-finger suppression
WINDOW_SIZE = 50                # Samples (1 second @ 50Hz)
```

### A.3 Finger Encoding

```python
# Combo string: 'ffeee' = [thumb=flexed, index=flexed, middle=extended, ...]
finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
combo_to_labels = lambda c: [1.0 if x == 'f' else 0.0 for x in c]
```

---

**Document History:**
- 2026-01-02: Created comprehensive design document
- Based on work from Dec 2025 - Jan 2026

**Next Update:** After additional data collection or significant model improvements
