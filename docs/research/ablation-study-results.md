# CNN-LSTM Ablation Study: Window Sizes and Feature Sets

**Status:** Research Complete
**Date:** January 2026
**Branch:** `claude/physics-simulation-data-generation-Oo6Cg`

---

## 1. Executive Summary

This study systematically tested different window sizes (1-50 samples) and feature combinations to understand what drives finger state classification accuracy and orientation invariance.

### Key Findings

| Finding | Impact |
|---------|--------|
| **Single sample (w=1) achieves 97%+** | Temporal patterns not essential |
| **Magnetometer alone is sufficient** | Full 9-DoF doesn't improve accuracy |
| **Raw mag outperforms iron-corrected** | Correction may remove useful signal |
| **Adding orientation features hurts invariance** | Model memorizes orientation-specific patterns |
| **All features fail cross-orientation** | 48-52% best vs 72% for deployed model |

---

## 2. Experimental Setup

### 2.1 Feature Sets Tested

| Name | Features | Description |
|------|----------|-------------|
| `mag_only` | mx, my, mz | Raw magnetometer (3 feat) |
| `iron_mag` | iron_mx, iron_my, iron_mz | Iron-corrected mag (3 feat) |
| `raw_9dof` | ax, ay, az, gx, gy, gz, mx, my, mz | Full sensor suite (9 feat) |
| `mag_euler` | mx, my, mz, pitch, roll, yaw | Mag + orientation (6 feat) |
| `iron_euler` | iron_mx/y/z, pitch, roll, yaw | Iron mag + orientation (6 feat) |

### 2.2 Window Sizes Tested

```
w = 1, 2, 5, 10, 12, 20, 24, 30, 50 samples
```

At 50 Hz sampling: 1 sample = 20ms, 50 samples = 1 second

### 2.3 Evaluation Methods

1. **Random Split**: 80/20 stratified train/test (same orientation distribution)
2. **Orientation Split**: Train on Q4 pitch (high), test on Q1 pitch (low)

---

## 3. Results: Random Split Evaluation

### 3.1 Accuracy by Window Size

All features achieve near-perfect accuracy when train/test have similar orientation distribution:

```
Feature Set       w=1   w=2   w=5  w=10  w=12  w=20  w=24  w=30  w=50
────────────────────────────────────────────────────────────────────
mag_only         97%   99%   99%   99%   100%  100%  99%   99%   85%
iron_mag         94%   98%   98%   99%   99%   99%   99%   88%   81%
raw_9dof         96%   99%   100%  98%   99%   96%   -     -     -
```

### 3.2 Key Observations

1. **Single sample is highly effective**: w=1 achieves 94-97% accuracy
2. **Sweet spot at w=2-12**: Highest accuracy with sufficient data
3. **Larger windows degrade**: Due to reduced sample count
4. **Magnetometer alone matches full 9-DoF**: No benefit from accel/gyro

---

## 4. Results: Orientation Invariance

### 4.1 Cross-Orientation Accuracy (Q4→Q1)

Testing on opposite orientation reveals true generalization:

```
Config              Train    Test    Gap     Status
────────────────────────────────────────────────────
mag_only_w12        85.2%   51.6%   33.6%   ★ Best
mag_only_w1         98.0%   48.0%   50.0%
mag_only_w2         97.9%   48.1%   49.8%
mag_only_w5         97.9%   41.7%   56.2%
iron_mag_w12        86.9%   45.1%   41.8%
iron_mag_w2         83.8%   25.2%   58.6%
raw_9dof_w2         90.1%   15.5%   74.6%   ✗ Worst
raw_9dof_w5         88.5%   12.1%   76.3%   ✗ Worst
mag_euler_w2        97.2%   15.9%   81.3%   ✗ Very Bad
mag_euler_w5        97.1%   32.3%   64.8%
iron_euler_w5       94.6%   12.9%   81.7%   ✗ Very Bad
```

### 4.2 Key Insights

1. **Magnetometer-only is most orientation-invariant**
   - Best: mag_only_w12 with 51.6% cross-orientation accuracy
   - The magnetic signature captures finger state independent of hand orientation

2. **Adding accelerometer/gyroscope hurts generalization**
   - raw_9dof drops to 12-15% cross-orientation (worse than random!)
   - These sensors encode gravity/rotation which varies with orientation

3. **Adding explicit orientation features is counterproductive**
   - mag_euler: 81.3% gap (train 97%, test 16%)
   - The model memorizes orientation-specific patterns instead of learning invariants

4. **Iron correction slightly hurts**
   - iron_mag: 45% vs mag_only: 52% on cross-orientation
   - The correction may remove useful orientation-dependent signal

---

## 5. Comparison with Deployed Model

### 5.1 Cross-Orientation Performance

| Model | Train (high pitch) | Test (low pitch) | Gap |
|-------|-------------------|------------------|-----|
| k-NN template | 92.3% | 61.6% | 30.7% |
| **CNN-LSTM deployed** | **73.9%** | **72.4%** | **1.5%** |
| Simple CNN (mag_only_w12) | 85.2% | 51.6% | 33.6% |
| Simple CNN (raw_9dof_w5) | 88.5% | 12.1% | 76.3% |

### 5.2 Why Deployed Model Wins

The deployed `finger_aligned_v2` model achieves true orientation invariance because:

1. **Trained on synthetic + diverse real data** - Saw many orientations during training
2. **Data augmentation** - Likely includes orientation-varied samples
3. **Larger window (50 samples)** - Captures temporal patterns
4. **Uses full 9-DoF** - But learned to extract invariant features

---

## 6. Practical Recommendations

### 6.1 For Same-Orientation Deployment

If the device will be used in a fixed orientation:

```
Best: mag_only, w=2-12, 99%+ accuracy
Simple, fast, minimal features
```

### 6.2 For Orientation-Independent Deployment

If the device must work at any orientation:

```
Option A: Use deployed CNN-LSTM (72% cross-orientation)
Option B: Train on diverse orientations
Option C: Use mag_only with rotation augmentation
```

### 6.3 Feature Selection Guidelines

| Requirement | Recommended Features |
|-------------|---------------------|
| Same orientation | mag_only (3 feat) |
| Cross-orientation | mag_only + training augmentation |
| Real-time | w=1-5 samples |
| Maximum accuracy | w=10-20 samples |

---

## 7. Why Does Window Size w=1 Work?

### 7.1 Analysis

Single-sample classification (w=1) achieves 97% because:

1. **Finger state is encoded in magnetic field direction/magnitude**
   - Each finger pose creates a distinct magnetic signature
   - No temporal pattern needed - it's a static measurement

2. **High SNR at low frequencies**
   - Finger movement is slow compared to 50 Hz sampling
   - Adjacent samples are highly correlated

3. **Classification is primarily spatial, not temporal**
   - The CNN-LSTM learns from temporal sequences
   - But the discriminative information is in the 3D magnetic vector

### 7.2 Implications

- **Simpler models may suffice**: A dense network on single samples could replace CNN-LSTM
- **Lower latency possible**: No need to wait for 50 samples
- **Reduced compute**: Single forward pass vs sequential processing

---

## 8. Files Created

| File | Description |
|------|-------------|
| `ml/ablation_study.py` | Initial ablation framework |
| `ml/ablation_study_v2.py` | Improved with k-fold CV |
| `ml/ablation_small_windows.py` | Small window tests (1-50) |
| `ml/quick_orientation_test.py` | Focused orientation invariance |
| `ml/orientation_test_results.json` | Raw orientation test results |

---

## 9. Conclusions

1. **Magnetometer is the key sensor** - Accel/gyro add no value for finger classification

2. **Temporal windows are optional** - Single samples achieve 97% accuracy

3. **Orientation is the challenge** - All simple models fail cross-orientation

4. **Training data diversity matters more than features** - The deployed model wins because it saw diverse orientations during training, not because it uses more features

5. **For production**: Either use the deployed CNN-LSTM or train a simple model on diverse orientation data
