# GAMBIT Calibration Deep Dive

## Overview

This document explains exactly where, when, and how calibration data is used throughout the GAMBIT pipeline, from real-time data collection through ML training and inference.

---

## Calibration Types

### 1. Hard Iron Calibration
**Purpose:** Remove constant magnetic offset from nearby ferromagnetic materials (rings, watches, device housing)

**Math:** `B_corrected = B_raw - offset`

**Implementation:** `calibration.js:161-210`
```javascript
// Find center of ellipsoid from min/max readings
hardIronOffset = {
    x: (maxX + minX) / 2,
    y: (maxY + minY) / 2,
    z: (maxZ + minZ) / 2
};
```

**Quality Metric:** Sphericity (0-1, higher is better)
- Excellent: > 0.9
- Good: > 0.7
- Poor: < 0.5

### 2. Soft Iron Calibration
**Purpose:** Correct field distortion from conductive materials that warp the magnetic field shape

**Math:** `B_corrected = M × (B_raw - hard_iron_offset)` where M is a 3×3 correction matrix

**Implementation:** `calibration.js:219-255`
```javascript
// Eigendecomposition to find principal axes scaling
softIronMatrix = [
    [avgScale / scaleX, 0, 0],
    [0, avgScale / scaleY, 0],
    [0, 0, avgScale / scaleZ]
];
```

**Quality Metric:** Eigenvalue ratio (min/max eigenvalue)
- Excellent: > 0.8
- Acceptable: > 0.5
- High distortion: < 0.3

### 3. Earth Field Calibration
**Purpose:** Capture the ambient Earth magnetic field vector in reference orientation

**Math:** `B_fingers = B_corrected - R(orientation) × B_earth`

**Implementation:** `calibration.js:259-302`
```javascript
// Average readings in reference orientation
earthField = {
    x: mean(samples.x),
    y: mean(samples.y),
    z: mean(samples.z)
};
```

**Quality Metric:** Standard deviation of samples
- Excellent: std < 1.0 μT
- Good: std < 3.0 μT
- Poor: std > 5.0 μT

---

## Data Flow: Where Calibration Is Applied

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐                                                          │
│  │ Magnetometer  │  Raw: {mx, my, mz} in device frame                      │
│  │ (LIS3MDL)     │  Range: ±400 μT @ 0.146 μT resolution                   │
│  └───────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: Hard Iron Correction                                          │ │
│  │ ═══════════════════════════                                           │ │
│  │                                                                       │ │
│  │ B₁ = B_raw - hardIronOffset                                          │ │
│  │                                                                       │ │
│  │ Removes constant bias from nearby ferromagnetic materials             │ │
│  │ (rings, watch, device housing)                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: Soft Iron Correction                                          │ │
│  │ ══════════════════════════                                            │ │
│  │                                                                       │ │
│  │ B₂ = softIronMatrix × B₁                                             │ │
│  │                                                                       │ │
│  │ Corrects ellipsoidal distortion to spherical                          │ │
│  │ Removes warping from conductive materials                             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: Earth Field Subtraction (REQUIRES ORIENTATION)                │ │
│  │ ══════════════════════════════════════════════════════                │ │
│  │                                                                       │ │
│  │ R = quaternion.toRotationMatrix()   // Device orientation             │ │
│  │ B_earth_rotated = R × earthField    // Earth field in device frame   │ │
│  │ B₃ = B₂ - B_earth_rotated           // Subtract Earth's contribution │ │
│  │                                                                       │ │
│  │ Result: Only finger magnet signals remain!                            │ │
│  │                                                                       │ │
│  │ ⚠️  REQUIRES: IMU sensor fusion to estimate device orientation       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 4: Kalman Filtering (Optional)                                   │ │
│  │ ═════════════════════════════════                                     │ │
│  │                                                                       │ │
│  │ B_filtered = KalmanFilter3D.update(B₃)                               │ │
│  │                                                                       │ │
│  │ Reduces noise, estimates velocity, smooths signal                     │ │
│  │ Process noise: 0.1, Measurement noise: 1.0                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          ▼                                                                  │
│  OUTPUT: Decorated telemetry with multiple field representations           │
│  {                                                                          │
│    mx, my, mz,                    // RAW (always preserved)                │
│    calibrated_mx, my, mz,         // Iron-corrected only                   │
│    fused_mx, my, mz,              // + Earth field subtracted (NEW)        │
│    filtered_mx, my, mz            // + Kalman smoothed                     │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## IMU Sensor Fusion for Orientation

### The Problem

Earth's magnetic field (~50 μT) dominates the magnetometer signal. As the device orientation changes, the Earth field projection onto sensor axes changes, creating apparent motion even when the hand is stationary.

**Without orientation compensation:**
- Rotating palm creates ~100 μT apparent signal change
- Finger magnets at 80mm create only ~15-35 μT
- Earth field noise drowns out finger signals

**With orientation compensation:**
- Track device orientation using accelerometer + gyroscope
- Project known Earth field into current device frame
- Subtract to isolate finger magnet contributions

### Sensor Fusion Approaches

#### 1. Complementary Filter (Simple, Fast)
```javascript
// High-pass gyro + low-pass accelerometer
orientation += gyro * dt * alpha + accel_orientation * (1 - alpha);
```
- Pro: Simple, low latency
- Con: Drift over time, not optimal

#### 2. Madgwick Filter (Recommended)
```javascript
// Gradient descent optimization of quaternion orientation
// Fuses accelerometer + gyroscope, optionally magnetometer
q = madgwick.update(ax, ay, az, gx, gy, gz, dt);
```
- Pro: No drift, handles gimbal lock, well-tested
- Con: More computation, tuning required

#### 3. Extended Kalman Filter (Best Accuracy)
```javascript
// Full state estimation with covariance
[q, bias] = ekf.update(accel, gyro, dt);
```
- Pro: Optimal estimation, bias tracking
- Con: Complex, highest computation

### Why We DON'T Use Magnetometer for Orientation

Normally, magnetometers help correct gyro drift in AHRS (Attitude Heading Reference Systems). However, for GAMBIT:

1. **Finger magnets corrupt the magnetometer** - the very signals we want to measure
2. **We use magnetometer OUTPUT, not INPUT** - it's a measurement target, not a reference
3. **Accelerometer + gyroscope are sufficient** for short-term orientation

---

## Implementation Status

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Hard Iron Calibration | `calibration.js:161-210` | ✅ Complete | Working in manual wizard |
| Soft Iron Calibration | `calibration.js:219-255` | ✅ Complete | Working in manual wizard |
| Earth Field Capture | `calibration.js:259-302` | ✅ Complete | Captures reference field |
| Quaternion Math | `calibration.js:78-120` | ✅ Complete | Rotation matrices working |
| Earth Field Subtraction | `calibration.js:322-333` | ⚠️ Exists but unused | Needs orientation input |
| IMU Sensor Fusion | `filters.js` | ❌ Missing | **NEEDS IMPLEMENTATION** |
| Orientation Integration | `collector.html` | ❌ Missing | Not passing orientation to correct() |
| Kalman Filtering | `filters.js` | ✅ Complete | For position, not orientation |

---

## Calibration in Session Data

### Current Decorated Fields
```json
{
  "mx": 45.2,                    // Raw magnetometer X
  "my": -12.3,                   // Raw magnetometer Y
  "mz": 88.1,                    // Raw magnetometer Z
  "calibrated_mx": 42.1,         // After iron correction
  "calibrated_my": -10.5,
  "calibrated_mz": 85.3,
  "filtered_mx": 42.3,           // After Kalman filter
  "filtered_my": -10.4,
  "filtered_mz": 85.5
}
```

### Proposed Additional Fields (After Sensor Fusion)
```json
{
  "fused_mx": 12.5,              // After Earth field subtraction
  "fused_my": -8.2,              // (requires orientation)
  "fused_mz": 5.1,
  "orientation_w": 0.707,        // Device orientation quaternion
  "orientation_x": 0.0,
  "orientation_y": 0.707,
  "orientation_z": 0.0
}
```

---

## Visualization Requirements

### Raw vs Corrected Comparison

Session visualizations should show multiple traces:

1. **Raw** (mx, my, mz) - Original sensor readings
2. **Iron Corrected** (calibrated_*) - After hard/soft iron
3. **Fused** (fused_*) - After Earth field subtraction
4. **Filtered** (filtered_*) - After Kalman smoothing

### Calibration Quality Overlay

Show calibration state/quality on recordings:
- Green: Full calibration applied
- Yellow: Partial calibration (iron only)
- Red: No calibration

### Orientation Track

Display device orientation during recording:
- 3D cube visualization
- Euler angle traces (roll, pitch, yaw)
- Highlight when orientation changes significantly

---

## ML Pipeline Integration

### Training Feature Selection

The data loader (`ml/data_loader.py`) selects features in priority order:

1. `fused_*` (best: Earth field removed)
2. `filtered_*` (good: smoothed)
3. `calibrated_*` (acceptable: iron corrected)
4. Raw `mx/my/mz` (baseline)

### Calibration File Discovery

```python
# Search order in data_loader.py
1. {data_dir}/gambit_calibration.json   # Per-dataset
2. ~/.gambit/calibration.json           # User global
3. None (use pre-decorated or raw)      # Fallback
```

### Training vs Inference Consistency

**Critical:** Training and inference must use the same calibration:

```javascript
// Inference must apply same corrections as training
if (modelTrainedWithFusion) {
    const orientation = sensorFusion.getOrientation();
    const corrected = calibration.correct(raw, orientation);
    const input = kalman.update(corrected);
}
```

---

## Visualization of Calibration Stages

### Available Data Fields

Session data now includes multiple representations of the magnetometer signal:

| Field | Description | When Available |
|-------|-------------|----------------|
| `mx, my, mz` | Raw magnetometer readings | Always |
| `calibrated_mx/my/mz` | After hard + soft iron correction | If calibrated |
| `fused_mx/my/mz` | After Earth field subtraction | If calibrated + orientation |
| `filtered_mx/my/mz` | After Kalman smoothing | Always |
| `orientation_w/x/y/z` | Device orientation quaternion | If IMU fusion enabled |
| `euler_roll/pitch/yaw` | Device orientation in degrees | If IMU fusion enabled |

### Recommended Visualization Approach

**1. Multi-trace Time Series Plot**

Show all signal stages overlaid to compare:
- Raw (gray, dashed)
- Iron corrected (blue)
- Fused (green)
- Filtered (red, bold)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, axis in enumerate(['x', 'y', 'z']):
    ax = axes[i]
    ax.plot(times, raw[axis], 'gray', alpha=0.5, label='Raw')
    ax.plot(times, calibrated[axis], 'b-', alpha=0.7, label='Iron corrected')
    ax.plot(times, fused[axis], 'g-', alpha=0.8, label='Fused (Earth subtracted)')
    ax.plot(times, filtered[axis], 'r-', linewidth=1.5, label='Filtered')
    ax.set_ylabel(f'M{axis.upper()} (μT)')
    ax.legend(loc='upper right')

plt.xlabel('Time (s)')
plt.suptitle('Magnetometer Signal Processing Stages')
```

**2. Calibration Status Indicator**

Color-code recording sections by calibration state:
- Green: Full calibration (fused available)
- Yellow: Partial (iron only)
- Red: No calibration

**3. Orientation Visualization**

Show device orientation during recording:
- 3D cube representation
- Euler angle traces (roll, pitch, yaw)
- Highlight orientation changes that affect Earth field projection

### Signal Quality Metrics

For each processing stage, compute and display:

| Metric | Formula | Target |
|--------|---------|--------|
| Noise floor | std(signal during rest) | < 1 μT |
| SNR | mean(signal) / std(signal) | > 10 dB |
| Drift | max(cumsum(signal - mean)) | < 5 μT |
| Earth field residual | magnitude of fused during rest | < 2 μT |

---

## References

- `src/web/GAMBIT/calibration.js` - Calibration implementation
- `src/web/GAMBIT/filters.js` - Kalman filters + IMU fusion
- `docs/calibration-filtering-guide.md` - User guide
- `docs/design/magnetic-finger-tracking-analysis.md` - Physics analysis

---

*Document created: 2024-12-10*
*SIMCAP Project - GAMBIT Calibration Deep Dive*

---

<link rel="stylesheet" href="../src/simcap.css">
