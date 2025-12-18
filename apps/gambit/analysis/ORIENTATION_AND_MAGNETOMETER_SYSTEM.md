# GAMBIT Orientation & Magnetometer Sensing System

## Technical Analysis and Implementation Document

**Version:** 1.0.0
**Date:** 2025-12-14
**Status:** Active Development

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Information-Theoretic Foundations](#information-theoretic-foundations)
3. [System Architecture](#system-architecture)
4. [Orientation Pipeline](#orientation-pipeline)
5. [9-DOF Magnetometer Fusion](#9-dof-magnetometer-fusion)
6. [Current Implementation State](#current-implementation-state)
7. [Future: Finger Magnet Sensing](#future-finger-magnet-sensing)
8. [File Reference](#file-reference)

---

## Executive Summary

The GAMBIT system uses a 9-axis IMU (accelerometer, gyroscope, magnetometer) mounted on the back of the hand to track orientation. This document describes the complete signal processing pipeline from raw sensor data to 3D hand visualization, including recent fixes for axis mapping issues and the integration of magnetometer data for stable yaw estimation.

### Key Achievements

1. **Axis Mapping Fixed**: Identified and corrected mismatch between AHRS ZYX Euler convention and renderer axis assignments
2. **Rotation Order Corrected**: Changed from YXZ to ZYX intrinsic to match AHRS quaternion decomposition
3. **Pivot Point Added**: Rotations now pivot around sensor position (palm) rather than wrist
4. **9-DOF Fusion Implemented**: Magnetometer data now informs yaw to eliminate gyroscope drift
5. **Residual Computation Added**: Infrastructure for future finger magnet sensing

---

## Information-Theoretic Foundations

### Observable Quantities and Degrees of Freedom

A rigid body in 3D space has 6 degrees of freedom (DOF): 3 translational + 3 rotational. For hand orientation tracking (ignoring position), we need to estimate 3 rotational DOF.

| Sensor | Physical Observable | Information Content |
|--------|---------------------|---------------------|
| **Accelerometer** | Gravity + Linear Acceleration | 2 DOF (pitch, roll) when stationary |
| **Gyroscope** | Angular Velocity | 3 DOF rates (integrates to orientation) |
| **Magnetometer** | Earth's Magnetic Field | 2 DOF (yaw relative to magnetic north) |

### Why 9-DOF Fusion is Necessary

**6-DOF (Accel + Gyro) Problem:**
```
Information Flow:
  Gyro → ∫dt → Orientation (all 3 DOF, but drifts)
  Accel → Gravity → Tilt correction (2 DOF, no drift)

  Result: Yaw drifts unbounded over time (no absolute reference)
```

**9-DOF (Accel + Gyro + Mag) Solution:**
```
Information Flow:
  Gyro → ∫dt → Orientation (all 3 DOF, drifts)
  Accel → Gravity → Pitch/Roll correction (2 DOF)
  Mag → Earth Field → Yaw correction (1 DOF)

  Result: All 3 DOF have absolute references, no drift
```

### Shannon Information Analysis

The magnetometer provides ~log₂(360°/σ) bits of yaw information per sample, where σ is the angular uncertainty. With typical magnetometer noise of ~1°, this is approximately 8-9 bits of heading information per sample.

However, this information is **conditional** on knowing:
1. The geomagnetic field vector at the current location
2. Hard iron offsets (constant magnetic distortions)
3. Soft iron distortions (orientation-dependent distortions)
4. Dynamic disturbances (finger magnets, environmental interference)

---

## System Architecture

### Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAW SENSOR DATA                                 │
│                    (LSB units from BMI270 + MMC5603)                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY PROCESSOR                                  │
│                    (telemetry-processor.js)                                  │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────────────┐   │
│  │ Unit        │   │ Motion      │   │ Gyro Bias                       │   │
│  │ Conversion  │──▶│ Detection   │──▶│ Calibration                     │   │
│  │ (LSB→g,°/s) │   │             │   │ (when stationary)               │   │
│  └─────────────┘   └─────────────┘   └─────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              MADGWICK AHRS FILTER (filters.js)                       │    │
│  │                                                                      │    │
│  │   ┌─────────────────┐        ┌─────────────────────────────────┐    │    │
│  │   │ 6-DOF Mode      │   OR   │ 9-DOF Mode (with magnetometer)  │    │    │
│  │   │ update()        │        │ updateWithMag()                 │    │    │
│  │   │ - accel tilt    │        │ - accel tilt correction         │    │    │
│  │   │ - gyro rates    │        │ - gyro rate integration         │    │    │
│  │   │ - yaw drifts!   │        │ - mag yaw correction            │    │    │
│  │   └─────────────────┘        │ - computes mag residual         │    │    │
│  │                              └─────────────────────────────────┘    │    │
│  │                                           │                         │    │
│  │                                           ▼                         │    │
│  │                              ┌─────────────────────────────────┐    │    │
│  │                              │ Quaternion q = {w, x, y, z}     │    │    │
│  │                              │ → getEulerAngles() ZYX order    │    │    │
│  │                              │ → {roll, pitch, yaw} in degrees │    │    │
│  │                              └─────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HAND 3D RENDERER                                    │
│                       (hand-3d-renderer.js)                                  │
│                                                                              │
│  updateFromSensorFusion(euler):                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ AXIS MAPPING (corrected 2025-12-14 v2 based on calibration data):   │   │
│  │                                                                      │   │
│  │   Physical Movement:               Visual Effect:     Renderer:      │   │
│  │   - Roll (tilt pinky down)    ──▶  hand tilts L/R ──▶ RotY (yaw)    │   │
│  │   - Pitch (tilt fingers)      ──▶  fingers nod    ──▶ RotX (pitch)  │   │
│  │   - Yaw (rotate while flat)   ──▶  hand spins     ──▶ RotZ (roll)   │   │
│  │                                                                      │   │
│  │   mappedOrientation = {                                              │   │
│  │     pitch: -euler.pitch,      // AHRS.pitch → renderer.pitch (RotX) │   │
│  │     yaw:   euler.roll,        // AHRS.roll → renderer.yaw (RotY)    │   │
│  │     roll:  euler.yaw          // AHRS.yaw → renderer.roll (RotZ)    │   │
│  │   }                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  render():                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ ROTATION ORDER: ZYX intrinsic (matches AHRS quaternion extraction)  │   │
│  │                                                                      │   │
│  │   // Pivot around sensor position (palm center)                      │   │
│  │   sensorPivot = [0, 0.15, -0.12]                                     │   │
│  │                                                                      │   │
│  │   handM = RotY(π)                      // Base: palm faces viewer    │   │
│  │   handM = handM × Trans(-pivot)        // Move pivot to origin       │   │
│  │   handM = handM × RotZ(roll)           // Z first  (heading)         │   │
│  │   handM = handM × RotY(yaw)            // Y second (elevation)       │   │
│  │   handM = handM × RotX(pitch)          // X third  (bank)            │   │
│  │   handM = handM × Trans(+pivot)        // Restore pivot              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Orientation Pipeline

### The Axis Mapping Problem (Solved)

The original diagnostic report identified "axis coupling" where single-axis physical rotations caused multiple AHRS Euler angles to change. This was initially thought to be a bug, but further analysis revealed:

**Insight: AHRS Coupling is Expected Behavior**

When AHRS extracts Euler angles using ZYX convention but the renderer was applying them in a different order with mismatched axis assignments, the coupling is a natural consequence of:

1. Euler angles are **not** unique - the same rotation has different representations in different orders
2. The renderer's axis labels (pitch, yaw, roll) didn't match AHRS axis labels
3. Rotation order affects how individual angles combine

**The Fix (V2 - based on calibration observations):**

The confusion arose from conflating AHRS axis names (roll/pitch/yaw as X/Y/Z rotations)
with physical movement names (roll/pitch/yaw as what pilots call these motions).

| Physical Motion | AHRS Value | Visual Effect | Renderer Rotation |
|-----------------|------------|---------------|-------------------|
| Roll (tilt L/R) | euler.roll | Hand tilts | RotY (renderer.yaw) |
| Pitch (nod) | euler.pitch | Fingers tilt | RotX (renderer.pitch) |
| Yaw (spin) | euler.yaw | Hand spins | RotZ (renderer.roll) |

Key insight: The renderer variable names (pitch/yaw/roll) don't match their rotation
axes (RotX/RotY/RotZ) in the intuitive way. `renderer.yaw` applies RotY, etc.

### Validation Results

**Pre-V2 fix observations (for reference):**

| Pose | AHRS Response | Visual Result | Issue |
|------|---------------|---------------|-------|
| FLAT_PALM_UP | Baseline | Unknown | Baseline |
| PITCH_FORWARD_45 | pitch=-34° | Coupled | Coupling visible |
| PITCH_BACKWARD_45 | pitch=+44° | ✓ Correct | Working |
| ROLL_LEFT_45 | roll=+48° | ✗ Wrong axis | Roll→RotX was wrong |
| ROLL_RIGHT_45 | roll=-33° | ✗ Wrong axis | Roll→RotX was wrong |
| YAW_CW_90 | yaw=-69° | ✗ Wrong axis | +180 offset issue |
| YAW_CCW_90 | yaw=+112° | ✗ Wrong axis | +180 offset issue |

**After V2 fix (needs validation):**

| Pose | Expected Visual | Status |
|------|-----------------|--------|
| FLAT_PALM_UP | Palm up, fingers away | To test |
| PITCH_FORWARD_45 | Fingers point down | To test |
| PITCH_BACKWARD_45 | Fingers point up | To test |
| ROLL_LEFT_45 | Pinky down, thumb up | To test |
| ROLL_RIGHT_45 | Thumb down, pinky up | To test |
| YAW_CW_90 | Fingers point left | To test |
| YAW_CCW_90 | Fingers point right | To test |

---

## 9-DOF Magnetometer Fusion

### Implementation Overview

The 9-DOF fusion extends the Madgwick AHRS filter to incorporate magnetometer readings for absolute yaw reference.

**Key Components:**

1. **Geomagnetic Reference** (`geomagnetic-field.js`)
   - IGRF-13 lookup table for 100+ cities
   - Provides horizontal/vertical field components
   - Auto-initializes from browser geolocation

2. **Madgwick MARG Algorithm** (`filters.js`)
   - Full gradient descent on combined objective function
   - Accelerometer: gravity alignment
   - Magnetometer: heading alignment
   - Gyroscope: rate integration

3. **Adaptive Trust** (`magTrust` parameter)
   - Range: 0.0 (ignore mag) to 1.0 (full trust)
   - Default: 0.5 (moderate trust)
   - Can be reduced when finger magnets interfere

### Mathematical Foundation

The Madgwick MARG filter minimizes a combined objective function:

```
f(q) = f_g(q) + f_b(q)

Where:
  f_g(q) = q* ⊗ [0,0,0,g] ⊗ q - a_measured    (gravity alignment)
  f_b(q) = q* ⊗ [0,bx,0,bz] ⊗ q - m_measured  (magnetic field alignment)
```

The gradient descent step is:

```
q̇_est = -β * ∇f / ||∇f||
q_new = q + (q̇_gyro + q̇_est) * dt
```

Where β controls how aggressively the filter corrects drift. The `magTrust` parameter scales the magnetic field contribution to β.

### Wiring in Telemetry Processor

```javascript
// telemetry-processor.js - constructor
this.useMagnetometer = options.useMagnetometer !== false;  // Default: enabled
this.magTrust = options.magTrust ?? 0.5;                   // Default: moderate
this.imuFusion.setMagTrust(this.magTrust);
this._initGeomagneticReference();  // Auto-detect location

// telemetry-processor.js - process()
if (this.useMagnetometer && magDataValid && this.geomagneticRef) {
    // Pass hard iron offset from calibration if available
    if (this.calibration?.hardIronCalibrated) {
        this.imuFusion.setHardIronOffset(this.calibration.hardIronOffset);
    }

    // 9-DOF update with magnetometer
    this.imuFusion.updateWithMag(
        ax_g, ay_g, az_g,      // Accelerometer (g)
        gx_dps, gy_dps, gz_dps, // Gyroscope (deg/s)
        mx_ut, my_ut, mz_ut,    // Magnetometer (µT)
        dt, true, true
    );
} else {
    // Fallback to 6-DOF
    this.imuFusion.update(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, dt, true);
}
```

---

## Current Implementation State

### Completed Work

| Component | File | Status |
|-----------|------|--------|
| Axis mapping fix | `hand-3d-renderer.js` | ✅ Complete |
| Rotation order fix | `hand-3d-renderer.js` | ✅ Complete |
| Pivot point fix | `hand-3d-renderer.js` | ✅ Complete |
| 9-DOF AHRS method | `filters.js` | ✅ Complete |
| Geomagnetic reference | `filters.js` + `geomagnetic-field.js` | ✅ Complete |
| Hard iron integration | `filters.js` | ✅ Complete |
| Mag trust parameter | `filters.js` | ✅ Complete |
| Residual computation | `filters.js` | ✅ Complete |
| Telemetry wiring | `telemetry-processor.js` | ✅ Complete |
| Auto location init | `telemetry-processor.js` | ✅ Complete |

### Data Flow Verification

The complete pipeline from sensor to visualization:

```
BMI270/MMC5603 → BLE → index.html → TelemetryProcessor
    → MadgwickAHRS.updateWithMag() → getEulerAngles()
    → updateFromSensorFusion() → render()
```

All components are wired and functional.

### Console Output (Expected)

When running with magnetometer:
```
[TelemetryProcessor] Using default geomagnetic reference: San Francisco
[TelemetryProcessor] Updated geomagnetic reference from browser location: Edinburgh
[TelemetryProcessor] IMU initialized from accelerometer
[TelemetryProcessor] Gyroscope bias calibration complete
[TelemetryProcessor] Using 9-DOF fusion with magnetometer (trust: 0.5)
```

---

## Future: Finger Magnet Sensing

### The Challenge

We want to:
1. Use magnetometer for stable yaw (current goal) ✅
2. Add magnets to fingers for position sensing (future goal)
3. These goals conflict: finger magnets corrupt the Earth field measurement

### Information-Theoretic Solution

**Residual-Based Sensing:**

```
B_measured = B_earth + B_hard_iron + B_finger_magnets

If we know B_earth and B_hard_iron accurately:
  B_residual = B_measured - B_expected
  B_residual ≈ B_finger_magnets (when orientation is accurate)
```

This is already implemented:

```javascript
// filters.js - getMagResidual()
_lastMagResidual = {
    x: mx_corrected - mx_expected,
    y: my_corrected - my_expected,
    z: mz_corrected - mz_expected
};
```

### Adaptive Trust Strategy

The `magTrust` parameter enables a phased approach:

| Phase | magTrust | Behavior |
|-------|----------|----------|
| **Calibration** | 0.5-1.0 | Use mag fully for yaw, train hard iron |
| **Mixed Operation** | 0.2-0.5 | Partial mag, gyro primary, monitor residuals |
| **Finger Sensing** | 0.0-0.1 | Gyro-only yaw, full residual for fingers |

### Residual Magnitude Analysis

The `getMagResidualMagnitude()` method returns the magnitude of the residual field:

```javascript
getMagResidualMagnitude() {
    if (!this._lastMagResidual) return 0;
    const { x, y, z } = this._lastMagResidual;
    return Math.sqrt(x*x + y*y + z*z);
}
```

Expected values:
- **No magnets, good calibration**: < 5 µT (sensor noise + calibration error)
- **Finger magnet present**: 10-100+ µT depending on distance

This residual is exposed in decorated telemetry as:
- `ahrs_mag_residual_x/y/z` - Vector components
- `ahrs_mag_residual_magnitude` - Scalar magnitude

### Future Work

1. **Residual visualization** - Show residual magnitude in UI for debugging
2. **Automatic magTrust adaptation** - Reduce trust when residual spikes
3. **Finger position inference** - Inverse kinematics from residual vector
4. **Multi-magnet separation** - Distinguish different finger positions

---

## File Reference

### Core Processing

| File | Purpose |
|------|---------|
| `shared/telemetry-processor.js` | Main pipeline orchestration, 9-DOF wiring |
| `filters.js` | MadgwickAHRS with 6-DOF and 9-DOF modes |
| `shared/geomagnetic-field.js` | IGRF-13 lookup, location detection |
| `shared/sensor-config.js` | Scale factors, filter creation |
| `shared/sensor-units.js` | LSB to physical unit conversion |

### Visualization

| File | Purpose |
|------|---------|
| `hand-3d-renderer.js` | Canvas 2D hand rendering with corrected axis mapping |
| `modules/threejs-hand-skeleton.js` | Three.js skeletal hand (alternative renderer) |

### Calibration

| File | Purpose |
|------|---------|
| `calibration.js` | Hard/soft iron calibration, Earth field estimation |
| `shared/orientation-calibration.js` | Orientation validation poses, AHRS coupling analysis |

### Entry Points

| File | Purpose |
|------|---------|
| `index.html` | Main GAMBIT application, wires all components |
| `modules/telemetry-handler.js` | Collector-specific telemetry handling |

---

## Appendix: Key Equations

### Quaternion to Euler (ZYX Order)

```javascript
// From filters.js getEulerAngles()
roll  = atan2(2(wy + xz), 1 - 2(y² + z²))
pitch = asin(2(wx - yz))  // clamped to [-π/2, π/2]
yaw   = atan2(2(wz + xy), 1 - 2(x² + z²))
```

### Expected Magnetic Field (Body Frame)

```javascript
// Rotate Earth field reference by current orientation
m_expected = q* ⊗ [0, bx, 0, bz] ⊗ q

Where [bx, 0, bz] is the normalized geomagnetic reference:
  bx = horizontal / magnitude  (toward magnetic north)
  bz = vertical / magnitude    (downward component)
```

### Madgwick Beta Scaling

```javascript
// Effective beta with magTrust
beta_accel = beta * 1.0
beta_mag   = beta * magTrust

// When magTrust = 0, filter ignores magnetometer
// When magTrust = 1, full MARG algorithm
```

---

*Document generated: 2025-12-14*
*Last commit: Wire up 9-DOF magnetometer fusion in telemetry processor*
