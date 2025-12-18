# GAMBIT Telemetry Data Flow Analysis

## Overview

I've traced the complete data flow from raw sensor input through to the visualizations. Here's a comprehensive breakdown:

---

## 1. Raw Telemetry Input (from GAMBIT device)

The device sends these **raw fields** (in LSB - Least Significant Bits):

| Field | Description | Unit |
|-------|-------------|------|
| `ax`, `ay`, `az` | Accelerometer | LSB |
| `gx`, `gy`, `gz` | Gyroscope | LSB |
| `mx`, `my`, `mz` | Magnetometer | LSB |
| `t` | Timestamp | ms |
| `b` | Battery level | % |
| `s` | Button state | - |
| `n` | Notification count | - |

---

## 2. TelemetryProcessor Pipeline (6 stages)

The `TelemetryProcessor.process()` method decorates raw data with computed fields:

### Stage 1: Unit Conversion
| Computed Field | Source | Formula |
|----------------|--------|---------|
| `ax_g`, `ay_g`, `az_g` | `ax`, `ay`, `az` | `raw / 8192` (g's) |
| `gx_dps`, `gy_dps`, `gz_dps` | `gx`, `gy`, `gz` | `raw / 114.28` (°/s) |
| `mx_ut`, `my_ut`, `mz_ut` | `mx`, `my`, `mz` | `magLsbToMicroTesla()` (µT) |

### Stage 2: Motion Detection
| Computed Field | Description |
|----------------|-------------|
| `isMoving` | Boolean - device in motion |
| `accelStd` | Accelerometer standard deviation |
| `gyroStd` | Gyroscope standard deviation |

### Stage 3: Gyroscope Bias Calibration
| Computed Field | Description |
|----------------|-------------|
| `gyroBiasCalibrated` | Boolean - bias calibration complete |

### Stage 4: IMU Sensor Fusion (Madgwick AHRS)
| Computed Field | Source | Description |
|----------------|--------|-------------|
| `orientation_w/x/y/z` | Accel + Gyro + Mag | Quaternion orientation |
| `euler_roll`, `euler_pitch`, `euler_yaw` | Quaternion | Euler angles (°) |
| `ahrs_mag_residual_x/y/z` | AHRS internal | Magnetometer residual from fusion |

### Stage 5: Magnetometer Calibration (UnifiedMagCalibration)
| Computed Field | Source | Description |
|----------------|--------|-------------|
| `iron_mx/my/mz` | `mx_ut/my_ut/mz_ut` | Hard/soft iron corrected |
| `residual_mx/my/mz` | Iron corrected + orientation | Earth field subtracted |
| `residual_magnitude` | Residual vector | `√(rx² + ry² + rz²)` |
| `mag_cal_ready` | Calibration state | Boolean |
| `mag_cal_confidence` | Calibration state | 0-1 |
| `magnet_status` | MagnetDetector | 'none'/'possible'/'likely'/'confirmed' |
| `magnet_confidence` | MagnetDetector | 0-1 |

### Stage 6: Kalman Filtering
| Computed Field | Source | Description |
|----------------|--------|-------------|
| `filtered_mx/my/mz` | `residual_mx/my/mz` (or `mx_ut`) | Noise-reduced magnetic field |

---

## 3. Visualization Field Usage

### A. **3D Cube Visualizations** (in `gambit-app.ts`)

There are **3 cubes** - each uses different fields:

#### Accelerometer Cube (`cubeA`)
```
Uses: prenorm.ax, prenorm.ay, prenorm.az (RAW LSB values)
Computes: accRoll = atan2(ay, az), accPitch = atan2(-ax, √(ay² + az²))
Filtered: LowPassFilter (alpha=0.4)
```

#### Gyroscope Cube (`cubeG`) - Actually shows FUSED orientation
```
Uses: euler.roll, euler.pitch, euler.yaw (COMPUTED from TelemetryProcessor)
Source: telemetryProcessor.getEulerAngles()
Filtered: LowPassFilter (alpha=0.3)
```

#### Magnetometer Cube (`cubeM`)
```
Uses: prenorm.mx, prenorm.my, prenorm.mz (RAW LSB values)
Computes: magAzimuth = atan2(my, mx), magElevation = asin(mz/magnitude)
Filtered: LowPassFilter (alpha=0.3)
```

### B. **Three.js Hand Skeleton** (`ThreeJSHandSkeleton`)

```
Uses: euler.roll, euler.pitch, euler.yaw (COMPUTED)
Source: Either from TelemetryProcessor OR from playback (prenorm.euler_roll/pitch/yaw)
```

The hand skeleton receives Euler angles and applies:
- Orientation offsets (configurable: roll=180°, pitch=180°, yaw=-180° default)
- Axis sign negation (configurable)
- Lerp smoothing (factor=0.15)

### C. **Magnetic Trajectory** (`MagneticTrajectory`)

```
Uses: decoratedData.residual_mx, residual_my, residual_mz (COMPUTED - Earth field subtracted)
```

This is the **fused/calibrated** magnetic field with Earth's field removed, showing only the finger magnet signal.

---

## 4. Data Stage Dropdown Analysis

The dropdown (`dataStageSelect`) controls **ONLY the sensor data table display**, NOT the visualizations.

| Stage | Table Shows | Visualizations Use |
|-------|-------------|-------------------|
| `raw` | `prenorm.mx/my/mz` (LSB) | Unchanged |
| `converted` | `mx_ut/my_ut/mz_ut` (µT) | Unchanged |
| `calibrated` | `calibrated_mx/my/mz` (iron-corrected) | Unchanged |
| `fused` | `residual_mx/my/mz` (Earth removed) | Unchanged |
| `filtered` | `filtered_mx/my/mz` (Kalman) | Unchanged |

**Key Finding**: The dropdown affects the numeric display in the sensor data panel but does **NOT** change which values are used by:
- The 3 cubes (always use raw for Acc/Mag cubes, fused Euler for Gyro cube)
- The hand skeleton (always uses fused Euler)
- The magnetic trajectory (always uses residual/fused magnetic field)

---

## Summary Table

| Visualization | Raw Fields Used | Computed Fields Used |
|---------------|-----------------|---------------------|
| **Accel Cube** | `ax`, `ay`, `az` | - |
| **Gyro Cube** | - | `euler_roll`, `euler_pitch`, `euler_yaw` |
| **Mag Cube** | `mx`, `my`, `mz` | - |
| **Hand Skeleton** | - | `euler_roll`, `euler_pitch`, `euler_yaw` |
| **Mag Trajectory** | - | `residual_mx`, `residual_my`, `residual_mz` |
| **Sensor Table** | Depends on dropdown | Depends on dropdown |


# Gyroscope Drift vs Magnetometer Heading Analysis

## The Problem

You're experiencing a classic IMU sensor fusion challenge:

1. **During motion**: Gyroscope provides excellent short-term orientation (responsive, no lag)
2. **After motion stops**: Orientation slowly drifts because:
   - Gyroscope has inherent drift (even with bias calibration)
   - Magnetometer is trying to "correct" the yaw
   - But the magnetometer reading changes when hand orientation changes (it measures Earth's field from different angles)

## Current Implementation Analysis

Looking at `MadgwickAHRS.updateWithMag()`:

```typescript
// Current behavior:
const effectiveBeta = this.beta * (1.0 + this.magTrust);
// magTrust default = 0.5 (from telemetry-processor.ts)
// beta default = 0.05 (from sensor-config.ts)
// effectiveBeta = 0.05 * 1.5 = 0.075
```

The magnetometer correction is applied **continuously** with constant strength, regardless of:
- Whether the device is moving or stationary
- The quality/reliability of the magnetometer reading
- Whether the magnetic field has changed due to finger magnets

## Root Cause

The Madgwick algorithm uses the magnetometer to correct **yaw drift** by comparing the measured magnetic field to an expected Earth field reference. However:

1. **When hand rotates**: The expected Earth field in device frame changes (correctly)
2. **Problem**: The algorithm doesn't distinguish between:
   - Legitimate yaw correction (gyro drift)
   - Magnetic field changes from finger magnets
   - Transient magnetic disturbances

## Solution Options

### Option 1: Motion-Adaptive Magnetometer Trust (Recommended)

Reduce magnetometer influence during/after motion, increase when stationary:

```typescript
// In TelemetryProcessor.process():
if (motionState.isMoving) {
    // During motion: trust gyro more, mag less
    this.imuFusion.setMagTrust(0.1);  // Low trust
} else {
    // Stationary: gradually increase mag trust
    // But only after settling period
    const timeSinceMotion = now - lastMotionTime;
    const settlingTime = 2000; // 2 seconds
    const trust = Math.min(0.5, 0.1 + (timeSinceMotion / settlingTime) * 0.4);
    this.imuFusion.setMagTrust(trust);
}
```

### Option 2: Magnetic Anomaly Detection

Only apply magnetometer correction when the field magnitude matches expected Earth field:

```typescript
// In MadgwickAHRS.updateWithMag():
const expectedMagnitude = Math.sqrt(
    this.geomagneticRef.horizontal ** 2 + 
    this.geomagneticRef.vertical ** 2
);
const measuredMagnitude = Math.sqrt(mx*mx + my*my + mz*mz);
const magnitudeError = Math.abs(measuredMagnitude - expectedMagnitude) / expectedMagnitude;

// If magnitude differs significantly, reduce trust
const anomalyFactor = magnitudeError > 0.2 ? 0.1 : 1.0;
const effectiveBeta = this.beta * (1.0 + this.magTrust * anomalyFactor);
```

### Option 3: Complementary Filter Approach

Use a slower time constant for magnetometer correction:

```typescript
// Instead of direct correction, use exponential smoothing
const magCorrectionAlpha = 0.01; // Very slow correction
this.yawFromMag = this.yawFromMag * (1 - magCorrectionAlpha) + measuredYaw * magCorrectionAlpha;
```

### Option 4: Disable Magnetometer for Yaw (Simplest)

Since you have finger magnets that disturb the field anyway:

```typescript
// In TelemetryProcessor constructor:
this.useMagnetometer = false; // Use 6-DOF only
```

This accepts yaw drift but eliminates magnetic interference.

### Option 5: Hybrid Approach (Best for GAMBIT)

Given that GAMBIT uses magnetometer for **finger tracking** (not heading), consider:

1. **Use 6-DOF for orientation** (accel + gyro only)
2. **Use magnetometer only for residual/finger detection**
3. **Implement periodic yaw reset** when user is in known pose

## Recommended Implementation

For GAMBIT specifically, I recommend **Option 1 + Option 2 combined**:

```typescript
// Enhanced TelemetryProcessor
private lastMotionTime: number = 0;
private magTrustSettlingTime: number = 2000; // ms

process(raw: RawTelemetry): DecoratedTelemetry {
    // ... existing code ...
    
    // Motion-adaptive magnetometer trust
    const now = performance.now();
    if (motionState.isMoving) {
        this.lastMotionTime = now;
        this.imuFusion.setMagTrust(0.05); // Minimal during motion
    } else {
        const timeSinceMotion = now - this.lastMotionTime;
        
        // Check for magnetic anomaly (finger magnets)
        const residualMag = decoratedData.residual_magnitude || 0;
        const hasFingerMagnets = residualMag > 15; // µT threshold
        
        if (hasFingerMagnets) {
            // Finger magnets detected - don't trust magnetometer for heading
            this.imuFusion.setMagTrust(0.0);
        } else {
            // Gradually restore trust after settling
            const settledRatio = Math.min(1, timeSinceMotion / this.magTrustSettlingTime);
            this.imuFusion.setMagTrust(0.05 + settledRatio * 0.45);
        }
    }
    
    // ... rest of processing ...
}
```

## Quick Test

To verify the issue, you can temporarily disable magnetometer fusion:

1. In `apps/gambit/shared/telemetry-processor.ts`, change:
   ```typescript
   this.useMagnetometer = false; // Line ~95
   ```

2. Test if drift behavior changes (it should drift more slowly but consistently)

> gyro drift is negligible. added `useMagnetometer: false,` to `new TelemetryProcessor()` in collector and gambit apps.