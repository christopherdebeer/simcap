# Orientation Validation Protocol

## Overview

This document describes the procedure for validating that the GAMBIT sensor orientation and hand visualization are correctly aligned.

## Coordinate System Conventions

### Sensor Placement on Hand

The Puck.js is positioned on the **back of the hand** with:
- **Battery (bottom)**: Facing the palm (skin side)
- **LEDs (top)**: Facing away from palm, toward fingers
- **MQBT42Q module/aerial**: Facing toward the wrist
- **Opposite the MQBT42Q**: Facing toward fingers

### Accelerometer & Gyroscope Axes (LSM6DS3)

| Axis | Sensor Definition | Hand Anatomy (Positive Direction) |
|------|-------------------|-----------------------------------|
| +X   | Toward aerial/MQBT42Q | Toward WRIST |
| +Y   | Toward IR LEDs | Toward FINGERS |
| +Z   | Into PCB (toward battery) | INTO PALM |

When the sensor is placed **face-up on a flat surface**:
- **Z axis**: Points UP (toward ceiling), gravity = +1g on Z
- **X/Y axes**: In the horizontal plane
- **Accelerometer reading**: `az ≈ +1.0g`, `ax ≈ 0`, `ay ≈ 0`

### Magnetometer Axes (LIS3MDL) - **CRITICAL: TRANSPOSED X/Y**

The magnetometer has **swapped X and Y axes** relative to accelerometer/gyroscope:

| Axis | Sensor Definition | Hand Anatomy (Positive Direction) |
|------|-------------------|-----------------------------------|
| +X   | Toward IR LEDs | Toward FINGERS (= Accel +Y) |
| +Y   | Toward aerial/MQBT42Q | Toward WRIST (= Accel +X) |
| +Z   | Into PCB (toward battery) | INTO PALM (= Accel +Z) |

**✅ FIXED**: Magnetometer axes are now aligned to accelerometer frame in `telemetry-processor.ts`:
```typescript
const mx_ut = my_ut_raw;   // Mag Y → aligned X (swap)
const my_ut = -mx_ut_raw;  // Mag X → aligned Y (swap + negate)
const mz_ut = mz_ut_raw;   // Z unchanged
```
The Y-axis negation ensures positive accel-mag correlation on all axes.

### Madgwick AHRS Euler Angles

- **Roll**: Rotation around X axis (tilting left/right)
- **Pitch**: Rotation around Y axis (tilting forward/back)
- **Yaw**: Rotation around Z axis (compass heading)
- When sensor is face-up and level: `roll ≈ 0°`, `pitch ≈ 0°`, `yaw = arbitrary`

### Hand Model Coordinate Frame

- **Fingers**: Extend in +Y direction
- **Palm face**: At +Z (toward viewer after base 180° Y rotation)
- **Base transform**: 180° Y rotation so palm faces viewer by default

## Expected Behavior

| Sensor Position | Expected Hand Visualization |
|-----------------|----------------------------|
| Face-up on desk | Palm UP (toward ceiling), fingers away from viewer |
| Tilted forward | Fingers tilt toward viewer |
| Tilted backward | Fingers tilt away from viewer |
| Tilted left | Palm tilts to face right |
| Tilted right | Palm tilts to face left |
| Rotated clockwise (yaw) | Hand rotates clockwise |

## Validation Procedure

### Prerequisites

1. Open GAMBIT index.html in browser
2. Connect to GAMBIT device
3. Ensure "Track IMU" checkbox is enabled

### Test 1: Face-Up Reference Pose

1. Place sensor face-up on a flat, level surface
2. Click "Get data" to start streaming
3. Wait for gyroscope bias calibration (✓ indicator)
4. **Verify**:
   - Euler debug shows: `Roll: ~0° | Pitch: ~0° | Yaw: varies`
   - Accelerometer shows: `ax ≈ 0, ay ≈ 0, az ≈ 1.0g`
   - Hand visualization shows: **Palm facing UP (toward ceiling)**

### Test 2: Forward Tilt

1. From face-up position, tilt sensor forward (toward you)
2. **Verify**:
   - Pitch increases (positive)
   - Hand fingers tilt toward viewer

### Test 3: Backward Tilt

1. From face-up position, tilt sensor backward (away from you)
2. **Verify**:
   - Pitch decreases (negative)
   - Hand fingers tilt away from viewer

### Test 4: Left Tilt

1. From face-up position, tilt sensor to the left
2. **Verify**:
   - Roll changes
   - Hand palm tilts to face right

### Test 5: Right Tilt

1. From face-up position, tilt sensor to the right
2. **Verify**:
   - Roll changes
   - Hand palm tilts to face left

### Test 6: Yaw Rotation

1. From face-up position, rotate sensor clockwise (around vertical axis)
2. **Verify**:
   - Yaw increases
   - Hand rotates clockwise

### Test 7: Stream Stop/Start Stability

1. With sensor stationary face-up, click "Stop" to stop streaming
2. **Do not move the sensor**
3. Click "Get data" to restart streaming
4. **Verify**:
   - Hand orientation should be similar to before (may differ in yaw)
   - No dramatic orientation jump

### Test 8: Reset Orientation

1. Move sensor to any arbitrary orientation
2. Click "Reset Orientation" button
3. **Verify**:
   - AHRS re-initializes from current accelerometer reading
   - Euler angles reset (roll/pitch from gravity, yaw to 0)

## Troubleshooting

### Hand shows wrong orientation on startup

- Click "Reset Orientation" button
- Ensure sensor is stationary during gyro bias calibration

### Orientation jumps when restarting stream

- This is expected if yaw drifted during the previous session
- Use "Reset Orientation" to re-initialize

### Tilt direction is inverted

- Check the axis mapping in `hand-3d-renderer.js` `updateFromSensorFusion()`
- The mapping includes a +90° pitch offset to show palm up when level

## Implementation Details

### Key Files

- `hand-3d-renderer.js`: `updateFromSensorFusion()` - maps IMU Euler to hand orientation
- `shared/telemetry-processor.js`: `process()` - runs AHRS, `reset()` - re-initializes
- `shared/sensor-config.js`: `createMadgwickAHRS()` - AHRS factory
- `filters.js`: `MadgwickAHRS` class - orientation estimation

### Axis Mapping (in updateFromSensorFusion)

**Current implementation in `hand-3d-renderer.js`:**

```javascript
const mappedOrientation = {
    pitch: euler.pitch + 90 + this.orientationOffset.pitch,   // palm up when sensor level
    yaw: euler.yaw + 180 + this.orientationOffset.yaw,        // fingers away from viewer
    roll: -euler.roll + 180 + this.orientationOffset.roll     // correct chirality
};
```

**Current implementation in `threejs-hand-skeleton.js`:**

```javascript
// Offsets set during initialization:
setOrientationOffsets({ roll: 180, pitch: 90, yaw: 180 });

// Applied in render loop:
const roll = (currentOrientation.roll + offsets.roll) * (Math.PI / 180);
const pitch = (currentOrientation.pitch + offsets.pitch) * (Math.PI / 180);
const yaw = (currentOrientation.yaw + offsets.yaw) * (Math.PI / 180);
handGroup.rotation.set(pitch, yaw, roll, "YXZ");
```

The offsets rotate the hand from "palm facing viewer" (default) to "palm facing up" (desired when sensor is face-up).

### TODO Items

- **[SENSOR-001]**: Magnetometer axis alignment not implemented (see `GAMBIT/index.html`)
- **[SENSOR-002]**: Create `magAlignToAccelFrame()` utility function (see `shared/sensor-units.js`)
- **[SENSOR-003]**: Calibration uses unaligned magnetometer data (see `calibration.js`)
- **[ORIENT-001]**: Validate both renderers produce consistent orientation
- **[ORIENT-002]**: Document expected behavior for each test pose
