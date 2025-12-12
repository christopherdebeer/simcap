# Orientation Validation Protocol

## Overview

This document describes the procedure for validating that the GAMBIT sensor orientation and hand visualization are correctly aligned.

## Coordinate System Conventions

### Sensor Coordinate Frame (Puck.js with LSM6DS3/LIS3MDL)

When the sensor is placed **face-up on a flat surface**:
- **Z axis**: Points UP (toward ceiling), gravity = +1g on Z
- **X/Y axes**: In the horizontal plane
- **Accelerometer reading**: `az ≈ +1.0g`, `ax ≈ 0`, `ay ≈ 0`

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

```javascript
const mappedOrientation = {
    pitch: -euler.pitch + 90 + this.orientationOffset.pitch,  // +90° for palm up
    yaw: euler.yaw + this.orientationOffset.yaw,
    roll: euler.roll + this.orientationOffset.roll
};
```

The +90° pitch offset rotates the hand from "palm facing viewer" (default) to "palm facing up" (desired when sensor is face-up).
