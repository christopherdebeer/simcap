# Orientation Mapping Diagnostic Report

## Executive Summary

Analysis of sensor data reveals **systematic axis coupling** that indicates a deeper problem than simple axis sign inversion. The root cause is likely one or more of:

1. **Sensor frame misalignment** - The physical sensor axes don't match the documented orientation
2. **Euler order mismatch** - ZYX vs YXZ convention conflicts
3. **Gimbal lock artifacts** - Near 90° pitch, roll and yaw exchange values

## Evidence of Axis Coupling

### Key Observations from `2025-12-13T21:35:42.588Z.json`:

| Movement | Expected Change | AHRS Roll Δ | AHRS Pitch Δ | AHRS Yaw Δ | Problem |
|----------|----------------|-------------|--------------|------------|---------|
| PITCH_NEG (forward tilt) | Pitch only | **+115.9°** | -41.2° | -55.8° | Roll > Pitch! |
| ROLL_NEG (left tilt) | Roll only | -175.9° | -5.8° | **-85.8°** | Huge yaw coupling |
| ROLL_POS (right tilt) | Roll only | +174.3° | -21.4° | **-80.9°** | Huge yaw coupling |

### Pattern Analysis

1. **YAW always couples with ROLL** - When rolling left/right, yaw changes 70-85°
2. **ROLL often dominates PITCH movements** - When pitching, roll sometimes changes more
3. **Accelerometer tilt generally agrees with expectations** - Problem is in AHRS output

## Root Cause Hypothesis

### Most Likely: Euler Order Mismatch

The Madgwick filter extracts Euler angles using **ZYX (aerospace)** convention:
- Roll (X) calculated first
- Pitch (Y) calculated second
- Yaw (Z) calculated third

But Three.js applies rotations in **YXZ** order:
- Yaw (Y) applied first
- Pitch (X) applied second
- Roll (Z) applied third

When you extract Euler angles in one order and apply them in another, you get **axis coupling**.

### Alternative: Sensor Frame Mismatch

The documented sensor orientation:
- X → toward wrist
- Y → toward fingers
- Z → into palm

If the actual mounting is different (rotated 90°), all the axis mappings would be shifted.

## Recommended Fix Strategy

### Option 1: Match Euler Orders (Preferred)

Change the Euler extraction in `filters.js` `getEulerAngles()` to use YXZ order (Three.js default):

```javascript
// Current ZYX extraction (WRONG for Three.js)
const sinr_cosp = 2 * (w * x + y * z);
...

// Should use YXZ extraction (matches Three.js)
// Pitch first (X), then Yaw (Y), then Roll (Z)
```

### Option 2: Test All 6 Permutations

Create a UI toggle to test different axis mappings:
- `roll → X, pitch → Y, yaw → Z` (current)
- `roll → Y, pitch → X, yaw → Z` (swap roll/pitch)
- `roll → Z, pitch → Y, yaw → X` (swap roll/yaw)
- etc.

### Option 3: Work in Quaternion Space

Bypass Euler angles entirely:
- Pass quaternion directly from AHRS to Three.js
- Let Three.js handle the conversion internally
- Eliminates all Euler order/gimbal lock issues

## Specific Code Changes Needed

### 1. In `filters.js` - Fix Euler extraction for YXZ

```javascript
getEulerAngles() {
    const { w, x, y, z } = this.q;

    // YXZ order (Three.js default)
    // Pitch (X-axis rotation) - computed from gimbal lock check
    const sinp = 2 * (w * x - y * z);
    let pitch = Math.abs(sinp) >= 1
        ? Math.sign(sinp) * Math.PI / 2
        : Math.asin(sinp);

    // Yaw (Y-axis rotation)
    const siny = 2 * (w * y + x * z);
    const cosy = 1 - 2 * (x * x + y * y);
    const yaw = Math.atan2(siny, cosy);

    // Roll (Z-axis rotation)
    const sinr = 2 * (w * z + x * y);
    const cosr = 1 - 2 * (x * x + z * z);
    const roll = Math.atan2(sinr, cosr);

    return {
        roll: roll * 180 / Math.PI,
        pitch: pitch * 180 / Math.PI,
        yaw: yaw * 180 / Math.PI
    };
}
```

### 2. In `threejs-hand-skeleton.js` - Use Quaternion Directly

```javascript
_applyOrientation(sensorData) {
    // Instead of Euler angles, use quaternion directly
    if (sensorData.quaternion) {
        this.handGroup.quaternion.set(
            sensorData.quaternion.x,
            sensorData.quaternion.y,
            sensorData.quaternion.z,
            sensorData.quaternion.w
        );
        return;
    }
    // Fallback to Euler if no quaternion
    ...
}
```

## Validation Test

After fixing, these conditions should hold:

1. **FLAT pose**: roll ≈ 0°, pitch ≈ 0°, yaw = any
2. **PITCH_FORWARD_45**: roll ≈ 0°, pitch ≈ -45°, yaw unchanged
3. **ROLL_LEFT_45**: roll ≈ -45°, pitch ≈ 0°, yaw unchanged
4. **YAW_CW_90**: roll ≈ 0°, pitch ≈ 0°, yaw ≈ +90°

No axis should change by more than ±15° when it's not the primary axis of rotation.
