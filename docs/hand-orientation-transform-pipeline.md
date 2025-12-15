# GAMBIT Hand Orientation Transform Pipeline
## Complete Technical Reference

**Document Version:** 1.0
**Date:** 2025-12-15
**Status:** Current Implementation

---

## Executive Summary

This document provides a complete technical reference for the GAMBIT hand orientation system, tracing the transformation pipeline from raw firmware sensor data through to 3D hand model orientation in Three.js. Every unit conversion, coordinate transform, axis mapping, and offset is documented with precise values and rationale.

**Key Pipeline Stages:**
1. Firmware sensor acquisition (LSB units)
2. Unit conversion (LSB → physical units: g, deg/s, µT)
3. IMU sensor fusion (Madgwick AHRS: physical units → quaternion)
4. Euler angle extraction (quaternion → roll/pitch/yaw degrees)
5. Axis mapping corrections (sensor frame → hand model frame)
6. 3D rendering (two implementations with consistent behavior)

---

## Table of Contents

1. [Physical Setup & Sensor Frames](#1-physical-setup--sensor-frames)
2. [Stage 1: Firmware Sensor Acquisition](#2-stage-1-firmware-sensor-acquisition)
3. [Stage 2: Unit Conversion](#3-stage-2-unit-conversion)
4. [Stage 3: Telemetry Processing](#4-stage-3-telemetry-processing)
5. [Stage 4: Madgwick AHRS Sensor Fusion](#5-stage-4-madgwick-ahrs-sensor-fusion)
6. [Stage 5: Euler Angle Extraction](#6-stage-5-euler-angle-extraction)
7. [Stage 6: Orientation Model Mapping](#7-stage-6-orientation-model-mapping)
8. [Stage 7A: Hand 3D Renderer](#8-stage-7a-hand-3d-renderer)
9. [Stage 7B: Three.js Hand Skeleton](#9-stage-7b-threejs-hand-skeleton)
10. [Complete Transform Summary](#10-complete-transform-summary)
11. [Simplification Analysis](#11-simplification-analysis)

---

## 1. Physical Setup & Sensor Frames

### 1.1 Device Mounting
- **Device:** Puck.js with LSM6DS3 (accel/gyro) and LIS3MDL (magnetometer)
- **Position:** Back of hand (dorsum), battery side toward palm
- **Orientation:**
  - MQBT42Q antenna edge → toward WRIST
  - IR LED edge → toward FINGERTIPS

### 1.2 Sensor Coordinate Frame (S)

```
When hand is palm-UP (palm facing ceiling):

     +Y (toward fingertips)
      ^
      |
      |
+X <--+---- (toward wrist)
      |
      v
     +Z (INTO palm, toward ceiling when palm-up)
```

**Accelerometer Reading (Palm-Up):** `ax≈0, ay≈0, az≈+1g`

### 1.3 Hand Model Coordinate Frame (H)

```
Three.js right-hand coordinate system:

     +Y (finger extension direction)
      ^
      |
      |
      +--->  +X (toward pinky, right hand)
     /
    v
   +Z (palm normal, toward viewer)
```

### 1.4 Frame Relationship

The sensor frame and hand model frame differ by a 180° rotation about the Y-axis:

| Sensor Axis | Hand Model Axis | Relationship |
|-------------|-----------------|--------------|
| +X (wrist)  | -X (thumb)      | Opposite     |
| +Y (fingers)| +Y (fingers)    | Same         |
| +Z (into palm)| -Z (back of hand)| Opposite     |

---

## 2. Stage 1: Firmware Sensor Acquisition

**File:** `/src/device/GAMBIT/app.js`

### 2.1 Sensor Hardware

#### Accelerometer & Gyroscope: LSM6DS3
- **Range:** ±2g (accel), ±245 dps (gyro)
- **Resolution:** 16-bit signed integer
- **Output:** Raw LSB (Least Significant Bit) values

#### Magnetometer: LIS3MDL
- **Range:** ±4 gauss
- **Resolution:** 16-bit signed integer
- **Output:** Raw LSB values

### 2.2 Firmware Output (line 173-211)

```javascript
function emit() {
    // Read accelerometer + gyroscope (single I2C read)
    var accel = Puck.accel();
    telemetry.ax = accel.acc.x;   // LSB
    telemetry.ay = accel.acc.y;   // LSB
    telemetry.az = accel.acc.z;   // LSB
    telemetry.gx = accel.gyro.x;  // LSB
    telemetry.gy = accel.gyro.y;  // LSB
    telemetry.gz = accel.gyro.z;  // LSB

    // Read magnetometer (every 2nd sample for power saving)
    if (sampleCount % 2 === 0) {
        var mag = Puck.mag();
        telemetry.mx = mag.x;      // LSB
        telemetry.my = mag.y;      // LSB
        telemetry.mz = mag.z;      // LSB
    }

    sendFrame('T', telemetry);
}
```

**Output Units:** All values in **LSB** (raw sensor counts)

**Transmission:** Bluetooth Low Energy via framed protocol

---

## 3. Stage 2: Unit Conversion

**File:** `/src/web/GAMBIT/shared/sensor-units.js`

### 3.1 Conversion Specifications

#### 3.1.1 Accelerometer (lines 28-37)

```javascript
export const ACCEL_SPEC = {
    sensor: 'LSM6DS3',
    rawUnit: 'LSB',
    convertedUnit: 'g',
    range: '±2g',
    resolution: 16,
    sensitivity: 8192,              // LSB per g
    conversionFactor: 1 / 8192,     // g per LSB = 0.0001220703125
};
```

**Formula:**
```
accel_g = accel_LSB × (1 / 8192)
accel_g = accel_LSB × 0.0001220703125
```

**Example:**
- Input: `az = 8192 LSB`
- Output: `az_g = 1.0 g` (gravity, palm-up)

#### 3.1.2 Gyroscope (lines 50-59)

```javascript
export const GYRO_SPEC = {
    sensor: 'LSM6DS3',
    rawUnit: 'LSB',
    convertedUnit: 'deg/s',
    range: '±245dps',
    resolution: 16,
    sensitivity: 114.28,            // LSB per deg/s
    conversionFactor: 1 / 114.28,   // deg/s per LSB = 0.008751093
};
```

**Formula:**
```
gyro_dps = gyro_LSB × (1 / 114.28)
gyro_dps = gyro_LSB × 0.008751093
```

**Example:**
- Input: `gx = 1143 LSB`
- Output: `gx_dps = 10.0 deg/s`

#### 3.1.3 Magnetometer (lines 78-94)

```javascript
export const MAG_SPEC = {
    sensor: 'LIS3MDL',
    rawUnit: 'LSB',
    convertedUnit: 'µT',
    range: '±4gauss',
    resolution: 16,
    sensitivity: 6842,              // LSB per gauss
    gaussToMicroTesla: 100,
    conversionFactor: 100 / 6842,   // µT per LSB = 0.014616
};
```

**Conversion Chain:**
```
1. LSB → gauss: mag_gauss = mag_LSB / 6842
2. gauss → µT:  mag_µT = mag_gauss × 100
3. Combined:    mag_µT = mag_LSB × (100 / 6842)
                mag_µT = mag_LSB × 0.014616
```

**Example:**
- Input: `mx = 3421 LSB`
- Output: `mx_µT = 50.0 µT` (Earth's field horizontal component)

### 3.2 Conversion Functions (lines 112-136)

```javascript
export function accelLsbToG(lsb) {
    return lsb * ACCEL_SPEC.conversionFactor;  // lsb × 0.0001220703125
}

export function gyroLsbToDps(lsb) {
    return lsb * GYRO_SPEC.conversionFactor;   // lsb × 0.008751093
}

export function magLsbToMicroTesla(lsb) {
    return lsb * MAG_SPEC.conversionFactor;    // lsb × 0.014616
}
```

**Output Units:**
- Accelerometer: **g** (standard gravity, 9.81 m/s²)
- Gyroscope: **deg/s** (degrees per second)
- Magnetometer: **µT** (microtesla)

---

## 4. Stage 3: Telemetry Processing

**File:** `/src/web/GAMBIT/shared/telemetry-processor.js`

### 4.1 Processing Pipeline (process method, lines 179-412)

```javascript
process(raw) {
    // 1. PRESERVE RAW DATA
    const decorated = { ...raw };  // Keep original LSB values

    // 2. UNIT CONVERSION (lines 191-218)
    const ax_g = accelLsbToG(raw.ax || 0);
    const ay_g = accelLsbToG(raw.ay || 0);
    const az_g = accelLsbToG(raw.az || 0);

    const gx_dps = gyroLsbToDps(raw.gx || 0);
    const gy_dps = gyroLsbToDps(raw.gy || 0);
    const gz_dps = gyroLsbToDps(raw.gz || 0);

    const mx_ut = magLsbToMicroTesla(raw.mx || 0);
    const my_ut = magLsbToMicroTesla(raw.my || 0);
    const mz_ut = magLsbToMicroTesla(raw.mz || 0);

    // Store as decorated fields
    decorated.ax_g = ax_g;
    decorated.gx_dps = gx_dps;
    decorated.mx_ut = mx_ut;
    // ... etc

    // 3. IMU SENSOR FUSION (lines 264-292)
    if (this.useMagnetometer && magDataValid) {
        // 9-DOF fusion
        this.imuFusion.updateWithMag(
            ax_g, ay_g, az_g,
            gx_dps, gy_dps, gz_dps,
            mx_ut, my_ut, mz_ut,
            dt, true, true  // gyroInDegrees=true, applyHardIron=true
        );
    } else {
        // 6-DOF fusion
        this.imuFusion.update(
            ax_g, ay_g, az_g,
            gx_dps, gy_dps, gz_dps,
            dt, true  // gyroInDegrees=true
        );
    }

    // 4. GET ORIENTATION (lines 295-309)
    const orientation = this.imuFusion.getQuaternion();  // {w, x, y, z}
    const euler = this.imuFusion.getEulerAngles();       // {roll, pitch, yaw} degrees

    return decorated;
}
```

**Key Operations:**
1. **Preserve raw data** (LSB values kept in output)
2. **Convert units** (LSB → g, deg/s, µT)
3. **Motion detection** (for gyro bias calibration)
4. **Gyro bias correction** (when stationary)
5. **IMU fusion** (convert sensor readings to orientation)

**Output:** Decorated telemetry with:
- Raw fields: `ax, ay, az, gx, gy, gz, mx, my, mz` (LSB)
- Converted fields: `ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, mx_ut, my_ut, mz_ut`
- Orientation: `orientation_w, orientation_x, orientation_y, orientation_z` (quaternion)
- Euler: `euler_roll, euler_pitch, euler_yaw` (degrees)

---

## 5. Stage 4: Madgwick AHRS Sensor Fusion

**File:** `/src/web/GAMBIT/filters.js`

### 5.1 AHRS Configuration (constructor, lines 32-66)

```javascript
class MadgwickAHRS {
    constructor(options = {}) {
        this.sampleFreq = options.sampleFreq || 50;  // Hz (default 50, actual 26)
        this.beta = options.beta || 0.1;              // Filter gain
        this.q = { w: 1, x: 0, y: 0, z: 0 };          // Identity quaternion
        this.gyroBias = { x: 0, y: 0, z: 0 };         // Gyro bias in rad/s
        this.magTrust = 1.0;                           // Magnetometer trust (0-1)
        this.hardIron = { x: 0, y: 0, z: 0 };         // Hard iron offset (µT)
    }
}
```

### 5.2 6-DOF Update (Accel + Gyro Only)

**Method:** `update(ax, ay, az, gx, gy, gz, dt, gyroInDegrees)`

#### 5.2.1 Input Processing (lines 80-93)

```javascript
// Time step
const deltaT = dt || (1.0 / this.sampleFreq);  // ~0.02s at 50Hz

// Convert gyroscope from deg/s to rad/s
if (gyroInDegrees) {
    gx = gx * Math.PI / 180;  // deg/s → rad/s
    gy = gy * Math.PI / 180;
    gz = gz * Math.PI / 180;
}

// Apply gyroscope bias correction
gx -= this.gyroBias.x;  // rad/s
gy -= this.gyroBias.y;
gz -= this.gyroBias.z;
```

**Transform:**
```
gyro_rad/s = gyro_deg/s × (π / 180)
gyro_rad/s = gyro_deg/s × 0.017453293
```

**Example:**
- Input: `gx_dps = 10.0 deg/s`
- Output: `gx = 0.17453293 rad/s`

#### 5.2.2 Quaternion Rate from Gyroscope (lines 95-101)

```javascript
let { w: q0, x: q1, y: q2, z: q3 } = this.q;

// Quaternion derivative from angular velocity
const qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
const qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
const qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
const qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);
```

**Mathematical Basis:** Quaternion kinematic equation
```
q̇ = 0.5 × q ⊗ ω
```
where ω = [0, gx, gy, gz] is angular velocity quaternion

#### 5.2.3 Accelerometer Correction (lines 103-140)

```javascript
// Normalize accelerometer
const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
const recipNorm = 1.0 / accelNorm;
ax *= recipNorm;
ay *= recipNorm;
az *= recipNorm;

// Gradient descent correction (Madgwick algorithm)
// Minimizes error between measured gravity and expected gravity
let s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
let s1 = _4q1 * q3q3 - _2q3 * ax + 4 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
// ... (gradient computation)

// Normalize gradient
const sNorm = 1.0 / Math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3);
s0 *= sNorm;
s1 *= sNorm;
s2 *= sNorm;
s3 *= sNorm;

// Apply correction with beta gain
q0 += (qDot1 - this.beta * s0) * deltaT;
q1 += (qDot2 - this.beta * s1) * deltaT;
q2 += (qDot3 - this.beta * s2) * deltaT;
q3 += (qDot4 - this.beta * s3) * deltaT;
```

**Purpose:** Correct gyro drift using gravity vector from accelerometer

**Beta Parameter:** `β = 0.05` (standard)
- Higher β = faster convergence, more noise
- Lower β = slower convergence, more filtering

#### 5.2.4 Quaternion Normalization (lines 154-161)

```javascript
// Ensure unit quaternion
const qNorm = 1.0 / Math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
this.q = {
    w: q0 * qNorm,
    x: q1 * qNorm,
    y: q2 * qNorm,
    z: q3 * qNorm
};
```

**Constraint:** `w² + x² + y² + z² = 1` (unit quaternion)

### 5.3 9-DOF Update (Accel + Gyro + Mag)

**Method:** `updateWithMag(ax, ay, az, gx, gy, gz, mx, my, mz, dt, gyroInDegrees, applyHardIron)`

#### 5.3.1 Additional Processing (lines 370-403)

```javascript
// Apply hard iron correction (if enabled)
if (applyHardIron) {
    mx -= this.hardIron.x;  // µT
    my -= this.hardIron.y;
    mz -= this.hardIron.z;
}

// Normalize magnetometer
const magNorm = Math.sqrt(mx * mx + my * my + mz * mz);
const recipMagNorm = 1.0 / magNorm;
mx *= recipMagNorm;
my *= recipMagNorm;
mz *= recipMagNorm;
```

#### 5.3.2 Magnetic Reference Direction (lines 427-433)

```javascript
// Compute reference magnetic field direction in earth frame
const hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + ...
const hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + ...

// Only horizontal component matters (Earth's field in horizontal plane)
const _2bx = Math.sqrt(hx * hx + hy * hy);
const _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + ...
```

**Purpose:** Extract Earth's magnetic field direction to correct yaw drift

#### 5.3.3 Combined Gradient (lines 436-446)

```javascript
// Gradient descent with BOTH accelerometer AND magnetometer errors
let s0 = -_2q2 * (2*q1q3 - _2q0q2 - ax) + _2q1 * (2*q0q1 + _2q2q3 - ay)
         - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + ...
// ... (full gradient computation)

// Apply with magnetometer trust weighting
const effectiveBeta = this.beta * (1.0 + this.magTrust);
```

**Key Difference from 6-DOF:** Magnetometer provides absolute yaw reference (prevents yaw drift)

### 5.4 Output State

**Quaternion:** `q = {w, x, y, z}` where `w² + x² + y² + z² = 1`

**Physical Meaning:**
- Represents rotation from earth frame to sensor frame
- `w` = scalar part (related to rotation angle)
- `(x, y, z)` = vector part (rotation axis × sin(θ/2))

---

## 6. Stage 5: Euler Angle Extraction

**File:** `/src/web/GAMBIT/filters.js`
**Method:** `getEulerAngles()` (lines 194-221)

### 6.1 Quaternion to Euler Conversion

```javascript
getEulerAngles() {
    const { w, x, y, z } = this.q;

    // Roll (rotation about X-axis)
    const sinr_cosp = 2 * (w * x + y * z);
    const cosr_cosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinr_cosp, cosr_cosp);

    // Pitch (rotation about Y-axis)
    const sinp = 2 * (w * y - z * x);
    let pitch;
    if (Math.abs(sinp) >= 1) {
        pitch = Math.sign(sinp) * Math.PI / 2;  // Gimbal lock protection
    } else {
        pitch = Math.asin(sinp);
    }

    // Yaw (rotation about Z-axis)
    const siny_cosp = 2 * (w * z + x * y);
    const cosy_cosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(siny_cosp, cosy_cosp);

    return {
        roll: roll * 180 / Math.PI,    // rad → degrees
        pitch: pitch * 180 / Math.PI,  // rad → degrees
        yaw: yaw * 180 / Math.PI       // rad → degrees
    };
}
```

### 6.2 Euler Angle Convention

**Convention:** ZYX intrinsic (Tait-Bryan angles)
- Applied in order: Yaw (Z) → Pitch (Y) → Roll (X)
- Rotations in **sensor body frame** (not world frame)

**Angle Definitions (Sensor Frame):**

| Angle | Axis | Physical Motion | Positive Direction |
|-------|------|-----------------|-------------------|
| Roll  | X (wrist) | Tilt hand left/right | Thumb-side UP |
| Pitch | Y (fingers) | Tilt fingers up/down | Fingers UP |
| Yaw   | Z (palm) | Rotate hand (spin) | Clockwise (viewed from above) |

**Range:**
- Roll: [-180°, +180°]
- Pitch: [-90°, +90°] (with gimbal lock protection at ±90°)
- Yaw: [-180°, +180°]

### 6.3 Transform Summary

```
Input:  Quaternion {w, x, y, z} (unit quaternion)
Process: Euler extraction using atan2 and asin
Output: {roll, pitch, yaw} in degrees
```

**Example:**
```javascript
// Input quaternion (identity)
q = {w: 1, x: 0, y: 0, z: 0}

// Output Euler
euler = {roll: 0, pitch: 0, yaw: 0}  // No rotation
```

---

## 7. Stage 6: Orientation Model Mapping

**File:** `/src/web/GAMBIT/shared/orientation-model.js`

### 7.1 Configuration (lines 143-156)

```javascript
export const ORIENTATION_CONFIG = {
    // Axis sign corrections (applied BEFORE offsets)
    negateRoll: false,   // UN-negated (was causing L/R inversion)
    negatePitch: true,   // NEGATED (fixes forward/back inversion)
    negateYaw: false,    // Unchanged (works correctly)

    // Offset values (degrees) to align neutral pose
    rollOffset: 180,
    pitchOffset: 180,
    yawOffset: -180,

    // Three.js Euler order
    eulerOrder: 'YXZ'
};
```

### 7.2 Mapping Function (lines 165-173)

```javascript
export function mapSensorToHand(sensorEuler, config = ORIENTATION_CONFIG) {
    const { roll: s_roll, pitch: s_pitch, yaw: s_yaw } = sensorEuler;

    return {
        roll:  (config.negateRoll  ? -s_roll  : s_roll)  + config.rollOffset,
        pitch: (config.negatePitch ? -s_pitch : s_pitch) + config.pitchOffset,
        yaw:   (config.negateYaw   ? -s_yaw   : s_yaw)   + config.yawOffset
    };
}
```

### 7.3 Transform Breakdown

#### Roll Transform
```
hand_roll = (negateRoll ? -sensor_roll : sensor_roll) + rollOffset
hand_roll = sensor_roll + 180°

Example:
  sensor_roll = 0°   → hand_roll = 180°
  sensor_roll = 30°  → hand_roll = 210°  (thumb tilted up)
  sensor_roll = -30° → hand_roll = 150°  (pinky tilted up)
```

#### Pitch Transform
```
hand_pitch = (negatePitch ? -sensor_pitch : sensor_pitch) + pitchOffset
hand_pitch = -sensor_pitch + 180°

Example:
  sensor_pitch = 0°   → hand_pitch = 180°
  sensor_pitch = 30°  → hand_pitch = 150°  (fingers point up)
  sensor_pitch = -30° → hand_pitch = 210°  (fingers point down)
```

#### Yaw Transform
```
hand_yaw = (negateYaw ? -sensor_yaw : sensor_yaw) + yawOffset
hand_yaw = sensor_yaw - 180°

Example:
  sensor_yaw = 0°   → hand_yaw = -180°
  sensor_yaw = 45°  → hand_yaw = -135°  (rotated CW)
  sensor_yaw = -45° → hand_yaw = -225° = 135° (rotated CCW)
```

### 7.4 Three.js Conversion (lines 183-193)

```javascript
export function mapSensorToThreeJS(sensorEuler, config = ORIENTATION_CONFIG) {
    const handAngles = mapSensorToHand(sensorEuler, config);
    const deg2rad = Math.PI / 180;

    return {
        x: handAngles.pitch * deg2rad,  // Three.js X rotation = pitch
        y: handAngles.yaw * deg2rad,    // Three.js Y rotation = yaw
        z: handAngles.roll * deg2rad,   // Three.js Z rotation = roll
        order: config.eulerOrder         // 'YXZ'
    };
}
```

**Unit Conversion:**
```
radians = degrees × (π / 180)
radians = degrees × 0.017453293
```

### 7.5 Rationale for Corrections

**Why negate pitch?**
- Sensor: +pitch = fingers point up
- Without negation: Model shows fingers pointing DOWN (inverted)
- With negation: Model correctly shows fingers pointing UP

**Why NOT negate roll?**
- Sensor: +roll = thumb-side up
- Old code negated it: Model showed OPPOSITE tilt (inverted)
- Without negation: Model correctly shows thumb-side UP

**Why offsets of ±180°?**
- Aligns hand model's "neutral" pose (palm facing viewer) with sensor's neutral (palm facing up)
- Compensates for 180° rotation between sensor and hand coordinate frames

---

## 8. Stage 7A: Hand 3D Renderer

**File:** `/src/web/GAMBIT/hand-3d-renderer.js`

### 8.1 Orientation Update (lines 222-267)

```javascript
updateFromSensorFusion(euler) {
    if (this.orientationMode !== 'sensor_fusion') return;
    if (!euler) return;

    // AXIS MAPPING (V2 corrected 2025-12-14):
    //
    // Physical movements → AHRS Reports → Renderer:
    //   - Physical ROLL (tilt pinky/thumb)  → AHRS roll  → renderer.yaw   → RotY
    //   - Physical PITCH (tilt fingers)     → AHRS pitch → renderer.pitch → RotX
    //   - Physical YAW (spin while flat)    → AHRS yaw   → renderer.roll  → RotZ
    //
    // Therefore:
    //   renderer.pitch = f(AHRS pitch)  - both are X-axis related for finger tilt
    //   renderer.yaw   = f(AHRS roll)   - AHRS roll causes RotY visual
    //   renderer.roll  = f(AHRS yaw)    - both are Z-axis related for spin

    const mappedOrientation = {
        pitch: -euler.pitch + this.orientationOffset.pitch,  // AHRS pitch → RotX
        yaw:   -euler.roll  + this.orientationOffset.yaw,    // AHRS roll  → RotY (negated)
        roll:   euler.yaw   + this.orientationOffset.roll    // AHRS yaw   → RotZ
    };

    this.setOrientation(mappedOrientation);
}
```

### 8.2 Axis Mapping Summary

| AHRS Output | Renderer Variable | Three.js Rotation | Visual Effect |
|-------------|-------------------|-------------------|---------------|
| euler.pitch | renderer.pitch    | RotX             | Fingers tilt forward/back |
| euler.roll  | renderer.yaw      | RotY             | Hand tilts left/right |
| euler.yaw   | renderer.roll     | RotZ             | Hand spins |

**Key Insight:** The axis SWAP between AHRS roll/yaw and renderer yaw/roll is the critical correction made in V2.

### 8.3 Rendering (lines 320-364)

```javascript
render() {
    const pitch = this._rad(this.orientation.pitch);  // degrees → radians
    const yaw = this._rad(this.orientation.yaw);
    const roll = this._rad(this.orientation.roll);

    // Base transform: 180° Y rotation (palm faces viewer)
    let handM = this._matRotY(Math.PI);

    // Pivot to sensor position
    const sensorPivot = [0, 0.15, -0.12];  // After base flip
    handM = this._matMul(handM, this._matTrans(-sensorPivot[0], -sensorPivot[1], -sensorPivot[2]));

    // Apply user rotations in ZYX order
    handM = this._matMul(handM, this._matRotZ(roll));   // Z first (yaw/heading)
    handM = this._matMul(handM, this._matRotY(yaw));    // Y second (pitch/elevation)
    handM = this._matMul(handM, this._matRotX(pitch));  // X third (roll/bank)

    // Pivot back
    handM = this._matMul(handM, this._matTrans(sensorPivot[0], sensorPivot[1], sensorPivot[2]));

    // ... render hand geometry with handM transform
}
```

**Rotation Order:** ZYX intrinsic (applied to local frame after each rotation)

**Pivot Point:** Rotations centered at sensor position (center of palm) for natural motion

### 8.4 Complete Transform Chain

```
1. AHRS Euler (degrees) → mapSensorToHand → handAngles (degrees)
2. handAngles → updateFromSensorFusion →
   {pitch: -euler.pitch, yaw: -euler.roll, roll: euler.yaw} (degrees)
3. degrees → radians (× π/180)
4. Build matrix: Base180° × Translate(-pivot) × RotZ × RotY × RotX × Translate(+pivot)
5. Apply matrix to all hand vertices
6. Project to 2D canvas
```

---

## 9. Stage 7B: Three.js Hand Skeleton

**File:** `/src/web/GAMBIT/shared/threejs-hand-skeleton.js`

### 9.1 Orientation Application (lines 399-457)

```javascript
_applyOrientation() {
    // Shortest-path lerp for smooth animation (wrap-aware)
    this.currentOrientation.roll = lerpAngleDeg(
        this.currentOrientation.roll, this.targetOrientation.roll, this.orientationLerpFactor);
    this.currentOrientation.pitch = lerpAngleDeg(
        this.currentOrientation.pitch, this.targetOrientation.pitch, this.orientationLerpFactor);
    this.currentOrientation.yaw = lerpAngleDeg(
        this.currentOrientation.yaw, this.targetOrientation.yaw, this.orientationLerpFactor);

    const offsets = this.orientationOffsets || { roll: 0, pitch: 0, yaw: 0 };
    const signs = this.axisSigns || { negateRoll: false, negatePitch: true, negateYaw: false };

    // AXIS MAPPING (corrected 2025-12-14):
    //
    // Sensor reports roll/pitch/yaw where:
    //   - Physical roll (tilt pinky/thumb) → sensor reports as "roll"
    //   - Physical pitch (tilt fingers) → sensor reports as "pitch"
    //   - Physical yaw (spin while flat) → sensor reports as "yaw"
    //
    // CORRECT MAPPING (physical → visual):
    //   - Sensor roll (physical roll) → RotY (yaw variable) → hand tilts L/R
    //   - Sensor pitch (physical pitch) → RotX (pitch variable) → fingers tilt
    //   - Sensor yaw (physical yaw) → RotZ (roll variable) → hand spins

    const sensorRoll = this.currentOrientation.roll;   // Physical roll
    const sensorPitch = this.currentOrientation.pitch; // Physical pitch
    const sensorYaw = this.currentOrientation.yaw;     // Physical yaw

    // Swap roll/yaw to correct axis assignment, with direction corrections
    // Roll needs negation for correct left/right tilt direction
    const pitch = ((signs.negatePitch ? -sensorPitch : sensorPitch) + offsets.pitch) * (Math.PI / 180);
    const yaw = (((signs.negateYaw ? 1 : -1) * sensorRoll) + offsets.yaw) * (Math.PI / 180);    // sensor ROLL → RotY (negated)
    const roll = ((signs.negateRoll ? -sensorYaw : sensorYaw) + offsets.roll) * (Math.PI / 180);  // sensor YAW → RotZ

    this.handGroup.rotation.set(
        pitch, // X - finger tilt (from sensor pitch)
        yaw,   // Y - hand tilt L/R (from sensor roll)
        roll,  // Z - hand spin (from sensor yaw)
        "YXZ"
    );
}
```

### 9.2 Axis Mapping Details

**The Axis Swap:**
```javascript
// What comes from sensor:
sensorRoll   // Tilt pinky/thumb up/down
sensorPitch  // Tilt fingers up/down
sensorYaw    // Spin hand clockwise/counterclockwise

// What Three.js needs:
rotation.x   // Pitch (finger tilt)
rotation.y   // Yaw (hand L/R tilt)
rotation.z   // Roll (hand spin)

// Mapping:
rotation.x = -sensorPitch   // Negated for correct direction
rotation.y = -sensorRoll    // Sensor roll → RotY, negated
rotation.z = sensorYaw      // Sensor yaw → RotZ
```

### 9.3 Three.js Rotation Order

**Order: 'YXZ'** (Euler angles applied in order: Y, then X, then Z)

```javascript
// Three.js internally applies rotations as:
// R = Ry(yaw) × Rx(pitch) × Rz(roll)
```

**Intrinsic vs Extrinsic:**
- Three.js Euler with 'YXZ' order = intrinsic YXZ (rotations in local frame)
- Equivalent to extrinsic ZXY (rotations in world frame)

### 9.4 Visual Smoothing

```javascript
function lerpAngleDeg(a, b, t) {
    // Shortest-path interpolation (avoids 360°/0° long-way spins)
    const delta = ((((b - a) % 360) + 540) % 360) - 180;
    return a + delta * t;
}
```

**Purpose:** Smooth orientation changes, prevent discontinuous jumps at angle wrapping

---

## 10. Complete Transform Summary

### 10.1 Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: FIRMWARE SENSOR ACQUISITION                                │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  Physical hand movement                                      │
│ Sensor: LSM6DS3 (accel/gyro) + LIS3MDL (mag)                       │
│ Output: {ax, ay, az, gx, gy, gz, mx, my, mz} in LSB               │
│ Units:  LSB (raw 16-bit sensor counts)                             │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: UNIT CONVERSION                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Accel:  LSB × (1/8192) → g                                         │
│ Gyro:   LSB × (1/114.28) → deg/s                                   │
│ Mag:    LSB × (100/6842) → µT                                      │
│ Output: {ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps,                │
│          mx_ut, my_ut, mz_ut}                                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TELEMETRY PROCESSING                                       │
├─────────────────────────────────────────────────────────────────────┤
│ - Motion detection (for gyro bias calibration)                     │
│ - Gyro bias estimation (when stationary)                           │
│ - Route to IMU fusion                                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MADGWICK AHRS SENSOR FUSION                                │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  {ax_g, ay_g, az_g} (g)                                     │
│         {gx_dps, gy_dps, gz_dps} (deg/s → rad/s)                   │
│         {mx_ut, my_ut, mz_ut} (µT, optional)                       │
│                                                                      │
│ Process:                                                             │
│   1. Gyro integration: q̇ = 0.5 × q ⊗ ω                            │
│   2. Accel correction: gradient descent on gravity error            │
│   3. Mag correction: gradient descent on field direction error      │
│   4. Quaternion normalization                                       │
│                                                                      │
│ Output: q = {w, x, y, z} (unit quaternion)                         │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: EULER ANGLE EXTRACTION                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  q = {w, x, y, z}                                           │
│ Process: ZYX Euler extraction using atan2/asin                      │
│ Output:  {roll, pitch, yaw} in degrees                             │
│          Convention: Sensor body frame, ZYX intrinsic              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: ORIENTATION MODEL MAPPING (optional, not always used)     │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  {roll, pitch, yaw} from AHRS (degrees)                     │
│ Process:                                                             │
│   hand_roll  = sensor_roll + 180°                                  │
│   hand_pitch = -sensor_pitch + 180°                                │
│   hand_yaw   = sensor_yaw - 180°                                   │
│ Output: {roll, pitch, yaw} in hand model frame (degrees)           │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                        ┌───────┴────────┐
                        ↓                ↓
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ STAGE 7A: Hand3DRenderer     │  │ STAGE 7B: ThreeJSHandSkeleton│
├──────────────────────────────┤  ├──────────────────────────────┤
│ Input: AHRS Euler (degrees)  │  │ Input: AHRS Euler (degrees)  │
│                              │  │                              │
│ AXIS MAPPING:                │  │ AXIS MAPPING:                │
│   renderer.pitch = -euler.pitch│ │   rotation.x = -sensorPitch  │
│   renderer.yaw   = -euler.roll │ │   rotation.y = -sensorRoll   │
│   renderer.roll  =  euler.yaw  │ │   rotation.z =  sensorYaw    │
│                              │  │                              │
│ Transform:                   │  │ Transform:                   │
│   1. Base 180° Y flip        │  │   1. Convert to radians      │
│   2. Translate to pivot      │  │   2. Apply low-pass filter   │
│   3. Apply ZYX rotations     │  │   3. Set rotation (YXZ order)│
│   4. Translate back          │  │                              │
│   5. Project to 2D canvas    │  │ Render: Three.js WebGL       │
│                              │  │                              │
│ Render: 2D Canvas projection │  │                              │
└──────────────────────────────┘  └──────────────────────────────┘
```

### 10.2 Unit Progression

| Stage | Units | Example Value |
|-------|-------|---------------|
| Firmware | LSB | ax = 8192 |
| Unit Conversion | g | ax_g = 1.0 |
| AHRS Input | g | ax_g = 1.0 |
| Gyro (pre-AHRS) | deg/s | gx_dps = 10.0 |
| Gyro (in AHRS) | rad/s | gx = 0.17453 |
| AHRS Output | quaternion | q = {1, 0, 0, 0} |
| Euler Angles | degrees | roll = 0° |
| Orientation Map | degrees | hand_roll = 180° |
| Renderer | radians | pitch_rad = π |

### 10.3 Coordinate Frame Transformations

```
Sensor Frame (S)         AHRS Processing          Hand Model Frame (H)
+X → wrist               (Quaternion fusion)      +X → pinky
+Y → fingers       ───────────────────────→       +Y → fingers
+Z → into palm                                    +Z → palm normal

Physical:                Euler Extraction          Visual:
Roll = thumb up          {roll, pitch, yaw}       RotY = hand tilt L/R
Pitch = fingers up       in sensor frame          RotX = fingers up/down
Yaw = spin CW                                     RotZ = hand spin

                         Axis Mapping
                         ───────────────
                         sensor.roll  → renderer.yaw (RotY)
                         sensor.pitch → renderer.pitch (RotX)
                         sensor.yaw   → renderer.roll (RotZ)
```

---

## 11. Simplification Analysis

### 11.1 Current Transform Complexity

#### Transform Count by Stage

| Stage | Transforms | Complexity |
|-------|-----------|------------|
| 1. Firmware | 0 (raw data) | Minimal |
| 2. Unit Conversion | 9 multiplications | Low |
| 3. Telemetry | Data routing | Low |
| 4. Madgwick AHRS | ~100 ops/sample | High |
| 5. Euler Extraction | 6 trig ops | Medium |
| 6. Orientation Model | 3 negations + 3 additions + 3 muls | Low |
| 7A/B. Rendering | Matrix ops or Three.js | Medium |

**Total Complexity:** Moderate to High (dominated by AHRS)

### 11.2 Potential Simplifications

#### 11.2.1 ❌ Skip Unit Conversion

**Proposal:** Pass raw LSB directly to AHRS

**Analysis:**
```javascript
// Current: LSB → physical units → AHRS
ax_g = ax_LSB / 8192
AHRS.update(ax_g, ...)

// Proposed: LSB → AHRS with adjusted parameters
AHRS_scaled.update(ax_LSB, ...)
```

**Pros:**
- Eliminates 9 multiplications per sample
- Reduces memory (no need to store converted values)

**Cons:**
- AHRS logic assumes normalized units (gravity = 1)
- Would require modifying well-tested AHRS algorithm
- Makes debugging harder (can't inspect physical units)
- Breaks separation of concerns (unit conversion tied to algorithm)

**Recommendation:** ❌ **DO NOT IMPLEMENT**
- Savings are negligible (~0.1 µs per sample at 26 Hz)
- Increases code coupling and reduces maintainability
- Loss of physical unit visibility is major debugging impediment

---

#### 11.2.2 ✅ Combine Orientation Model Mapping with Renderer Axis Mapping

**Current Flow:**
```
AHRS Euler → orientation-model.js (negations + offsets)
           → renderer (axis swap + more negations)
```

**Observation:** The orientation-model.js step is NOT always used! Looking at the code:

1. **hand-3d-renderer.js** (line 222): Receives AHRS Euler directly, applies its own mapping
2. **threejs-hand-skeleton.js** (line 341): Receives AHRS Euler directly, applies its own mapping
3. **orientation-model.js**: Exists as a separate module but may not be in the hot path

**Analysis:**
```javascript
// Current (orientation-model.js):
hand_roll  = sensor_roll + 180°
hand_pitch = -sensor_pitch + 180°
hand_yaw   = sensor_yaw - 180°

// Then renderer (hand-3d-renderer.js):
renderer.pitch = -euler.pitch
renderer.yaw   = -euler.roll
renderer.roll  =  euler.yaw

// Combined (simplified):
renderer.pitch = sensor_pitch + 180°  // No double negation!
renderer.yaw   = -sensor_roll + 180°
renderer.roll  = sensor_yaw - 180°
```

**Pros:**
- Eliminates intermediate step if orientation-model.js not used in hot path
- Reduces number of negations (from 4 to 2)
- Single source of truth for axis mapping
- Clearer data flow

**Cons:**
- Only beneficial if orientation-model.js is truly redundant
- Need to verify all usage paths

**Investigation Required:**
```bash
# Check where orientation-model.js mapSensorToHand is called
grep -r "mapSensorToHand" src/web/GAMBIT/
```

**Recommendation:** ✅ **VERIFY AND POTENTIALLY IMPLEMENT**
- If orientation-model.js is not in the critical path, remove it
- Consolidate all mapping logic directly in renderers
- Update documentation to reflect single transform

---

#### 11.2.3 ⚠️ Use Quaternion Directly Instead of Euler Angles

**Proposal:** Skip Euler extraction, use quaternion for rendering

**Current:**
```
Quaternion → Euler (atan2/asin) → degrees → radians → rotation matrix
```

**Proposed:**
```
Quaternion → rotation matrix (direct)
```

**Code Example:**
```javascript
// Current
const euler = ahrs.getEulerAngles();  // 6 trig operations
const pitch_rad = euler.pitch * Math.PI / 180;
const yaw_rad = euler.yaw * Math.PI / 180;
const roll_rad = euler.roll * Math.PI / 180;
// Build rotation matrix from Euler angles (9 trig operations)

// Proposed
const q = ahrs.getQuaternion();
const rotMatrix = quaternionToRotationMatrix(q);  // 9 multiplications, no trig
```

**Quaternion to Rotation Matrix:**
```javascript
function quaternionToRotationMatrix(q) {
    const { w, x, y, z } = q;
    return [
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ];
}
```

**Pros:**
- Eliminates ~15 trig operations per frame
- More numerically stable (no gimbal lock)
- No singularities at ±90° pitch
- More efficient (9 multiplications vs 15 trig ops)

**Cons:**
- Quaternions less intuitive for debugging
- Axis mapping corrections must be redone in quaternion space
- Three.js supports both, but canvas renderer uses matrices
- Major refactor of axis mapping logic

**Performance Comparison:**
```
Euler path:  ~50 µs (trig operations)
Quaternion:  ~5 µs (multiplications only)
Savings:     ~45 µs per frame × 26 Hz = 1.17 ms per second
```

**Recommendation:** ⚠️ **CONSIDER FOR FUTURE OPTIMIZATION**
- Significant performance gain (~10x faster rotation computation)
- Requires re-engineering axis mapping corrections
- Better numerical properties
- Defer until axis mapping is fully stabilized

---

#### 11.2.4 ✅ Unify Two Renderer Implementations

**Observation:** Two separate renderer implementations exist:

1. **hand-3d-renderer.js** - Custom 2D canvas projection
2. **threejs-hand-skeleton.js** - Three.js WebGL rendering

Both duplicate axis mapping logic with slight variations.

**Proposal:** Create shared axis mapping module

```javascript
// shared/hand-orientation-mapper.js
export function mapAHRSToHandOrientation(ahrsEuler) {
    return {
        pitch: -ahrsEuler.pitch,
        yaw: -ahrsEuler.roll,
        roll: ahrsEuler.yaw
    };
}
```

**Then both renderers use:**
```javascript
const handOrientation = mapAHRSToHandOrientation(ahrsEuler);
// Apply to rendering
```

**Pros:**
- Single source of truth for axis mapping
- Easier to maintain and update
- Consistent behavior across renderers
- Reduces code duplication

**Cons:**
- Minimal (just refactoring)

**Recommendation:** ✅ **IMPLEMENT**
- High value, low risk
- Improves maintainability significantly
- Can be done incrementally

---

#### 11.2.5 ❌ Simplify Madgwick AHRS Algorithm

**Proposal:** Replace Madgwick with simpler complementary filter

**Complementary Filter:**
```javascript
roll = alpha * (roll + gx * dt) + (1 - alpha) * accel_roll;
pitch = alpha * (pitch + gy * dt) + (1 - alpha) * accel_pitch;
```

**Pros:**
- Much simpler (~10 lines vs ~200 lines)
- Faster execution (~5 µs vs ~50 µs)
- Easier to understand

**Cons:**
- Less accurate orientation estimation
- No magnetometer support (yaw drifts)
- Accumulates errors over time
- Poor performance in dynamic motion
- Can't handle 9-DOF fusion

**Recommendation:** ❌ **DO NOT IMPLEMENT**
- Madgwick accuracy is essential for hand tracking
- 9-DOF magnetometer fusion needed for absolute yaw
- Performance cost is acceptable (50 µs << 20 ms frame budget)
- Complementary filter suitable only for low-accuracy applications

---

#### 11.2.6 ⚠️ Pre-compute Conversion Factors

**Current:**
```javascript
// Computed every sample
const ax_g = ax_LSB * (1 / 8192);
const gx_dps = gx_LSB * (1 / 114.28);
```

**Proposed:**
```javascript
// Pre-computed constants
const ACCEL_FACTOR = 1 / 8192;        // = 0.0001220703125
const GYRO_FACTOR = 1 / 114.28;       // = 0.008751093
const MAG_FACTOR = 100 / 6842;        // = 0.014616

// Use in hot path
const ax_g = ax_LSB * ACCEL_FACTOR;
const gx_dps = gx_LSB * GYRO_FACTOR;
```

**Analysis:**
- JavaScript engines likely already optimize constant divisions
- Modern JIT compilers (V8, SpiderMonkey) constant-fold these
- Minimal to zero performance gain
- Slightly better code clarity

**Recommendation:** ⚠️ **ALREADY IMPLEMENTED**
- Check current code: sensor-units.js already exports conversionFactor
- If not using constants, switch to them (low effort, no risk)

---

#### 11.2.7 ✅ Remove Dead Code: Orientation Offsets

**Observation:** Both renderers support orientation offsets:

```javascript
// hand-3d-renderer.js
this.orientationOffset = {
    pitch: options.pitchOffset || 0,
    yaw: options.yawOffset || 0,
    roll: options.rollOffset || 0
};

// Then applied:
pitch: -euler.pitch + this.orientationOffset.pitch
```

**Investigation:** Are these offsets ever set to non-zero values?

```bash
# Search for calls to setOrientationOffset
grep -r "setOrientationOffset\|pitchOffset\|yawOffset\|rollOffset" src/web/GAMBIT/
```

**If offsets are always zero:**
- Remove offset parameters
- Simplify arithmetic (one less addition per axis)
- Cleaner code

**If offsets are used:**
- Keep them (they provide calibration capability)

**Recommendation:** ✅ **INVESTIGATE AND POTENTIALLY SIMPLIFY**
- Check if offsets are actually used
- If not, remove them from both renderers
- If yes, document when/why they're needed

---

### 11.3 Summary of Recommendations

| Simplification | Recommendation | Effort | Risk | Value |
|----------------|----------------|--------|------|-------|
| 11.2.1 Skip Unit Conversion | ❌ No | Low | High | Negative |
| 11.2.2 Combine Orientation Mapping | ✅ Yes | Low | Low | High |
| 11.2.3 Use Quaternion Directly | ⚠️ Future | High | Medium | High |
| 11.2.4 Unify Renderer Mapping | ✅ Yes | Low | Low | High |
| 11.2.5 Simplify AHRS | ❌ No | Low | High | Negative |
| 11.2.6 Pre-compute Factors | ⚠️ Check | Low | None | Low |
| 11.2.7 Remove Dead Offset Code | ✅ Investigate | Low | None | Medium |

**Priority Implementation Order:**
1. **11.2.4** - Unify renderer axis mapping (biggest maintainability win)
2. **11.2.7** - Remove unused offset code (if applicable)
3. **11.2.2** - Combine orientation mapping steps (if redundant)
4. **11.2.6** - Ensure constant conversion factors used
5. **11.2.3** - Consider quaternion rendering (future optimization)

**DO NOT Implement:**
- 11.2.1 (Skip unit conversion) - breaks debugging
- 11.2.5 (Simplify AHRS) - degrades accuracy

---

## 12. Verification and Testing

### 12.1 Test Cases for Orientation Mapping

From `orientation-model.js:getValidationTestCases()` (lines 256-358):

| Test Case | Sensor Input | Expected Behavior |
|-----------|--------------|-------------------|
| FLAT_PALM_UP | roll:0, pitch:0, yaw:0 | Palm facing UP, fingers away |
| TIP_FORWARD_30 | pitch:-30 | Fingers point DOWN and forward |
| TIP_BACKWARD_30 | pitch:+30 | Fingers point UP toward ceiling |
| TILT_LEFT_30 | roll:-30 | Pinky side DOWN |
| TILT_RIGHT_30 | roll:+30 | Thumb side DOWN |
| ROTATE_CW_45 | yaw:+45 | Fingers point LEFT |
| ROTATE_CCW_45 | yaw:-45 | Fingers point RIGHT |
| PALM_DOWN | roll:180 | Palm facing DOWN/toward camera |
| FINGERS_UP_90 | pitch:+90 | Fingers point straight UP |

### 12.2 Unit Conversion Validation

```javascript
import { validateRawSensorUnits } from './sensor-units.js';

// Example usage:
const raw = { ax: 8192, ay: 0, az: 0, gx: 1143, gy: 0, gz: 0, mx: 3421, my: 0, mz: 0 };
const validation = validateRawSensorUnits(raw);
console.log(validation);
// {
//   valid: true,
//   warnings: [],
//   accelMaxLsb: 8192,
//   gyroMaxLsb: 1143,
//   magMaxLsb: 3421,
//   magMagnitudeUt: 50.0
// }
```

### 12.3 End-to-End Verification

**Test Procedure:**
1. Place device flat on table, palm up
2. Observe: Hand model should show palm UP
3. Tilt device forward (fingers down)
4. Observe: Hand model fingers should tilt FORWARD
5. Tilt device left (pinky down)
6. Observe: Hand model should tilt LEFT
7. Rotate device clockwise
8. Observe: Hand model should spin CLOCKWISE

---

## 13. References

### 13.1 Source Files

| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| `/src/device/GAMBIT/app.js` | Firmware sensor acquisition | 173-211 |
| `/src/web/GAMBIT/shared/sensor-units.js` | Unit conversion specs | 28-136, 309-329 |
| `/src/web/GAMBIT/shared/sensor-config.js` | Factory functions | 16-59, 100-112 |
| `/src/web/GAMBIT/shared/telemetry-processor.js` | Processing pipeline | 179-412 |
| `/src/web/GAMBIT/filters.js` | Madgwick AHRS | 31-534 |
| `/src/web/GAMBIT/shared/orientation-model.js` | Orientation mapping | 143-193, 256-358 |
| `/src/web/GAMBIT/hand-3d-renderer.js` | 2D canvas renderer | 222-267, 320-364 |
| `/src/web/GAMBIT/shared/threejs-hand-skeleton.js` | Three.js renderer | 399-457 |

### 13.2 External References

1. **LSM6DS3 Datasheet**
   https://www.st.com/resource/en/datasheet/lsm6ds3.pdf
   Accelerometer/Gyroscope specifications

2. **LIS3MDL Datasheet**
   https://www.st.com/resource/en/datasheet/lis3mdl.pdf
   Magnetometer specifications

3. **Madgwick AHRS Paper**
   "An efficient orientation filter for inertial and inertial/magnetic sensor arrays"
   Sebastian O.H. Madgwick (2010)
   https://x-io.co.uk/res/doc/madgwick_internal_report.pdf

4. **Three.js Euler Documentation**
   https://threejs.org/docs/#api/en/math/Euler
   Rotation order conventions

### 13.3 Git History

Recent commits addressing axis mapping:
```
953f4c6 - Merge PR #50: Fix hand orientation mapping (2025-12-14)
b357931 - Negate roll for correct left/right tilt direction
36ac662 - Fix axis mapping in both renderers: swap roll/yaw sources
87c9e60 - Update documentation with V2 axis mapping fix
b86205f - Fix axis mapping: swap roll/pitch and remove incorrect offsets
```

---

## Appendix A: Quick Reference Tables

### A.1 Unit Conversions

| Sensor | Input Unit | Output Unit | Conversion Factor | Formula |
|--------|-----------|-------------|-------------------|---------|
| Accel | LSB | g | 1/8192 = 0.0001220703125 | g = LSB / 8192 |
| Gyro | LSB | deg/s | 1/114.28 = 0.008751093 | dps = LSB / 114.28 |
| Mag | LSB | µT | 100/6842 = 0.014616 | µT = LSB × 100 / 6842 |
| Gyro | deg/s | rad/s | π/180 = 0.017453293 | rad/s = deg/s × π/180 |
| Angle | degrees | radians | π/180 = 0.017453293 | rad = deg × π/180 |
| Angle | radians | degrees | 180/π = 57.29578 | deg = rad × 180/π |

### A.2 Axis Mapping (Current Implementation)

| Physical Motion | AHRS Output | Renderer Variable | Three.js Axis | Visual Result |
|-----------------|-------------|-------------------|---------------|---------------|
| Tilt pinky/thumb up/down | euler.roll | renderer.yaw (negated) | RotY | Hand tilts L/R |
| Tilt fingers up/down | euler.pitch | renderer.pitch (negated) | RotX | Fingers tilt |
| Spin hand CW/CCW | euler.yaw | renderer.roll | RotZ | Hand spins |

### A.3 Configuration Constants

```javascript
// Sensor specifications
ACCEL_SCALE = 8192;           // LSB per g
GYRO_SCALE = 114.28;          // LSB per deg/s
MAG_SCALE_LSB_TO_UT = 0.014616;  // µT per LSB

// AHRS parameters
SAMPLE_FREQ = 26;             // Hz (firmware accelOn rate)
BETA = 0.05;                  // Madgwick filter gain (standard)

// Orientation mapping
negateRoll = false;
negatePitch = true;
negateYaw = false;

rollOffset = 180;             // degrees
pitchOffset = 180;            // degrees
yawOffset = -180;             // degrees

eulerOrder = 'YXZ';           // Three.js rotation order
```

---

## Appendix B: Debugging Checklist

### B.1 Orientation Issues

**Symptom: Hand orientation is inverted**

Check:
1. Axis sign configuration (`negateRoll`, `negatePitch`, `negateYaw`)
2. Axis mapping (roll/pitch/yaw assignment)
3. Rotation order (YXZ vs ZYX vs XYZ)
4. Offset values (should be ±180° or 0°)

**Symptom: Hand spins wrong direction**

Check:
1. Yaw axis mapping (AHRS yaw → renderer roll)
2. Yaw sign (should NOT be negated in current config)
3. Rotation matrix multiplication order

**Symptom: Hand tilts wrong axis**

Check:
1. Roll/yaw swap (AHRS roll → renderer yaw, not renderer roll)
2. Pitch mapping (AHRS pitch → renderer pitch)
3. Physical sensor mounting orientation

### B.2 Unit Conversion Issues

**Symptom: Erratic orientation, doesn't stabilize**

Check:
1. Unit conversions applied correctly (LSB → g, deg/s, µT)
2. Gyro input in correct units (AHRS expects deg/s, converts to rad/s internally)
3. Accelerometer magnitude (should be ~1g when stationary)
4. Magnetometer magnitude (should be 25-65 µT for Earth's field)

**Symptom: Hand drifts over time**

Check:
1. Gyroscope bias calibration enabled
2. Device stationary for initial calibration (~1 second)
3. Magnetometer fusion enabled for yaw stability
4. Hard iron calibration applied (if magnets present)

### B.3 Sensor Data Validation

```javascript
// Check raw sensor values are in expected range
console.log('Accel LSB range: -16384 to +16384');
console.log('Gyro LSB range: -28000 to +28000');
console.log('Mag LSB range: -27368 to +27368');

// Check converted values
console.log('Accel g range: -2 to +2');
console.log('Gyro dps range: -245 to +245');
console.log('Mag µT magnitude: 25-65 (Earth) or higher (interference)');

// Check quaternion
console.log('Quaternion norm:', Math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2));
// Should be very close to 1.0

// Check Euler angles
console.log('Roll range: -180 to +180');
console.log('Pitch range: -90 to +90');
console.log('Yaw range: -180 to +180');
```

---

**End of Document**
