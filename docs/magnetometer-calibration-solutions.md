# Magnetometer Calibration: Implementation Solutions
**Date:** 2025-12-11
**Related:** `magnetometer-calibration-investigation.md`

---

## Overview

This document proposes concrete implementation solutions to address the critical issues identified in the magnetometer calibration system investigation.

Solutions are prioritized by:
1. **Impact** - how much improvement this will provide
2. **Difficulty** - implementation complexity
3. **Dependencies** - what must be fixed first

---

## Solution Priority Matrix

| Issue | Priority | Difficulty | Est. Time | Dependencies |
|-------|----------|-----------|-----------|--------------|
| Fix Python Earth Subtraction | 🔴 **P0** | Low | 30 min | None |
| Debug Real-Time Calibration | 🔴 **P0** | Medium | 1-2 hours | None |
| Add Accelerometer Noise Rejection | 🟡 **P1** | Medium | 2-3 hours | Real-time calibration working |
| Tune Kalman Filter | 🟢 **P2** | Low | 30 min | Above fixes |
| Add Motion Detection | 🟢 **P3** | Medium | 1-2 hours | Above fixes |

---

## P0-1: Fix Python Earth Field Subtraction with Orientation

### Problem
Python `calibration.py` subtracts a static Earth field vector, causing massive noise (up to 466 µT) during device movement.

### Solution
Add orientation-based Earth field rotation to match JavaScript implementation.

### Implementation

**File:** `ml/calibration.py`

Add quaternion rotation utilities:

```python
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: dict with keys 'w', 'x', 'y', 'z' or numpy array [w, x, y, z]

    Returns:
        3x3 rotation matrix as numpy array
    """
    if isinstance(q, dict):
        w, x, y, z = q['w'], q['x'], q['y'], q['z']
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]

    # Rotation matrix from quaternion
    # See: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R
```

Modify `correct()` method to accept orientation:

```python
def correct(self, measurement, orientation=None):
    """
    Apply all calibrations to a magnetometer reading.

    Args:
        measurement: {x, y, z} magnetometer reading
        orientation: Optional orientation quaternion (dict with w, x, y, z keys)
                    If provided, Earth field will be rotated before subtraction

    Returns:
        Corrected {x, y, z} reading
    """
    # Convert to numpy vector
    m = np.array([measurement['x'], measurement['y'], measurement['z']])

    # 1. Remove hard iron offset
    m = m - self.hard_iron_offset

    # 2. Apply soft iron correction
    m = self.soft_iron_matrix @ m

    # 3. Remove earth field (with orientation compensation if available)
    if orientation is not None:
        # Rotate Earth field to current sensor frame
        R = quaternion_to_rotation_matrix(orientation)
        earth_rotated = R @ self.earth_field
        m = m - earth_rotated
    else:
        # Fall back to static subtraction (only valid if orientation unchanged)
        m = m - self.earth_field

    return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}
```

Add convenience method for iron-only correction:

```python
def correct_iron_only(self, measurement):
    """
    Apply only hard and soft iron corrections, no Earth field subtraction.

    Use this when:
    - Orientation is not available
    - You want to see iron-corrected signal before Earth compensation

    Args:
        measurement: {x, y, z} magnetometer reading

    Returns:
        Iron-corrected {x, y, z} reading
    """
    m = np.array([measurement['x'], measurement['y'], measurement['z']])

    # 1. Remove hard iron offset
    m = m - self.hard_iron_offset

    # 2. Apply soft iron correction
    m = self.soft_iron_matrix @ m

    return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}
```

Update `decorate_telemetry_with_calibration()` to use orientation:

```python
def decorate_telemetry_with_calibration(telemetry_data: List[Dict],
                                       calibration: EnvironmentalCalibration,
                                       use_orientation: bool = True) -> List[Dict]:
    """
    Decorate telemetry data with calibrated and fused magnetometer fields.

    IMPORTANT: Preserves raw data, only adds decorated fields.

    Args:
        telemetry_data: List of telemetry dictionaries with mx, my, mz fields
        calibration: EnvironmentalCalibration instance
        use_orientation: If True and orientation fields present, apply orientation-based
                        Earth subtraction. If False, use static subtraction.

    Returns:
        List of telemetry dictionaries with added fields:
        - calibrated_mx/my/mz: Iron corrected only
        - fused_mx/my/mz: Iron + orientation-compensated Earth subtraction
    """
    decorated = []

    for sample in telemetry_data:
        # Create decorated copy
        decorated_sample = sample.copy()

        # Check if we have calibration
        has_iron_cal = (calibration.has_calibration('hard_iron') and
                       calibration.has_calibration('soft_iron'))
        has_earth_cal = calibration.has_calibration('earth_field')

        if has_iron_cal:
            try:
                # Iron correction only (always safe)
                iron_corrected = calibration.correct_iron_only({
                    'x': sample['mx'],
                    'y': sample['my'],
                    'z': sample['mz']
                })
                decorated_sample['calibrated_mx'] = iron_corrected['x']
                decorated_sample['calibrated_my'] = iron_corrected['y']
                decorated_sample['calibrated_mz'] = iron_corrected['z']

                # Fused (iron + Earth subtraction with orientation if available)
                if has_earth_cal:
                    orientation = None
                    if use_orientation and 'orientation_w' in sample:
                        orientation = {
                            'w': sample['orientation_w'],
                            'x': sample['orientation_x'],
                            'y': sample['orientation_y'],
                            'z': sample['orientation_z']
                        }

                    fused = calibration.correct({
                        'x': sample['mx'],
                        'y': sample['my'],
                        'z': sample['mz']
                    }, orientation=orientation)

                    decorated_sample['fused_mx'] = fused['x']
                    decorated_sample['fused_my'] = fused['y']
                    decorated_sample['fused_mz'] = fused['z']

            except Exception as e:
                # Calibration failed, skip decoration
                pass

        decorated.append(decorated_sample)

    return decorated
```

### Testing

Create test script `ml/test_calibration.py`:

```python
#!/usr/bin/env python3
"""Test orientation-based Earth field subtraction."""

import numpy as np
from calibration import EnvironmentalCalibration, quaternion_to_rotation_matrix

def test_static_orientation():
    """Test: Static orientation should give same result as before."""
    cal = EnvironmentalCalibration()
    cal.earth_field = np.array([20, 370, -285])
    cal.calibrations['earth_field'] = True

    measurement = {'x': 100, 'y': 500, 'z': -200}
    identity_quat = {'w': 1, 'x': 0, 'y': 0, 'z': 0}

    corrected = cal.correct(measurement, orientation=identity_quat)

    expected_x = 100 - 20
    expected_y = 500 - 370
    expected_z = -200 - (-285)

    assert abs(corrected['x'] - expected_x) < 0.01
    assert abs(corrected['y'] - expected_y) < 0.01
    assert abs(corrected['z'] - expected_z) < 0.01

    print("✅ Static orientation test passed")

def test_90deg_rotation():
    """Test: 90° rotation should rotate Earth field correctly."""
    cal = EnvironmentalCalibration()
    cal.earth_field = np.array([0, 100, 0])  # Earth field in +Y
    cal.calibrations['earth_field'] = True

    # Quaternion for 90° rotation around Z axis (swaps X and Y)
    q_90z = {'w': 0.707, 'x': 0, 'y': 0, 'z': 0.707}

    # Sensor measures +X (which is Earth field rotated by 90°)
    measurement = {'x': 100, 'y': 0, 'z': 0}

    corrected = cal.correct(measurement, orientation=q_90z)

    # After rotation, Earth field should be in -X direction in sensor frame
    # So measurement.x - rotated_earth.x should be near 0
    assert abs(corrected['x']) < 10  # Should be nearly zero after subtraction

    print("✅ 90° rotation test passed")

def test_no_orientation_fallback():
    """Test: Without orientation, should use static subtraction."""
    cal = EnvironmentalCalibration()
    cal.earth_field = np.array([20, 370, -285])
    cal.calibrations['earth_field'] = True

    measurement = {'x': 100, 'y': 500, 'z': -200}

    corrected = cal.correct(measurement, orientation=None)

    expected_x = 100 - 20
    assert abs(corrected['x'] - expected_x) < 0.01

    print("✅ No orientation fallback test passed")

if __name__ == '__main__':
    test_static_orientation()
    test_90deg_rotation()
    test_no_orientation_fallback()
    print("\n✅ All tests passed!")
```

Run: `python ml/test_calibration.py`

### Expected Improvement
- **Fused signal std dev:** ~220 µT → **< 20 µT** (11x improvement)
- **Noise floor:** 61 µT → **< 10 µT** (6x improvement)
- Enables accurate tracking during dynamic hand movement

---

## P0-2: Debug and Fix Real-Time Calibration Application

### Problem
Real-time JavaScript calibration pipeline not executing during data collection, resulting in no `calibrated_` or `fused_` fields in session data.

### Diagnostic Steps

#### Step 1: Add Debug Logging

**File:** `src/web/GAMBIT/modules/telemetry-handler.js`

Add logging around calibration conditional:

```javascript
// Apply calibration correction (adds calibrated_ fields - iron correction only)
if (deps.calibrationInstance &&
    deps.calibrationInstance.hardIronCalibrated &&
    deps.calibrationInstance.softIronCalibrated) {

    console.debug('[Calibration] Applying iron correction');

    try {
        // Iron correction only (no Earth field subtraction yet)
        const ironCorrected = deps.calibrationInstance.correctIronOnly({
            x: telemetry.mx,
            y: telemetry.my,
            z: telemetry.mz
        });
        decoratedTelemetry.calibrated_mx = ironCorrected.x;
        decoratedTelemetry.calibrated_my = ironCorrected.y;
        decoratedTelemetry.calibrated_mz = ironCorrected.z;

        // Full correction with Earth field subtraction (requires orientation)
        if (deps.calibrationInstance.earthFieldCalibrated && orientation) {
            console.debug('[Calibration] Applying Earth field subtraction');

            // Create Quaternion object for calibration.correct()
            const quatOrientation = new Quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            );
            const fused = deps.calibrationInstance.correct(
                { x: telemetry.mx, y: telemetry.my, z: telemetry.mz },
                quatOrientation
            );
            decoratedTelemetry.fused_mx = fused.x;
            decoratedTelemetry.fused_my = fused.y;
            decoratedTelemetry.fused_mz = fused.z;
        } else {
            console.debug('[Calibration] Skipping Earth subtraction:', {
                earthFieldCalibrated: deps.calibrationInstance.earthFieldCalibrated,
                hasOrientation: !!orientation
            });
        }
    } catch (e) {
        // Calibration failed, skip decoration
        console.error('[Calibration] Correction failed:', e.message);
    }
} else {
    console.debug('[Calibration] Skipping calibration:', {
        hasInstance: !!deps.calibrationInstance,
        hardIronCalibrated: deps.calibrationInstance?.hardIronCalibrated,
        softIronCalibrated: deps.calibrationInstance?.softIronCalibrated
    });
}
```

#### Step 2: Verify Calibration Loading

**File:** Check where `deps.calibrationInstance` is initialized

Search for calibration instance creation:
```bash
grep -r "calibrationInstance.*=" src/web/GAMBIT/
```

Ensure calibration is loaded from file at startup:

```javascript
// Pseudo-code for expected flow
async function initializeCalibration() {
    try {
        const calibrationData = await loadCalibrationFromStorage();
        deps.calibrationInstance = new Calibration();
        deps.calibrationInstance.load(calibrationData);

        console.log('[Calibration] Loaded calibration:', {
            hardIronCalibrated: deps.calibrationInstance.hardIronCalibrated,
            softIronCalibrated: deps.calibrationInstance.softIronCalibrated,
            earthFieldCalibrated: deps.calibrationInstance.earthFieldCalibrated
        });
    } catch (e) {
        console.error('[Calibration] Failed to load calibration:', e);
    }
}
```

#### Step 3: Check Calibration Persistence

Verify that `gambit_calibration.json` is accessible to the web application:

- If running in browser: Check localStorage or IndexedDB
- If running in Electron: Check file system access
- Ensure calibration file path is correct

#### Step 4: Test Calibration Methods Directly

Add unit test in browser console:

```javascript
// Test calibration instance
console.log('Calibration instance:', deps.calibrationInstance);
console.log('Flags:', {
    hardIron: deps.calibrationInstance.hardIronCalibrated,
    softIron: deps.calibrationInstance.softIronCalibrated,
    earthField: deps.calibrationInstance.earthFieldCalibrated
});

// Test correction
const testReading = { x: 93, y: 383, z: -338 };
const testQuat = new Quaternion(1, 0, 0, 0);
const corrected = deps.calibrationInstance.correct(testReading, testQuat);
console.log('Test correction:', corrected);
```

### Likely Root Causes

#### Cause 1: Calibration Not Loaded at Startup
**Symptom:** `deps.calibrationInstance` is undefined or null

**Fix:** Ensure calibration is loaded before telemetry handler starts

```javascript
// In main initialization sequence
async function initializeGAMBIT() {
    // ... other initialization ...

    // Load calibration BEFORE starting telemetry
    await initializeCalibration();

    // Start telemetry handler
    startTelemetryCapture();
}
```

#### Cause 2: Calibration Flags Not Persisted
**Symptom:** `deps.calibrationInstance` exists but flags are false

**Fix:** Ensure calibration.load() sets flags correctly

```javascript
// In calibration.js load() method
load(data) {
    // ... load offsets and matrices ...

    // IMPORTANT: Set flags from loaded data
    this.hardIronCalibrated = data.hardIronCalibrated || false;
    this.softIronCalibrated = data.softIronCalibrated || false;
    this.earthFieldCalibrated = data.earthFieldCalibrated || false;

    console.log('[Calibration] Loaded with flags:', {
        hardIron: this.hardIronCalibrated,
        softIron: this.softIronCalibrated,
        earthField: this.earthFieldCalibrated
    });
}
```

#### Cause 3: Orientation Not Available at Start
**Symptom:** `calibrated_` fields exist but `fused_` fields don't

**Fix:** Already handled by conditional logic, but verify IMU initialization

```javascript
// Ensure IMU initializes quickly
if (!imuInitialized) {
    const accelMag = Math.abs(telemetry.ax) + Math.abs(telemetry.ay) + Math.abs(telemetry.az);
    if (accelMag > 0.5) {  // Lower threshold from 0.5 to 0.3?
        deps.imuFusion.initFromAccelerometer(telemetry.ax, telemetry.ay, telemetry.az);
        imuInitialized = true;
        console.log('[IMU] Initialized from accelerometer');
    }
}
```

### Expected Improvement
- Session data will contain `calibrated_mx/my/mz` and `fused_mx/my/mz` fields
- Visualizations will show all 4 calibration stages
- Enables real-time monitoring of calibration quality

---

## P1: Add Accelerometer Noise Rejection to IMU Fusion

### Problem
Madgwick AHRS treats all accelerometer input as gravity, causing orientation errors during dynamic movement (linear acceleration, vibration).

### Solution
Add accelerometer magnitude check to reject invalid accelerometer readings.

### Implementation

**File:** `src/web/GAMBIT/filters.js`

Modify Madgwick AHRS `update()` method:

```javascript
update(ax, ay, az, gx, gy, gz, dt = null, gyroInDegrees = true) {
    const deltaT = dt || (1.0 / this.sampleFreq);

    // Convert gyroscope to rad/s if needed
    if (gyroInDegrees) {
        gx = gx * Math.PI / 180;
        gy = gy * Math.PI / 180;
        gz = gz * Math.PI / 180;
    }

    // Apply gyroscope bias correction
    gx -= this.gyroBias.x;
    gy -= this.gyroBias.y;
    gz -= this.gyroBias.z;

    let { w: q0, x: q1, y: q2, z: q3 } = this.q;

    // Rate of change of quaternion from gyroscope
    const qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    const qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    const qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    const qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    // Compute feedback only if accelerometer measurement valid
    const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);

    // NEW: Check if accelerometer is measuring primarily gravity
    // Reject if magnitude is far from 1g (indicates linear acceleration/vibration)
    const GRAVITY_NOMINAL = 8192;  // Adjust based on your accelerometer scale
    const GRAVITY_TOLERANCE = 0.3;  // Accept ±30% variation
    const accelMagDeviation = Math.abs(accelNorm - GRAVITY_NOMINAL) / GRAVITY_NOMINAL;

    const accelIsValid = (accelNorm > 0.01) && (accelMagDeviation < GRAVITY_TOLERANCE);

    if (accelIsValid) {
        // Normalize accelerometer
        const recipNorm = 1.0 / accelNorm;
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        // ... rest of gradient descent algorithm ...
        // (unchanged)

    } else {
        // Only integrate gyroscope (ignore accelerometer)
        q0 += qDot1 * deltaT;
        q1 += qDot2 * deltaT;
        q2 += qDot3 * deltaT;
        q3 += qDot4 * deltaT;

        // Optional: Log rejection for debugging
        if (accelNorm > 0.01) {
            console.debug('[IMU] Rejecting accelerometer:', {
                norm: accelNorm,
                nominal: GRAVITY_NOMINAL,
                deviation: (accelMagDeviation * 100).toFixed(1) + '%'
            });
        }
    }

    // Normalize quaternion
    const qNorm = 1.0 / Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    this.q = {
        w: q0 * qNorm,
        x: q1 * qNorm,
        y: q2 * qNorm,
        z: q3 * qNorm
    };

    return this.q;
}
```

### Configuration

Add configuration option:

```javascript
constructor(options = {}) {
    const {
        sampleFreq = 50,
        beta = 0.1,
        accelRejectionThreshold = 0.3  // NEW: Accelerometer validity threshold
    } = options;

    this.sampleFreq = sampleFreq;
    this.beta = beta;
    this.accelRejectionThreshold = accelRejectionThreshold;

    // ... rest of constructor ...
}
```

### Alternative: Adaptive Beta

Instead of rejecting accelerometer, reduce its influence during motion:

```javascript
// Compute adaptive beta based on acceleration magnitude
const accelMagDeviation = Math.abs(accelNorm - GRAVITY_NOMINAL) / GRAVITY_NOMINAL;
const adaptiveBeta = this.beta * Math.exp(-5 * accelMagDeviation);
// When deviation = 0: beta = 0.1
// When deviation = 0.2: beta = 0.037
// When deviation = 0.5: beta = 0.008

// Use adaptiveBeta instead of this.beta in gradient descent
q0 += (qDot1 - adaptiveBeta * s0) * deltaT;
q1 += (qDot2 - adaptiveBeta * s1) * deltaT;
q2 += (qDot3 - adaptiveBeta * s2) * deltaT;
q3 += (qDot4 - adaptiveBeta * s3) * deltaT;
```

### Expected Improvement
- More stable orientation estimates during dynamic movement
- Reduced Earth field subtraction errors
- Better separation of Earth field from finger magnet signals

---

## P2: Tune Kalman Filter Parameters

### Problem
Current parameters (Q=0.1, R=1.0) cause over-smoothing, lag, and drift.

### Solution
Adjust Q/R ratio to balance responsiveness vs. smoothing.

### Implementation

**File:** `src/web/GAMBIT/filters.js:342-345`

Recommended new parameters:

```javascript
const {
    processNoise = 1.0,        // Q - increased from 0.1
    measurementNoise = 1.0,    // R - unchanged
    initialCovariance = 100    // P0 - unchanged
} = options;
```

**Rationale:**
- Q/R ratio: 10:1 → 1:1
- More trust in measurements, less trust in process model
- Faster response to real changes
- Less drift during noisy periods

### Adaptive Approach

Better: Adjust parameters based on signal characteristics:

```javascript
class KalmanFilter3D {
    constructor(options = {}) {
        // ... existing initialization ...

        // Track measurement innovation for adaptive tuning
        this.recentInnovations = [];
        this.adaptiveMode = options.adaptiveMode || false;
    }

    update(measurement, dt = null) {
        // ... existing prediction step ...

        // Compute innovation (measurement - prediction)
        const innovation = {
            x: measurement.x - this.state[0],
            y: measurement.y - this.state[1],
            z: measurement.z - this.state[2]
        };
        const innovationMag = Math.sqrt(
            innovation.x**2 + innovation.y**2 + innovation.z**2
        );

        // Adaptive R: Increase measurement noise during high innovation
        let R_adaptive = this.R;
        if (this.adaptiveMode) {
            this.recentInnovations.push(innovationMag);
            if (this.recentInnovations.length > 10) {
                this.recentInnovations.shift();
            }

            const avgInnovation = this.recentInnovations.reduce((a, b) => a + b, 0) /
                                 this.recentInnovations.length;

            // Scale R based on innovation magnitude
            R_adaptive = this.R * (1 + avgInnovation / 50);
        }

        // ... rest of update step using R_adaptive ...
    }
}
```

### Expected Improvement
- Faster response to real magnetic field changes
- Less drift during noisy periods
- Better tracking of finger magnet signals

---

## P3: Add Motion Detection and Mode Switching

### Problem
Single set of parameters cannot handle both static calibration and dynamic tracking.

### Solution
Detect motion state and switch between parameter sets.

### Implementation

Add motion detector:

```javascript
class MotionDetector {
    constructor(options = {}) {
        this.accelThreshold = options.accelThreshold || 2000;  // LSB units
        this.gyroThreshold = options.gyroThreshold || 500;     // deg/s * 100
        this.windowSize = options.windowSize || 10;            // samples

        this.recentAccel = [];
        this.recentGyro = [];
    }

    update(ax, ay, az, gx, gy, gz) {
        // Compute magnitudes
        const accelMag = Math.sqrt(ax*ax + ay*ay + az*az);
        const gyroMag = Math.sqrt(gx*gx + gy*gy + gz*gz);

        // Add to history
        this.recentAccel.push(accelMag);
        this.recentGyro.push(gyroMag);

        if (this.recentAccel.length > this.windowSize) {
            this.recentAccel.shift();
            this.recentGyro.shift();
        }

        // Compute std dev as motion indicator
        const accelStd = this._std(this.recentAccel);
        const gyroStd = this._std(this.recentGyro);

        // Detect motion
        const isMoving = (accelStd > this.accelThreshold) ||
                        (gyroStd > this.gyroThreshold);

        return {
            isMoving,
            accelStd,
            gyroStd
        };
    }

    _std(arr) {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((a, b) => a + (b - mean)**2, 0) / arr.length;
        return Math.sqrt(variance);
    }
}
```

Use motion state to switch modes:

```javascript
// In telemetry handler
const motionState = motionDetector.update(
    telemetry.ax, telemetry.ay, telemetry.az,
    telemetry.gx, telemetry.gy, telemetry.gz
);

if (motionState.isMoving) {
    // Dynamic mode: Reject accelerometer in IMU fusion
    deps.imuFusion.setAccelRejection(true);

    // Dynamic mode: Higher Kalman filter Q
    deps.magFilter.setProcessNoise(1.0);

} else {
    // Static mode: Trust accelerometer for drift correction
    deps.imuFusion.setAccelRejection(false);

    // Static mode: Lower Kalman filter Q for smoothing
    deps.magFilter.setProcessNoise(0.1);
}

// Add motion state to telemetry
decoratedTelemetry.motion_state = motionState.isMoving ? 1 : 0;
decoratedTelemetry.motion_accel_std = motionState.accelStd;
decoratedTelemetry.motion_gyro_std = motionState.gyroStd;
```

### Expected Improvement
- Optimal performance in both static and dynamic scenarios
- Better calibration data collection (static mode)
- Better tracking during hand movement (dynamic mode)
- Visible motion state in data for debugging

---

## Implementation Plan

### Phase 1: Quick Wins (Day 1)
1. ✅ **Fix Python Earth subtraction** (30 min)
   - Add orientation rotation to `ml/calibration.py`
   - Update `decorate_telemetry_with_calibration()`
   - Test with existing session data

2. ✅ **Debug real-time calibration** (1-2 hours)
   - Add debug logging to telemetry handler
   - Verify calibration loading
   - Fix initialization sequence

3. ✅ **Test end-to-end** (30 min)
   - Collect new session with calibration active
   - Verify `calibrated_` and `fused_` fields present
   - Generate visualization with all 4 stages

### Phase 2: Robustness Improvements (Day 2-3)
4. ✅ **Add accelerometer rejection** (2-3 hours)
   - Implement magnitude check in Madgwick AHRS
   - Test with dynamic movement data
   - Tune rejection threshold

5. ✅ **Tune Kalman parameters** (30 min)
   - Adjust Q/R ratio
   - Test with calibrated data
   - Measure improvement in tracking

### Phase 3: Advanced Features (Day 4-5)
6. ✅ **Add motion detection** (1-2 hours)
   - Implement MotionDetector class
   - Add mode switching logic
   - Test static/dynamic transitions

7. ✅ **Validate improvements** (2-3 hours)
   - Collect test dataset with various movements
   - Measure SNR improvements
   - Compare before/after visualizations

---

## Success Metrics

### Target Performance

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Fused Std Dev | ~220 µT | < 20 µT | **11x better** |
| Noise Floor | 61 µT | < 10 µT | **6x better** |
| SNR (dB) | ~3 dB | > 15 dB | **12 dB gain** |
| Earth Removal | 45% | > 95% | **2x better** |
| Orientation Error | ~10° | < 2° | **5x better** |

### Validation Tests

1. **Static test:** Device still, no magnets
   - Fused magnitude should be < 5 µT
   - All axes std dev < 2 µT

2. **Rotation test:** Device rotated 360°, no magnets
   - Fused magnitude should remain < 10 µT throughout
   - No correlation between orientation and fused magnitude

3. **Magnet test:** Single finger magnet at various distances
   - Clear dipole signal in fused data
   - SNR > 10 dB at 5cm distance

4. **Dynamic test:** Hand movement with all magnets
   - Individual finger magnets distinguishable
   - No Earth field artifacts during motion

---

## Appendix: Code References

### A.1 Quaternion to Rotation Matrix (Python)
```python
def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix."""
    if isinstance(q, dict):
        w, x, y, z = q['w'], q['x'], q['y'], q['z']
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R
```

### A.2 Accelerometer Validity Check (JavaScript)
```javascript
const GRAVITY_NOMINAL = 8192;
const GRAVITY_TOLERANCE = 0.3;
const accelMagDeviation = Math.abs(accelNorm - GRAVITY_NOMINAL) / GRAVITY_NOMINAL;
const accelIsValid = (accelNorm > 0.01) && (accelMagDeviation < GRAVITY_TOLERANCE);
```

### A.3 Motion Detection (JavaScript)
```javascript
const accelStd = this._std(this.recentAccel);
const gyroStd = this._std(this.recentGyro);
const isMoving = (accelStd > ACCEL_THRESHOLD) || (gyroStd > GYRO_THRESHOLD);
```

---

**End of Solutions Document**
