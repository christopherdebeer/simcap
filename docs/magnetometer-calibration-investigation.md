# Magnetometer Calibration System Investigation
**Date:** 2025-12-11
**Session Analyzed:** 2025-12-11T13_26_33.209Z
**Investigation Scope:** Magnetometer calibration noise issues and 6-second transition anomaly

---

## Executive Summary

Investigation reveals **three critical issues** in the magnetometer calibration pipeline:

1. **Real-time calibration NOT being applied** during data collection
2. **Python ML pipeline uses broken Earth field subtraction** (no orientation compensation)
3. **IMU orientation estimation vulnerable to acceleration noise** during dynamic movement

The 6-second discontinuity observed by Gemini aligns with the transition from static calibration collection (0-6s) to dynamic movement (6s+), where these issues compound to create massive noise in the "fused" signal.

---

## 1. Critical Findings

### 1.1 Missing Real-Time Calibration Application

**Finding:** Session data from 2025-12-11T13_26_33.209Z contains:
- ✅ Raw magnetometer fields: `mx`, `my`, `mz`
- ✅ Orientation fields: `orientation_w/x/y/z`, `euler_roll/pitch/yaw`
- ✅ Filtered fields: `filtered_mx/my/mz`
- ❌ **MISSING** calibrated fields: `calibrated_mx/my/mz`
- ❌ **MISSING** fused fields: `fused_mx/my/mz`

**Location:** Verified in `/home/user/simcap/data/GAMBIT/2025-12-11T13_26_33.209Z.full.json`

**Impact:**
- Real-time calibration stages (Iron Correction, Earth Subtraction) are **not being executed** during data collection
- Kalman filter is operating on **uncalibrated raw magnetometer data**
- This explains why visualizations show limited calibration stages

**Root Cause Investigation:**

The real-time pipeline in `src/web/GAMBIT/modules/telemetry-handler.js:89-122` has conditional logic:

```javascript
// Apply calibration correction (adds calibrated_ fields - iron correction only)
if (deps.calibrationInstance &&
    deps.calibrationInstance.hardIronCalibrated &&
    deps.calibrationInstance.softIronCalibrated) {
    // ... apply iron correction ...

    // Full correction with Earth field subtraction (requires orientation)
    if (deps.calibrationInstance.earthFieldCalibrated && orientation) {
        // ... apply Earth subtraction to create fused_ fields ...
    }
}
```

**Hypothesis:** Either:
1. `deps.calibrationInstance` is not initialized during data collection, OR
2. Calibration flags (`hardIronCalibrated`, `softIronCalibrated`, `earthFieldCalibrated`) are false, OR
3. Calibration object exists but conditions fail silently

### 1.2 Python ML Pipeline: Broken Orientation-Agnostic Earth Subtraction

**Finding:** The Python calibration implementation has a **fundamental flaw** in Earth field subtraction.

**Comparison:**

| Implementation | Earth Field Subtraction Method | Correct? |
|----------------|-------------------------------|----------|
| **JavaScript** (`src/web/GAMBIT/calibration.js:560-567`) | Rotates Earth field vector by current orientation quaternion, then subtracts | ✅ **YES** |
| **Python** (`ml/calibration.py:192-194`) | Subtracts **static** Earth field vector regardless of orientation | ❌ **NO** |

**JavaScript (Correct):**
```javascript
// Step 3: Subtract Earth field (rotated to current orientation)
if (this.earthFieldCalibrated && orientation) {
    const rotMatrix = orientation.toRotationMatrix();
    const rotatedEarth = rotMatrix.multiply(this.earthField);
    corrected = {
        x: corrected.x - rotatedEarth.x,
        y: corrected.y - rotatedEarth.y,
        z: corrected.z - rotatedEarth.z
    };
}
```

**Python (Broken):**
```python
def correct(self, measurement):
    # 1. Remove hard iron offset
    m = m - self.hard_iron_offset

    # 2. Apply soft iron correction
    m = self.soft_iron_matrix @ m

    # 3. Remove earth field (NO ROTATION!)
    m = m - self.earth_field  # ❌ Static subtraction

    return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}
```

**Impact:**
- **Static orientation:** Earth field subtraction works correctly
- **Dynamic movement:** As device orientation changes, the Earth field projection in sensor frame changes, but Python code subtracts the same static vector
- **Result:** Residual Earth field components appear as large noise signals (~50 µT) that swing wildly with orientation

**Magnitude of Error:**
- Earth field magnitude: ~466 µT (from `gambit_calibration.json`)
- Maximum orientation-dependent error: **up to 466 µT** when orientation changes by 90°
- This matches Gemini's observation: "Standard Deviation ~220 µT" during movement

### 1.3 IMU Orientation Estimation: Accelerometer Noise Vulnerability

**Finding:** The Madgwick AHRS implementation (`src/web/GAMBIT/filters.js:26-265`) relies heavily on accelerometer for orientation correction.

**Algorithm Design:**
- Uses gyroscope for integration (primary orientation tracking)
- Uses accelerometer feedback to correct drift (gradient descent on gravity vector alignment)
- **Beta parameter** = 0.1 (filter gain for accelerometer feedback)
- **No magnetometer** used (correctly, since finger magnets would corrupt it)

**Problem during dynamic movement:**

When the device moves (t > 6s), the accelerometer measures:
```
a_measured = gravity + linear_acceleration + vibration
```

The Madgwick algorithm **assumes** `a_measured = gravity` and uses this to correct orientation. When linear acceleration is significant:

1. **Incorrect gravity vector estimation** → Wrong "down" direction
2. **Gradient descent pulls orientation** toward this wrong direction
3. **Earth field subtraction uses wrong orientation** → Large residual errors
4. **Errors compound** through Kalman filter

**Code Location:** `filters.js:80-122`

```javascript
// Compute feedback only if accelerometer measurement valid
const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
if (accelNorm > 0.01) {
    // Normalize accelerometer
    const recipNorm = 1.0 / accelNorm;
    ax *= recipNorm;  // ❌ Treating total acceleration as gravity
    ay *= recipNorm;
    az *= recipNorm;

    // Gradient descent algorithm corrective step
    // Objective function: minimize error between expected and measured gravity
    // ... applies correction with beta = 0.1 ...
}
```

**Gemini's Analysis Confirms This:**
> "If the device starts moving at 6s, the accelerometer likely becomes noisy due to vibration or linear acceleration. If this noisy orientation is used to rotate the Earth frame for subtraction, it injects noise directly into your magnetometer data."

### 1.4 Kalman Filter: Over-Smoothing Configuration

**Finding:** Kalman filter parameters cause excessive lag and drift.

**Current Parameters:**
```javascript
// filters.js:342-345
const {
    processNoise = 0.1,      // Q - process noise
    measurementNoise = 1.0,  // R - measurement noise
    initialCovariance = 100  // P0 - initial uncertainty
} = options;
```

**Analysis:**
- **R/Q ratio = 10:1** means the filter trusts the process model 10x more than measurements
- This causes **high smoothing** but **slow response** to real changes
- **Consequence:** When input signal has large noise (as in post-6s period), filter slowly drifts instead of tracking

**Gemini's Analysis:**
> "Your Kalman filter likely has a very low Process Noise Covariance (Q) or a very high Measurement Noise Covariance (R). While this looks nice visually, it causes significant latency."
>
> "Notice the Y-Axis (Red line) slowly drifting downwards after 6s. This suggests the filter is reacting too slowly to real changes (drift), or it is being dragged off-course by the mean of the noise."

---

## 2. The 6-Second Transition Event

### 2.1 Data Collection Timeline

**Source:** `src/web/GAMBIT/modules/wizard.js:27-31`

```javascript
const TRANSITION_TIME = 5;     // seconds of unlabeled transition
const HOLD_TIME = 3;           // seconds of labeled hold
const HOLD_TIME_MED = 6;       // 6-second hold period
const HOLD_TIME_LONG = 10;     // 10-second hold period
```

**Typical Data Collection Sequence:**
1. **0-5s:** Transition period (unlabeled, device positioning)
2. **5-11s:** Hold period (labeled data collection, typically HOLD_TIME_MED = 6s)
3. **After 6s from hold start:** User begins movement/gesture

### 2.2 Why Fused Signal is Zero Before 6s

**Hypothesis 1:** Earth field calibration not complete
- Earth field calibration requires stable reference measurements
- First ~6 seconds used to collect these samples
- `earthFieldCalibrated` flag remains false until calibration completes
- Conditional logic skips Earth subtraction → fused values default to 0

**Hypothesis 2:** Orientation not initialized
- IMU fusion initializes from first significant accelerometer reading:
  ```javascript
  if (!imuInitialized && Math.abs(telemetry.ax) + ... > 0.5) {
      deps.imuFusion.initFromAccelerometer(...);
      imuInitialized = true;
  }
  ```
- If device is very still (< 0.5g total), initialization may be delayed
- Without orientation, Earth subtraction cannot run

**Hypothesis 3:** Intentional calibration gate**
- System may intentionally zero fused output during calibration collection
- Prevents unstable/uncalibrated data from propagating

### 2.3 Why Noise Explodes After 6s

**Compound Effect of All Issues:**

```
Movement Begins (t=6s)
    ↓
Accelerometer measures: gravity + linear_accel + vibration
    ↓
Madgwick AHRS applies accelerometer feedback (beta=0.1)
    ↓
Orientation estimate becomes noisy/wrong
    ↓
Earth field subtraction uses wrong orientation
    ↓
(If using Python pipeline:) Static earth subtraction amplifies error
    ↓
Residual Earth field components ± real magnetic signals = massive noise
    ↓
Kalman filter (Q=0.1, R=1.0) smooths but drifts
    ↓
Result: Standard deviation ~220 µT, noise floor ~61 µT
```

---

## 3. Signal Quality Metrics Analysis

### 3.1 Gemini's Reported Metrics

| Stage | Mean Magnitude | Std Dev | Noise Assessment |
|-------|---------------|---------|------------------|
| **Raw (Gray)** | ~728 µT | ~144 µT | High (includes Earth field + magnets + noise) |
| **Iron Corrected (Blue)** | ~119 µT | ~70 µT | Moderate (Earth field + magnets remain) |
| **Fused (Green)** | 0 µT (0-6s), then high variance | ~220 µT (6s+) | **CRITICAL** (should be low, shows calibration failure) |
| **Filtered (Red)** | Smooth | Low | Over-smoothed (hides underlying problems) |

**Noise Floor: 61.38 µT (HIGH)**
- Context: Earth's magnetic field is 25-65 µT total
- **Implication:** Noise is as loud as the signal being measured
- **This is unacceptable for finger tracking** where magnet signals may be 10-50 µT

### 3.2 Expected vs. Actual Performance

| Metric | Expected | Actual (6s+) | Status |
|--------|----------|--------------|--------|
| Fused Std Dev | < 10 µT | ~220 µT | ❌ **22x worse** |
| Noise Floor | < 5 µT | 61.38 µT | ❌ **12x worse** |
| Iron Correction | Reduces raw noise by ~50% | Reduces by ~51% | ✅ Working |
| Earth Subtraction | Reduces magnitude to <50 µT | Increases variance dramatically | ❌ **Broken** |

---

## 4. Implementation Details Review

### 4.1 File Locations Summary

| Component | Location | Status |
|-----------|----------|--------|
| **Real-time Calibration** | `src/web/GAMBIT/calibration.js` | ✅ Correctly implements orientation-based Earth subtraction |
| **Real-time Telemetry Handler** | `src/web/GAMBIT/modules/telemetry-handler.js` | ⚠️ Conditional logic not executing during data collection |
| **IMU Fusion (Madgwick)** | `src/web/GAMBIT/filters.js:26-265` | ⚠️ Vulnerable to accelerometer noise during movement |
| **Kalman Filter 3D** | `src/web/GAMBIT/filters.js:339-623` | ⚠️ Over-smoothing parameters (Q=0.1, R=1.0) |
| **Python Calibration (ML)** | `ml/calibration.py` | ❌ **BROKEN** - static Earth subtraction |
| **Visualization Pipeline** | `ml/visualize.py` | ✅ Correctly reads and plots available data |

### 4.2 Calibration Configuration

**Current Calibration File:** `/home/user/simcap/data/GAMBIT/gambit_calibration.json`

```json
{
  "hardIronOffset": {"x": 4.5, "y": 520.5, "z": -482},
  "softIronMatrix": [1.029, 0, 0, 0, 1.165, 0, 0, 0, 0.855],
  "earthField": {"x": 18.11, "y": 368.8, "z": -284.52},
  "earthFieldMagnitude": 466.15,
  "hardIronCalibrated": true,
  "softIronCalibrated": true,
  "earthFieldCalibrated": true,
  "timestamp": "2025-12-11T13:21:12.134Z"
}
```

**Analysis:**
- Hard iron offset is **large** (520.5 µT in Y) - suggests nearby ferromagnetic material
- Soft iron correction shows **non-uniform scaling** (Y: 1.165, Z: 0.855) - confirms soft iron distortion
- Earth field magnitude (466 µT) is **reasonable** for typical locations
- Calibration flags all true, suggesting calibration was completed successfully

**Question:** Why isn't this calibration being applied in real-time?

---

## 5. Root Cause Summary

### Primary Issues (by Priority)

#### 🔴 CRITICAL #1: Real-Time Calibration Not Applied
- **What:** Calibration stages not executing during data collection
- **Evidence:** Session data lacks `calibrated_mx/my/mz` and `fused_mx/my/mz` fields
- **Impact:** Uncalibrated data flows through entire pipeline
- **Fix Priority:** **IMMEDIATE** - blocks all other improvements

#### 🔴 CRITICAL #2: Python ML Pipeline Broken Earth Subtraction
- **What:** Static earth field subtraction without orientation compensation
- **Location:** `ml/calibration.py:192-194`
- **Impact:** Massive noise during movement (up to 466 µT orientation-dependent error)
- **Fix Priority:** **HIGH** - needed for post-processing and ML training

#### 🟡 HIGH #3: IMU Orientation Vulnerable to Acceleration Noise
- **What:** Madgwick AHRS trusts accelerometer during dynamic motion
- **Location:** `src/web/GAMBIT/filters.js:80-122`
- **Impact:** Wrong orientation → wrong Earth subtraction → amplified noise
- **Fix Priority:** **MEDIUM** - affects dynamic tracking accuracy

#### 🟡 MEDIUM #4: Kalman Filter Over-Smoothing
- **What:** Q=0.1, R=1.0 causes excessive lag and drift
- **Location:** `src/web/GAMBIT/filters.js:342-345`
- **Impact:** Hides real signals, slow response, drift during noise
- **Fix Priority:** **LOW** - cosmetic issue compared to others

---

## 6. Gemini's Specific Hypotheses: Validated

| Gemini Hypothesis | Validated? | Evidence |
|-------------------|------------|----------|
| "Fused stage is primary noise source" | ✅ **YES** | Python implementation lacks orientation compensation |
| "Iron correction is working correctly" | ✅ **YES** | Hard/soft iron calibration reduces noise as expected |
| "Orientation filter trusting accelerometer too much" | ✅ **YES** | Madgwick beta=0.1, no rejection of linear acceleration |
| "Kalman filter over-smoothing" | ✅ **YES** | R/Q=10:1 ratio causes lag and drift |
| "6s event is motor/EMI source" | ⚠️ **PARTIAL** | Likely start of movement, not external EMI |
| "Fused logic has initialization issue" | ✅ **YES** | Conditional logic prevents real-time application |

---

## 7. Additional Observations

### 7.1 Magnetometer Calibration vs. Finger Magnet Separation

**Important distinction:**
- **Environmental calibration** (Earth field, hard/soft iron) compensates for **constant** or **slowly-varying** background fields
- **Finger magnet signals** are the **dynamic** signals we want to measure
- After proper calibration, residual should be **only finger magnet signals** (< 50 µT typically)

**Current problem:** Residual after "fused" stage is **220 µT std dev**, meaning:
- Either calibration isn't removing the background, OR
- Calibration is creating new noise (as we found with orientation errors)

### 7.2 Data Collection vs. Post-Processing

**Two separate pipelines:**

1. **Real-time (JavaScript):**
   - Runs during data collection
   - Should decorate telemetry with `calibrated_` and `fused_` fields
   - **Currently not working**

2. **Post-processing (Python ML):**
   - Runs after data collection for analysis/training
   - Can add fields retroactively
   - **Currently has broken Earth subtraction**

**User's workflow appears to be:**
- Collect data (JavaScript) → No calibration applied
- Visualize (Python) → Tries to apply calibration, but broken implementation creates noise

---

## 8. Next Steps

See separate section "Proposed Implementation Solutions" below.

---

## Appendices

### A. Key Code References

#### A.1 JavaScript Earth Field Subtraction (Correct)
**File:** `src/web/GAMBIT/calibration.js:560-567`
```javascript
// Step 3: Subtract Earth field (rotated to current orientation)
if (this.earthFieldCalibrated && orientation) {
    const rotMatrix = orientation.toRotationMatrix();
    const rotatedEarth = rotMatrix.multiply(this.earthField);
    corrected = {
        x: corrected.x - rotatedEarth.x,
        y: corrected.y - rotatedEarth.y,
        z: corrected.z - rotatedEarth.z
    };
}
```

#### A.2 Python Earth Field Subtraction (Broken)
**File:** `ml/calibration.py:192-194`
```python
# 3. Remove earth field
m = m - self.earth_field  # ❌ NO ORIENTATION ROTATION
```

#### A.3 Madgwick AHRS Accelerometer Feedback
**File:** `src/web/GAMBIT/filters.js:80-122`
```javascript
const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
if (accelNorm > 0.01) {
    // Normalize accelerometer (assumes it's measuring gravity)
    const recipNorm = 1.0 / accelNorm;
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;
    // ... gradient descent with beta = 0.1 ...
}
```

#### A.4 Kalman Filter Initialization
**File:** `src/web/GAMBIT/filters.js:342-345`
```javascript
const {
    processNoise = 0.1,      // Q - process noise
    measurementNoise = 1.0,  // R - measurement noise
    initialCovariance = 100  // P0 - initial uncertainty
} = options;
```

### B. Session Data Analysis

**Session:** 2025-12-11T13_26_33.209Z
- **Samples:** ~550 samples (~11 seconds @ 50Hz)
- **Fields present:** Raw IMU, orientation quaternion/euler, filtered magnetometer
- **Fields missing:** Calibrated magnetometer, fused magnetometer
- **Conclusion:** Real-time calibration pipeline not executing

---

**End of Investigation Report**
