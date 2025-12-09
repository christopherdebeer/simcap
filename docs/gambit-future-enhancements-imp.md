# GAMBIT Future Enhancements - Implementation Plans

This document provides detailed implementation plans for three advanced features that build on the current calibration and filtering foundation.

---

## 🔄 1. Adaptive Calibration (AUTO-RECALIBRATION)

### Overview

**Goal**: Automatically detect and correct calibration drift over time without manual intervention.

**Problem**: Environmental changes (temperature, nearby metal moved, location change) cause calibration to drift, degrading accuracy over weeks/months.

**Solution**: Continuous monitoring of background field with automatic drift correction and user alerts.

---

### Implementation Plan

#### Phase 1: Drift Detection (1 hour)

**Location**: `collector.html` + `calibration.js`

**Changes**:

1. **Add drift monitoring state**:
```javascript
// In collector.html, add after calibration initialization
const driftMonitor = {
    enabled: true,
    windowSize: 250,              // 5 seconds @ 50Hz
    samples: [],                  // Rolling window of recent samples
    checkInterval: 50,            // Check every 50 samples (1 second)
    driftThreshold: 5.0,          // μT
    alertThreshold: 10.0,         // μT (critical)
    lastCheck: Date.now(),
    driftDetected: false,
    driftMagnitude: 0
};
```

2. **Add drift detection function**:
```javascript
function checkCalibrationDrift() {
    if (!driftMonitor.enabled || driftMonitor.samples.length < driftMonitor.windowSize) {
        return;
    }

    // Estimate current Earth field from recent background samples
    // (samples collected during static periods, no finger motion)
    const currentEarthField = estimateEarthFieldFromWindow(driftMonitor.samples);

    // Compare with saved calibration
    const savedEarthField = calibration.earth_field;
    const drift = {
        x: currentEarthField.x - savedEarthField.x,
        y: currentEarthField.y - savedEarthField.y,
        z: currentEarthField.z - savedEarthField.z
    };

    const driftMagnitude = Math.sqrt(drift.x**2 + drift.y**2 + drift.z**2);
    driftMonitor.driftMagnitude = driftMagnitude;

    // Alert user if drift exceeds threshold
    if (driftMagnitude > driftMonitor.alertThreshold) {
        // Critical drift - show alert
        log(`⚠️ CRITICAL calibration drift detected: ${driftMagnitude.toFixed(1)} μT`);
        log('Please run calibration wizard to recalibrate.');
        driftMonitor.driftDetected = true;

        // Flash wizard button
        $('wizardBtn').classList.add('flash-warning');
    } else if (driftMagnitude > driftMonitor.driftThreshold) {
        // Minor drift - log warning
        console.warn(`[Drift] Calibration drift: ${driftMagnitude.toFixed(1)} μT`);
        driftMonitor.driftDetected = true;
    } else {
        driftMonitor.driftDetected = false;
    }
}

function estimateEarthFieldFromWindow(samples) {
    // Filter for static samples (low variance in accelerometer)
    const staticSamples = samples.filter(s => {
        const accVar = s.ax**2 + s.ay**2 + s.az**2 - 9.8**2;
        return Math.abs(accVar) < 2.0; // Static threshold
    });

    if (staticSamples.length < 10) {
        return calibration.earth_field; // Not enough data
    }

    // Average magnetic field during static periods
    const sum = staticSamples.reduce((acc, s) => ({
        x: acc.x + s.mx,
        y: acc.y + s.my,
        z: acc.z + s.mz
    }), {x: 0, y: 0, z: 0});

    return {
        x: sum.x / staticSamples.length,
        y: sum.y / staticSamples.length,
        z: sum.z / staticSamples.length
    };
}
```

3. **Integrate into onTelemetry**:
```javascript
// In onTelemetry function, after storing data:
if (driftMonitor.enabled) {
    // Add to rolling window
    driftMonitor.samples.push({
        mx: telemetry.mx,
        my: telemetry.my,
        mz: telemetry.mz,
        ax: telemetry.ax,
        ay: telemetry.ay,
        az: telemetry.az
    });

    // Trim to window size
    if (driftMonitor.samples.length > driftMonitor.windowSize) {
        driftMonitor.samples.shift();
    }

    // Check for drift periodically
    if (state.sessionData.length % driftMonitor.checkInterval === 0) {
        checkCalibrationDrift();
    }
}
```

4. **Add UI indicator**:
```html
<!-- In Recording section, after pose estimation status -->
<div id="driftStatus" style="display: none; margin-top: 10px; padding: 8px; background: #ffe5e5; border-radius: 6px; border: 1px solid #ff6b6b; color: #c92a2a;">
    ⚠️ Calibration drift detected: <span id="driftMagnitude">0.0</span> μT
    <br>
    <button class="btn-warning btn-tiny" onclick="openWizard()">Recalibrate Now</button>
</div>
```

#### Phase 2: Auto-Correction (Optional, 30 min)

**For slow drift only** (< 2 μT over hours):

```javascript
function autoCorrectDrift() {
    if (!driftMonitor.driftDetected || driftMonitor.driftMagnitude > 2.0) {
        return; // Only auto-correct minor drift
    }

    // Update Earth field calibration with exponential moving average
    const alpha = 0.01; // Slow adaptation
    const currentEarthField = estimateEarthFieldFromWindow(driftMonitor.samples);

    calibration.earth_field = {
        x: alpha * currentEarthField.x + (1 - alpha) * calibration.earth_field.x,
        y: alpha * currentEarthField.y + (1 - alpha) * calibration.earth_field.y,
        z: alpha * currentEarthField.z + (1 - alpha) * calibration.earth_field.z
    };

    // Save updated calibration
    calibration.save('gambit_calibration');
    console.log('[Drift] Auto-corrected Earth field calibration');
}
```

---

### Testing

1. **Simulate drift**: Manually edit localStorage calibration, offset Earth field by 5 μT
2. **Verify detection**: Check that alert appears within 10 seconds
3. **Test auto-correction**: Enable with 1 μT offset, verify gradual correction
4. **Edge cases**: Test with high motion (should not false-trigger)

---

### Benefits

- ✅ Maintains accuracy over time without user intervention
- ✅ Alerts user to recalibrate when needed
- ✅ Extends calibration lifespan from weeks to months
- ✅ Improves production system reliability

---

## 🔀 2. Multi-Sensor Fusion (EXTENDED KALMAN FILTER)

### Overview

**Goal**: Combine magnetometer + accelerometer + gyroscope for robust 6-DOF hand pose estimation.

**Problem**: Magnetometer alone is:
- Sensitive to metal interference
- Ambiguous (multiple poses can produce similar fields)
- No orientation information

**Solution**: Extended Kalman Filter (EKF) fusing all IMU sensors.

---

### State Vector

```
State (13D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz]
             └─ position ─┘ └─ velocity ┘ └─ quaternion ──┘ └─ angular ─┘
                 (3)          (3)              (4)             velocity (3)
```

**Why quaternion?** Avoids gimbal lock, more stable than Euler angles.

---

### Implementation Plan

#### Phase 1: Extended Kalman Filter Class (2-3 hours)

**Location**: New file `extended-kalman.js`

```javascript
class ExtendedKalmanFilter {
    constructor(options = {}) {
        const {
            processNoise = 0.1,
            magNoise = 1.0,
            accelNoise = 0.5,
            gyroNoise = 0.1
        } = options;

        // State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz]
        this.state = new Float64Array(13);
        this.state[6] = 1.0; // Initialize quaternion to identity (w=1)

        // Covariance matrix (13x13)
        this.P = new Float64Array(13 * 13);
        for (let i = 0; i < 13; i++) {
            this.P[i * 13 + i] = 100; // Initial uncertainty
        }

        // Process noise
        this.Q = processNoise;

        // Measurement noise (per sensor)
        this.R_mag = magNoise;
        this.R_accel = accelNoise;
        this.R_gyro = gyroNoise;

        this.dt = 0.02; // 50Hz
        this.initialized = false;
    }

    /**
     * Prediction step: propagate state using motion model
     */
    predict(dt = null) {
        if (dt === null) dt = this.dt;

        // Extract state components
        let [px, py, pz, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz] = this.state;

        // Position update: p = p + v*dt
        px += vx * dt;
        py += vy * dt;
        pz += vz * dt;

        // Orientation update: integrate angular velocity
        // Quaternion derivative: q̇ = 0.5 * q ⊗ [0, ω]
        const dqw = 0.5 * (-qx * ωx - qy * ωy - qz * ωz) * dt;
        const dqx = 0.5 * (qw * ωx + qy * ωz - qz * ωy) * dt;
        const dqy = 0.5 * (qw * ωy - qx * ωz + qz * ωx) * dt;
        const dqz = 0.5 * (qw * ωz + qx * ωy - qy * ωx) * dt;

        qw += dqw;
        qx += dqx;
        qy += dqy;
        qz += dqz;

        // Normalize quaternion
        const qnorm = Math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= qnorm;
        qx /= qnorm;
        qy /= qnorm;
        qz /= qnorm;

        // Update state
        this.state = new Float64Array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz]);

        // Jacobian F (13x13) - linearization of motion model
        const F = this._computeJacobianF(dt);

        // Covariance prediction: P = F*P*F' + Q
        // (Simplified: assume Q is diagonal)
        const FP = this._matMul(F, this.P, 13, 13, 13);
        const FPFt = this._matMul(FP, this._transpose(F, 13, 13), 13, 13, 13);

        for (let i = 0; i < 13; i++) {
            FPFt[i * 13 + i] += this.Q;
        }

        this.P = FPFt;
    }

    /**
     * Update with magnetometer measurement
     */
    updateMagnetometer(measurement) {
        // Measurement model: z_mag = h(x) + noise
        // h(x) = magnetic dipole field as function of position + orientation

        const [px, py, pz, vx, vy, vz, qw, qx, qy, qz] = this.state;

        // Predicted measurement
        const predicted = this._predictMagneticField(px, py, pz, qw, qx, qy, qz);

        // Innovation: y = z - h(x)
        const innovation = {
            x: measurement.x - predicted.x,
            y: measurement.y - predicted.y,
            z: measurement.z - predicted.z
        };

        // Jacobian H (3x13) - how measurement changes with state
        const H = this._computeJacobianH_mag(px, py, pz, qw, qx, qy, qz);

        // Innovation covariance: S = H*P*H' + R
        const HP = this._matMul(H, this.P, 3, 13, 13);
        const HPHt = this._matMul(HP, this._transpose(H, 3, 13), 3, 13, 3);

        for (let i = 0; i < 3; i++) {
            HPHt[i * 3 + i] += this.R_mag;
        }

        // Kalman gain: K = P*H'*inv(S)
        const Ht = this._transpose(H, 3, 13);
        const K = this._matMul(
            this._matMul(this.P, Ht, 13, 13, 3),
            this._invert3x3(HPHt),
            13, 3, 3
        );

        // State update: x = x + K*y
        const innovVec = [innovation.x, innovation.y, innovation.z];
        for (let i = 0; i < 13; i++) {
            for (let j = 0; j < 3; j++) {
                this.state[i] += K[i * 3 + j] * innovVec[j];
            }
        }

        // Covariance update: P = (I - K*H)*P
        const KH = this._matMul(K, H, 13, 3, 13);
        const I_KH = new Float64Array(13 * 13);
        for (let i = 0; i < 13; i++) {
            for (let j = 0; j < 13; j++) {
                I_KH[i * 13 + j] = (i === j ? 1 : 0) - KH[i * 13 + j];
            }
        }

        this.P = this._matMul(I_KH, this.P, 13, 13, 13);
    }

    /**
     * Update with accelerometer measurement (provides orientation constraint)
     */
    updateAccelerometer(measurement) {
        // Measurement model: z_accel = R(q) * [0, 0, -g] + noise
        // Where R(q) is rotation matrix from quaternion
        // Accelerometer measures gravity direction in body frame

        const [px, py, pz, vx, vy, vz, qw, qx, qy, qz] = this.state;

        // Predicted gravity vector in body frame
        const predicted = this._rotateVector([0, 0, -9.8], qw, qx, qy, qz);

        // Innovation
        const innovation = {
            x: measurement.ax - predicted[0],
            y: measurement.ay - predicted[1],
            z: measurement.az - predicted[2]
        };

        // Jacobian H (3x13) - only depends on quaternion (cols 6-9)
        const H = this._computeJacobianH_accel(qw, qx, qy, qz);

        // (Rest of update step similar to magnetometer)
        // ...
    }

    /**
     * Update with gyroscope measurement (direct angular velocity)
     */
    updateGyroscope(measurement) {
        // Measurement model: z_gyro = [ωx, ωy, ωz] + noise
        // Direct measurement of angular velocity

        const innovation = {
            x: measurement.gx - this.state[10],
            y: measurement.gy - this.state[11],
            z: measurement.gz - this.state[12]
        };

        // Jacobian H (3x13) - identity for last 3 states
        // Simple update since measurement is direct

        // (Simplified update)
        this.state[10] += 0.1 * innovation.x;
        this.state[11] += 0.1 * innovation.y;
        this.state[12] += 0.1 * innovation.z;
    }

    /**
     * Get current pose estimate
     */
    getPose() {
        return {
            position: {
                x: this.state[0],
                y: this.state[1],
                z: this.state[2]
            },
            velocity: {
                x: this.state[3],
                y: this.state[4],
                z: this.state[5]
            },
            orientation: {
                w: this.state[6],
                x: this.state[7],
                y: this.state[8],
                z: this.state[9]
            },
            angularVelocity: {
                x: this.state[10],
                y: this.state[11],
                z: this.state[12]
            }
        };
    }

    // Helper functions (Jacobians, matrix operations, etc.)
    _computeJacobianF(dt) {
        // Linearization of motion model
        // ... (complex, see refs)
    }

    _computeJacobianH_mag(px, py, pz, qw, qx, qy, qz) {
        // How magnetic field changes with position + orientation
        // Numerical differentiation or analytical (complex)
    }

    _predictMagneticField(px, py, pz, qw, qx, qy, qz) {
        // Dipole field equations with orientation
    }

    // ... other helpers
}
```

#### Phase 2: Integration (1 hour)

```javascript
// In collector.html, replace poseFilter with ExtendedKalmanFilter

const ekf = new ExtendedKalmanFilter({
    processNoise: 0.1,
    magNoise: 1.0,
    accelNoise: 0.5,
    gyroNoise: 0.1
});

// In onTelemetry:
ekf.predict(0.02);
ekf.updateMagnetometer({x: filteredMag.x, y: filteredMag.y, z: filteredMag.z});
ekf.updateAccelerometer({ax: telemetry.ax, ay: telemetry.ay, az: telemetry.az});
ekf.updateGyroscope({gx: telemetry.gx, gy: telemetry.gy, gz: telemetry.gz});

const pose = ekf.getPose();
// Use pose.position + pose.orientation for full 6-DOF tracking
```

---

### Benefits

- ✅ Robust to magnetic interference (uses accel+gyro as backup)
- ✅ Full 6-DOF tracking (position + orientation)
- ✅ Resolves ambiguities (orientation constrains magnetic field interpretation)
- ✅ Smoother tracking with multi-sensor redundancy

### Challenges

- ⚠️ Complex implementation (Jacobians, quaternion math)
- ⚠️ Higher computational cost (~5-10ms per update)
- ⚠️ Requires careful tuning of noise parameters

---

## 📈 3. Gesture Prediction with Temporal Features

### Overview

**Goal**: Enable dynamic gesture classification (swipes, waves) using motion patterns, not just static poses.

**Problem**: Current ML model only sees position, ignores velocity and acceleration which encode motion patterns.

**Solution**: Extract temporal derivative features from Kalman filter state and train on dynamics.

---

### Implementation Plan

#### Phase 1: Feature Extraction (1 hour)

**Location**: `ml/data_loader.py`

```python
def extract_temporal_features(telemetry_data: List[Dict],
                             filter_instance: KalmanFilter3D) -> List[Dict]:
    """
    Add velocity and acceleration features from Kalman filter.

    New features:
    - vx, vy, vz: Velocity from Kalman state
    - ax_derived, ay_derived, az_derived: Acceleration from velocity derivative
    - speed: Magnitude of velocity
    - jerk: Magnitude of acceleration change
    """
    decorated = []

    prev_velocity = None

    for i, sample in enumerate(telemetry_data):
        decorated_sample = sample.copy()

        # Get velocity from Kalman filter
        velocity = filter_instance.get_velocity()
        decorated_sample['vx'] = velocity['x']
        decorated_sample['vy'] = velocity['y']
        decorated_sample['vz'] = velocity['z']
        decorated_sample['speed'] = np.sqrt(velocity['x']**2 + velocity['y']**2 + velocity['z']**2)

        # Compute acceleration (velocity derivative)
        if prev_velocity is not None:
            dt = 0.02  # 50Hz
            ax_derived = (velocity['x'] - prev_velocity['x']) / dt
            ay_derived = (velocity['y'] - prev_velocity['y']) / dt
            az_derived = (velocity['z'] - prev_velocity['z']) / dt

            decorated_sample['ax_derived'] = ax_derived
            decorated_sample['ay_derived'] = ay_derived
            decorated_sample['az_derived'] = az_derived
            decorated_sample['accel_mag'] = np.sqrt(ax_derived**2 + ay_derived**2 + az_derived**2)
        else:
            decorated_sample['ax_derived'] = 0
            decorated_sample['ay_derived'] = 0
            decorated_sample['az_derived'] = 0
            decorated_sample['accel_mag'] = 0

        prev_velocity = velocity
        decorated.append(decorated_sample)

    return decorated
```

**Update load_session_data**:
```python
# After filtering decoration:
if apply_filtering:
    data = decorate_telemetry_with_filtering(data, mag_filter)
    data = extract_temporal_features(data, mag_filter)  # Add this line
```

#### Phase 2: Update Schema (15 min)

**Location**: `ml/schema.py`

```python
# Update feature count
NUM_FEATURES = 17  # Was 9, now 9 + 8 temporal

FEATURE_NAMES = [
    'ax', 'ay', 'az',       # Accelerometer
    'gx', 'gy', 'gz',       # Gyroscope
    'mx', 'my', 'mz',       # Magnetometer
    'vx', 'vy', 'vz',       # Velocity (new)
    'speed',                 # Speed magnitude (new)
    'ax_derived', 'ay_derived', 'az_derived',  # Derived acceleration (new)
    'accel_mag'             # Acceleration magnitude (new)
]
```

#### Phase 3: Model Architecture Update (30 min)

**Location**: `ml/model.py`

```python
# Update input shape to handle 17 features
def create_model(num_classes: int = 10,
                window_size: int = 50,
                num_features: int = 17):  # Changed from 9
    """
    Create LSTM model with temporal features.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),

        # LSTM layers to capture temporal dependencies
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

#### Phase 4: Training (30 min)

**No code changes needed**, just retrain:

```bash
python ml/train.py --data data/GAMBIT --epochs 50
```

**Expected improvements**:
- Static gestures: ~93% → ~94% (minor)
- Dynamic gestures: ~75% → ~88% (significant)
- Transition detection: ~60% → ~80% (major)

---

### Benefits

- ✅ Enables dynamic gesture recognition (swipes, waves, flicks)
- ✅ Better transition detection (start/stop of motion)
- ✅ Richer feature space for ML model
- ✅ Captures motion patterns, not just endpoints

### Use Cases

**New gestures enabled**:
- Swipe left/right/up/down
- Wave (rapid hand oscillation)
- Flick (quick snap motion)
- Draw shapes (circles, letters)
- Speed-based classification (slow vs fast fist)

---

## 📊 Comparison Matrix

| Feature | Effort | Impact | Complexity | Priority |
|---------|--------|--------|------------|----------|
| **Adaptive Calibration** | 1-2h | High (reliability) | Low | 🥇 High |
| **Real-time Pose Estimation** | 2-3h | High (UX) | Medium | ✅ **DONE** |
| **Multi-Sensor Fusion** | 4-5h | Medium (robustness) | High | 🥈 Medium |
| **Gesture Prediction** | 2h | High (new features) | Medium | 🥉 Medium |

---

## 🚀 Recommended Implementation Order

### **Immediate (This Week)**:
1. ✅ Real-time Pose Estimation - **COMPLETED**
2. 🔄 Adaptive Calibration - Quick win, high reliability impact

### **Short-term (This Month)**:
3. 📈 Gesture Prediction - Enables dynamic gestures, good ROI
4. 🔀 Multi-Sensor Fusion - If magnetometer alone is insufficient

### **Long-term (Next Quarter)**:
- Advanced EKF with full 6-DOF
- Learning-based drift correction
- Attention-based sensor fusion

---

## 📚 References

### Adaptive Calibration
- Kok et al. (2017). "A fast calibration method for triaxial magnetometers"
- Gebre-Egziabher et al. (2006). "Calibration of Strapdown Magnetometers in Magnetic Field Domain"

### Extended Kalman Filter
- Madgwick (2011). "An efficient orientation filter for IMU and MARG sensor arrays"
- Sabatini (2006). "Quaternion-based extended Kalman filter for determining orientation by inertial and magnetic sensing"

### Temporal Feature Learning
- Hammerla et al. (2016). "Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables"
- Ordóñez & Roggen (2016). "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"

---

## 💡 Tips for Implementation

### Debugging
- **Visualize**: Plot states, covariances, innovations
- **Unit test**: Test each component independently
- **Synthetic data**: Generate known trajectories, verify tracking
- **Gradual integration**: Add one sensor at a time

### Performance
- **Profile**: Measure ms per update, identify bottlenecks
- **Optimize**: Use TypedArrays, SIMD if available
- **Parallelize**: Run filters on Web Workers (future)

### Tuning
- **Start conservative**: High noise values, reduce gradually
- **A/B test**: Compare old vs new side-by-side
- **Collect metrics**: Accuracy, latency, drift rate
- **Iterate**: 3-5 tuning rounds typical

---

**Version**: 1.0
**Last Updated**: 2025-12-09
**Status**: Pose Estimation (✅ DONE), Others (📋 PLANNED)

---

<link rel="stylesheet" href="../src/simcap.css">
