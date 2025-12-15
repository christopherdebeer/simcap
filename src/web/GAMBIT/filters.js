/**
 * Extended Filtering for Magnetic Finger Tracking
 *
 * Provides:
 * - IMU Sensor Fusion (Madgwick AHRS for orientation estimation)
 * - Multi-dimensional Kalman Filter (for 3D position/velocity tracking)
 * - Extended Kalman Filter (for non-linear magnetic field model)
 * - Particle Filter (for multi-hypothesis finger tracking)
 *
 * Based on analysis: magnetic-finger-tracking-analysis.md
 */

/**
 * Madgwick AHRS (Attitude and Heading Reference System)
 *
 * Estimates device orientation from accelerometer and gyroscope data.
 * Uses gradient descent optimization to fuse sensor readings into a quaternion.
 *
 * Reference: "An efficient orientation filter for inertial and inertial/magnetic sensor arrays"
 * by Sebastian O.H. Madgwick (2010)
 *
 * MAGNETOMETER SUPPORT:
 * - 6-DOF mode (update): Uses accel + gyro only. Yaw will drift over time.
 * - 9-DOF mode (updateWithMag): Uses accel + gyro + mag for absolute yaw reference.
 *
 * The magTrust parameter (0-1) controls magnetometer influence:
 * - Set to 1.0 during development (no finger magnets) for stable yaw
 * - Reduce when finger magnets are added to prevent interference
 * - The residual (measured - expected) field is computed for later finger sensing
 */
class MadgwickAHRS {
    constructor(options = {}) {
        const {
            sampleFreq = 50,    // Hz
            beta = 0.1,         // Filter gain (higher = faster convergence, more noise)
            // Geomagnetic reference for 9-DOF mode (from geomagnetic-field.js)
            geomagneticRef = null  // { horizontal: µT, vertical: µT, declination: degrees }
        } = options;

        this.sampleFreq = sampleFreq;
        this.beta = beta;

        // Quaternion representing orientation (w, x, y, z)
        // Initialized to identity (no rotation)
        this.q = { w: 1, x: 0, y: 0, z: 0 };

        // For gyroscope bias estimation
        this.gyroBias = { x: 0, y: 0, z: 0 };
        this.biasAlpha = 0.001; // Slow adaptation rate

        // Geomagnetic reference for magnetometer fusion
        // If not provided, will use first magnetometer reading as reference
        this.geomagneticRef = geomagneticRef;
        this.magRefNormalized = null;  // Normalized reference in world frame

        // Magnetometer trust factor (1.0 = full trust, 0.0 = ignore mag)
        // Use 1.0 now (no finger magnets), reduce later when magnets added
        this.magTrust = 1.0;

        // Hard iron offset (learned during calibration)
        this.hardIron = { x: 0, y: 0, z: 0 };

        // Last computed expected/residual for external access
        this._lastMagExpected = null;
        this._lastMagResidual = null;
    }

    /**
     * Update orientation estimate with new IMU data
     *
     * @param {number} ax - Accelerometer X (g or m/s²)
     * @param {number} ay - Accelerometer Y
     * @param {number} az - Accelerometer Z
     * @param {number} gx - Gyroscope X (rad/s or deg/s)
     * @param {number} gy - Gyroscope Y
     * @param {number} gz - Gyroscope Z
     * @param {number} dt - Time step (seconds), if null uses 1/sampleFreq
     * @param {boolean} gyroInDegrees - If true, convert gyro from deg/s to rad/s
     */
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
        if (accelNorm > 0.01) {
            // Normalize accelerometer
            const recipNorm = 1.0 / accelNorm;
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Auxiliary variables to avoid repeated arithmetic
            const _2q0 = 2 * q0;
            const _2q1 = 2 * q1;
            const _2q2 = 2 * q2;
            const _2q3 = 2 * q3;
            const _4q0 = 4 * q0;
            const _4q1 = 4 * q1;
            const _4q2 = 4 * q2;
            const _8q1 = 8 * q1;
            const _8q2 = 8 * q2;
            const q0q0 = q0 * q0;
            const q1q1 = q1 * q1;
            const q2q2 = q2 * q2;
            const q3q3 = q3 * q3;

            // Gradient descent algorithm corrective step
            // Objective function: minimize error between expected and measured gravity
            let s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
            let s1 = _4q1 * q3q3 - _2q3 * ax + 4 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
            let s2 = 4 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
            let s3 = 4 * q1q1 * q3 - _2q1 * ax + 4 * q2q2 * q3 - _2q2 * ay;

            // Normalize step magnitude
            const sNorm = 1.0 / Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= sNorm;
            s1 *= sNorm;
            s2 *= sNorm;
            s3 *= sNorm;

            // Apply feedback step
            q0 += (qDot1 - this.beta * s0) * deltaT;
            q1 += (qDot2 - this.beta * s1) * deltaT;
            q2 += (qDot3 - this.beta * s2) * deltaT;
            q3 += (qDot4 - this.beta * s3) * deltaT;
        } else {
            // Only integrate gyroscope
            q0 += qDot1 * deltaT;
            q1 += qDot2 * deltaT;
            q2 += qDot3 * deltaT;
            q3 += qDot4 * deltaT;
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

    /**
     * Update gyroscope bias estimate when device is stationary
     * Call this when accelerometer shows near-zero motion
     */
    updateGyroBias(gx, gy, gz, gyroInDegrees = true) {
        if (gyroInDegrees) {
            gx = gx * Math.PI / 180;
            gy = gy * Math.PI / 180;
            gz = gz * Math.PI / 180;
        }

        // Exponential moving average of gyro readings when stationary
        this.gyroBias.x += this.biasAlpha * (gx - this.gyroBias.x);
        this.gyroBias.y += this.biasAlpha * (gy - this.gyroBias.y);
        this.gyroBias.z += this.biasAlpha * (gz - this.gyroBias.z);
    }

    /**
     * Get current orientation as quaternion
     */
    getQuaternion() {
        return { ...this.q };
    }

    /**
     * Get current orientation as Euler angles (in degrees)
     * Returns: { roll, pitch, yaw }
     */
    getEulerAngles() {
        const { w, x, y, z } = this.q;

        // Roll (x-axis rotation)
        const sinr_cosp = 2 * (w * x + y * z);
        const cosr_cosp = 1 - 2 * (x * x + y * y);
        const roll = Math.atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        const sinp = 2 * (w * y - z * x);
        let pitch;
        if (Math.abs(sinp) >= 1) {
            pitch = Math.sign(sinp) * Math.PI / 2; // Use 90 degrees if out of range
        } else {
            pitch = Math.asin(sinp);
        }

        // Yaw (z-axis rotation)
        const siny_cosp = 2 * (w * z + x * y);
        const cosy_cosp = 1 - 2 * (y * y + z * z);
        const yaw = Math.atan2(siny_cosp, cosy_cosp);

        return {
            roll: roll * 180 / Math.PI,
            pitch: pitch * 180 / Math.PI,
            yaw: yaw * 180 / Math.PI
        };
    }

    /**
     * Get rotation matrix (3x3) from current quaternion
     * For transforming vectors from world to device frame
     */
    getRotationMatrix() {
        const { w, x, y, z } = this.q;

        return [
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ];
    }

    /**
     * Transform a vector from world frame to device frame
     * @param {Object} v - Vector {x, y, z} in world frame
     * @returns {Object} Vector {x, y, z} in device frame
     */
    transformToDeviceFrame(v) {
        const R = this.getRotationMatrix();
        return {
            x: R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
            y: R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
            z: R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z
        };
    }

    /**
     * Reset to initial orientation
     */
    reset() {
        this.q = { w: 1, x: 0, y: 0, z: 0 };
        this.gyroBias = { x: 0, y: 0, z: 0 };
    }

    /**
     * Set orientation from accelerometer only (for initialization)
     * Assumes device is stationary
     */
    initFromAccelerometer(ax, ay, az) {
        const norm = Math.sqrt(ax * ax + ay * ay + az * az);
        if (norm < 0.01) return;

        ax /= norm;
        ay /= norm;
        az /= norm;

        // Calculate roll and pitch from gravity vector
        const roll = Math.atan2(ay, az);
        const pitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az));

        // Convert to quaternion (yaw = 0)
        const cr = Math.cos(roll / 2);
        const sr = Math.sin(roll / 2);
        const cp = Math.cos(pitch / 2);
        const sp = Math.sin(pitch / 2);

        this.q = {
            w: cr * cp,
            x: sr * cp,
            y: cr * sp,
            z: -sr * sp
        };
    }

    // =========================================================================
    // 9-DOF MAGNETOMETER FUSION
    // =========================================================================
    // These methods add magnetometer support for absolute yaw reference.
    // Currently used for orientation; later will also compute residual for
    // finger magnet sensing.

    /**
     * Set geomagnetic reference from location data
     * @param {Object} geoRef - From geomagnetic-field.js { horizontal, vertical, declination }
     */
    setGeomagneticReference(geoRef) {
        this.geomagneticRef = geoRef;

        // Compute normalized reference vector in NED (North-East-Down) frame
        // Horizontal component points toward magnetic north
        // Vertical component points down (positive = down)
        if (geoRef) {
            const h = geoRef.horizontal;
            const v = geoRef.vertical;
            const mag = Math.sqrt(h * h + v * v);
            // Reference in world frame: [North, East, Down] = [horizontal, 0, vertical]
            this.magRefNormalized = {
                x: h / mag,  // North component (normalized)
                y: 0,        // East component (magnetic north has no east component)
                z: v / mag   // Down component (normalized)
            };
        }
    }

    /**
     * Set hard iron calibration offset
     * @param {Object} offset - { x, y, z } in same units as magnetometer
     */
    setHardIronOffset(offset) {
        this.hardIron = { ...offset };
    }

    /**
     * Set magnetometer trust factor
     * @param {number} trust - 0.0 (ignore mag) to 1.0 (full trust)
     */
    setMagTrust(trust) {
        this.magTrust = Math.max(0, Math.min(1, trust));
    }

    /**
     * Update orientation with 9-DOF fusion (accel + gyro + magnetometer)
     *
     * This is the Madgwick MARG algorithm that uses magnetometer for yaw correction.
     * The magTrust parameter allows gradual reduction of magnetometer influence
     * when finger magnets are present.
     *
     * @param {number} ax,ay,az - Accelerometer readings
     * @param {number} gx,gy,gz - Gyroscope readings
     * @param {number} mx,my,mz - Magnetometer readings (after hard iron correction if using raw)
     * @param {number} dt - Time step (seconds)
     * @param {boolean} gyroInDegrees - If true, convert gyro from deg/s to rad/s
     * @param {boolean} applyHardIron - If true, subtract this.hardIron from mag readings
     */
    updateWithMag(ax, ay, az, gx, gy, gz, mx, my, mz, dt = null, gyroInDegrees = true, applyHardIron = true) {
        // If magTrust is 0 or magnetometer invalid, fall back to 6-DOF
        const magNorm = Math.sqrt(mx * mx + my * my + mz * mz);
        if (this.magTrust < 0.01 || magNorm < 0.01) {
            return this.update(ax, ay, az, gx, gy, gz, dt, gyroInDegrees);
        }

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

        // Apply hard iron correction
        if (applyHardIron) {
            mx -= this.hardIron.x;
            my -= this.hardIron.y;
            mz -= this.hardIron.z;
        }

        let { w: q0, x: q1, y: q2, z: q3 } = this.q;

        // Rate of change of quaternion from gyroscope
        const qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
        const qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
        const qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
        const qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

        // Normalize accelerometer
        const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
        if (accelNorm < 0.01) {
            // Only integrate gyroscope
            q0 += qDot1 * deltaT;
            q1 += qDot2 * deltaT;
            q2 += qDot3 * deltaT;
            q3 += qDot4 * deltaT;
        } else {
            const recipAccelNorm = 1.0 / accelNorm;
            ax *= recipAccelNorm;
            ay *= recipAccelNorm;
            az *= recipAccelNorm;

            // Normalize magnetometer
            const recipMagNorm = 1.0 / magNorm;
            mx *= recipMagNorm;
            my *= recipMagNorm;
            mz *= recipMagNorm;

            // Auxiliary variables
            const _2q0mx = 2 * q0 * mx;
            const _2q0my = 2 * q0 * my;
            const _2q0mz = 2 * q0 * mz;
            const _2q1mx = 2 * q1 * mx;
            const _2q0 = 2 * q0;
            const _2q1 = 2 * q1;
            const _2q2 = 2 * q2;
            const _2q3 = 2 * q3;
            const _2q0q2 = 2 * q0 * q2;
            const _2q2q3 = 2 * q2 * q3;
            const q0q0 = q0 * q0;
            const q0q1 = q0 * q1;
            const q0q2 = q0 * q2;
            const q0q3 = q0 * q3;
            const q1q1 = q1 * q1;
            const q1q2 = q1 * q2;
            const q1q3 = q1 * q3;
            const q2q2 = q2 * q2;
            const q2q3 = q2 * q3;
            const q3q3 = q3 * q3;

            // Reference direction of Earth's magnetic field (simplified: horizontal plane)
            const hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
            const hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
            const _2bx = Math.sqrt(hx * hx + hy * hy);
            const _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
            const _4bx = 2 * _2bx;
            const _4bz = 2 * _2bz;

            // Gradient descent algorithm corrective step (combined accel + mag)
            let s0 = -_2q2 * (2 * q1q3 - _2q0q2 - ax) + _2q1 * (2 * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
            let s1 = _2q3 * (2 * q1q3 - _2q0q2 - ax) + _2q0 * (2 * q0q1 + _2q2q3 - ay) - 4 * q1 * (1 - 2 * q1q1 - 2 * q2q2 - az) + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
            let s2 = -_2q0 * (2 * q1q3 - _2q0q2 - ax) + _2q3 * (2 * q0q1 + _2q2q3 - ay) - 4 * q2 * (1 - 2 * q1q1 - 2 * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
            let s3 = _2q1 * (2 * q1q3 - _2q0q2 - ax) + _2q2 * (2 * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);

            // Normalize step magnitude
            const sNorm = 1.0 / Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= sNorm;
            s1 *= sNorm;
            s2 *= sNorm;
            s3 *= sNorm;

            // Apply feedback step with magTrust weighting
            // Full trust uses full beta, zero trust ignores magnetic correction
            const effectiveBeta = this.beta * (1.0 + this.magTrust);  // Slightly boost when mag available

            q0 += (qDot1 - effectiveBeta * s0) * deltaT;
            q1 += (qDot2 - effectiveBeta * s1) * deltaT;
            q2 += (qDot3 - effectiveBeta * s2) * deltaT;
            q3 += (qDot4 - effectiveBeta * s3) * deltaT;
        }

        // Normalize quaternion
        const qNorm = 1.0 / Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        this.q = {
            w: q0 * qNorm,
            x: q1 * qNorm,
            y: q2 * qNorm,
            z: q3 * qNorm
        };

        // Compute expected magnetic field and residual for later finger magnet sensing
        this._computeMagResidual(mx * magNorm, my * magNorm, mz * magNorm);

        return this.q;
    }

    /**
     * Compute expected magnetic field and residual
     * This is used later for finger magnet sensing
     * @private
     */
    _computeMagResidual(mx, my, mz) {
        if (!this.geomagneticRef) {
            this._lastMagExpected = null;
            this._lastMagResidual = null;
            return;
        }

        // Expected field = R(q) × B_earth (rotate earth field to sensor frame)
        const expected = this.transformToDeviceFrame({
            x: this.geomagneticRef.horizontal,  // North component
            y: 0,                                 // East component
            z: this.geomagneticRef.vertical      // Down component
        });

        // Add hard iron (it's in sensor frame)
        expected.x += this.hardIron.x;
        expected.y += this.hardIron.y;
        expected.z += this.hardIron.z;

        this._lastMagExpected = expected;

        // Residual = measured - expected
        // This is what finger magnets will contribute
        this._lastMagResidual = {
            x: mx - expected.x,
            y: my - expected.y,
            z: mz - expected.z
        };
    }

    /**
     * Get the expected magnetic field at current orientation
     * @returns {Object|null} { x, y, z } in sensor frame (µT)
     */
    getExpectedMagField() {
        return this._lastMagExpected;
    }

    /**
     * Get the magnetic field residual (measured - expected)
     * This is the signal from finger magnets (after subtracting earth field)
     * @returns {Object|null} { x, y, z } in sensor frame (µT)
     */
    getMagResidual() {
        return this._lastMagResidual;
    }

    /**
     * Get residual magnitude (useful for detecting magnet proximity)
     * @returns {number} Magnitude of residual field (µT)
     */
    getMagResidualMagnitude() {
        if (!this._lastMagResidual) return 0;
        const r = this._lastMagResidual;
        return Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    }
}

/**
 * Complementary Filter (Simple Alternative)
 *
 * Faster and simpler than Madgwick, but less accurate.
 * Good for quick prototyping.
 */
class ComplementaryFilter {
    constructor(alpha = 0.98) {
        this.alpha = alpha;  // Weight for gyroscope (0.98 = 98% gyro, 2% accel)
        this.roll = 0;
        this.pitch = 0;
        this.yaw = 0;  // Cannot be estimated from accel alone
    }

    update(ax, ay, az, gx, gy, gz, dt, gyroInDegrees = true) {
        // Convert gyro to deg/s if needed
        if (!gyroInDegrees) {
            gx = gx * 180 / Math.PI;
            gy = gy * 180 / Math.PI;
            gz = gz * 180 / Math.PI;
        }

        // Accelerometer angles
        const accelRoll = Math.atan2(ay, az) * 180 / Math.PI;
        const accelPitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az)) * 180 / Math.PI;

        // Complementary filter
        this.roll = this.alpha * (this.roll + gx * dt) + (1 - this.alpha) * accelRoll;
        this.pitch = this.alpha * (this.pitch + gy * dt) + (1 - this.alpha) * accelPitch;
        this.yaw += gz * dt;  // Gyro only (drifts over time)

        return this.getEulerAngles();
    }

    getEulerAngles() {
        return {
            roll: this.roll,
            pitch: this.pitch,
            yaw: this.yaw
        };
    }

    getQuaternion() {
        // Convert Euler to quaternion
        const cr = Math.cos(this.roll * Math.PI / 360);
        const sr = Math.sin(this.roll * Math.PI / 360);
        const cp = Math.cos(this.pitch * Math.PI / 360);
        const sp = Math.sin(this.pitch * Math.PI / 360);
        const cy = Math.cos(this.yaw * Math.PI / 360);
        const sy = Math.sin(this.yaw * Math.PI / 360);

        return {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy
        };
    }

    reset() {
        this.roll = 0;
        this.pitch = 0;
        this.yaw = 0;
    }
}

/**
 * Motion Detector
 *
 * Detects device motion state based on IMU sensor readings.
 * Uses accelerometer and gyroscope variance over a sliding window.
 */
class MotionDetector {
    constructor(options = {}) {
        const {
            accelThreshold = 2000,    // LSB units (sensor-specific)
            gyroThreshold = 500,       // LSB units (sensor-specific)
            windowSize = 10           // Number of samples in moving window
        } = options;

        this.accelThreshold = accelThreshold;
        this.gyroThreshold = gyroThreshold;
        this.windowSize = windowSize;

        this.recentAccel = [];
        this.recentGyro = [];

        this.isMoving = false;
        this.accelStd = 0;
        this.gyroStd = 0;
    }

    /**
     * Update motion state with new IMU reading
     *
     * @param {number} ax - Accelerometer X
     * @param {number} ay - Accelerometer Y
     * @param {number} az - Accelerometer Z
     * @param {number} gx - Gyroscope X
     * @param {number} gy - Gyroscope Y
     * @param {number} gz - Gyroscope Z
     * @returns {Object} Motion state {isMoving, accelStd, gyroStd}
     */
    update(ax, ay, az, gx, gy, gz) {
        // Compute magnitudes
        const accelMag = Math.sqrt(ax*ax + ay*ay + az*az);
        const gyroMag = Math.sqrt(gx*gx + gy*gy + gz*gz);

        // Add to history
        this.recentAccel.push(accelMag);
        this.recentGyro.push(gyroMag);

        // Maintain window size
        if (this.recentAccel.length > this.windowSize) {
            this.recentAccel.shift();
            this.recentGyro.shift();
        }

        // Need minimum samples for valid detection
        if (this.recentAccel.length < this.windowSize / 2) {
            return this.getState();
        }

        // Compute standard deviation as motion indicator
        this.accelStd = this._std(this.recentAccel);
        this.gyroStd = this._std(this.recentGyro);

        // Detect motion: high variance in either sensor
        this.isMoving = (this.accelStd > this.accelThreshold) ||
                       (this.gyroStd > this.gyroThreshold);

        return this.getState();
    }

    /**
     * Get current motion state
     */
    getState() {
        return {
            isMoving: this.isMoving,
            accelStd: this.accelStd,
            gyroStd: this.gyroStd
        };
    }

    /**
     * Reset motion detector state
     */
    reset() {
        this.recentAccel = [];
        this.recentGyro = [];
        this.isMoving = false;
        this.accelStd = 0;
        this.gyroStd = 0;
    }

    /**
     * Compute standard deviation of array
     * @private
     */
    _std(arr) {
        if (arr.length === 0) return 0;
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / arr.length;
        return Math.sqrt(variance);
    }
}

/**
 * Multi-dimensional Kalman Filter
 *
 * Tracks state vector with position and velocity for each dimension.
 * State: [x, y, z, vx, vy, vz]
 */
class KalmanFilter3D {
    constructor(options = {}) {
        const {
            processNoise = 1.0,      // Q - process noise (increased from 0.1 for better responsiveness)
            measurementNoise = 1.0,  // R - measurement noise
            initialCovariance = 100  // P0 - initial uncertainty
        } = options;

        // State dimension: position (3) + velocity (3) = 6
        this.stateDim = 6;
        this.measDim = 3;

        // State vector [x, y, z, vx, vy, vz]
        this.state = new Float64Array(6);

        // Covariance matrix (6x6)
        this.P = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            this.P[i * 6 + i] = initialCovariance;
        }

        // Process noise covariance (6x6 diagonal)
        this.Q = processNoise;

        // Measurement noise covariance
        this.R = measurementNoise;

        // Time step
        this.dt = 0.02; // 50Hz default

        this.initialized = false;
    }

    /**
     * State transition matrix F
     * Models constant velocity: x_new = x + v*dt
     */
    _getF(dt) {
        return new Float64Array([
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
        ]);
    }

    /**
     * Measurement matrix H
     * We only measure position, not velocity
     */
    _getH() {
        return new Float64Array([
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0
        ]);
    }

    /**
     * Matrix multiplication: C = A * B
     */
    _matMul(A, B, rowsA, colsA, colsB) {
        const C = new Float64Array(rowsA * colsB);
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i * colsA + k] * B[k * colsB + j];
                }
                C[i * colsB + j] = sum;
            }
        }
        return C;
    }

    /**
     * Matrix transpose
     */
    _transpose(A, rows, cols) {
        const AT = new Float64Array(cols * rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                AT[j * rows + i] = A[i * cols + j];
            }
        }
        return AT;
    }

    /**
     * Add matrices
     */
    _matAdd(A, B, size) {
        const C = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
        return C;
    }

    /**
     * Subtract matrices
     */
    _matSub(A, B, size) {
        const C = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            C[i] = A[i] - B[i];
        }
        return C;
    }

    /**
     * 3x3 matrix inversion (for Kalman gain calculation)
     */
    _invert3x3(M) {
        const det = M[0] * (M[4] * M[8] - M[5] * M[7])
                  - M[1] * (M[3] * M[8] - M[5] * M[6])
                  + M[2] * (M[3] * M[7] - M[4] * M[6]);

        if (Math.abs(det) < 1e-10) {
            // Return identity if singular
            return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
        }

        const invDet = 1 / det;
        return new Float64Array([
            (M[4] * M[8] - M[5] * M[7]) * invDet,
            (M[2] * M[7] - M[1] * M[8]) * invDet,
            (M[1] * M[5] - M[2] * M[4]) * invDet,
            (M[5] * M[6] - M[3] * M[8]) * invDet,
            (M[0] * M[8] - M[2] * M[6]) * invDet,
            (M[2] * M[3] - M[0] * M[5]) * invDet,
            (M[3] * M[7] - M[4] * M[6]) * invDet,
            (M[1] * M[6] - M[0] * M[7]) * invDet,
            (M[0] * M[4] - M[1] * M[3]) * invDet
        ]);
    }

    /**
     * Initialize filter with first measurement
     */
    initialize(measurement) {
        this.state[0] = measurement.x;
        this.state[1] = measurement.y;
        this.state[2] = measurement.z;
        this.state[3] = 0; // Initial velocity = 0
        this.state[4] = 0;
        this.state[5] = 0;
        this.initialized = true;
    }

    /**
     * Predict step
     */
    predict(dt = null) {
        if (!this.initialized) return null;

        const deltaT = dt || this.dt;
        const F = this._getF(deltaT);

        // State prediction: x = F * x
        const newState = new Float64Array(6);
        for (let i = 0; i < 6; i++) {
            newState[i] = 0;
            for (let j = 0; j < 6; j++) {
                newState[i] += F[i * 6 + j] * this.state[j];
            }
        }
        this.state = newState;

        // Covariance prediction: P = F * P * F' + Q
        const FP = this._matMul(F, this.P, 6, 6, 6);
        const FT = this._transpose(F, 6, 6);
        const FPFT = this._matMul(FP, FT, 6, 6, 6);

        // Add process noise (diagonal)
        for (let i = 0; i < 6; i++) {
            FPFT[i * 6 + i] += this.Q;
        }
        this.P = FPFT;

        return this.getPosition();
    }

    /**
     * Update step with measurement
     */
    update(measurement) {
        if (!this.initialized) {
            this.initialize(measurement);
            return this.getPosition();
        }

        const H = this._getH();
        const HT = this._transpose(H, 3, 6);

        // Innovation: y = z - H * x
        const z = new Float64Array([measurement.x, measurement.y, measurement.z]);
        const Hx = new Float64Array(3);
        for (let i = 0; i < 3; i++) {
            Hx[i] = 0;
            for (let j = 0; j < 6; j++) {
                Hx[i] += H[i * 6 + j] * this.state[j];
            }
        }
        const y = this._matSub(z, Hx, 3);

        // Innovation covariance: S = H * P * H' + R
        const HP = this._matMul(H, this.P, 3, 6, 6);
        const HPHT = this._matMul(HP, HT, 3, 6, 3);
        for (let i = 0; i < 3; i++) {
            HPHT[i * 3 + i] += this.R;
        }

        // Kalman gain: K = P * H' * S^-1
        const PHT = this._matMul(this.P, HT, 6, 6, 3);
        const Sinv = this._invert3x3(HPHT);
        const K = this._matMul(PHT, Sinv, 6, 3, 3);

        // State update: x = x + K * y
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 3; j++) {
                this.state[i] += K[i * 3 + j] * y[j];
            }
        }

        // Covariance update: P = (I - K * H) * P
        const KH = this._matMul(K, H, 6, 3, 6);
        const I_KH = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                I_KH[i * 6 + j] = (i === j ? 1 : 0) - KH[i * 6 + j];
            }
        }
        this.P = this._matMul(I_KH, this.P, 6, 6, 6);

        return this.getPosition();
    }

    /**
     * Get current position estimate
     */
    getPosition() {
        return {
            x: this.state[0],
            y: this.state[1],
            z: this.state[2]
        };
    }

    /**
     * Get current velocity estimate
     */
    getVelocity() {
        return {
            x: this.state[3],
            y: this.state[4],
            z: this.state[5]
        };
    }

    /**
     * Get full state
     */
    getState() {
        return {
            position: this.getPosition(),
            velocity: this.getVelocity(),
            covariance: Array.from(this.P)
        };
    }

    /**
     * Reset filter
     */
    reset() {
        this.state = new Float64Array(6);
        this.P = new Float64Array(36);
        for (let i = 0; i < 6; i++) {
            this.P[i * 6 + i] = 100;
        }
        this.initialized = false;
    }
}

/**
 * Multi-Finger Kalman Filter
 *
 * Tracks 5 fingers independently, each with 3D position.
 */
class MultiFingerKalmanFilter {
    constructor(options = {}) {
        this.fingers = {
            thumb: new KalmanFilter3D(options),
            index: new KalmanFilter3D(options),
            middle: new KalmanFilter3D(options),
            ring: new KalmanFilter3D(options),
            pinky: new KalmanFilter3D(options)
        };
    }

    /**
     * Update specific finger with measurement
     */
    updateFinger(fingerName, measurement) {
        if (this.fingers[fingerName]) {
            return this.fingers[fingerName].update(measurement);
        }
        return null;
    }

    /**
     * Predict all fingers
     */
    predictAll(dt = null) {
        const predictions = {};
        for (const [name, filter] of Object.entries(this.fingers)) {
            predictions[name] = filter.predict(dt);
        }
        return predictions;
    }

    /**
     * Get all finger positions
     */
    getAllPositions() {
        const positions = {};
        for (const [name, filter] of Object.entries(this.fingers)) {
            positions[name] = filter.getPosition();
        }
        return positions;
    }

    /**
     * Reset all filters
     */
    resetAll() {
        for (const filter of Object.values(this.fingers)) {
            filter.reset();
        }
    }
}

/**
 * Particle Filter for Hand Pose Estimation
 *
 * Maintains multiple hypotheses about hand pose,
 * useful for handling multi-modal distributions and
 * ambiguity in magnetic field measurements.
 */
class ParticleFilter {
    constructor(options = {}) {
        const {
            numParticles = 500,
            positionNoise = 2.0,  // mm
            velocityNoise = 5.0,  // mm/s
            resampleThreshold = 0.5
        } = options;

        this.numParticles = numParticles;
        this.positionNoise = positionNoise;
        this.velocityNoise = velocityNoise;
        this.resampleThreshold = resampleThreshold;

        // Each particle represents a hand pose hypothesis
        // State per finger: [x, y, z, vx, vy, vz]
        // 5 fingers × 6 state = 30 values per particle
        this.particles = [];
        this.weights = new Float64Array(numParticles).fill(1 / numParticles);

        this.initialized = false;
    }

    /**
     * Initialize particles around initial pose estimate
     */
    initialize(initialPose) {
        this.particles = [];

        for (let i = 0; i < this.numParticles; i++) {
            const particle = {};

            for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
                const base = initialPose[finger] || { x: 0, y: 0, z: 0 };
                particle[finger] = {
                    x: base.x + this._randn() * this.positionNoise * 5,
                    y: base.y + this._randn() * this.positionNoise * 5,
                    z: base.z + this._randn() * this.positionNoise * 5,
                    vx: this._randn() * this.velocityNoise,
                    vy: this._randn() * this.velocityNoise,
                    vz: this._randn() * this.velocityNoise
                };
            }

            this.particles.push(particle);
        }

        this.weights.fill(1 / this.numParticles);
        this.initialized = true;
    }

    /**
     * Prediction step: propagate particles with motion model
     */
    predict(dt = 0.02) {
        if (!this.initialized) return;

        for (const particle of this.particles) {
            for (const finger of Object.values(particle)) {
                // Position update with velocity
                finger.x += finger.vx * dt + this._randn() * this.positionNoise;
                finger.y += finger.vy * dt + this._randn() * this.positionNoise;
                finger.z += finger.vz * dt + this._randn() * this.positionNoise;

                // Velocity random walk
                finger.vx += this._randn() * this.velocityNoise * dt;
                finger.vy += this._randn() * this.velocityNoise * dt;
                finger.vz += this._randn() * this.velocityNoise * dt;
            }
        }
    }

    /**
     * Update step: weight particles based on measurement likelihood
     *
     * @param {Object} measurement - Observed magnetic field {x, y, z}
     * @param {Function} likelihoodFn - Function(particle, measurement) => likelihood
     */
    update(measurement, likelihoodFn) {
        if (!this.initialized) return;

        let sumWeights = 0;

        for (let i = 0; i < this.numParticles; i++) {
            const likelihood = likelihoodFn(this.particles[i], measurement);
            this.weights[i] *= likelihood;
            sumWeights += this.weights[i];
        }

        // Normalize weights
        if (sumWeights > 0) {
            for (let i = 0; i < this.numParticles; i++) {
                this.weights[i] /= sumWeights;
            }
        } else {
            // All weights zero - reinitialize
            this.weights.fill(1 / this.numParticles);
        }

        // Resample if effective sample size too low
        const nEff = this._effectiveSampleSize();
        if (nEff < this.numParticles * this.resampleThreshold) {
            this._resample();
        }
    }

    /**
     * Get weighted average estimate
     */
    estimate() {
        if (!this.initialized) return null;

        const result = {};

        for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
            let x = 0, y = 0, z = 0;

            for (let i = 0; i < this.numParticles; i++) {
                const w = this.weights[i];
                x += this.particles[i][finger].x * w;
                y += this.particles[i][finger].y * w;
                z += this.particles[i][finger].z * w;
            }

            result[finger] = { x, y, z };
        }

        return result;
    }

    /**
     * Get particle diversity (spread of hypotheses)
     */
    getDiversity() {
        if (!this.initialized) return 0;

        const est = this.estimate();
        let totalVar = 0;

        for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
            let varX = 0, varY = 0, varZ = 0;

            for (let i = 0; i < this.numParticles; i++) {
                const w = this.weights[i];
                const dx = this.particles[i][finger].x - est[finger].x;
                const dy = this.particles[i][finger].y - est[finger].y;
                const dz = this.particles[i][finger].z - est[finger].z;
                varX += dx * dx * w;
                varY += dy * dy * w;
                varZ += dz * dz * w;
            }

            totalVar += varX + varY + varZ;
        }

        return Math.sqrt(totalVar / 5);
    }

    /**
     * Systematic resampling
     */
    _resample() {
        const newParticles = [];
        const cumSum = new Float64Array(this.numParticles);

        cumSum[0] = this.weights[0];
        for (let i = 1; i < this.numParticles; i++) {
            cumSum[i] = cumSum[i - 1] + this.weights[i];
        }

        const step = 1 / this.numParticles;
        let u = Math.random() * step;
        let j = 0;

        for (let i = 0; i < this.numParticles; i++) {
            while (u > cumSum[j] && j < this.numParticles - 1) {
                j++;
            }

            // Deep copy particle
            const copy = {};
            for (const finger of Object.keys(this.particles[j])) {
                copy[finger] = { ...this.particles[j][finger] };
            }
            newParticles.push(copy);

            u += step;
        }

        this.particles = newParticles;
        this.weights.fill(1 / this.numParticles);
    }

    /**
     * Effective sample size
     */
    _effectiveSampleSize() {
        let sumSq = 0;
        for (let i = 0; i < this.numParticles; i++) {
            sumSq += this.weights[i] * this.weights[i];
        }
        return 1 / sumSq;
    }

    /**
     * Standard normal random number
     */
    _randn() {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Reset filter
     */
    reset() {
        this.particles = [];
        this.weights = new Float64Array(this.numParticles).fill(1 / this.numParticles);
        this.initialized = false;
    }
}

/**
 * Simple magnetic field likelihood model
 *
 * Given a particle (hand pose hypothesis) and measured field,
 * compute how likely the measurement is.
 */
function magneticLikelihood(particle, measurement, magnetConfig = null) {
    // Default magnet configuration if not provided
    // Assumes small cylindrical magnets (e.g., 3mm x 2mm N52 neodymium)
    // Magnetic moment ≈ 0.01 A·m² for small magnets
    if (!magnetConfig) {
        magnetConfig = {
            thumb: {moment: {x: 0, y: 0, z: 0.01}},   // Z-axis oriented
            index: {moment: {x: 0, y: 0, z: 0.01}},
            middle: {moment: {x: 0, y: 0, z: 0.01}},
            ring: {moment: {x: 0, y: 0, z: 0.01}},
            pinky: {moment: {x: 0, y: 0, z: 0.01}}
        };
    }

    // Sensor position (reference frame origin)
    const sensorPos = {x: 0, y: 0, z: 0};

    // Calculate expected field as sum of all dipole contributions
    let expectedX = 0, expectedY = 0, expectedZ = 0;

    for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
        if (particle[finger] && magnetConfig[finger]) {
            // Compute dipole field from this finger's magnet
            const magnetPos = particle[finger];
            const magnetMoment = magnetConfig[finger].moment;

            // Position vector from magnet to sensor (in mm, converted to m)
            const rx = (sensorPos.x - magnetPos.x) * 0.001; // mm to m
            const ry = (sensorPos.y - magnetPos.y) * 0.001;
            const rz = (sensorPos.z - magnetPos.z) * 0.001;

            // Distance
            const r = Math.sqrt(rx * rx + ry * ry + rz * rz);

            // Avoid singularity at r=0
            if (r >= 0.001) { // 1mm threshold
                // Unit vector r̂
                const rx_hat = rx / r;
                const ry_hat = ry / r;
                const rz_hat = rz / r;

                // Dot product m·r̂
                const m_dot_r = magnetMoment.x * rx_hat + magnetMoment.y * ry_hat + magnetMoment.z * rz_hat;

                // Dipole field: B = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³
                // Using simplified constant k = 1.0 (units absorbed into calibration)
                const k = 1.0;
                const r3 = r * r * r;

                expectedX += k * (3 * m_dot_r * rx_hat - magnetMoment.x) / r3;
                expectedY += k * (3 * m_dot_r * ry_hat - magnetMoment.y) / r3;
                expectedZ += k * (3 * m_dot_r * rz_hat - magnetMoment.z) / r3;
            }
        }
    }

    // Compute residual (measurement - expected)
    const dx = measurement.x - expectedX;
    const dy = measurement.y - expectedY;
    const dz = measurement.z - expectedZ;

    // Euclidean distance
    const residual = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Gaussian likelihood with standard deviation sigma
    // Sigma represents measurement noise + model uncertainty
    const sigma = 10.0; // Tunable parameter (adjust based on calibration data)

    // Likelihood: exp(-residual² / (2σ²))
    const likelihood = Math.exp(-(residual * residual) / (2 * sigma * sigma));

    return likelihood;
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MadgwickAHRS,
        ComplementaryFilter,
        MotionDetector,
        KalmanFilter3D,
        MultiFingerKalmanFilter,
        ParticleFilter,
        magneticLikelihood
    };
}
