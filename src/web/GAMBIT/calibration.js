/**
 * Environmental Calibration for Magnetic Finger Tracking
 *
 * Handles hard iron, soft iron, and Earth field compensation
 * for magnetometer data from the GAMBIT device.
 *
 * Usage:
 *   const calibration = new EnvironmentalCalibration();
 *   await calibration.runHardIronCalibration(collectSamples);
 *   const corrected = calibration.correct(rawMag, orientation);
 *
 * =====================================================================
 * TODO: [SENSOR-003] Magnetometer Axis Alignment Required
 * =====================================================================
 *
 * The LIS3MDL magnetometer has TRANSPOSED X/Y axes relative to the
 * LSM6DS3 accelerometer/gyroscope:
 *
 *   Magnetometer:  +X → fingers, +Y → wrist, +Z → palm
 *   Accel/Gyro:    +X → wrist,   +Y → fingers, +Z → palm
 *
 * CURRENT STATE: This calibration module receives magnetometer data
 * WITHOUT axis alignment. The hard iron offset and Earth field vectors
 * are stored in the magnetometer's native coordinate frame, which does
 * NOT match the IMU orientation quaternion's coordinate frame.
 *
 * IMPACT:
 *   - Earth field subtraction (correct() method) may produce incorrect
 *     residuals because orientation is in accel/gyro frame but Earth
 *     field reference is in magnetometer frame
 *   - Hard/soft iron calibration itself is self-consistent (calibrates
 *     in mag frame, corrects in mag frame) so finger tracking still works
 *
 * RECOMMENDED FIX:
 *   1. Align mag data to accel frame BEFORE passing to calibration
 *   2. OR add axis swap inside this module for orientation-dependent ops
 *
 * See: GAMBIT/index.html for full coordinate system documentation
 * =====================================================================
 */

/**
 * 3x3 Matrix operations
 */
class Matrix3 {
    constructor(data = null) {
        this.data = data || [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
    }

    static identity() {
        return new Matrix3();
    }

    static fromArray(arr) {
        return new Matrix3([
            [arr[0], arr[1], arr[2]],
            [arr[3], arr[4], arr[5]],
            [arr[6], arr[7], arr[8]]
        ]);
    }

    multiply(vec) {
        return {
            x: this.data[0][0] * vec.x + this.data[0][1] * vec.y + this.data[0][2] * vec.z,
            y: this.data[1][0] * vec.x + this.data[1][1] * vec.y + this.data[1][2] * vec.z,
            z: this.data[2][0] * vec.x + this.data[2][1] * vec.y + this.data[2][2] * vec.z
        };
    }

    multiplyMatrix(other) {
        const result = new Matrix3();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                result.data[i][j] = 0;
                for (let k = 0; k < 3; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    transpose() {
        return new Matrix3([
            [this.data[0][0], this.data[1][0], this.data[2][0]],
            [this.data[0][1], this.data[1][1], this.data[2][1]],
            [this.data[0][2], this.data[1][2], this.data[2][2]]
        ]);
    }

    toArray() {
        return [
            this.data[0][0], this.data[0][1], this.data[0][2],
            this.data[1][0], this.data[1][1], this.data[1][2],
            this.data[2][0], this.data[2][1], this.data[2][2]
        ];
    }
}

/**
 * Quaternion for orientation representation
 */
class Quaternion {
    constructor(w = 1, x = 0, y = 0, z = 0) {
        this.w = w;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    static fromEuler(roll, pitch, yaw) {
        // Convert degrees to radians
        const r = roll * Math.PI / 180;
        const p = pitch * Math.PI / 180;
        const y = yaw * Math.PI / 180;

        const cr = Math.cos(r / 2);
        const sr = Math.sin(r / 2);
        const cp = Math.cos(p / 2);
        const sp = Math.sin(p / 2);
        const cy = Math.cos(y / 2);
        const sy = Math.sin(y / 2);

        return new Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    toRotationMatrix() {
        const { w, x, y, z } = this;
        return new Matrix3([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ]);
    }

    normalize() {
        const mag = Math.sqrt(this.w*this.w + this.x*this.x + this.y*this.y + this.z*this.z);
        return new Quaternion(this.w/mag, this.x/mag, this.y/mag, this.z/mag);
    }
}

/**
 * Environmental Calibration Class
 *
 * Implements calibration for:
 * 1. Hard Iron - constant offset from nearby ferromagnetic materials
 * 2. Soft Iron - distortion from nearby conductive materials
 * 3. Earth Field - compensation for Earth's magnetic field
 */
class EnvironmentalCalibration {
    constructor() {
        // Hard iron offset (constant bias)
        this.hardIronOffset = { x: 0, y: 0, z: 0 };

        // Soft iron correction matrix
        this.softIronMatrix = Matrix3.identity();

        // Earth field reference (in local coordinates)
        this.earthField = { x: 0, y: 0, z: 0 };
        this.earthFieldMagnitude = 50; // μT typical

        // Calibration status
        this.hardIronCalibrated = false;
        this.softIronCalibrated = false;
        this.earthFieldCalibrated = false;

        // Sensor scale factors (for LIS3MDL: 6842 LSB/gauss @ ±4 gauss)
        this.scaleFactorLSBToUT = 100 / 6842; // Convert to μT

        // Calibration samples
        this.calibrationSamples = [];
    }

    /**
     * Run hard iron calibration
     * Requires samples collected while rotating the sensor through all orientations
     *
     * @param {Array} samples - Array of {x, y, z} magnetometer readings
     * @returns {Object} Calibration result with offset and quality metrics
     */
    runHardIronCalibration(samples) {
        if (samples.length < 100) {
            throw new Error('Need at least 100 samples for hard iron calibration');
        }

        // Find min/max for each axis
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (const s of samples) {
            minX = Math.min(minX, s.x);
            maxX = Math.max(maxX, s.x);
            minY = Math.min(minY, s.y);
            maxY = Math.max(maxY, s.y);
            minZ = Math.min(minZ, s.z);
            maxZ = Math.max(maxZ, s.z);
        }

        // Hard iron offset is the center of the ellipsoid
        this.hardIronOffset = {
            x: (maxX + minX) / 2,
            y: (maxY + minY) / 2,
            z: (maxZ + minZ) / 2
        };

        // Calculate quality metrics
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;
        const avgRange = (rangeX + rangeY + rangeZ) / 3;

        // Sphericity: how close to a sphere (1.0 = perfect sphere)
        const sphericity = Math.min(rangeX, rangeY, rangeZ) / Math.max(rangeX, rangeY, rangeZ);

        // Coverage: how well the samples cover the sphere
        const coverage = this._calculateCoverage(samples);

        this.hardIronCalibrated = true;

        return {
            offset: { ...this.hardIronOffset },
            ranges: { x: rangeX, y: rangeY, z: rangeZ },
            sphericity,
            coverage,
            sampleCount: samples.length,
            quality: sphericity, // Numeric quality metric (0.0 to 1.0)
            qualityLevel: sphericity > 0.9 && coverage > 0.7 ? 'good' :
                          sphericity > 0.7 && coverage > 0.5 ? 'acceptable' : 'poor'
        };
    }

    /**
     * Run soft iron calibration
     * Uses ellipsoid fitting to determine the distortion matrix
     *
     * @param {Array} samples - Array of {x, y, z} magnetometer readings
     * @returns {Object} Calibration result
     */
    runSoftIronCalibration(samples) {
        if (samples.length < 200) {
            throw new Error('Need at least 200 samples for soft iron calibration');
        }

        // First apply hard iron correction
        const corrected = samples.map(s => ({
            x: s.x - this.hardIronOffset.x,
            y: s.y - this.hardIronOffset.y,
            z: s.z - this.hardIronOffset.z
        }));

        // Calculate covariance matrix
        const cov = this._calculateCovariance(corrected);

        // Eigendecomposition for principal axes
        // For simplicity, use diagonal scaling (assumes axes aligned)
        const scaleX = Math.sqrt(cov[0][0]);
        const scaleY = Math.sqrt(cov[1][1]);
        const scaleZ = Math.sqrt(cov[2][2]);
        const avgScale = (scaleX + scaleY + scaleZ) / 3;

        // Create correction matrix to make ellipsoid into sphere
        this.softIronMatrix = new Matrix3([
            [avgScale / scaleX, 0, 0],
            [0, avgScale / scaleY, 0],
            [0, 0, avgScale / scaleZ]
        ]);

        this.softIronCalibrated = true;

        // Calculate quality based on how uniform the scaling is (1.0 = perfect sphere)
        const minScale = Math.min(scaleX, scaleY, scaleZ);
        const maxScale = Math.max(scaleX, scaleY, scaleZ);
        const quality = minScale / maxScale; // 1.0 = perfect sphere, lower = more distortion

        return {
            matrix: this.softIronMatrix.toArray(),
            scales: { x: scaleX, y: scaleY, z: scaleZ },
            correction: { x: avgScale / scaleX, y: avgScale / scaleY, z: avgScale / scaleZ },
            quality: quality // Numeric quality metric (0.0 to 1.0)
        };
    }

    /**
     * Calibrate Earth field reference
     * Should be done with sensor in a known orientation (e.g., flat on table)
     *
     * IMPORTANT: Earth field is stored in WORLD frame for proper orientation compensation.
     * The referenceOrientation parameter is required to transform from sensor to world frame.
     *
     * @param {Array} samples - Array of {x, y, z} readings in reference orientation
     * @param {Quaternion} referenceOrientation - Current device orientation (from IMU fusion)
     * @returns {Object} Earth field estimate with detailed diagnostics
     */
    runEarthFieldCalibration(samples, referenceOrientation = null) {
        if (samples.length < 50) {
            throw new Error('Need at least 50 samples for Earth field calibration');
        }

        // Apply hard/soft iron corrections first
        const corrected = samples.map(s => this._applyIronCorrection(s));

        // Average the corrected readings (in sensor frame)
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const s of corrected) {
            sumX += s.x;
            sumY += s.y;
            sumZ += s.z;
        }

        const earthFieldSensor = {
            x: sumX / corrected.length,
            y: sumY / corrected.length,
            z: sumZ / corrected.length
        };

        // Transform to world frame if orientation provided
        // R transforms from sensor to world, so: B_world = R @ B_sensor
        if (referenceOrientation) {
            const rotMatrix = referenceOrientation.toRotationMatrix();
            this.earthField = rotMatrix.multiply(earthFieldSensor);
            console.log('[Calibration] Earth field stored in WORLD frame');
        } else {
            // Fallback: assume sensor frame = world frame (device flat, aligned with world)
            this.earthField = earthFieldSensor;
            console.warn('[Calibration] No orientation provided - assuming device is flat and aligned');
        }

        this.earthFieldMagnitude = Math.sqrt(
            this.earthField.x ** 2 +
            this.earthField.y ** 2 +
            this.earthField.z ** 2
        );

        // === ENHANCED DIAGNOSTICS ===
        
        // Per-axis statistics
        const deviations = corrected.map(s => ({
            x: s.x - this.earthField.x,
            y: s.y - this.earthField.y,
            z: s.z - this.earthField.z,
            magnitude: Math.sqrt(
                (s.x - this.earthField.x) ** 2 +
                (s.y - this.earthField.y) ** 2 +
                (s.z - this.earthField.z) ** 2
            )
        }));

        // Per-axis standard deviation
        const axisStats = {
            x: this._calculateAxisStats(deviations.map(d => d.x)),
            y: this._calculateAxisStats(deviations.map(d => d.y)),
            z: this._calculateAxisStats(deviations.map(d => d.z))
        };

        // Overall deviation stats
        const magnitudeDeviations = deviations.map(d => d.magnitude);
        const avgDeviation = magnitudeDeviations.reduce((a, b) => a + b, 0) / magnitudeDeviations.length;
        const maxDeviation = Math.max(...magnitudeDeviations);
        const minDeviation = Math.min(...magnitudeDeviations);
        
        // Variance (sum of squared deviations)
        const variance = magnitudeDeviations.reduce((sum, d) => sum + d * d, 0);
        const stdDev = Math.sqrt(variance / magnitudeDeviations.length);

        // Outlier detection (samples > 2 std devs from mean)
        const outlierThreshold = avgDeviation + 2 * stdDev;
        const outliers = deviations.map((d, i) => ({ index: i, deviation: d.magnitude }))
            .filter(o => o.deviation > outlierThreshold);
        const outlierPercentage = (outliers.length / corrected.length) * 100;

        // Temporal analysis - detect drift and sudden jumps
        const temporalAnalysis = this._analyzeTemporalStability(corrected);

        // Quality: 1.0 if deviation is very small, decreases with larger deviation
        // Typical good reading has <5% deviation
        const quality = Math.max(0, Math.min(1, 1 - (avgDeviation / this.earthFieldMagnitude) * 10));
        
        // Determine dominant issue for diagnostics
        const diagnostics = this._generateEarthFieldDiagnostics({
            quality,
            avgDeviation,
            stdDev,
            axisStats,
            outlierPercentage,
            temporalAnalysis,
            earthFieldMagnitude: this.earthFieldMagnitude
        });

        console.log(`Earth field calibration details:`, {
            quality, 
            avgDeviation, 
            stdDev,
            variance, 
            axisStats,
            outliers: outliers.length,
            temporalAnalysis,
            diagnostics,
            earthField: this.earthField, 
            earthFieldMagnitude: this.earthFieldMagnitude
        });
        
        this.earthFieldCalibrated = true;

        return {
            field: { ...this.earthField },
            magnitude: this.earthFieldMagnitude,
            // Convert to μT using scale factor
            magnitudeUT: this.earthFieldMagnitude * this.scaleFactorLSBToUT,
            quality: quality, // Numeric quality metric (0.0 to 1.0)
            avgDeviation: avgDeviation,
            // Enhanced diagnostics
            diagnostics: {
                stdDev,
                maxDeviation,
                minDeviation,
                axisStats,
                outlierCount: outliers.length,
                outlierPercentage: outlierPercentage.toFixed(1),
                temporalAnalysis,
                recommendations: diagnostics.recommendations,
                dominantIssue: diagnostics.dominantIssue
            }
        };
    }

    /**
     * Calculate statistics for a single axis
     * @private
     */
    _calculateAxisStats(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
        const stdDev = Math.sqrt(variance);
        const max = Math.max(...values);
        const min = Math.min(...values);
        const range = max - min;
        
        return { mean, stdDev, max, min, range };
    }

    /**
     * Analyze temporal stability of samples
     * @private
     */
    _analyzeTemporalStability(samples) {
        if (samples.length < 10) {
            return { stable: true, drift: 0, jumps: 0 };
        }

        // Calculate magnitude for each sample
        const magnitudes = samples.map(s => Math.sqrt(s.x**2 + s.y**2 + s.z**2));
        
        // Check for drift (compare first 10% vs last 10%)
        const windowSize = Math.max(5, Math.floor(samples.length * 0.1));
        const firstWindow = magnitudes.slice(0, windowSize);
        const lastWindow = magnitudes.slice(-windowSize);
        const firstAvg = firstWindow.reduce((a, b) => a + b, 0) / windowSize;
        const lastAvg = lastWindow.reduce((a, b) => a + b, 0) / windowSize;
        const drift = lastAvg - firstAvg;
        const driftPercent = (Math.abs(drift) / firstAvg) * 100;

        // Detect sudden jumps (sample-to-sample changes > 3x median change)
        const changes = [];
        for (let i = 1; i < magnitudes.length; i++) {
            changes.push(Math.abs(magnitudes[i] - magnitudes[i-1]));
        }
        changes.sort((a, b) => a - b);
        const medianChange = changes[Math.floor(changes.length / 2)];
        const jumpThreshold = Math.max(medianChange * 3, 1); // At least 1 unit
        const jumps = changes.filter(c => c > jumpThreshold).length;
        const jumpPercentage = (jumps / changes.length) * 100;

        // Check for periodic noise (simple autocorrelation check)
        let periodicScore = 0;
        if (samples.length >= 50) {
            // Check correlation at common interference frequencies (50Hz, 60Hz harmonics)
            for (const lag of [1, 2, 5, 10]) {
                if (lag < samples.length / 2) {
                    let correlation = 0;
                    for (let i = lag; i < magnitudes.length; i++) {
                        correlation += (magnitudes[i] - firstAvg) * (magnitudes[i - lag] - firstAvg);
                    }
                    correlation /= (magnitudes.length - lag);
                    periodicScore = Math.max(periodicScore, Math.abs(correlation));
                }
            }
        }

        return {
            stable: driftPercent < 2 && jumpPercentage < 5,
            drift: drift.toFixed(2),
            driftPercent: driftPercent.toFixed(1),
            jumps,
            jumpPercentage: jumpPercentage.toFixed(1),
            periodicScore: periodicScore.toFixed(2),
            firstAvg: firstAvg.toFixed(2),
            lastAvg: lastAvg.toFixed(2)
        };
    }

    /**
     * Generate diagnostic recommendations based on calibration results
     * @private
     */
    _generateEarthFieldDiagnostics(stats) {
        const recommendations = [];
        let dominantIssue = 'unknown';

        const deviationPercent = (stats.avgDeviation / stats.earthFieldMagnitude) * 100;
        const { temporalAnalysis, axisStats, outlierPercentage } = stats;

        // Check for movement during calibration
        if (temporalAnalysis.jumpPercentage > 10) {
            recommendations.push('🚶 Movement detected: Keep device completely still during calibration');
            dominantIssue = 'movement';
        }

        // Check for drift (device slowly moving or warming up)
        if (parseFloat(temporalAnalysis.driftPercent) > 3) {
            recommendations.push(`📈 Drift detected (${temporalAnalysis.driftPercent}%): Device may be moving slowly or sensor warming up. Wait 30s after connecting before calibrating.`);
            if (dominantIssue === 'unknown') dominantIssue = 'drift';
        }

        // Check for high outlier percentage (interference spikes)
        if (outlierPercentage > 5) {
            recommendations.push(`⚡ ${outlierPercentage.toFixed(1)}% outliers detected: Possible electromagnetic interference. Move away from electronics, motors, or power cables.`);
            if (dominantIssue === 'unknown') dominantIssue = 'interference';
        }

        // Check for axis-specific issues
        const axisStdDevs = [
            { axis: 'X', stdDev: axisStats.x.stdDev },
            { axis: 'Y', stdDev: axisStats.y.stdDev },
            { axis: 'Z', stdDev: axisStats.z.stdDev }
        ].sort((a, b) => b.stdDev - a.stdDev);

        const worstAxis = axisStdDevs[0];
        const bestAxis = axisStdDevs[2];
        
        if (worstAxis.stdDev > bestAxis.stdDev * 3) {
            recommendations.push(`📊 ${worstAxis.axis}-axis has ${(worstAxis.stdDev / bestAxis.stdDev).toFixed(1)}x more noise than ${bestAxis.axis}-axis. Check sensor orientation or nearby magnetic interference along ${worstAxis.axis}-axis.`);
            if (dominantIssue === 'unknown') dominantIssue = 'axis_noise';
        }

        // Check for periodic interference
        if (parseFloat(temporalAnalysis.periodicScore) > 100) {
            recommendations.push('🔄 Periodic noise detected: Possible 50/60Hz interference from power lines or electronics.');
            if (dominantIssue === 'unknown') dominantIssue = 'periodic_noise';
        }

        // General high deviation
        if (deviationPercent > 5 && recommendations.length === 0) {
            recommendations.push(`📏 High deviation (${deviationPercent.toFixed(1)}%): Ensure device is stationary and away from magnets (>50cm).`);
            dominantIssue = 'high_deviation';
        }

        // Check earth field magnitude sanity
        if (stats.earthFieldMagnitude < 200 || stats.earthFieldMagnitude > 700) {
            recommendations.push(`🧲 Unusual earth field magnitude (${stats.earthFieldMagnitude.toFixed(0)}). Expected 250-650 for typical locations. Check for nearby strong magnets or ferromagnetic materials.`);
            if (dominantIssue === 'unknown') dominantIssue = 'magnitude_anomaly';
        }

        // Success case
        if (stats.quality > 0.9) {
            recommendations.push('✅ Excellent calibration quality!');
            dominantIssue = 'none';
        } else if (stats.quality > 0.7) {
            recommendations.push('⚠️ Acceptable quality, but could be improved by reducing movement/interference.');
        }

        return { recommendations, dominantIssue };
    }

    /**
     * Apply full calibration correction to raw magnetometer reading
     *
     * @param {Object} raw - Raw {x, y, z} magnetometer reading
     * @param {Quaternion} orientation - Current orientation (optional)
     * @returns {Object} Corrected {x, y, z} reading
     */
    correct(raw, orientation = null) {
        // Step 1: Apply hard iron correction
        let corrected = {
            x: raw.x - this.hardIronOffset.x,
            y: raw.y - this.hardIronOffset.y,
            z: raw.z - this.hardIronOffset.z
        };

        // Step 2: Apply soft iron correction
        if (this.softIronCalibrated) {
            corrected = this.softIronMatrix.multiply(corrected);
        }

        // Step 3: Subtract Earth field (rotated to current sensor frame)
        // earthField is stored in WORLD frame during calibration
        // To subtract it, we transform world→sensor using R.T (transpose)
        if (this.earthFieldCalibrated && orientation) {
            const rotMatrix = orientation.toRotationMatrix();
            // R.T transforms from world frame to current sensor frame
            const rotatedEarth = rotMatrix.transpose().multiply(this.earthField);
            corrected = {
                x: corrected.x - rotatedEarth.x,
                y: corrected.y - rotatedEarth.y,
                z: corrected.z - rotatedEarth.z
            };
        }

        return corrected;
    }

    /**
     * Apply only iron corrections (no Earth field subtraction)
     * Use this when orientation is not available or when you want
     * to see the iron-corrected signal before Earth field compensation.
     *
     * @param {Object} raw - Raw {x, y, z} magnetometer reading
     * @returns {Object} Iron-corrected {x, y, z} reading
     */
    correctIronOnly(raw) {
        return this._applyIronCorrection(raw);
    }

    /**
     * Apply only iron corrections (internal method)
     */
    _applyIronCorrection(raw) {
        let corrected = {
            x: raw.x - this.hardIronOffset.x,
            y: raw.y - this.hardIronOffset.y,
            z: raw.z - this.hardIronOffset.z
        };

        if (this.softIronCalibrated) {
            corrected = this.softIronMatrix.multiply(corrected);
        }

        return corrected;
    }

    /**
     * Calculate coverage metric for calibration samples
     */
    _calculateCoverage(samples) {
        // Divide sphere into octants and count presence
        const octants = new Array(8).fill(0);

        for (const s of samples) {
            const cx = s.x - this.hardIronOffset.x;
            const cy = s.y - this.hardIronOffset.y;
            const cz = s.z - this.hardIronOffset.z;

            const idx = (cx >= 0 ? 4 : 0) + (cy >= 0 ? 2 : 0) + (cz >= 0 ? 1 : 0);
            octants[idx]++;
        }

        const nonEmpty = octants.filter(c => c > 0).length;
        return nonEmpty / 8;
    }

    /**
     * Calculate covariance matrix
     */
    _calculateCovariance(samples) {
        const n = samples.length;
        let sumX = 0, sumY = 0, sumZ = 0;

        for (const s of samples) {
            sumX += s.x;
            sumY += s.y;
            sumZ += s.z;
        }

        const meanX = sumX / n;
        const meanY = sumY / n;
        const meanZ = sumZ / n;

        let cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

        for (const s of samples) {
            const dx = s.x - meanX;
            const dy = s.y - meanY;
            const dz = s.z - meanZ;

            cov[0][0] += dx * dx;
            cov[0][1] += dx * dy;
            cov[0][2] += dx * dz;
            cov[1][1] += dy * dy;
            cov[1][2] += dy * dz;
            cov[2][2] += dz * dz;
        }

        // Make symmetric
        cov[1][0] = cov[0][1];
        cov[2][0] = cov[0][2];
        cov[2][1] = cov[1][2];

        // Normalize
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                cov[i][j] /= (n - 1);
            }
        }

        return cov;
    }

    /**
     * Check if a specific calibration type has been completed
     * @param {string} type - 'hard_iron', 'soft_iron', or 'earth_field'
     * @returns {boolean} Whether the calibration has been completed
     */
    hasCalibration(type) {
        switch (type) {
            case 'hard_iron':
                return this.hardIronCalibrated;
            case 'soft_iron':
                return this.softIronCalibrated;
            case 'earth_field':
                return this.earthFieldCalibrated;
            default:
                return false;
        }
    }

    /**
     * Save calibration to JSON
     */
    toJSON() {
        return {
            hardIronOffset: this.hardIronOffset,
            softIronMatrix: this.softIronMatrix.toArray(),
            earthField: this.earthField,
            earthFieldMagnitude: this.earthFieldMagnitude,
            hardIronCalibrated: this.hardIronCalibrated,
            softIronCalibrated: this.softIronCalibrated,
            earthFieldCalibrated: this.earthFieldCalibrated,
            timestamp: new Date().toISOString(),
            // Unit metadata (CRITICAL: documents what units these values are in)
            units: {
                hardIronOffset: 'µT',
                earthField: 'µT',
                earthFieldMagnitude: 'µT',
                softIronMatrix: 'dimensionless',
                note: 'Calibration operates in µT units. Raw sensor data (LSB) must be converted before applying calibration.'
            }
        };
    }

    /**
     * Load calibration from JSON
     */
    static fromJSON(json) {
        const cal = new EnvironmentalCalibration();
        cal.hardIronOffset = json.hardIronOffset || { x: 0, y: 0, z: 0 };
        cal.softIronMatrix = json.softIronMatrix ?
            Matrix3.fromArray(json.softIronMatrix) : Matrix3.identity();
        cal.earthField = json.earthField || { x: 0, y: 0, z: 0 };
        cal.earthFieldMagnitude = json.earthFieldMagnitude || 50;
        cal.hardIronCalibrated = json.hardIronCalibrated || false;
        cal.softIronCalibrated = json.softIronCalibrated || false;
        cal.earthFieldCalibrated = json.earthFieldCalibrated || false;
        return cal;
    }

    /**
     * Save to localStorage
     */
    save(key = 'gambit_calibration') {
        localStorage.setItem(key, JSON.stringify(this.toJSON()));
    }

    /**
     * Load calibration from localStorage into this instance
     * @param {string} key - localStorage key
     * @returns {boolean} Whether calibration was loaded successfully
     */
    load(key = 'gambit_calibration') {
        const json = localStorage.getItem(key);
        if (json) {
            const data = JSON.parse(json);
            this.hardIronOffset = data.hardIronOffset || { x: 0, y: 0, z: 0 };
            this.softIronMatrix = data.softIronMatrix ?
                Matrix3.fromArray(data.softIronMatrix) : Matrix3.identity();
            this.earthField = data.earthField || { x: 0, y: 0, z: 0 };
            this.earthFieldMagnitude = data.earthFieldMagnitude || 50;
            this.hardIronCalibrated = data.hardIronCalibrated || false;
            this.softIronCalibrated = data.softIronCalibrated || false;
            this.earthFieldCalibrated = data.earthFieldCalibrated || false;
            return true;
        }
        return false;
    }

    /**
     * Load from localStorage (static factory method)
     */
    static load(key = 'gambit_calibration') {
        const json = localStorage.getItem(key);
        if (json) {
            return EnvironmentalCalibration.fromJSON(JSON.parse(json));
        }
        return new EnvironmentalCalibration();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EnvironmentalCalibration, Matrix3, Quaternion };
}
