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
            quality: sphericity > 0.9 && coverage > 0.7 ? 'good' :
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

        return {
            matrix: this.softIronMatrix.toArray(),
            scales: { x: scaleX, y: scaleY, z: scaleZ },
            correction: { x: avgScale / scaleX, y: avgScale / scaleY, z: avgScale / scaleZ }
        };
    }

    /**
     * Calibrate Earth field reference
     * Should be done with sensor in a known orientation (e.g., flat on table)
     *
     * @param {Array} samples - Array of {x, y, z} readings in reference orientation
     * @returns {Object} Earth field estimate
     */
    runEarthFieldCalibration(samples) {
        if (samples.length < 50) {
            throw new Error('Need at least 50 samples for Earth field calibration');
        }

        // Apply hard/soft iron corrections first
        const corrected = samples.map(s => this._applyIronCorrection(s));

        // Average the corrected readings
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const s of corrected) {
            sumX += s.x;
            sumY += s.y;
            sumZ += s.z;
        }

        this.earthField = {
            x: sumX / corrected.length,
            y: sumY / corrected.length,
            z: sumZ / corrected.length
        };

        this.earthFieldMagnitude = Math.sqrt(
            this.earthField.x ** 2 +
            this.earthField.y ** 2 +
            this.earthField.z ** 2
        );

        this.earthFieldCalibrated = true;

        return {
            field: { ...this.earthField },
            magnitude: this.earthFieldMagnitude,
            // Convert to μT using scale factor
            magnitudeUT: this.earthFieldMagnitude * this.scaleFactorLSBToUT
        };
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

        return corrected;
    }

    /**
     * Apply only iron corrections (no Earth field subtraction)
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
            earthFieldCalibrated: this.earthFieldCalibrated
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
     * Load from localStorage
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
