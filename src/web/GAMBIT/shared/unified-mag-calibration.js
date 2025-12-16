/**
 * Unified Magnetometer Calibration
 *
 * Single calibration system for GAMBIT magnetometer data that replaces the
 * dual EnvironmentalCalibration + IncrementalCalibration approach.
 *
 * Key design decisions based on investigation (see /docs/technical/earth-field-subtraction-investigation.md):
 *
 * 1. SLIDING WINDOW EARTH ESTIMATION (200 samples)
 *    - World-frame averaging: transform raw readings to world frame, average
 *    - 200-sample sliding window outperforms cumulative (SNR 4.07x vs 3.34x)
 *    - Improvements visible at 50 samples (~1 second at 50Hz)
 *
 * 2. NO HARD IRON FROM MIN/MAX when magnets present
 *    - Min/max method captures magnet extremes, not true hard iron
 *    - Optional hard iron from stored calibration (magnet-free environment)
 *
 * 3. ORIENTATION-COMPENSATED SUBTRACTION
 *    - Earth field stored in world frame
 *    - Rotated to sensor frame using quaternion before subtraction
 *
 * @module shared/unified-mag-calibration
 */

/**
 * UnifiedMagCalibration class
 *
 * Provides real-time Earth field estimation and subtraction using
 * orientation-compensated world-frame averaging.
 */
export class UnifiedMagCalibration {
    /**
     * Create a UnifiedMagCalibration instance
     * @param {Object} options - Configuration options
     * @param {number} [options.windowSize=200] - Sliding window size for Earth estimation
     * @param {number} [options.minSamples=50] - Minimum samples before Earth field is computed
     * @param {Object} [options.hardIronOffset] - Optional hard iron offset {x, y, z} in µT
     * @param {boolean} [options.debug=false] - Enable debug logging
     */
    constructor(options = {}) {
        // Configuration
        this.windowSize = options.windowSize || 200;
        this.minSamples = options.minSamples || 50;
        this.debug = options.debug || false;

        // Hard iron offset (optional, from stored calibration)
        this._hardIronOffset = options.hardIronOffset || { x: 0, y: 0, z: 0 };
        this._hardIronEnabled = !!(options.hardIronOffset);

        // Sliding window buffer for world-frame samples
        this._worldSamples = [];

        // Computed Earth field in world frame
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;

        // Sample count for statistics
        this._totalSamples = 0;

        // Recent residuals for confidence tracking
        this._recentResiduals = [];
        this._maxResidualHistory = 100;

        // Debug logging state
        this._loggedFirstSample = false;
        this._loggedEarthComputed = false;
    }

    /**
     * Update calibration with a new sample
     * Call this for every magnetometer reading
     *
     * @param {number} mx_ut - Magnetometer X in µT
     * @param {number} my_ut - Magnetometer Y in µT
     * @param {number} mz_ut - Magnetometer Z in µT
     * @param {Object} orientation - Quaternion {w, x, y, z} from IMU fusion
     * @returns {Object} Current state {earthReady, confidence, earthMagnitude}
     */
    update(mx_ut, my_ut, mz_ut, orientation) {
        if (!orientation || orientation.w === undefined) {
            return { earthReady: false, confidence: 0, earthMagnitude: 0 };
        }

        this._totalSamples++;

        // Debug first sample
        if (this.debug && !this._loggedFirstSample) {
            const rawMag = Math.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2);
            console.log(`[UnifiedMagCal] First sample:
  Mag (µT): [${mx_ut.toFixed(1)}, ${my_ut.toFixed(1)}, ${mz_ut.toFixed(1)}] |${rawMag.toFixed(1)}| µT
  Orientation: w=${orientation.w.toFixed(3)} x=${orientation.x.toFixed(3)} y=${orientation.y.toFixed(3)} z=${orientation.z.toFixed(3)}`);
            this._loggedFirstSample = true;
        }

        // Apply hard iron correction if enabled
        const corrected = {
            x: mx_ut - this._hardIronOffset.x,
            y: my_ut - this._hardIronOffset.y,
            z: mz_ut - this._hardIronOffset.z
        };

        // Transform to world frame using R^T (sensor → world)
        const R = this._quaternionToRotationMatrix(orientation);
        const worldSample = {
            x: R[0][0] * corrected.x + R[1][0] * corrected.y + R[2][0] * corrected.z,
            y: R[0][1] * corrected.x + R[1][1] * corrected.y + R[2][1] * corrected.z,
            z: R[0][2] * corrected.x + R[1][2] * corrected.y + R[2][2] * corrected.z
        };

        // Add to sliding window
        this._worldSamples.push(worldSample);
        if (this._worldSamples.length > this.windowSize) {
            this._worldSamples.shift();
        }

        // Recompute Earth field estimate
        this._computeEarthField();

        // Track residual if Earth field is ready
        if (this._earthFieldMagnitude > 0) {
            const residual = this.getResidual(mx_ut, my_ut, mz_ut, orientation);
            if (residual) {
                this._recentResiduals.push(residual.magnitude);
                if (this._recentResiduals.length > this._maxResidualHistory) {
                    this._recentResiduals.shift();
                }
            }
        }

        return {
            earthReady: this._earthFieldMagnitude > 0,
            confidence: this.getConfidence(),
            earthMagnitude: this._earthFieldMagnitude
        };
    }

    /**
     * Compute Earth field from sliding window
     * @private
     */
    _computeEarthField() {
        if (this._worldSamples.length < this.minSamples) {
            return;
        }

        const n = this._worldSamples.length;
        let sumX = 0, sumY = 0, sumZ = 0;
        const magnitudes = [];

        for (const s of this._worldSamples) {
            sumX += s.x;
            sumY += s.y;
            sumZ += s.z;
            magnitudes.push(Math.sqrt(s.x**2 + s.y**2 + s.z**2));
        }

        // Average direction in world frame
        const avgX = sumX / n;
        const avgY = sumY / n;
        const avgZ = sumZ / n;

        // Use average of individual magnitudes (more robust with incomplete coverage)
        const avgMagnitude = magnitudes.reduce((a, b) => a + b, 0) / n;

        // Scale direction vector to correct magnitude
        const vectorMagnitude = Math.sqrt(avgX**2 + avgY**2 + avgZ**2);

        if (vectorMagnitude > 0.1) {
            const scale = avgMagnitude / vectorMagnitude;
            this._earthFieldWorld = {
                x: avgX * scale,
                y: avgY * scale,
                z: avgZ * scale
            };
        } else {
            this._earthFieldWorld = { x: avgX, y: avgY, z: avgZ };
        }

        const prevMagnitude = this._earthFieldMagnitude;
        this._earthFieldMagnitude = avgMagnitude;

        // Log when Earth field is first computed
        if (this.debug && prevMagnitude === 0 && this._earthFieldMagnitude > 0 && !this._loggedEarthComputed) {
            console.log(`[UnifiedMagCal] Earth field computed at sample ${this._totalSamples}:
  Earth (world): [${this._earthFieldWorld.x.toFixed(1)}, ${this._earthFieldWorld.y.toFixed(1)}, ${this._earthFieldWorld.z.toFixed(1)}] µT
  Magnitude: ${this._earthFieldMagnitude.toFixed(1)} µT
  Window size: ${n} samples`);
            this._loggedEarthComputed = true;
        }
    }

    /**
     * Get residual (Earth-subtracted reading) for a sample
     *
     * @param {number} mx_ut - Magnetometer X in µT
     * @param {number} my_ut - Magnetometer Y in µT
     * @param {number} mz_ut - Magnetometer Z in µT
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object|null} {x, y, z, magnitude} residual in µT, or null if not ready
     */
    getResidual(mx_ut, my_ut, mz_ut, orientation) {
        if (this._earthFieldMagnitude === 0 || !orientation) {
            return null;
        }

        // Apply hard iron correction
        const corrected = {
            x: mx_ut - this._hardIronOffset.x,
            y: my_ut - this._hardIronOffset.y,
            z: mz_ut - this._hardIronOffset.z
        };

        // Rotate Earth field from world to sensor frame using R
        const R = this._quaternionToRotationMatrix(orientation);
        const earthSensor = {
            x: R[0][0] * this._earthFieldWorld.x + R[0][1] * this._earthFieldWorld.y + R[0][2] * this._earthFieldWorld.z,
            y: R[1][0] * this._earthFieldWorld.x + R[1][1] * this._earthFieldWorld.y + R[1][2] * this._earthFieldWorld.z,
            z: R[2][0] * this._earthFieldWorld.x + R[2][1] * this._earthFieldWorld.y + R[2][2] * this._earthFieldWorld.z
        };

        // Residual = measured - expected (both in sensor frame)
        const residual = {
            x: corrected.x - earthSensor.x,
            y: corrected.y - earthSensor.y,
            z: corrected.z - earthSensor.z
        };

        residual.magnitude = Math.sqrt(residual.x**2 + residual.y**2 + residual.z**2);

        return residual;
    }

    /**
     * Convert quaternion to 3x3 rotation matrix
     * @private
     */
    _quaternionToRotationMatrix(q) {
        const { w, x, y, z } = q;
        return [
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ];
    }

    /**
     * Get calibration confidence (0.0 - 1.0)
     * Based on residual magnitude - lower is better
     * @returns {number}
     */
    getConfidence() {
        if (this._recentResiduals.length < 10) {
            // Not enough data yet, estimate from sample count
            return Math.min(0.5, this._totalSamples / (this.minSamples * 2));
        }

        const meanResidual = this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;

        // Linear confidence: 0 µT → 100%, 20 µT → 0%
        return Math.max(0, Math.min(1, 1 - meanResidual / 20));
    }

    /**
     * Get mean residual magnitude
     * @returns {number} Mean residual in µT (lower is better)
     */
    getMeanResidual() {
        if (this._recentResiduals.length === 0) {
            return Infinity;
        }
        return this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;
    }

    /**
     * Check if Earth field estimation is ready
     * @returns {boolean}
     */
    isReady() {
        return this._earthFieldMagnitude > 0;
    }

    /**
     * Get Earth field vector in world frame
     * @returns {Object} {x, y, z} in µT
     */
    getEarthFieldWorld() {
        return { ...this._earthFieldWorld };
    }

    /**
     * Get Earth field magnitude
     * @returns {number} Magnitude in µT
     */
    getEarthFieldMagnitude() {
        return this._earthFieldMagnitude;
    }

    /**
     * Get hard iron offset
     * @returns {Object} {x, y, z} in µT
     */
    getHardIronOffset() {
        return { ...this._hardIronOffset };
    }

    /**
     * Set hard iron offset (from external calibration)
     * @param {Object} offset - {x, y, z} in µT
     */
    setHardIronOffset(offset) {
        if (offset && typeof offset.x === 'number') {
            this._hardIronOffset = { ...offset };
            this._hardIronEnabled = true;

            // Reset Earth estimation since hard iron changed
            this._worldSamples = [];
            this._earthFieldWorld = { x: 0, y: 0, z: 0 };
            this._earthFieldMagnitude = 0;
            this._loggedEarthComputed = false;

            if (this.debug) {
                console.log(`[UnifiedMagCal] Hard iron offset set: [${offset.x.toFixed(1)}, ${offset.y.toFixed(1)}, ${offset.z.toFixed(1)}] µT`);
            }
        }
    }

    /**
     * Clear hard iron offset
     */
    clearHardIronOffset() {
        this._hardIronOffset = { x: 0, y: 0, z: 0 };
        this._hardIronEnabled = false;

        // Reset Earth estimation
        this._worldSamples = [];
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._loggedEarthComputed = false;
    }

    /**
     * Get current state for telemetry decoration
     * @returns {Object}
     */
    getState() {
        return {
            ready: this.isReady(),
            confidence: this.getConfidence(),
            meanResidual: this.getMeanResidual(),
            earthMagnitude: this._earthFieldMagnitude,
            earthWorld: { ...this._earthFieldWorld },
            hardIronOffset: { ...this._hardIronOffset },
            hardIronEnabled: this._hardIronEnabled,
            windowSize: this._worldSamples.length,
            totalSamples: this._totalSamples
        };
    }

    /**
     * Reset calibration state
     */
    reset() {
        this._worldSamples = [];
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._totalSamples = 0;
        this._recentResiduals = [];
        this._loggedFirstSample = false;
        this._loggedEarthComputed = false;

        if (this.debug) {
            console.log('[UnifiedMagCal] Reset complete');
        }
    }

    /**
     * Export calibration state for saving
     * @returns {Object}
     */
    toJSON() {
        return {
            earthFieldWorld: this._earthFieldWorld,
            earthFieldMagnitude: this._earthFieldMagnitude,
            hardIronOffset: this._hardIronOffset,
            hardIronEnabled: this._hardIronEnabled,
            confidence: this.getConfidence(),
            meanResidual: this.getMeanResidual(),
            windowSize: this.windowSize,
            minSamples: this.minSamples,
            totalSamples: this._totalSamples,
            timestamp: new Date().toISOString(),
            units: {
                earthFieldWorld: 'µT',
                earthFieldMagnitude: 'µT',
                hardIronOffset: 'µT',
                meanResidual: 'µT'
            }
        };
    }
}

/**
 * Create a UnifiedMagCalibration instance
 * @param {Object} options - Configuration options
 * @returns {UnifiedMagCalibration}
 */
export function createUnifiedMagCalibration(options = {}) {
    return new UnifiedMagCalibration(options);
}

export default {
    UnifiedMagCalibration,
    createUnifiedMagCalibration
};
