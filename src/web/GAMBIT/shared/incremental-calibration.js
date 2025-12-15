/**
 * Incremental Magnetometer Calibration
 *
 * Builds calibration from streaming data rather than relying on stored values.
 * This approach was validated by analyze_raw_magnetic.py which achieved < 5 µT
 * residual on 12/14 sessions by computing calibration fresh per-session.
 *
 * Key insight: The stored calibration had incorrect Earth field magnitude (6.8 µT
 * vs expected 25-65 µT), causing ~50 µT residuals. Computing from streaming data
 * with orientation compensation yields correct results.
 *
 * @module shared/incremental-calibration
 */

/**
 * IncrementalCalibration class
 *
 * Accumulates magnetometer samples and orientation data to build calibration
 * incrementally. Tracks confidence metrics based on:
 * - Sample count
 * - Orientation coverage (octants visited)
 * - Sphericity (how spherical the data distribution is)
 * - Earth field stability (consistency of estimates)
 */
export class IncrementalCalibration {
    constructor(options = {}) {
        // Configuration
        this.minSamplesHardIron = options.minSamplesHardIron || 100;
        this.minSamplesEarthField = options.minSamplesEarthField || 50;
        this.windowSize = options.windowSize || 500; // Rolling window for estimates

        // Hard iron estimation (min/max method)
        this.hardIron = {
            minX: Infinity, maxX: -Infinity,
            minY: Infinity, maxY: -Infinity,
            minZ: Infinity, maxZ: -Infinity,
            sampleCount: 0
        };

        // Earth field estimation (world frame average)
        this.earthField = {
            sumX: 0, sumY: 0, sumZ: 0,
            sumSqX: 0, sumSqY: 0, sumSqZ: 0,
            sampleCount: 0,
            recentMagnitudes: [] // Track recent magnitude estimates for stability
        };

        // Orientation coverage tracking (8 octants)
        this.octantCounts = new Array(8).fill(0);

        // Rolling buffer for recent samples (for recalibration)
        this.recentSamples = [];

        // Computed calibration values
        this._hardIronOffset = { x: 0, y: 0, z: 0 };
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;

        // Confidence metrics
        this._hardIronConfidence = 0;
        this._earthFieldConfidence = 0;
    }

    /**
     * Add a sample to the incremental calibration
     * @param {Object} mag - Magnetometer reading in µT {x, y, z}
     * @param {Object} orientation - Quaternion {w, x, y, z} from IMU fusion
     */
    addSample(mag, orientation) {
        if (!mag || mag.x === undefined) return;

        // Update hard iron bounds (min/max)
        this._updateHardIronBounds(mag);

        // If we have orientation, update Earth field estimate
        if (orientation && orientation.w !== undefined) {
            this._updateEarthFieldEstimate(mag, orientation);
        }

        // Store in rolling buffer
        this.recentSamples.push({ mag: { ...mag }, orientation: orientation ? { ...orientation } : null });
        if (this.recentSamples.length > this.windowSize) {
            this.recentSamples.shift();
        }

        // Recompute calibration values
        this._computeCalibration();
    }

    /**
     * Update hard iron min/max bounds
     * @private
     */
    _updateHardIronBounds(mag) {
        const hi = this.hardIron;

        hi.minX = Math.min(hi.minX, mag.x);
        hi.maxX = Math.max(hi.maxX, mag.x);
        hi.minY = Math.min(hi.minY, mag.y);
        hi.maxY = Math.max(hi.maxY, mag.y);
        hi.minZ = Math.min(hi.minZ, mag.z);
        hi.maxZ = Math.max(hi.maxZ, mag.z);
        hi.sampleCount++;

        // Update octant coverage (for confidence)
        const offset = this._hardIronOffset;
        const cx = mag.x - offset.x;
        const cy = mag.y - offset.y;
        const cz = mag.z - offset.z;
        const octant = (cx >= 0 ? 4 : 0) + (cy >= 0 ? 2 : 0) + (cz >= 0 ? 1 : 0);
        this.octantCounts[octant]++;
    }

    /**
     * Update Earth field estimate using orientation to transform to world frame
     * @private
     */
    _updateEarthFieldEstimate(mag, q) {
        // Apply current hard iron correction
        const corrected = {
            x: mag.x - this._hardIronOffset.x,
            y: mag.y - this._hardIronOffset.y,
            z: mag.z - this._hardIronOffset.z
        };

        // Convert quaternion to rotation matrix (sensor to world)
        // R transforms world->sensor, so R.T transforms sensor->world
        const R = this._quaternionToRotationMatrix(q);

        // Transform magnetometer reading to world frame
        // mag_world = R.T @ mag_sensor (transpose for sensor->world)
        const magWorld = {
            x: R[0][0] * corrected.x + R[1][0] * corrected.y + R[2][0] * corrected.z,
            y: R[0][1] * corrected.x + R[1][1] * corrected.y + R[2][1] * corrected.z,
            z: R[0][2] * corrected.x + R[1][2] * corrected.y + R[2][2] * corrected.z
        };

        // Accumulate for running average
        const ef = this.earthField;
        ef.sumX += magWorld.x;
        ef.sumY += magWorld.y;
        ef.sumZ += magWorld.z;
        ef.sumSqX += magWorld.x * magWorld.x;
        ef.sumSqY += magWorld.y * magWorld.y;
        ef.sumSqZ += magWorld.z * magWorld.z;
        ef.sampleCount++;

        // Track recent magnitude for stability metric
        const magnitude = Math.sqrt(magWorld.x ** 2 + magWorld.y ** 2 + magWorld.z ** 2);
        ef.recentMagnitudes.push(magnitude);
        if (ef.recentMagnitudes.length > 100) {
            ef.recentMagnitudes.shift();
        }
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
     * Compute calibration values from accumulated data
     * @private
     */
    _computeCalibration() {
        const hi = this.hardIron;
        const ef = this.earthField;

        // Hard iron offset = center of min/max bounds
        if (hi.sampleCount >= this.minSamplesHardIron) {
            this._hardIronOffset = {
                x: (hi.maxX + hi.minX) / 2,
                y: (hi.maxY + hi.minY) / 2,
                z: (hi.maxZ + hi.minZ) / 2
            };
        }

        // Earth field = average of world-frame readings
        if (ef.sampleCount >= this.minSamplesEarthField) {
            this._earthFieldWorld = {
                x: ef.sumX / ef.sampleCount,
                y: ef.sumY / ef.sampleCount,
                z: ef.sumZ / ef.sampleCount
            };
            this._earthFieldMagnitude = Math.sqrt(
                this._earthFieldWorld.x ** 2 +
                this._earthFieldWorld.y ** 2 +
                this._earthFieldWorld.z ** 2
            );
        }

        // Compute confidence metrics
        this._computeConfidence();
    }

    /**
     * Compute confidence metrics
     * @private
     */
    _computeConfidence() {
        // Hard iron confidence based on:
        // 1. Sample count (more = better)
        // 2. Sphericity (should be roughly spherical)
        // 3. Coverage (should cover all octants)

        const hi = this.hardIron;
        if (hi.sampleCount < this.minSamplesHardIron) {
            this._hardIronConfidence = 0;
        } else {
            const rangeX = hi.maxX - hi.minX;
            const rangeY = hi.maxY - hi.minY;
            const rangeZ = hi.maxZ - hi.minZ;

            // Sphericity: min range / max range (1.0 = perfect sphere)
            const minRange = Math.min(rangeX, rangeY, rangeZ);
            const maxRange = Math.max(rangeX, rangeY, rangeZ);
            const sphericity = maxRange > 0 ? minRange / maxRange : 0;

            // Coverage: fraction of octants visited
            const occupiedOctants = this.octantCounts.filter(c => c > 0).length;
            const coverage = occupiedOctants / 8;

            // Sample factor: ramps up to 1.0 at 500 samples
            const sampleFactor = Math.min(1, hi.sampleCount / 500);

            this._hardIronConfidence = sphericity * coverage * sampleFactor;
        }

        // Earth field confidence based on:
        // 1. Sample count
        // 2. Stability (low variance in recent magnitude estimates)
        // 3. Magnitude sanity (should be 25-65 µT)

        const ef = this.earthField;
        if (ef.sampleCount < this.minSamplesEarthField) {
            this._earthFieldConfidence = 0;
        } else {
            // Sample factor
            const sampleFactor = Math.min(1, ef.sampleCount / 200);

            // Magnitude sanity (peak at 45 µT, drops off outside 25-65 range)
            const mag = this._earthFieldMagnitude;
            let magnitudeSanity = 0;
            if (mag >= 25 && mag <= 65) {
                magnitudeSanity = 1;
            } else if (mag >= 15 && mag < 25) {
                magnitudeSanity = (mag - 15) / 10;
            } else if (mag > 65 && mag <= 80) {
                magnitudeSanity = 1 - (mag - 65) / 15;
            }

            // Stability: based on variance of recent magnitudes
            let stability = 0;
            if (ef.recentMagnitudes.length >= 10) {
                const mean = ef.recentMagnitudes.reduce((a, b) => a + b, 0) / ef.recentMagnitudes.length;
                const variance = ef.recentMagnitudes.reduce((sum, m) => sum + (m - mean) ** 2, 0) / ef.recentMagnitudes.length;
                const stdDev = Math.sqrt(variance);
                // Good stability if std dev < 5% of mean
                stability = Math.max(0, 1 - (stdDev / mean) * 10);
            }

            this._earthFieldConfidence = sampleFactor * Math.max(0.3, magnitudeSanity) * Math.max(0.5, stability);
        }
    }

    /**
     * Get hard iron offset
     * @returns {Object} {x, y, z} in µT
     */
    getHardIronOffset() {
        return { ...this._hardIronOffset };
    }

    /**
     * Get Earth field in world frame
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
     * Get overall confidence (0.0 - 1.0)
     * @returns {number}
     */
    getConfidence() {
        return Math.min(this._hardIronConfidence, this._earthFieldConfidence);
    }

    /**
     * Get detailed confidence breakdown
     * @returns {Object}
     */
    getConfidenceDetails() {
        const hi = this.hardIron;
        const ef = this.earthField;

        return {
            overall: this.getConfidence(),
            hardIron: {
                confidence: this._hardIronConfidence,
                sampleCount: hi.sampleCount,
                offset: this._hardIronOffset,
                ranges: {
                    x: hi.maxX - hi.minX,
                    y: hi.maxY - hi.minY,
                    z: hi.maxZ - hi.minZ
                },
                sphericity: this._computeSphericity(),
                coverage: this.octantCounts.filter(c => c > 0).length / 8
            },
            earthField: {
                confidence: this._earthFieldConfidence,
                sampleCount: ef.sampleCount,
                vector: this._earthFieldWorld,
                magnitude: this._earthFieldMagnitude,
                stability: this._computeEarthFieldStability()
            }
        };
    }

    /**
     * Compute sphericity metric
     * @private
     */
    _computeSphericity() {
        const hi = this.hardIron;
        if (hi.sampleCount < 10) return 0;

        const rangeX = hi.maxX - hi.minX;
        const rangeY = hi.maxY - hi.minY;
        const rangeZ = hi.maxZ - hi.minZ;
        const minRange = Math.min(rangeX, rangeY, rangeZ);
        const maxRange = Math.max(rangeX, rangeY, rangeZ);

        return maxRange > 0 ? minRange / maxRange : 0;
    }

    /**
     * Compute Earth field stability metric
     * @private
     */
    _computeEarthFieldStability() {
        const mags = this.earthField.recentMagnitudes;
        if (mags.length < 10) return 0;

        const mean = mags.reduce((a, b) => a + b, 0) / mags.length;
        const variance = mags.reduce((sum, m) => sum + (m - mean) ** 2, 0) / mags.length;
        const stdDev = Math.sqrt(variance);

        return {
            mean,
            stdDev,
            coefficientOfVariation: mean > 0 ? (stdDev / mean) * 100 : 0 // as percentage
        };
    }

    /**
     * Compute residual for a sample using current calibration
     * @param {Object} mag - Magnetometer reading in µT {x, y, z}
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object} {residual: {x, y, z}, magnitude: number}
     */
    computeResidual(mag, orientation) {
        if (!mag || !orientation) return null;

        // Apply hard iron correction
        const corrected = {
            x: mag.x - this._hardIronOffset.x,
            y: mag.y - this._hardIronOffset.y,
            z: mag.z - this._hardIronOffset.z
        };

        // Rotate Earth field from world to sensor frame
        const R = this._quaternionToRotationMatrix(orientation);
        // R transforms world->sensor directly
        const earthSensor = {
            x: R[0][0] * this._earthFieldWorld.x + R[0][1] * this._earthFieldWorld.y + R[0][2] * this._earthFieldWorld.z,
            y: R[1][0] * this._earthFieldWorld.x + R[1][1] * this._earthFieldWorld.y + R[1][2] * this._earthFieldWorld.z,
            z: R[2][0] * this._earthFieldWorld.x + R[2][1] * this._earthFieldWorld.y + R[2][2] * this._earthFieldWorld.z
        };

        // Residual = measured - expected
        const residual = {
            x: corrected.x - earthSensor.x,
            y: corrected.y - earthSensor.y,
            z: corrected.z - earthSensor.z
        };

        const magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);

        return { residual, magnitude };
    }

    /**
     * Reset calibration state
     */
    reset() {
        this.hardIron = {
            minX: Infinity, maxX: -Infinity,
            minY: Infinity, maxY: -Infinity,
            minZ: Infinity, maxZ: -Infinity,
            sampleCount: 0
        };

        this.earthField = {
            sumX: 0, sumY: 0, sumZ: 0,
            sumSqX: 0, sumSqY: 0, sumSqZ: 0,
            sampleCount: 0,
            recentMagnitudes: []
        };

        this.octantCounts = new Array(8).fill(0);
        this.recentSamples = [];

        this._hardIronOffset = { x: 0, y: 0, z: 0 };
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._hardIronConfidence = 0;
        this._earthFieldConfidence = 0;
    }

    /**
     * Export calibration for saving
     * @returns {Object}
     */
    toJSON() {
        return {
            hardIronOffset: this._hardIronOffset,
            earthField: this._earthFieldWorld,
            earthFieldMagnitude: this._earthFieldMagnitude,
            confidence: {
                overall: this.getConfidence(),
                hardIron: this._hardIronConfidence,
                earthField: this._earthFieldConfidence
            },
            stats: {
                hardIronSamples: this.hardIron.sampleCount,
                earthFieldSamples: this.earthField.sampleCount,
                sphericity: this._computeSphericity(),
                coverage: this.octantCounts.filter(c => c > 0).length / 8
            },
            timestamp: new Date().toISOString(),
            units: {
                hardIronOffset: 'µT',
                earthField: 'µT',
                earthFieldMagnitude: 'µT'
            }
        };
    }
}

/**
 * Create an IncrementalCalibration instance
 * @param {Object} options - Configuration options
 * @returns {IncrementalCalibration}
 */
export function createIncrementalCalibration(options = {}) {
    return new IncrementalCalibration(options);
}

export default {
    IncrementalCalibration,
    createIncrementalCalibration
};
