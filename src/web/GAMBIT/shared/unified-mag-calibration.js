/**
 * Unified Magnetometer Calibration
 *
 * Single class handling all magnetometer calibration:
 * 1. Hard Iron - constant offset from nearby ferromagnetic materials (wizard)
 * 2. Soft Iron - distortion from nearby conductive materials (wizard)
 * 3. Earth Field - real-time estimation using orientation-compensated averaging
 * 4. Extended Baseline - session-start capture with fingers extended (automatic)
 *
 * Design based on investigations:
 * - /docs/technical/earth-field-subtraction-investigation.md
 * - /docs/technical/magnetometer-calibration-complete-analysis.md
 *
 * @module shared/unified-mag-calibration
 */

/**
 * 3x3 Matrix for soft iron correction
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

    toArray() {
        return [
            this.data[0][0], this.data[0][1], this.data[0][2],
            this.data[1][0], this.data[1][1], this.data[1][2],
            this.data[2][0], this.data[2][1], this.data[2][2]
        ];
    }
}

/**
 * UnifiedMagCalibration class
 *
 * Handles iron calibration (wizard) and Earth field estimation (real-time).
 */
export class UnifiedMagCalibration {
    /**
     * Create instance
     * @param {Object} options - Configuration
     * @param {number} [options.windowSize=200] - Sliding window for Earth estimation
     * @param {number} [options.minSamples=50] - Minimum samples before Earth field computed
     * @param {boolean} [options.debug=false] - Enable debug logging
     * @param {boolean} [options.extendedBaselineEnabled=true] - Enable Extended Baseline capture
     * @param {Object} [options.extendedBaseline=null] - Pre-computed baseline {x, y, z} in µT
     * @param {number} [options.baselineMagnitudeThreshold=100] - Max magnitude for valid baseline (µT)
     * @param {number} [options.baselineMinSamples=50] - Min samples for baseline capture (~1 second)
     * @param {number} [options.confidenceResidualThreshold=50] - Residual threshold for 0% confidence (µT)
     */
    constructor(options = {}) {
        // Configuration
        this.windowSize = options.windowSize || 200;
        this.minSamples = options.minSamples || 50;
        this.debug = options.debug || false;

        // Extended Baseline configuration
        this.extendedBaselineEnabled = options.extendedBaselineEnabled !== false; // default true
        this.baselineMagnitudeThreshold = options.baselineMagnitudeThreshold || 100; // µT
        this.baselineMinSamples = options.baselineMinSamples || 50; // ~1 second at 50Hz
        this.autoBaseline = options.autoBaseline !== false; // default true - auto-capture at session start

        // Confidence calculation configuration
        // Threshold of 50µT is more forgiving during motion (Earth field ~50µT)
        this.confidenceResidualThreshold = options.confidenceResidualThreshold || 50; // µT

        // Iron calibration (from wizard)
        this.hardIronOffset = { x: 0, y: 0, z: 0 };
        this.softIronMatrix = Matrix3.identity();
        this.hardIronCalibrated = false;
        this.softIronCalibrated = false;

        // Extended Baseline state (session-start capture)
        this._extendedBaseline = { x: 0, y: 0, z: 0 };
        this._extendedBaselineActive = false;
        this._baselineCapturing = false;
        this._baselineCaptureSamples = [];
        this._autoBaselineAttempted = false; // tracks if auto-baseline has been attempted
        this._autoBaselineRetryCount = 0;
        this._autoBaselineMaxRetries = 5; // retry up to 5 times if quality gate fails

        // Apply pre-computed baseline if provided
        if (options.extendedBaseline) {
            this.setExtendedBaseline(options.extendedBaseline);
            this._autoBaselineAttempted = true; // skip auto if baseline provided
        }

        // Earth field estimation (real-time)
        this._worldSamples = [];
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;

        // Statistics
        this._totalSamples = 0;
        this._recentResiduals = [];
        this._maxResidualHistory = 100;

        // Debug state
        this._loggedFirstSample = false;
        this._loggedEarthComputed = false;
    }

    // =========================================================================
    // IRON CALIBRATION (Wizard)
    // =========================================================================

    /**
     * Run hard iron calibration from collected samples
     * @param {Array} samples - Array of {x, y, z} magnetometer readings
     * @returns {Object} Calibration result with offset and quality
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
            minX = Math.min(minX, s.x); maxX = Math.max(maxX, s.x);
            minY = Math.min(minY, s.y); maxY = Math.max(maxY, s.y);
            minZ = Math.min(minZ, s.z); maxZ = Math.max(maxZ, s.z);
        }

        // Hard iron offset is center of ellipsoid
        this.hardIronOffset = {
            x: (maxX + minX) / 2,
            y: (maxY + minY) / 2,
            z: (maxZ + minZ) / 2
        };

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;

        // Sphericity: how close to sphere (1.0 = perfect)
        const sphericity = Math.min(rangeX, rangeY, rangeZ) / Math.max(rangeX, rangeY, rangeZ);
        const coverage = this._calculateCoverage(samples);

        this.hardIronCalibrated = true;

        // Reset Earth estimation since iron calibration changed
        this._resetEarthEstimation();

        return {
            offset: { ...this.hardIronOffset },
            ranges: { x: rangeX, y: rangeY, z: rangeZ },
            sphericity,
            coverage,
            sampleCount: samples.length,
            quality: sphericity,
            qualityLevel: sphericity > 0.9 && coverage > 0.7 ? 'good' :
                          sphericity > 0.7 && coverage > 0.5 ? 'acceptable' : 'poor'
        };
    }

    /**
     * Run soft iron calibration from collected samples
     * @param {Array} samples - Array of {x, y, z} magnetometer readings
     * @returns {Object} Calibration result
     */
    runSoftIronCalibration(samples) {
        if (samples.length < 200) {
            throw new Error('Need at least 200 samples for soft iron calibration');
        }

        // Apply hard iron correction first
        const corrected = samples.map(s => ({
            x: s.x - this.hardIronOffset.x,
            y: s.y - this.hardIronOffset.y,
            z: s.z - this.hardIronOffset.z
        }));

        // Calculate covariance matrix
        const cov = this._calculateCovariance(corrected);

        // Diagonal scaling (assumes axes aligned)
        const scaleX = Math.sqrt(cov[0][0]);
        const scaleY = Math.sqrt(cov[1][1]);
        const scaleZ = Math.sqrt(cov[2][2]);
        const avgScale = (scaleX + scaleY + scaleZ) / 3;

        // Correction matrix to make ellipsoid into sphere
        this.softIronMatrix = new Matrix3([
            [avgScale / scaleX, 0, 0],
            [0, avgScale / scaleY, 0],
            [0, 0, avgScale / scaleZ]
        ]);

        this.softIronCalibrated = true;

        // Reset Earth estimation
        this._resetEarthEstimation();

        const minScale = Math.min(scaleX, scaleY, scaleZ);
        const maxScale = Math.max(scaleX, scaleY, scaleZ);
        const quality = minScale / maxScale;

        return {
            matrix: this.softIronMatrix.toArray(),
            scales: { x: scaleX, y: scaleY, z: scaleZ },
            correction: { x: avgScale / scaleX, y: avgScale / scaleY, z: avgScale / scaleZ },
            quality
        };
    }

    /**
     * Apply iron correction (hard + soft) to raw reading
     * @param {Object} raw - {x, y, z} in µT
     * @returns {Object} Iron-corrected {x, y, z}
     */
    applyIronCorrection(raw) {
        if (!this.hardIronCalibrated) {
            return { x: raw.x, y: raw.y, z: raw.z };
        }

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
     * Check if any iron calibration available
     * @returns {boolean}
     */
    hasIronCalibration() {
        return this.hardIronCalibrated || this.softIronCalibrated;
    }

    // =========================================================================
    // EXTENDED BASELINE (Session-start capture)
    // =========================================================================

    /**
     * Start baseline capture phase
     * Call this at session start, prompt user to extend fingers and rotate hand.
     * Samples collected during capture are averaged to form the Extended Baseline.
     */
    startBaselineCapture() {
        if (!this.extendedBaselineEnabled) {
            if (this.debug) console.log('[UnifiedMagCal] Extended Baseline disabled, skipping capture');
            return;
        }
        this._baselineCapturing = true;
        this._baselineCaptureSamples = [];
        if (this.debug) console.log('[UnifiedMagCal] Baseline capture started');
    }

    /**
     * End baseline capture and compute Extended Baseline
     * @returns {Object} Result with success, magnitude, quality info
     */
    endBaselineCapture() {
        if (!this._baselineCapturing) {
            return { success: false, reason: 'not_capturing' };
        }

        this._baselineCapturing = false;
        const samples = this._baselineCaptureSamples;

        if (samples.length < this.baselineMinSamples) {
            if (this.debug) console.log(`[UnifiedMagCal] Baseline capture failed: only ${samples.length}/${this.baselineMinSamples} samples`);
            return {
                success: false,
                reason: 'insufficient_samples',
                sampleCount: samples.length,
                required: this.baselineMinSamples
            };
        }

        // Compute mean residual
        const sumX = samples.reduce((s, r) => s + r.x, 0);
        const sumY = samples.reduce((s, r) => s + r.y, 0);
        const sumZ = samples.reduce((s, r) => s + r.z, 0);
        const n = samples.length;
        const baseline = { x: sumX / n, y: sumY / n, z: sumZ / n };
        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

        // Quality gate: reject high-magnitude baselines (fingers not extended)
        if (magnitude > this.baselineMagnitudeThreshold) {
            if (this.debug) console.log(`[UnifiedMagCal] Baseline rejected: magnitude ${magnitude.toFixed(1)} µT > ${this.baselineMagnitudeThreshold} µT threshold`);
            return {
                success: false,
                reason: 'magnitude_too_high',
                magnitude,
                threshold: this.baselineMagnitudeThreshold,
                baseline,
                suggestion: 'Extend fingers further from palm sensor'
            };
        }

        // Accept baseline
        this._extendedBaseline = baseline;
        this._extendedBaselineActive = true;

        if (this.debug) console.log(`[UnifiedMagCal] Baseline captured: [${baseline.x.toFixed(1)}, ${baseline.y.toFixed(1)}, ${baseline.z.toFixed(1)}] |${magnitude.toFixed(1)}| µT`);

        return {
            success: true,
            baseline: { ...baseline },
            magnitude,
            sampleCount: n,
            quality: magnitude < 60 ? 'excellent' : magnitude < 80 ? 'good' : 'acceptable'
        };
    }

    /**
     * Manually set Extended Baseline (e.g., from stored calibration)
     * @param {Object} baseline - {x, y, z} in µT
     * @returns {Object} Result with success and magnitude
     */
    setExtendedBaseline(baseline) {
        if (!baseline || typeof baseline.x !== 'number') {
            return { success: false, reason: 'invalid_baseline' };
        }

        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

        // Optional quality check (can be bypassed by providing baseline directly)
        if (magnitude > this.baselineMagnitudeThreshold * 2) {
            if (this.debug) console.log(`[UnifiedMagCal] Warning: provided baseline magnitude ${magnitude.toFixed(1)} µT is very high`);
        }

        this._extendedBaseline = { x: baseline.x, y: baseline.y, z: baseline.z };
        this._extendedBaselineActive = true;

        if (this.debug) console.log(`[UnifiedMagCal] Extended Baseline set: [${baseline.x.toFixed(1)}, ${baseline.y.toFixed(1)}, ${baseline.z.toFixed(1)}] |${magnitude.toFixed(1)}| µT`);

        return { success: true, magnitude };
    }

    /**
     * Clear Extended Baseline
     * Also resets auto-baseline state so it can retry
     */
    clearExtendedBaseline() {
        this._extendedBaseline = { x: 0, y: 0, z: 0 };
        this._extendedBaselineActive = false;
        this._baselineCapturing = false;
        this._baselineCaptureSamples = [];
        this._autoBaselineAttempted = false;
        this._autoBaselineRetryCount = 0;
        if (this.debug) console.log('[UnifiedMagCal] Extended Baseline cleared (auto-baseline can retry)');
    }

    /**
     * Check if Extended Baseline is active
     * @returns {boolean}
     */
    hasExtendedBaseline() {
        return this._extendedBaselineActive;
    }

    /**
     * Get current Extended Baseline
     * @returns {Object} {x, y, z, magnitude, active}
     */
    getExtendedBaseline() {
        const mag = Math.sqrt(
            this._extendedBaseline.x ** 2 +
            this._extendedBaseline.y ** 2 +
            this._extendedBaseline.z ** 2
        );
        return {
            x: this._extendedBaseline.x,
            y: this._extendedBaseline.y,
            z: this._extendedBaseline.z,
            magnitude: mag,
            active: this._extendedBaselineActive
        };
    }

    /**
     * Check if currently capturing baseline
     * @returns {boolean}
     */
    isCapturingBaseline() {
        return this._baselineCapturing;
    }

    // =========================================================================
    // EARTH FIELD ESTIMATION (Real-time)
    // =========================================================================

    /**
     * Update with new sample - applies iron correction and estimates Earth field
     * @param {number} mx_ut - Raw magnetometer X in µT
     * @param {number} my_ut - Raw magnetometer Y in µT
     * @param {number} mz_ut - Raw magnetometer Z in µT
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object} {earthReady, confidence, earthMagnitude}
     */
    update(mx_ut, my_ut, mz_ut, orientation) {
        if (!orientation || orientation.w === undefined) {
            return { earthReady: false, confidence: 0, earthMagnitude: 0 };
        }

        this._totalSamples++;

        // Auto-start baseline capture on first sample if enabled and not already captured
        if (this.autoBaseline && this.extendedBaselineEnabled &&
            !this._extendedBaselineActive && !this._baselineCapturing &&
            this._autoBaselineRetryCount < this._autoBaselineMaxRetries) {
            this._baselineCapturing = true;
            this._baselineCaptureSamples = [];
            if (this.debug) console.log('[UnifiedMagCal] Auto-baseline capture started');
        }

        // Apply iron correction
        const ironCorrected = this.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

        // Debug first sample
        if (this.debug && !this._loggedFirstSample) {
            const mag = Math.sqrt(ironCorrected.x**2 + ironCorrected.y**2 + ironCorrected.z**2);
            console.log(`[UnifiedMagCal] First sample: [${ironCorrected.x.toFixed(1)}, ${ironCorrected.y.toFixed(1)}, ${ironCorrected.z.toFixed(1)}] |${mag.toFixed(1)}| µT`);
            this._loggedFirstSample = true;
        }

        // Transform to world frame using R^T (sensor → world)
        const R = this._quaternionToRotationMatrix(orientation);
        const worldSample = {
            x: R[0][0] * ironCorrected.x + R[1][0] * ironCorrected.y + R[2][0] * ironCorrected.z,
            y: R[0][1] * ironCorrected.x + R[1][1] * ironCorrected.y + R[2][1] * ironCorrected.z,
            z: R[0][2] * ironCorrected.x + R[1][2] * ironCorrected.y + R[2][2] * ironCorrected.z
        };

        // Add to sliding window
        this._worldSamples.push(worldSample);
        if (this._worldSamples.length > this.windowSize) {
            this._worldSamples.shift();
        }

        // Recompute Earth field
        this._computeEarthField();

        // Track residual and collect baseline samples
        let autoBaselineResult = null;
        if (this._earthFieldMagnitude > 0) {
            const residual = this._getEarthResidual(mx_ut, my_ut, mz_ut, orientation);
            if (residual) {
                this._recentResiduals.push(residual.magnitude);
                if (this._recentResiduals.length > this._maxResidualHistory) {
                    this._recentResiduals.shift();
                }

                // Collect samples during baseline capture phase
                if (this._baselineCapturing) {
                    this._baselineCaptureSamples.push({ x: residual.x, y: residual.y, z: residual.z });

                    // Auto-complete baseline when we have enough samples
                    if (this.autoBaseline && this._baselineCaptureSamples.length >= this.baselineMinSamples) {
                        autoBaselineResult = this._attemptAutoBaseline();
                    }
                }
            }
        }

        return {
            earthReady: this._earthFieldMagnitude > 0,
            confidence: this.getConfidence(),
            earthMagnitude: this._earthFieldMagnitude,
            capturingBaseline: this._baselineCapturing,
            baselineSampleCount: this._baselineCaptureSamples.length,
            autoBaselineResult // null or {success, magnitude, quality} if auto-baseline completed
        };
    }

    /**
     * Attempt automatic baseline completion
     * @private
     * @returns {Object} Result with success, magnitude, quality info
     */
    _attemptAutoBaseline() {
        const samples = this._baselineCaptureSamples;
        const n = samples.length;

        // Compute mean residual
        const sumX = samples.reduce((s, r) => s + r.x, 0);
        const sumY = samples.reduce((s, r) => s + r.y, 0);
        const sumZ = samples.reduce((s, r) => s + r.z, 0);
        const baseline = { x: sumX / n, y: sumY / n, z: sumZ / n };
        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

        // Quality gate: reject high-magnitude baselines (magnets present)
        if (magnitude > this.baselineMagnitudeThreshold) {
            this._autoBaselineRetryCount++;
            if (this.debug) {
                console.log(`[UnifiedMagCal] Auto-baseline rejected (attempt ${this._autoBaselineRetryCount}/${this._autoBaselineMaxRetries}): magnitude ${magnitude.toFixed(1)} µT > ${this.baselineMagnitudeThreshold} µT`);
            }

            // Clear samples and retry if we haven't exceeded max retries
            if (this._autoBaselineRetryCount < this._autoBaselineMaxRetries) {
                this._baselineCaptureSamples = [];
                // Keep capturing
            } else {
                // Give up on auto-baseline
                this._baselineCapturing = false;
                if (this.debug) console.log('[UnifiedMagCal] Auto-baseline gave up after max retries');
            }

            return {
                success: false,
                reason: 'magnitude_too_high',
                magnitude,
                threshold: this.baselineMagnitudeThreshold,
                attempt: this._autoBaselineRetryCount
            };
        }

        // Accept baseline
        this._extendedBaseline = baseline;
        this._extendedBaselineActive = true;
        this._baselineCapturing = false;
        this._autoBaselineAttempted = true;

        if (this.debug) {
            const quality = magnitude < 60 ? 'excellent' : magnitude < 80 ? 'good' : 'acceptable';
            console.log(`[UnifiedMagCal] Auto-baseline captured: [${baseline.x.toFixed(1)}, ${baseline.y.toFixed(1)}, ${baseline.z.toFixed(1)}] |${magnitude.toFixed(1)}| µT (${quality})`);
        }

        return {
            success: true,
            baseline: { ...baseline },
            magnitude,
            sampleCount: n,
            quality: magnitude < 60 ? 'excellent' : magnitude < 80 ? 'good' : 'acceptable'
        };
    }

    /**
     * Get residual (fully corrected: iron + Earth + Extended Baseline subtracted)
     * @param {number} mx_ut - Raw magnetometer X in µT
     * @param {number} my_ut - Raw magnetometer Y in µT
     * @param {number} mz_ut - Raw magnetometer Z in µT
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object|null} {x, y, z, magnitude} or null if not ready
     */
    getResidual(mx_ut, my_ut, mz_ut, orientation) {
        // Get Earth-subtracted residual
        const earthResidual = this._getEarthResidual(mx_ut, my_ut, mz_ut, orientation);
        if (!earthResidual) {
            return null;
        }

        // Apply Extended Baseline if active
        if (this._extendedBaselineActive) {
            const residual = {
                x: earthResidual.x - this._extendedBaseline.x,
                y: earthResidual.y - this._extendedBaseline.y,
                z: earthResidual.z - this._extendedBaseline.z
            };
            residual.magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);
            return residual;
        }

        return earthResidual;
    }

    /**
     * Get Earth-only residual (iron + Earth subtracted, no Extended Baseline)
     * @private
     * @param {number} mx_ut - Raw magnetometer X in µT
     * @param {number} my_ut - Raw magnetometer Y in µT
     * @param {number} mz_ut - Raw magnetometer Z in µT
     * @param {Object} orientation - Quaternion {w, x, y, z}
     * @returns {Object|null} {x, y, z, magnitude} or null if not ready
     */
    _getEarthResidual(mx_ut, my_ut, mz_ut, orientation) {
        if (this._earthFieldMagnitude === 0 || !orientation) {
            return null;
        }

        // Apply iron correction
        const ironCorrected = this.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

        // Rotate Earth field from world to sensor frame
        const R = this._quaternionToRotationMatrix(orientation);
        const earthSensor = {
            x: R[0][0] * this._earthFieldWorld.x + R[0][1] * this._earthFieldWorld.y + R[0][2] * this._earthFieldWorld.z,
            y: R[1][0] * this._earthFieldWorld.x + R[1][1] * this._earthFieldWorld.y + R[1][2] * this._earthFieldWorld.z,
            z: R[2][0] * this._earthFieldWorld.x + R[2][1] * this._earthFieldWorld.y + R[2][2] * this._earthFieldWorld.z
        };

        // Residual = iron-corrected - expected Earth field
        const residual = {
            x: ironCorrected.x - earthSensor.x,
            y: ironCorrected.y - earthSensor.y,
            z: ironCorrected.z - earthSensor.z
        };
        residual.magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);

        return residual;
    }

    /**
     * Check if Earth field estimation ready
     * @returns {boolean}
     */
    isReady() {
        return this._earthFieldMagnitude > 0;
    }

    /**
     * Get confidence (0-1) based on residual magnitude
     *
     * Note: This measures instantaneous residual accuracy, not calibration quality.
     * During motion, residuals naturally increase due to Earth estimate lag.
     * See getCalibrationQuality() for orientation diversity metric.
     *
     * @returns {number}
     */
    getConfidence() {
        if (this._recentResiduals.length < 10) {
            return Math.min(0.5, this._totalSamples / (this.minSamples * 2));
        }
        const meanResidual = this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;
        // Use configurable threshold (default 50µT) instead of hardcoded 20µT
        return Math.max(0, Math.min(1, 1 - meanResidual / this.confidenceResidualThreshold));
    }

    /**
     * Get calibration quality (0-1) based on orientation diversity
     *
     * This is a better metric for "how well calibrated" the system is,
     * as it measures the diversity of orientations in the sample window
     * rather than instantaneous residual accuracy.
     *
     * @returns {Object} {quality: number, diversityRatio: number, windowFill: number}
     */
    getCalibrationQuality() {
        const windowFill = Math.min(1, this._worldSamples.length / this.windowSize);

        // Measure orientation diversity via variance in world-frame samples
        if (this._worldSamples.length < this.minSamples) {
            return { quality: windowFill * 0.5, diversityRatio: 0, windowFill };
        }

        const samples = this._worldSamples;
        const n = samples.length;

        // Compute variance in each axis
        const meanX = samples.reduce((s, p) => s + p.x, 0) / n;
        const meanY = samples.reduce((s, p) => s + p.y, 0) / n;
        const meanZ = samples.reduce((s, p) => s + p.z, 0) / n;

        const varX = samples.reduce((s, p) => s + (p.x - meanX) ** 2, 0) / n;
        const varY = samples.reduce((s, p) => s + (p.y - meanY) ** 2, 0) / n;
        const varZ = samples.reduce((s, p) => s + (p.z - meanZ) ** 2, 0) / n;

        // Total variance - higher means more diverse orientations
        const totalVar = Math.sqrt(varX + varY + varZ);

        // For ideal calibration with diverse rotations, we'd expect variance
        // roughly equal to Earth field magnitude (samples spread around sphere)
        // Static device has near-zero variance
        const diversityRatio = Math.min(1, totalVar / (this._earthFieldMagnitude || 50));

        // Quality combines window fill and orientation diversity
        const quality = windowFill * (0.5 + 0.5 * diversityRatio);

        return { quality, diversityRatio, windowFill };
    }

    /**
     * Get mean residual magnitude
     * @returns {number}
     */
    getMeanResidual() {
        if (this._recentResiduals.length === 0) return Infinity;
        return this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;
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
     * @returns {number}
     */
    getEarthFieldMagnitude() {
        return this._earthFieldMagnitude;
    }

    // =========================================================================
    // STATE & PERSISTENCE
    // =========================================================================

    /**
     * Get current state
     * @returns {Object}
     */
    getState() {
        const baselineMag = Math.sqrt(
            this._extendedBaseline.x ** 2 +
            this._extendedBaseline.y ** 2 +
            this._extendedBaseline.z ** 2
        );
        const calibrationQuality = this.getCalibrationQuality();
        return {
            ready: this.isReady(),
            confidence: this.getConfidence(),
            meanResidual: this.getMeanResidual(),
            earthMagnitude: this._earthFieldMagnitude,
            earthWorld: { ...this._earthFieldWorld },
            hardIronCalibrated: this.hardIronCalibrated,
            softIronCalibrated: this.softIronCalibrated,
            extendedBaselineActive: this._extendedBaselineActive,
            extendedBaseline: { ...this._extendedBaseline },
            extendedBaselineMagnitude: baselineMag,
            capturingBaseline: this._baselineCapturing,
            baselineSampleCount: this._baselineCaptureSamples.length,
            windowSize: this._worldSamples.length,
            totalSamples: this._totalSamples,
            // Calibration quality based on orientation diversity (better metric during motion)
            calibrationQuality: calibrationQuality.quality,
            diversityRatio: calibrationQuality.diversityRatio,
            windowFill: calibrationQuality.windowFill,
            // Auto-baseline status
            autoBaselineEnabled: this.autoBaseline,
            autoBaselineRetryCount: this._autoBaselineRetryCount,
            autoBaselineMaxRetries: this._autoBaselineMaxRetries
        };
    }

    /**
     * Reset Earth field estimation (keeps iron calibration)
     */
    resetEarthEstimation() {
        this._resetEarthEstimation();
    }

    /**
     * Full reset (clears everything including iron calibration and Extended Baseline)
     */
    reset() {
        this.hardIronOffset = { x: 0, y: 0, z: 0 };
        this.softIronMatrix = Matrix3.identity();
        this.hardIronCalibrated = false;
        this.softIronCalibrated = false;
        this._resetEarthEstimation();
        this.clearExtendedBaseline();
        this._totalSamples = 0;
        this._loggedFirstSample = false;

        if (this.debug) console.log('[UnifiedMagCal] Full reset');
    }

    /**
     * Save to localStorage
     * @param {string} key
     */
    save(key = 'gambit_calibration') {
        localStorage.setItem(key, JSON.stringify(this.toJSON()));
    }

    /**
     * Load from localStorage
     * @param {string} key
     * @returns {boolean} Success
     */
    load(key = 'gambit_calibration') {
        const json = localStorage.getItem(key);
        if (json) {
            this.fromJSON(JSON.parse(json));
            return true;
        }
        return false;
    }

    /**
     * Export to JSON
     * @returns {Object}
     */
    toJSON() {
        const baselineMag = Math.sqrt(
            this._extendedBaseline.x ** 2 +
            this._extendedBaseline.y ** 2 +
            this._extendedBaseline.z ** 2
        );
        return {
            hardIronOffset: this.hardIronOffset,
            softIronMatrix: this.softIronMatrix.toArray(),
            hardIronCalibrated: this.hardIronCalibrated,
            softIronCalibrated: this.softIronCalibrated,
            // Extended Baseline (persisted for session continuity)
            extendedBaseline: this._extendedBaselineActive ? { ...this._extendedBaseline } : null,
            extendedBaselineMagnitude: this._extendedBaselineActive ? baselineMag : null,
            // Earth field state (informational, will be re-estimated)
            earthFieldWorld: this._earthFieldWorld,
            earthFieldMagnitude: this._earthFieldMagnitude,
            timestamp: new Date().toISOString(),
            units: {
                hardIronOffset: 'µT',
                softIronMatrix: 'dimensionless',
                extendedBaseline: 'µT',
                earthFieldWorld: 'µT',
                earthFieldMagnitude: 'µT'
            }
        };
    }

    /**
     * Import from JSON
     * @param {Object} json
     */
    fromJSON(json) {
        this.hardIronOffset = json.hardIronOffset || { x: 0, y: 0, z: 0 };
        this.softIronMatrix = json.softIronMatrix ? Matrix3.fromArray(json.softIronMatrix) : Matrix3.identity();
        this.hardIronCalibrated = json.hardIronCalibrated || false;
        this.softIronCalibrated = json.softIronCalibrated || false;

        // Restore Extended Baseline if available
        if (json.extendedBaseline) {
            this.setExtendedBaseline(json.extendedBaseline);
        } else {
            this.clearExtendedBaseline();
        }

        // Don't restore Earth field - let it re-estimate in real-time
        this._resetEarthEstimation();
    }

    // =========================================================================
    // PRIVATE METHODS
    // =========================================================================

    _resetEarthEstimation() {
        this._worldSamples = [];
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._recentResiduals = [];
        this._loggedEarthComputed = false;
    }

    _computeEarthField() {
        if (this._worldSamples.length < this.minSamples) return;

        const n = this._worldSamples.length;
        let sumX = 0, sumY = 0, sumZ = 0;
        const magnitudes = [];

        for (const s of this._worldSamples) {
            sumX += s.x; sumY += s.y; sumZ += s.z;
            magnitudes.push(Math.sqrt(s.x**2 + s.y**2 + s.z**2));
        }

        const avgX = sumX / n, avgY = sumY / n, avgZ = sumZ / n;
        const avgMagnitude = magnitudes.reduce((a, b) => a + b, 0) / n;
        const vectorMagnitude = Math.sqrt(avgX**2 + avgY**2 + avgZ**2);

        if (vectorMagnitude > 0.1) {
            const scale = avgMagnitude / vectorMagnitude;
            this._earthFieldWorld = { x: avgX * scale, y: avgY * scale, z: avgZ * scale };
        } else {
            this._earthFieldWorld = { x: avgX, y: avgY, z: avgZ };
        }

        const prevMagnitude = this._earthFieldMagnitude;
        this._earthFieldMagnitude = avgMagnitude;

        if (this.debug && prevMagnitude === 0 && this._earthFieldMagnitude > 0 && !this._loggedEarthComputed) {
            console.log(`[UnifiedMagCal] Earth field: [${this._earthFieldWorld.x.toFixed(1)}, ${this._earthFieldWorld.y.toFixed(1)}, ${this._earthFieldWorld.z.toFixed(1)}] |${this._earthFieldMagnitude.toFixed(1)}| µT`);
            this._loggedEarthComputed = true;
        }
    }

    _quaternionToRotationMatrix(q) {
        const { w, x, y, z } = q;
        return [
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ];
    }

    _calculateCoverage(samples) {
        const octants = new Array(8).fill(0);
        for (const s of samples) {
            const cx = s.x - this.hardIronOffset.x;
            const cy = s.y - this.hardIronOffset.y;
            const cz = s.z - this.hardIronOffset.z;
            const idx = (cx >= 0 ? 4 : 0) + (cy >= 0 ? 2 : 0) + (cz >= 0 ? 1 : 0);
            octants[idx]++;
        }
        return octants.filter(c => c > 0).length / 8;
    }

    _calculateCovariance(samples) {
        const n = samples.length;
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const s of samples) { sumX += s.x; sumY += s.y; sumZ += s.z; }
        const meanX = sumX / n, meanY = sumY / n, meanZ = sumZ / n;

        const cov = [[0,0,0], [0,0,0], [0,0,0]];
        for (const s of samples) {
            const dx = s.x - meanX, dy = s.y - meanY, dz = s.z - meanZ;
            cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz;
            cov[1][1] += dy*dy; cov[1][2] += dy*dz; cov[2][2] += dz*dz;
        }
        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] /= (n - 1);
        return cov;
    }
}

/**
 * Create instance
 * @param {Object} options
 * @returns {UnifiedMagCalibration}
 */
export function createUnifiedMagCalibration(options = {}) {
    return new UnifiedMagCalibration(options);
}

export default { UnifiedMagCalibration, createUnifiedMagCalibration };
