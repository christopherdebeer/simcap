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

import type { Quaternion } from '@core/types';

// ===== Type Definitions =====

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface CalibrationOptions {
  windowSize?: number;
  minSamples?: number;
  debug?: boolean;
  extendedBaselineEnabled?: boolean;
  extendedBaseline?: Vector3 | null;
  baselineMagnitudeThreshold?: number;
  baselineMinSamples?: number;
  confidenceResidualThreshold?: number;
  autoBaseline?: boolean;
}

export interface HardIronResult {
  offset: Vector3;
  ranges: Vector3;
  sphericity: number;
  coverage: number;
  sampleCount: number;
  quality: number;
  qualityLevel: 'good' | 'acceptable' | 'poor';
}

export interface SoftIronResult {
  matrix: number[];
  scales: Vector3;
  correction: Vector3;
  quality: number;
}

export interface BaselineCaptureResult {
  success: boolean;
  reason?: string;
  sampleCount?: number;
  required?: number;
  magnitude?: number;
  threshold?: number;
  baseline?: Vector3;
  suggestion?: string;
  quality?: 'excellent' | 'good' | 'acceptable';
  attempt?: number;
}

export interface UpdateResult {
  earthReady: boolean;
  confidence: number;
  earthMagnitude: number;
  capturingBaseline?: boolean;
  baselineSampleCount?: number;
  autoBaselineResult?: BaselineCaptureResult | null;
}

export interface ResidualResult extends Vector3 {
  magnitude: number;
}

export interface CalibrationState {
  ready: boolean;
  confidence: number;
  meanResidual: number;
  earthMagnitude: number;
  earthWorld: Vector3;
  hardIronCalibrated: boolean;
  softIronCalibrated: boolean;
  extendedBaselineActive: boolean;
  extendedBaseline: Vector3;
  extendedBaselineMagnitude: number;
  capturingBaseline: boolean;
  baselineSampleCount: number;
  windowSize: number;
  totalSamples: number;
  calibrationQuality: number;
  diversityRatio: number;
  windowFill: number;
  autoBaselineEnabled: boolean;
  autoBaselineRetryCount: number;
  autoBaselineMaxRetries: number;
}

export interface CalibrationQuality {
  quality: number;
  diversityRatio: number;
  windowFill: number;
}

export interface CalibrationJSON {
  hardIronOffset: Vector3;
  softIronMatrix: number[];
  hardIronCalibrated: boolean;
  softIronCalibrated: boolean;
  extendedBaseline: Vector3 | null;
  extendedBaselineMagnitude: number | null;
  earthFieldWorld: Vector3;
  earthFieldMagnitude: number;
  timestamp: string;
  units: {
    hardIronOffset: string;
    softIronMatrix: string;
    extendedBaseline: string;
    earthFieldWorld: string;
    earthFieldMagnitude: string;
  };
}

export interface ExtendedBaselineState extends Vector3 {
  magnitude: number;
  active: boolean;
}

type Matrix3x3 = [[number, number, number], [number, number, number], [number, number, number]];

// ===== Matrix3 Class =====

/**
 * 3x3 Matrix for soft iron correction
 */
class Matrix3 {
    data: Matrix3x3;

    constructor(data: Matrix3x3 | null = null) {
        this.data = data || [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
    }

    static identity(): Matrix3 {
        return new Matrix3();
    }

    static fromArray(arr: number[]): Matrix3 {
        return new Matrix3([
            [arr[0], arr[1], arr[2]],
            [arr[3], arr[4], arr[5]],
            [arr[6], arr[7], arr[8]]
        ]);
    }

    multiply(vec: Vector3): Vector3 {
        return {
            x: this.data[0][0] * vec.x + this.data[0][1] * vec.y + this.data[0][2] * vec.z,
            y: this.data[1][0] * vec.x + this.data[1][1] * vec.y + this.data[1][2] * vec.z,
            z: this.data[2][0] * vec.x + this.data[2][1] * vec.y + this.data[2][2] * vec.z
        };
    }

    toArray(): number[] {
        return [
            this.data[0][0], this.data[0][1], this.data[0][2],
            this.data[1][0], this.data[1][1], this.data[1][2],
            this.data[2][0], this.data[2][1], this.data[2][2]
        ];
    }
}

// ===== UnifiedMagCalibration Class =====

/**
 * UnifiedMagCalibration class
 *
 * Handles iron calibration (wizard) and Earth field estimation (real-time).
 */
export class UnifiedMagCalibration {
    // Configuration
    private windowSize: number;
    private minSamples: number;
    private debug: boolean;

    // Extended Baseline configuration
    private extendedBaselineEnabled: boolean;
    private baselineMagnitudeThreshold: number;
    private baselineMinSamples: number;
    private autoBaseline: boolean;
    private confidenceResidualThreshold: number;

    // Iron calibration (from wizard)
    private hardIronOffset: Vector3 = { x: 0, y: 0, z: 0 };
    private softIronMatrix: Matrix3 = Matrix3.identity();
    hardIronCalibrated: boolean = false;
    softIronCalibrated: boolean = false;

    // Extended Baseline state (session-start capture)
    private _extendedBaseline: Vector3 = { x: 0, y: 0, z: 0 };
    private _extendedBaselineActive: boolean = false;
    private _baselineCapturing: boolean = false;
    private _baselineCaptureSamples: Vector3[] = [];
    private _autoBaselineAttempted: boolean = false;
    private _autoBaselineRetryCount: number = 0;
    private _autoBaselineMaxRetries: number = 5;

    // Earth field estimation (real-time)
    private _worldSamples: Vector3[] = [];
    private _earthFieldWorld: Vector3 = { x: 0, y: 0, z: 0 };
    private _earthFieldMagnitude: number = 0;

    // Statistics
    private _totalSamples: number = 0;
    private _recentResiduals: number[] = [];
    private _maxResidualHistory: number = 100;

    // Debug state
    private _loggedFirstSample: boolean = false;
    private _loggedEarthComputed: boolean = false;

    /**
     * Create instance
     */
    constructor(options: CalibrationOptions = {}) {
        // Configuration
        this.windowSize = options.windowSize || 200;
        this.minSamples = options.minSamples || 50;
        this.debug = options.debug || false;

        // Extended Baseline configuration
        this.extendedBaselineEnabled = options.extendedBaselineEnabled !== false;
        this.baselineMagnitudeThreshold = options.baselineMagnitudeThreshold || 100;
        this.baselineMinSamples = options.baselineMinSamples || 50;
        this.autoBaseline = options.autoBaseline !== false;
        this.confidenceResidualThreshold = options.confidenceResidualThreshold || 50;

        // Apply pre-computed baseline if provided
        if (options.extendedBaseline) {
            this.setExtendedBaseline(options.extendedBaseline);
            this._autoBaselineAttempted = true;
        }
    }

    // =========================================================================
    // IRON CALIBRATION (Wizard)
    // =========================================================================

    /**
     * Run hard iron calibration from collected samples
     */
    runHardIronCalibration(samples: Vector3[]): HardIronResult {
        if (samples.length < 100) {
            throw new Error('Need at least 100 samples for hard iron calibration');
        }

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (const s of samples) {
            minX = Math.min(minX, s.x); maxX = Math.max(maxX, s.x);
            minY = Math.min(minY, s.y); maxY = Math.max(maxY, s.y);
            minZ = Math.min(minZ, s.z); maxZ = Math.max(maxZ, s.z);
        }

        this.hardIronOffset = {
            x: (maxX + minX) / 2,
            y: (maxY + minY) / 2,
            z: (maxZ + minZ) / 2
        };

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;

        const sphericity = Math.min(rangeX, rangeY, rangeZ) / Math.max(rangeX, rangeY, rangeZ);
        const coverage = this._calculateCoverage(samples);

        this.hardIronCalibrated = true;
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
     */
    runSoftIronCalibration(samples: Vector3[]): SoftIronResult {
        if (samples.length < 200) {
            throw new Error('Need at least 200 samples for soft iron calibration');
        }

        const corrected = samples.map(s => ({
            x: s.x - this.hardIronOffset.x,
            y: s.y - this.hardIronOffset.y,
            z: s.z - this.hardIronOffset.z
        }));

        const cov = this._calculateCovariance(corrected);

        const scaleX = Math.sqrt(cov[0][0]);
        const scaleY = Math.sqrt(cov[1][1]);
        const scaleZ = Math.sqrt(cov[2][2]);
        const avgScale = (scaleX + scaleY + scaleZ) / 3;

        this.softIronMatrix = new Matrix3([
            [avgScale / scaleX, 0, 0],
            [0, avgScale / scaleY, 0],
            [0, 0, avgScale / scaleZ]
        ]);

        this.softIronCalibrated = true;
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
     */
    applyIronCorrection(raw: Vector3): Vector3 {
        if (!this.hardIronCalibrated) {
            return { x: raw.x, y: raw.y, z: raw.z };
        }

        let corrected: Vector3 = {
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
     */
    hasIronCalibration(): boolean {
        return this.hardIronCalibrated || this.softIronCalibrated;
    }

    // =========================================================================
    // EXTENDED BASELINE (Session-start capture)
    // =========================================================================

    /**
     * Start baseline capture phase
     */
    startBaselineCapture(): void {
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
     */
    endBaselineCapture(): BaselineCaptureResult {
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

        const sumX = samples.reduce((s, r) => s + r.x, 0);
        const sumY = samples.reduce((s, r) => s + r.y, 0);
        const sumZ = samples.reduce((s, r) => s + r.z, 0);
        const n = samples.length;
        const baseline = { x: sumX / n, y: sumY / n, z: sumZ / n };
        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

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
     * Manually set Extended Baseline
     */
    setExtendedBaseline(baseline: Vector3): { success: boolean; reason?: string; magnitude?: number } {
        if (!baseline || typeof baseline.x !== 'number') {
            return { success: false, reason: 'invalid_baseline' };
        }

        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

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
     */
    clearExtendedBaseline(): void {
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
     */
    hasExtendedBaseline(): boolean {
        return this._extendedBaselineActive;
    }

    /**
     * Get current Extended Baseline
     */
    getExtendedBaseline(): ExtendedBaselineState {
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
     */
    isCapturingBaseline(): boolean {
        return this._baselineCapturing;
    }

    // =========================================================================
    // EARTH FIELD ESTIMATION (Real-time)
    // =========================================================================

    /**
     * Update with new sample
     */
    update(mx_ut: number, my_ut: number, mz_ut: number, orientation: Quaternion): UpdateResult {
        if (!orientation || orientation.w === undefined) {
            return { earthReady: false, confidence: 0, earthMagnitude: 0 };
        }

        this._totalSamples++;

        if (this.autoBaseline && this.extendedBaselineEnabled &&
            !this._extendedBaselineActive && !this._baselineCapturing &&
            this._autoBaselineRetryCount < this._autoBaselineMaxRetries) {
            this._baselineCapturing = true;
            this._baselineCaptureSamples = [];
            if (this.debug) console.log('[UnifiedMagCal] Auto-baseline capture started');
        }

        const ironCorrected = this.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

        if (this.debug && !this._loggedFirstSample) {
            const mag = Math.sqrt(ironCorrected.x**2 + ironCorrected.y**2 + ironCorrected.z**2);
            console.log(`[UnifiedMagCal] First sample: [${ironCorrected.x.toFixed(1)}, ${ironCorrected.y.toFixed(1)}, ${ironCorrected.z.toFixed(1)}] |${mag.toFixed(1)}| µT`);
            this._loggedFirstSample = true;
        }

        const R = this._quaternionToRotationMatrix(orientation);
        const worldSample: Vector3 = {
            x: R[0][0] * ironCorrected.x + R[1][0] * ironCorrected.y + R[2][0] * ironCorrected.z,
            y: R[0][1] * ironCorrected.x + R[1][1] * ironCorrected.y + R[2][1] * ironCorrected.z,
            z: R[0][2] * ironCorrected.x + R[1][2] * ironCorrected.y + R[2][2] * ironCorrected.z
        };

        this._worldSamples.push(worldSample);
        if (this._worldSamples.length > this.windowSize) {
            this._worldSamples.shift();
        }

        this._computeEarthField();

        let autoBaselineResult: BaselineCaptureResult | null = null;
        if (this._earthFieldMagnitude > 0) {
            const residual = this._getEarthResidual(mx_ut, my_ut, mz_ut, orientation);
            if (residual) {
                this._recentResiduals.push(residual.magnitude);
                if (this._recentResiduals.length > this._maxResidualHistory) {
                    this._recentResiduals.shift();
                }

                if (this._baselineCapturing) {
                    this._baselineCaptureSamples.push({ x: residual.x, y: residual.y, z: residual.z });

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
            autoBaselineResult
        };
    }

    /**
     * Attempt automatic baseline completion
     */
    private _attemptAutoBaseline(): BaselineCaptureResult {
        const samples = this._baselineCaptureSamples;
        const n = samples.length;

        const sumX = samples.reduce((s, r) => s + r.x, 0);
        const sumY = samples.reduce((s, r) => s + r.y, 0);
        const sumZ = samples.reduce((s, r) => s + r.z, 0);
        const baseline = { x: sumX / n, y: sumY / n, z: sumZ / n };
        const magnitude = Math.sqrt(baseline.x ** 2 + baseline.y ** 2 + baseline.z ** 2);

        if (magnitude > this.baselineMagnitudeThreshold) {
            this._autoBaselineRetryCount++;
            if (this.debug) {
                console.log(`[UnifiedMagCal] Auto-baseline rejected (attempt ${this._autoBaselineRetryCount}/${this._autoBaselineMaxRetries}): magnitude ${magnitude.toFixed(1)} µT > ${this.baselineMagnitudeThreshold} µT`);
            }

            if (this._autoBaselineRetryCount < this._autoBaselineMaxRetries) {
                this._baselineCaptureSamples = [];
            } else {
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
     * Get residual (fully corrected)
     */
    getResidual(mx_ut: number, my_ut: number, mz_ut: number, orientation: Quaternion): ResidualResult | null {
        const earthResidual = this._getEarthResidual(mx_ut, my_ut, mz_ut, orientation);
        if (!earthResidual) {
            return null;
        }

        if (this._extendedBaselineActive) {
            const residual: ResidualResult = {
                x: earthResidual.x - this._extendedBaseline.x,
                y: earthResidual.y - this._extendedBaseline.y,
                z: earthResidual.z - this._extendedBaseline.z,
                magnitude: 0
            };
            residual.magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);
            return residual;
        }

        return earthResidual;
    }

    /**
     * Get Earth-only residual
     */
    private _getEarthResidual(mx_ut: number, my_ut: number, mz_ut: number, orientation: Quaternion): ResidualResult | null {
        if (this._earthFieldMagnitude === 0 || !orientation) {
            return null;
        }

        const ironCorrected = this.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

        const R = this._quaternionToRotationMatrix(orientation);
        const earthSensor: Vector3 = {
            x: R[0][0] * this._earthFieldWorld.x + R[0][1] * this._earthFieldWorld.y + R[0][2] * this._earthFieldWorld.z,
            y: R[1][0] * this._earthFieldWorld.x + R[1][1] * this._earthFieldWorld.y + R[1][2] * this._earthFieldWorld.z,
            z: R[2][0] * this._earthFieldWorld.x + R[2][1] * this._earthFieldWorld.y + R[2][2] * this._earthFieldWorld.z
        };

        const residual: ResidualResult = {
            x: ironCorrected.x - earthSensor.x,
            y: ironCorrected.y - earthSensor.y,
            z: ironCorrected.z - earthSensor.z,
            magnitude: 0
        };
        residual.magnitude = Math.sqrt(residual.x ** 2 + residual.y ** 2 + residual.z ** 2);

        return residual;
    }

    /**
     * Check if Earth field estimation ready
     */
    isReady(): boolean {
        return this._earthFieldMagnitude > 0;
    }

    /**
     * Get confidence (0-1) based on residual magnitude
     */
    getConfidence(): number {
        if (this._recentResiduals.length < 10) {
            return Math.min(0.5, this._totalSamples / (this.minSamples * 2));
        }
        const meanResidual = this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;
        return Math.max(0, Math.min(1, 1 - meanResidual / this.confidenceResidualThreshold));
    }

    /**
     * Get calibration quality based on orientation diversity
     */
    getCalibrationQuality(): CalibrationQuality {
        const windowFill = Math.min(1, this._worldSamples.length / this.windowSize);

        if (this._worldSamples.length < this.minSamples) {
            return { quality: windowFill * 0.5, diversityRatio: 0, windowFill };
        }

        const samples = this._worldSamples;
        const n = samples.length;

        const meanX = samples.reduce((s, p) => s + p.x, 0) / n;
        const meanY = samples.reduce((s, p) => s + p.y, 0) / n;
        const meanZ = samples.reduce((s, p) => s + p.z, 0) / n;

        const varX = samples.reduce((s, p) => s + (p.x - meanX) ** 2, 0) / n;
        const varY = samples.reduce((s, p) => s + (p.y - meanY) ** 2, 0) / n;
        const varZ = samples.reduce((s, p) => s + (p.z - meanZ) ** 2, 0) / n;

        const totalVar = Math.sqrt(varX + varY + varZ);
        const diversityRatio = Math.min(1, totalVar / (this._earthFieldMagnitude || 50));
        const quality = windowFill * (0.5 + 0.5 * diversityRatio);

        return { quality, diversityRatio, windowFill };
    }

    /**
     * Get mean residual magnitude
     */
    getMeanResidual(): number {
        if (this._recentResiduals.length === 0) return Infinity;
        return this._recentResiduals.reduce((a, b) => a + b, 0) / this._recentResiduals.length;
    }

    /**
     * Get Earth field in world frame
     */
    getEarthFieldWorld(): Vector3 {
        return { ...this._earthFieldWorld };
    }

    /**
     * Get Earth field magnitude
     */
    getEarthFieldMagnitude(): number {
        return this._earthFieldMagnitude;
    }

    // =========================================================================
    // STATE & PERSISTENCE
    // =========================================================================

    /**
     * Get current state
     */
    getState(): CalibrationState {
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
            calibrationQuality: calibrationQuality.quality,
            diversityRatio: calibrationQuality.diversityRatio,
            windowFill: calibrationQuality.windowFill,
            autoBaselineEnabled: this.autoBaseline,
            autoBaselineRetryCount: this._autoBaselineRetryCount,
            autoBaselineMaxRetries: this._autoBaselineMaxRetries
        };
    }

    /**
     * Reset Earth field estimation (keeps iron calibration)
     */
    resetEarthEstimation(): void {
        this._resetEarthEstimation();
    }

    /**
     * Full reset
     */
    reset(): void {
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
     */
    save(key: string = 'gambit_calibration'): void {
        localStorage.setItem(key, JSON.stringify(this.toJSON()));
    }

    /**
     * Load from localStorage
     */
    load(key: string = 'gambit_calibration'): boolean {
        const json = localStorage.getItem(key);
        if (json) {
            this.fromJSON(JSON.parse(json));
            return true;
        }
        return false;
    }

    /**
     * Export to JSON
     */
    toJSON(): CalibrationJSON {
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
            extendedBaseline: this._extendedBaselineActive ? { ...this._extendedBaseline } : null,
            extendedBaselineMagnitude: this._extendedBaselineActive ? baselineMag : null,
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
     */
    fromJSON(json: Partial<CalibrationJSON>): void {
        this.hardIronOffset = json.hardIronOffset || { x: 0, y: 0, z: 0 };
        this.softIronMatrix = json.softIronMatrix ? Matrix3.fromArray(json.softIronMatrix) : Matrix3.identity();
        this.hardIronCalibrated = json.hardIronCalibrated || false;
        this.softIronCalibrated = json.softIronCalibrated || false;

        if (json.extendedBaseline) {
            this.setExtendedBaseline(json.extendedBaseline);
        } else {
            this.clearExtendedBaseline();
        }

        this._resetEarthEstimation();
    }

    // =========================================================================
    // PRIVATE METHODS
    // =========================================================================

    private _resetEarthEstimation(): void {
        this._worldSamples = [];
        this._earthFieldWorld = { x: 0, y: 0, z: 0 };
        this._earthFieldMagnitude = 0;
        this._recentResiduals = [];
        this._loggedEarthComputed = false;
    }

    private _computeEarthField(): void {
        if (this._worldSamples.length < this.minSamples) return;

        const n = this._worldSamples.length;
        let sumX = 0, sumY = 0, sumZ = 0;
        const magnitudes: number[] = [];

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

    private _quaternionToRotationMatrix(q: Quaternion): Matrix3x3 {
        const { w, x, y, z } = q;
        return [
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ];
    }

    private _calculateCoverage(samples: Vector3[]): number {
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

    private _calculateCovariance(samples: Vector3[]): Matrix3x3 {
        const n = samples.length;
        let sumX = 0, sumY = 0, sumZ = 0;
        for (const s of samples) { sumX += s.x; sumY += s.y; sumZ += s.z; }
        const meanX = sumX / n, meanY = sumY / n, meanZ = sumZ / n;

        const cov: Matrix3x3 = [[0,0,0], [0,0,0], [0,0,0]];
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

// ===== Factory Function =====

/**
 * Create instance
 */
export function createUnifiedMagCalibration(options: CalibrationOptions = {}): UnifiedMagCalibration {
    return new UnifiedMagCalibration(options);
}

// ===== Default Export =====

export default { UnifiedMagCalibration, createUnifiedMagCalibration };
