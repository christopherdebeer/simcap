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
  autoHardIron?: boolean;           // Enable auto hard iron estimation from residual feedback
  autoHardIronMinSamples?: number;  // Min samples before applying auto hard iron
  autoHardIronAlpha?: number;       // Exponential smoothing factor for auto hard iron (0-1)
}

export interface GeomagneticReference {
  horizontal: number;  // Horizontal component (µT)
  vertical: number;    // Vertical/down component (µT)
  declination: number; // Magnetic declination (degrees)
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
  autoHardIronReady?: boolean;
  autoHardIronEstimate?: Vector3 | null;
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
  autoHardIronEnabled: boolean;
  autoHardIronReady: boolean;
  autoHardIronEstimate: Vector3;
  autoHardIronSampleCount: number;
  autoHardIronRanges: Vector3;
  autoHardIronProgress: number;
  autoSoftIronScale: Vector3;
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

    // Auto Hard Iron estimation (min-max method, orientation-independent)
    private _autoHardIronEnabled: boolean = false;
    private _autoHardIronMinSamples: number = 100;
    private _autoHardIronAlpha: number = 0.02;  // Slow adaptation rate (legacy, unused in min-max)
    private _autoHardIronEstimate: Vector3 = { x: 0, y: 0, z: 0 };
    private _autoHardIronSampleCount: number = 0;
    private _autoHardIronReady: boolean = false;
    private _loggedAutoHardIron: boolean = false;

    // Min-max tracking for orientation-independent hard iron estimation
    private _autoHardIronMin: Vector3 = { x: Infinity, y: Infinity, z: Infinity };
    private _autoHardIronMax: Vector3 = { x: -Infinity, y: -Infinity, z: -Infinity };
    private _autoHardIronMinRangeRequired: number = 80;  // Minimum range (µT) per axis - needs ~80% of full rotation

    // Auto soft iron scale factors (computed from min-max ranges)
    private _autoSoftIronScale: Vector3 = { x: 1, y: 1, z: 1 };
    private _autoSoftIronEnabled: boolean = true;  // Apply soft iron correction when auto hard iron is ready

    // Full 3x3 soft iron matrix for orientation-aware calibration
    private _autoSoftIronMatrix: Matrix3 = Matrix3.identity();
    private _useFullSoftIronMatrix: boolean = false;  // Use full matrix when orientation-aware cal is ready

    // Orientation-aware calibration state
    private _orientationAwareCalEnabled: boolean = true;
    private _orientationAwareCalReady: boolean = false;
    private _orientationAwareCalSamples: { mx: number; my: number; mz: number; ax: number; ay: number; az: number }[] = [];
    private _orientationAwareCalMinSamples: number = 200;  // Need diverse orientations
    private _orientationAwareCalMaxSamples: number = 500;  // Cap to limit memory
    private _loggedOrientationAwareCal: boolean = false;

    // Geomagnetic reference (from tables, not estimated)
    private _geomagneticRef: GeomagneticReference | null = null;
    private _geomagEarthWorld: Vector3 | null = null;  // Earth field in world frame from geomag

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

        // Auto Hard Iron configuration
        this._autoHardIronEnabled = options.autoHardIron !== false;  // Enabled by default
        this._autoHardIronMinSamples = options.autoHardIronMinSamples || 100;
        this._autoHardIronAlpha = options.autoHardIronAlpha || 0.02;

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
     * Uses wizard calibration if available, otherwise falls back to auto estimate (when ready)
     */
    applyIronCorrection(raw: Vector3): Vector3 {
        // Determine which offset to use: wizard calibration takes priority
        let offset: Vector3;
        if (this.hardIronCalibrated) {
            offset = this.hardIronOffset;
        } else if (this._autoHardIronReady) {
            offset = this._autoHardIronEstimate;
        } else {
            // No calibration available
            return { x: raw.x, y: raw.y, z: raw.z };
        }

        let corrected: Vector3 = {
            x: raw.x - offset.x,
            y: raw.y - offset.y,
            z: raw.z - offset.z
        };

        if (this.softIronCalibrated) {
            corrected = this.softIronMatrix.multiply(corrected);
        }

        return corrected;
    }

    /**
     * Apply progressive iron correction using current best estimate
     * Unlike applyIronCorrection, this ALWAYS applies current estimate even if not "ready"
     * For use with progressive 9-DOF fusion during calibration warmup
     *
     * Applies both hard iron (offset) and soft iron (scale) correction:
     * 1. Hard iron: subtract offset to center the ellipsoid at origin
     * 2. Soft iron: scale each axis to normalize ellipsoid to sphere
     */
    applyProgressiveIronCorrection(raw: Vector3): Vector3 {
        // Wizard calibration always takes priority
        if (this.hardIronCalibrated) {
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

        // Use current auto estimate even if not "ready"
        // This enables progressive calibration - estimate improves as rotation occurs
        const offset = this._autoHardIronEstimate;
        let corrected: Vector3 = {
            x: raw.x - offset.x,
            y: raw.y - offset.y,
            z: raw.z - offset.z
        };

        // Apply auto soft iron scale factors (normalizes ellipsoid to sphere)
        if (this._autoSoftIronEnabled && this._autoHardIronReady) {
            corrected = {
                x: corrected.x * this._autoSoftIronScale.x,
                y: corrected.y * this._autoSoftIronScale.y,
                z: corrected.z * this._autoSoftIronScale.z
            };
        }

        return corrected;
    }

    /**
     * Check if any iron calibration available (wizard or auto)
     */
    hasIronCalibration(): boolean {
        return this.hardIronCalibrated || this.softIronCalibrated || this._autoHardIronReady;
    }

    /**
     * Check if using auto hard iron (vs wizard calibration)
     */
    isUsingAutoHardIron(): boolean {
        return !this.hardIronCalibrated && this._autoHardIronReady;
    }

    /**
     * Get current auto hard iron estimate
     */
    getAutoHardIronEstimate(): Vector3 | null {
        return this._autoHardIronReady ? { ...this._autoHardIronEstimate } : null;
    }

    /**
     * Get current auto hard iron estimate (even if not ready)
     * For progressive calibration - use the best estimate we have so far
     */
    getCurrentAutoHardIronEstimate(): Vector3 {
        return { ...this._autoHardIronEstimate };
    }

    /**
     * Get auto hard iron calibration progress (0.0 to 1.0)
     * Based on min-max range coverage across all axes
     */
    getAutoHardIronProgress(): number {
        if (this.hardIronCalibrated) return 1.0;  // Wizard calibration = 100%
        if (this._autoHardIronSampleCount < 10) return 0.0;  // Not enough samples yet

        const rangeX = this._autoHardIronMax.x - this._autoHardIronMin.x;
        const rangeY = this._autoHardIronMax.y - this._autoHardIronMin.y;
        const rangeZ = this._autoHardIronMax.z - this._autoHardIronMin.z;

        // Use geomagnetic reference to set target, or fallback to 80 µT
        let rangeThreshold = this._autoHardIronMinRangeRequired;
        if (this._geomagneticRef) {
            const expectedMag = Math.sqrt(
                this._geomagneticRef.horizontal ** 2 +
                this._geomagneticRef.vertical ** 2
            );
            rangeThreshold = Math.max(rangeThreshold, expectedMag * 1.6);
        }

        // Progress is based on the MINIMUM range (limiting axis)
        const minRange = Math.min(rangeX, rangeY, rangeZ);
        return Math.min(1.0, minRange / rangeThreshold);
    }

    /**
     * Set geomagnetic reference (from location tables)
     * Used for auto hard iron estimation with known Earth field
     */
    setGeomagneticReference(ref: GeomagneticReference): void {
        this._geomagneticRef = ref;

        // Compute Earth field in world frame
        // IMPORTANT: Use magnetic north frame (X = magnetic north, Y = east, Z = down)
        // This matches the AHRS convention where the quaternion represents rotation
        // from magnetic north frame to sensor frame. Declination is NOT applied here
        // because the AHRS heading is relative to magnetic north, not true north.
        this._geomagEarthWorld = {
            x: ref.horizontal,  // Magnetic north component (all horizontal in X)
            y: 0,               // East component = 0 (same as AHRS)
            z: ref.vertical     // Down component
        };

        if (this.debug) {
            const mag = Math.sqrt(
                this._geomagEarthWorld.x ** 2 +
                this._geomagEarthWorld.y ** 2 +
                this._geomagEarthWorld.z ** 2
            );
            console.log(`[UnifiedMagCal] Geomagnetic ref set: H=${ref.horizontal.toFixed(1)} V=${ref.vertical.toFixed(1)} (|E|=${mag.toFixed(1)} µT, magnetic north frame)`);
        }
    }

    /**
     * Check if geomagnetic reference is available
     */
    hasGeomagneticReference(): boolean {
        return this._geomagEarthWorld !== null;
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

        // Use progressive iron correction for Earth field estimation
        // This ensures consistency with AHRS fusion and residual calculation
        const ironCorrected = this.applyProgressiveIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

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

        // Auto Hard Iron estimation using MIN-MAX method (orientation-independent)
        // Called early so it can build estimate even before Earth field is estimated
        // Min-max method doesn't require geomagnetic reference - it tracks min/max of raw readings
        if (this._autoHardIronEnabled && !this.hardIronCalibrated) {
            this._updateAutoHardIron(mx_ut, my_ut, mz_ut, orientation);
        }

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
            autoBaselineResult,
            autoHardIronReady: this._autoHardIronReady,
            autoHardIronEstimate: this._autoHardIronReady ? { ...this._autoHardIronEstimate } : null
        };
    }

    /**
     * Collect sample for orientation-aware calibration
     * Called from update() when accelerometer data is available
     */
    collectOrientationAwareSample(mx: number, my: number, mz: number, ax: number, ay: number, az: number): void {
        if (!this._orientationAwareCalEnabled || this._orientationAwareCalReady) return;
        if (this.hardIronCalibrated) return;  // Wizard cal takes priority

        // Only collect if we have valid accelerometer data
        const accelMag = Math.sqrt(ax**2 + ay**2 + az**2);
        if (accelMag < 0.5 || accelMag > 2.0) return;  // Invalid accel (should be ~1g)

        this._orientationAwareCalSamples.push({ mx, my, mz, ax, ay, az });

        // Cap samples to limit memory
        if (this._orientationAwareCalSamples.length > this._orientationAwareCalMaxSamples) {
            // Remove oldest samples
            this._orientationAwareCalSamples = this._orientationAwareCalSamples.slice(-this._orientationAwareCalMaxSamples);
        }

        // Check if we have enough samples and enough rotation coverage
        if (this._orientationAwareCalSamples.length >= this._orientationAwareCalMinSamples && this._autoHardIronReady) {
            this._runOrientationAwareCalibration();
        }
    }

    /**
     * Run orientation-aware calibration optimization
     * Uses accelerometer to determine expected Earth field direction
     */
    private _runOrientationAwareCalibration(): void {
        if (this._orientationAwareCalReady) return;
        if (!this._geomagneticRef) return;  // Need geomagnetic reference

        const samples = this._orientationAwareCalSamples;
        const n = samples.length;

        // Earth field in world frame (magnetic north)
        const earthWorld = {
            x: this._geomagneticRef.horizontal,
            y: 0,
            z: this._geomagneticRef.vertical
        };

        // Helper: get roll/pitch from accelerometer
        const accelToRollPitch = (ax: number, ay: number, az: number): { roll: number; pitch: number } => {
            const aMag = Math.sqrt(ax**2 + ay**2 + az**2);
            if (aMag < 0.1) return { roll: 0, pitch: 0 };
            const axn = ax / aMag, ayn = ay / aMag, azn = az / aMag;
            return {
                roll: Math.atan2(ayn, azn),
                pitch: Math.atan2(-axn, Math.sqrt(ayn**2 + azn**2))
            };
        };

        // Helper: tilt-compensate magnetometer
        const tiltCompensate = (mx: number, my: number, mz: number, roll: number, pitch: number) => {
            const cr = Math.cos(roll), sr = Math.sin(roll);
            const cp = Math.cos(pitch), sp = Math.sin(pitch);
            const mx_h = mx * cp + my * sr * sp + mz * cr * sp;
            const my_h = my * cr - mz * sr;
            return { mx_h, my_h };
        };

        // Helper: ZYX rotation matrix
        const eulerToRotationMatrix = (roll: number, pitch: number, yaw: number): number[][] => {
            const cy = Math.cos(yaw), sy = Math.sin(yaw);
            const cp = Math.cos(pitch), sp = Math.sin(pitch);
            const cr = Math.cos(roll), sr = Math.sin(roll);
            return [
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ];
        };

        // Simple gradient descent optimization
        // Parameters: offset (3) + soft iron matrix (9) = 12 parameters
        // Start from min-max estimate
        let offset = { ...this._autoHardIronEstimate };
        let S = [
            [this._autoSoftIronScale.x, 0, 0],
            [0, this._autoSoftIronScale.y, 0],
            [0, 0, this._autoSoftIronScale.z]
        ];

        // Compute residual for current parameters
        const computeResidual = (off: Vector3, softIron: number[][]): number => {
            let totalResidual = 0;
            for (const sample of samples) {
                // Apply calibration
                const centered = {
                    x: sample.mx - off.x,
                    y: sample.my - off.y,
                    z: sample.mz - off.z
                };
                const corrected = {
                    x: softIron[0][0] * centered.x + softIron[0][1] * centered.y + softIron[0][2] * centered.z,
                    y: softIron[1][0] * centered.x + softIron[1][1] * centered.y + softIron[1][2] * centered.z,
                    z: softIron[2][0] * centered.x + softIron[2][1] * centered.y + softIron[2][2] * centered.z
                };

                // Get orientation from accelerometer
                const { roll, pitch } = accelToRollPitch(sample.ax, sample.ay, sample.az);
                const { mx_h, my_h } = tiltCompensate(corrected.x, corrected.y, corrected.z, roll, pitch);
                const yaw = Math.atan2(-my_h, mx_h);

                // Compute expected Earth field in device frame
                const R = eulerToRotationMatrix(roll, pitch, yaw);
                // R^T * earthWorld (transpose to go from world to device)
                const earthDevice = {
                    x: R[0][0] * earthWorld.x + R[1][0] * earthWorld.y + R[2][0] * earthWorld.z,
                    y: R[0][1] * earthWorld.x + R[1][1] * earthWorld.y + R[2][1] * earthWorld.z,
                    z: R[0][2] * earthWorld.x + R[1][2] * earthWorld.y + R[2][2] * earthWorld.z
                };

                // Residual = difference between corrected mag and expected Earth
                const dx = corrected.x - earthDevice.x;
                const dy = corrected.y - earthDevice.y;
                const dz = corrected.z - earthDevice.z;
                totalResidual += dx*dx + dy*dy + dz*dz;
            }
            return Math.sqrt(totalResidual / n);
        };

        // Gradient descent with numerical gradients
        // Use small learning rate and regularization to prevent divergence
        const learningRateOffset = 0.1;  // Smaller for stability
        const learningRateMatrix = 0.01;  // Even smaller for matrix (more sensitive)
        const epsilon = 0.5;  // For numerical gradient
        const maxIterations = 50;
        const convergenceThreshold = 0.1;

        let prevResidual = computeResidual(offset, S);
        let bestResidual = prevResidual;
        let bestOffset = { ...offset };
        let bestS = S.map(row => [...row]);

        for (let iter = 0; iter < maxIterations; iter++) {
            // Compute gradients for offset
            const gradOffset = { x: 0, y: 0, z: 0 };
            for (const axis of ['x', 'y', 'z'] as const) {
                const offsetPlus = { ...offset, [axis]: offset[axis] + epsilon };
                const offsetMinus = { ...offset, [axis]: offset[axis] - epsilon };
                gradOffset[axis] = (computeResidual(offsetPlus, S) - computeResidual(offsetMinus, S)) / (2 * epsilon);
            }

            // Compute gradients for soft iron matrix
            const gradS = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const Splus = S.map(row => [...row]);
                    const Sminus = S.map(row => [...row]);
                    Splus[i][j] += epsilon;
                    Sminus[i][j] -= epsilon;
                    gradS[i][j] = (computeResidual(offset, Splus) - computeResidual(offset, Sminus)) / (2 * epsilon);
                }
            }

            // Update parameters with gradient clipping
            const maxGrad = 10;
            offset.x -= learningRateOffset * Math.max(-maxGrad, Math.min(maxGrad, gradOffset.x));
            offset.y -= learningRateOffset * Math.max(-maxGrad, Math.min(maxGrad, gradOffset.y));
            offset.z -= learningRateOffset * Math.max(-maxGrad, Math.min(maxGrad, gradOffset.z));

            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    S[i][j] -= learningRateMatrix * Math.max(-maxGrad, Math.min(maxGrad, gradS[i][j]));
                }
            }

            // Regularization: keep soft iron matrix close to identity-like
            // Diagonal elements should be positive and near 1
            // Off-diagonal elements should be small
            for (let i = 0; i < 3; i++) {
                // Clamp diagonal to reasonable range [0.5, 2.0]
                S[i][i] = Math.max(0.5, Math.min(2.0, S[i][i]));
                for (let j = 0; j < 3; j++) {
                    if (i !== j) {
                        // Clamp off-diagonal to [-0.5, 0.5]
                        S[i][j] = Math.max(-0.5, Math.min(0.5, S[i][j]));
                    }
                }
            }

            // Check convergence and track best
            const newResidual = computeResidual(offset, S);
            if (newResidual < bestResidual) {
                bestResidual = newResidual;
                bestOffset = { ...offset };
                bestS = S.map(row => [...row]);
            }

            if (Math.abs(prevResidual - newResidual) < convergenceThreshold) {
                break;
            }
            prevResidual = newResidual;
        }

        // Use best result found
        offset = bestOffset;
        S = bestS;

        // Store results
        this._autoHardIronEstimate = offset;
        this._autoSoftIronMatrix = new Matrix3([
            [S[0][0], S[0][1], S[0][2]],
            [S[1][0], S[1][1], S[1][2]],
            [S[2][0], S[2][1], S[2][2]]
        ]);
        this._useFullSoftIronMatrix = true;
        this._orientationAwareCalReady = true;

        // Log results
        if (this.debug && !this._loggedOrientationAwareCal) {
            const finalResidual = computeResidual(offset, S);
            console.log(`[UnifiedMagCal] Orientation-aware calibration complete:`);
            console.log(`  Samples: ${n}`);
            console.log(`  Final residual: ${finalResidual.toFixed(1)} µT`);
            console.log(`  Hard iron: [${offset.x.toFixed(2)}, ${offset.y.toFixed(2)}, ${offset.z.toFixed(2)}] µT`);
            console.log(`  Soft iron matrix:`);
            console.log(`    [${S[0][0].toFixed(4)}, ${S[0][1].toFixed(4)}, ${S[0][2].toFixed(4)}]`);
            console.log(`    [${S[1][0].toFixed(4)}, ${S[1][1].toFixed(4)}, ${S[1][2].toFixed(4)}]`);
            console.log(`    [${S[2][0].toFixed(4)}, ${S[2][1].toFixed(4)}, ${S[2][2].toFixed(4)}]`);
            this._loggedOrientationAwareCal = true;
        }
    }

    /**
     * Update auto hard iron estimate using MIN-MAX method
     *
     * This approach is ORIENTATION-INDEPENDENT - it doesn't rely on accurate yaw
     * from 6-DOF fusion. As the device rotates, we track min/max for each axis.
     * The hard iron offset is (max + min) / 2.
     *
     * Why this works:
     * - Raw magnetometer = Earth_field_in_sensor_frame + hard_iron
     * - As device rotates, Earth field components oscillate around zero (in body frame)
     * - Hard iron is constant offset in body frame
     * - Min/max captures the range of Earth field + hard iron
     * - (max + min) / 2 = hard_iron (since Earth field averages to 0 over rotation)
     *
     * Requirements:
     * - Device must be rotated to get sufficient coverage
     * - Minimum range per axis ensures rotation has occurred
     */
    private _updateAutoHardIron(mx_ut: number, my_ut: number, mz_ut: number, _orientation: Quaternion): void {
        // Update min/max tracking
        this._autoHardIronMin.x = Math.min(this._autoHardIronMin.x, mx_ut);
        this._autoHardIronMin.y = Math.min(this._autoHardIronMin.y, my_ut);
        this._autoHardIronMin.z = Math.min(this._autoHardIronMin.z, mz_ut);

        this._autoHardIronMax.x = Math.max(this._autoHardIronMax.x, mx_ut);
        this._autoHardIronMax.y = Math.max(this._autoHardIronMax.y, my_ut);
        this._autoHardIronMax.z = Math.max(this._autoHardIronMax.z, mz_ut);

        this._autoHardIronSampleCount++;

        // Compute ranges
        const rangeX = this._autoHardIronMax.x - this._autoHardIronMin.x;
        const rangeY = this._autoHardIronMax.y - this._autoHardIronMin.y;
        const rangeZ = this._autoHardIronMax.z - this._autoHardIronMin.z;

        // Compute hard iron estimate as center of min-max bounds
        this._autoHardIronEstimate = {
            x: (this._autoHardIronMax.x + this._autoHardIronMin.x) / 2,
            y: (this._autoHardIronMax.y + this._autoHardIronMin.y) / 2,
            z: (this._autoHardIronMax.z + this._autoHardIronMin.z) / 2
        };

        // Check if ready: sufficient samples AND sufficient rotation coverage
        // We need at least minRangeRequired on each axis to ensure device was rotated
        // If we have geomagnetic reference, use 1.6x expected magnitude as threshold
        // (Earth field of 50 µT should give ~100 µT range, so 80 µT = 80% coverage)
        let rangeThreshold = this._autoHardIronMinRangeRequired;
        if (this._geomagneticRef) {
            const expectedMag = Math.sqrt(
                this._geomagneticRef.horizontal ** 2 +
                this._geomagneticRef.vertical ** 2
            );
            rangeThreshold = Math.max(rangeThreshold, expectedMag * 1.6);  // 80% of full swing (2x magnitude)
        }

        const hasEnoughSamples = this._autoHardIronSampleCount >= this._autoHardIronMinSamples;
        const hasEnoughRotation = rangeX >= rangeThreshold &&
                                  rangeY >= rangeThreshold &&
                                  rangeZ >= rangeThreshold;

        // Only mark ready once we have both conditions
        if (!this._autoHardIronReady && hasEnoughSamples && hasEnoughRotation) {
            this._autoHardIronReady = true;

            // Calculate sphericity for quality check
            const minRange = Math.min(rangeX, rangeY, rangeZ);
            const maxRange = Math.max(rangeX, rangeY, rangeZ);
            const sphericity = minRange / maxRange;

            // Compute soft iron scale factors to normalize ellipsoid to sphere
            // Strategy: scale each axis so the corrected magnitude matches expected Earth field
            //
            // The range on each axis should be 2x the Earth field magnitude (full swing from -E to +E)
            // So expected range = 2 * expectedMag
            //
            // Scale factor = expectedRange / actualRange = (2 * expectedMag) / actualRange
            //
            // This ensures the corrected magnitude matches the expected Earth field magnitude

            // Get expected magnitude from geomagnetic reference, or use 50 µT default
            let expectedMag = 50.0;  // Default Earth field magnitude
            if (this._geomagneticRef) {
                expectedMag = Math.sqrt(
                    this._geomagneticRef.horizontal ** 2 +
                    this._geomagneticRef.vertical ** 2
                );
            }
            const expectedRange = 2 * expectedMag;

            // Scale factors: expectedRange / actualRange
            // This normalizes each axis to the expected Earth field swing
            // If range is too small, clamp to avoid extreme scaling
            const minAllowedRange = 20;  // µT - prevent division by very small numbers
            this._autoSoftIronScale = {
                x: rangeX > minAllowedRange ? expectedRange / rangeX : 1,
                y: rangeY > minAllowedRange ? expectedRange / rangeY : 1,
                z: rangeZ > minAllowedRange ? expectedRange / rangeZ : 1
            };

            if (this.debug && !this._loggedAutoHardIron) {
                const est = this._autoHardIronEstimate;
                const estMag = Math.sqrt(est.x**2 + est.y**2 + est.z**2);
                console.log(`[UnifiedMagCal] Auto hard iron ready (min-max method):`);
                console.log(`  Offset: [${est.x.toFixed(1)}, ${est.y.toFixed(1)}, ${est.z.toFixed(1)}] µT (|offset|=${estMag.toFixed(1)} µT)`);
                console.log(`  Ranges: [${rangeX.toFixed(1)}, ${rangeY.toFixed(1)}, ${rangeZ.toFixed(1)}] µT`);
                console.log(`  Sphericity: ${sphericity.toFixed(2)} (${sphericity > 0.7 ? 'good' : 'fair'})`);

                // Soft iron diagnostics
                console.log(`  Soft iron scale: [${this._autoSoftIronScale.x.toFixed(3)}, ${this._autoSoftIronScale.y.toFixed(3)}, ${this._autoSoftIronScale.z.toFixed(3)}]`);
                console.log(`  Target range: ${expectedRange.toFixed(1)} µT (2x expected Earth field)`);

                // Axis asymmetry investigation
                const xDeviation = ((rangeX - expectedRange) / expectedRange * 100).toFixed(1);
                const yDeviation = ((rangeY - expectedRange) / expectedRange * 100).toFixed(1);
                const zDeviation = ((rangeZ - expectedRange) / expectedRange * 100).toFixed(1);
                console.log(`  Axis deviation from target: X=${xDeviation}%, Y=${yDeviation}%, Z=${zDeviation}%`);

                // Min/max values for investigation
                console.log(`  Min values: [${this._autoHardIronMin.x.toFixed(1)}, ${this._autoHardIronMin.y.toFixed(1)}, ${this._autoHardIronMin.z.toFixed(1)}] µT`);
                console.log(`  Max values: [${this._autoHardIronMax.x.toFixed(1)}, ${this._autoHardIronMax.y.toFixed(1)}, ${this._autoHardIronMax.z.toFixed(1)}] µT`);

                this._loggedAutoHardIron = true;
            }
        }

        // Log progress periodically
        if (this.debug && !this._autoHardIronReady && this._autoHardIronSampleCount % 50 === 0) {
            const minRangeAchieved = Math.min(rangeX, rangeY, rangeZ);
            const coverage = Math.min(100, (minRangeAchieved / rangeThreshold) * 100);
            console.log(`[UnifiedMagCal] Auto hard iron progress: samples=${this._autoHardIronSampleCount}, ranges=[${rangeX.toFixed(0)}, ${rangeY.toFixed(0)}, ${rangeZ.toFixed(0)}] µT, coverage=${coverage.toFixed(0)}% (need ${rangeThreshold.toFixed(0)} µT per axis)`);
        }
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

        // Use progressive iron correction (includes soft iron scaling when available)
        // This ensures residual calculation matches what's used in AHRS fusion
        const ironCorrected = this.applyProgressiveIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

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
            autoBaselineMaxRetries: this._autoBaselineMaxRetries,
            autoHardIronEnabled: this._autoHardIronEnabled,
            autoHardIronReady: this._autoHardIronReady,
            autoHardIronEstimate: { ...this._autoHardIronEstimate },
            autoHardIronSampleCount: this._autoHardIronSampleCount,
            autoHardIronRanges: {
                x: this._autoHardIronMax.x - this._autoHardIronMin.x,
                y: this._autoHardIronMax.y - this._autoHardIronMin.y,
                z: this._autoHardIronMax.z - this._autoHardIronMin.z
            },
            autoHardIronProgress: this.getAutoHardIronProgress(),
            autoSoftIronScale: { ...this._autoSoftIronScale }
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

        // Reset auto hard iron
        this._autoHardIronEstimate = { x: 0, y: 0, z: 0 };
        this._autoHardIronSampleCount = 0;
        this._autoHardIronReady = false;
        this._loggedAutoHardIron = false;
        this._autoHardIronMin = { x: Infinity, y: Infinity, z: Infinity };
        this._autoHardIronMax = { x: -Infinity, y: -Infinity, z: -Infinity };

        // Reset auto soft iron
        this._autoSoftIronScale = { x: 1, y: 1, z: 1 };

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
