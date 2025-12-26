/**
 * Telemetry Processor
 *
 * Unified telemetry processing pipeline for GAMBIT sensor data.
 * Handles unit conversion, calibration, filtering, and orientation estimation.
 *
 * Reference: index.html updateData() + collector telemetry-handler.js
 *
 * @module shared/telemetry-processor
 */

import {
    ACCEL_SCALE,
    GYRO_SCALE,
    STATIONARY_SAMPLES_FOR_CALIBRATION,
    accelLsbToG,
    gyroLsbToDps,
    createMadgwickAHRS,
    createKalmanFilter3D,
    createMotionDetector,
    createGyroBiasState
} from './sensor-config.js';

// Import actual types from filter modules
import type { MadgwickAHRS as MadgwickAHRSClass, MotionDetector as MotionDetectorClass } from '@filters';
import type { KalmanFilter3D as KalmanFilter3DClass } from '@filters';

import {
    magLsbToMicroTesla,
    getSensorUnitMetadata
} from './sensor-units.js';

import {
    getDefaultLocation,
    getBrowserLocation
} from './geomagnetic-field.js';

import {
    UnifiedMagCalibration
} from './unified-mag-calibration.js';

import {
    MagnetDetector,
    createMagnetDetector
} from './magnet-detector.js';

import type {
  EulerAngles,
  Quaternion,
  RawTelemetry,
  DecoratedTelemetry,
  GeomagneticReference,
  MotionDetectorState,
  GyroBiasCalibrationState,
} from '@core/types';
import type { GeomagneticLocation } from './geomagnetic-field';
import type { MagnetDetectorState } from './magnet-detector';

// Re-export core types for consumers
export type { RawTelemetry, DecoratedTelemetry } from '@core/types';

export interface TelemetryProcessorOptions {
  onProcessed?: ((telemetry: DecoratedTelemetry) => void) | null;
  onOrientationUpdate?: ((euler: EulerAngles, quaternion: Quaternion) => void) | null;
  onGyroBiasCalibrated?: (() => void) | null;
  onMagnetStatusChange?: ((state: MagnetDetectorState) => void) | null;
  onLog?: ((message: string) => void) | null;  // Callback for diagnostic logging
  useMagnetometer?: boolean;
  magTrust?: number;
  magCalibrationDebug?: boolean;
  calibration?: any;
}

// Use GeomagneticReference from core types
type GeomagneticRef = GeomagneticReference;

// Re-export motion/gyro state types with local aliases for backward compatibility
export type MotionState = MotionDetectorState;
export type GyroBiasState = GyroBiasCalibrationState;

interface MadgwickAHRS {
  update: (ax: number, ay: number, az: number, gx: number, gy: number, gz: number, dt?: number | null, gyroInDegrees?: boolean) => void;
  updateWithMag: (ax: number, ay: number, az: number, gx: number, gy: number, gz: number, mx: number, my: number, mz: number, dt?: number | null, gyroInDegrees?: boolean, applyHardIron?: boolean) => void;
  initFromAccelerometer: (ax: number, ay: number, az: number) => void;
  getQuaternion: () => Quaternion;
  getEulerAngles: () => EulerAngles;
  updateGyroBias: (gx: number, gy: number, gz: number, isStationary: boolean) => void;
  setMagTrust: (trust: number) => void;
  setGeomagneticReference: (ref: GeomagneticRef) => void;
  getMagResidual: () => { x: number; y: number; z: number } | null;
  getMagResidualMagnitude: () => number;
  reset: () => void;
}

interface KalmanFilter3D {
  update: (input: { x: number; y: number; z: number }) => { x: number; y: number; z: number };
  reset: () => void;
}

interface MotionDetectorInstance {
  update: (ax: number, ay: number, az: number, gx: number, gy: number, gz: number) => MotionState;
  getState: () => MotionState;
  reset: () => void;
}

// ===== TelemetryProcessor Class =====

/**
 * TelemetryProcessor class
 *
 * Processes raw sensor telemetry through a complete pipeline:
 * 1. Unit conversion (LSB → physical units)
 * 2. IMU sensor fusion (orientation estimation)
 * 3. Gyroscope bias calibration (when stationary)
 * 4. Magnetometer calibration (hard/soft iron, Earth field)
 * 5. Kalman filtering (noise reduction)
 *
 * Emits decorated telemetry with both raw and processed fields.
 */
export class TelemetryProcessor {
    private options: TelemetryProcessorOptions;
    private magCalibration: UnifiedMagCalibration;
    private magnetDetector: MagnetDetector;
    private imuFusion: MadgwickAHRSClass;
    private magFilter: KalmanFilter3DClass;
    private motionDetector: MotionDetectorClass;
    private gyroBiasState: GyroBiasState;
    private imuInitialized: boolean = false;
    private useMagnetometer: boolean;
    private magTrust: number;
    private geomagneticRef: GeomagneticRef | null = null;
    private lastTimestamp: number | null = null;
    private onProcessed: ((telemetry: DecoratedTelemetry) => void) | null;
    private onOrientationUpdate: ((euler: EulerAngles, quaternion: Quaternion) => void) | null;
    private onGyroBiasCalibrated: (() => void) | null;
    private onLog: ((message: string) => void) | null;
    private _loggedCalibrationMissing: boolean = false;
    private _loggedMagFusion: boolean = false;
    private _loggedMagFusionDisabled: boolean = false;  // Track 6-DOF disabled message
    private _loggedMagnetDetection: boolean = false;

    // Diagnostic tracking for magnetometer residual drift
    private _sampleCount: number = 0;
    private _magResidualHistory: { x: number; y: number; z: number; mag: number; yaw: number }[] = [];
    private _lastDiagnosticLog: number = 0;

    /**
     * Create a TelemetryProcessor instance
     */
    constructor(options: TelemetryProcessorOptions = {}) {
        this.options = options;

        // Unified magnetometer calibration (iron correction + Earth field estimation)
        this.magCalibration = new UnifiedMagCalibration({
            windowSize: 200,  // Optimal based on investigation
            minSamples: 50,   // ~1 second at 50Hz
            debug: options.magCalibrationDebug || false,
            onLog: options.onLog || undefined  // Pass through diagnostic logging callback
        });

        // Load stored iron calibration from localStorage
        this.magCalibration.load('gambit_calibration');

        // Magnet detector (detects finger magnet presence from residual magnitude)
        // Wrap the callback to match the expected signature
        const magnetStatusCallback = options.onMagnetStatusChange
            ? (_newStatus: string, _oldStatus: string, state: MagnetDetectorState) => options.onMagnetStatusChange!(state)
            : null;
        this.magnetDetector = createMagnetDetector({
            onStatusChange: magnetStatusCallback
        });

        // Create signal processing components
        this.imuFusion = createMadgwickAHRS();
        this.magFilter = createKalmanFilter3D();
        this.motionDetector = createMotionDetector();

        // Gyroscope bias calibration state
        this.gyroBiasState = createGyroBiasState();

        // 9-DOF magnetometer fusion configuration
        this.useMagnetometer = options.useMagnetometer !== false; // Default: enabled
        this.magTrust = options.magTrust ?? 0.5; // Default: moderate trust

        // Initialize magnetometer trust on AHRS
        this.imuFusion.setMagTrust(this.magTrust);

        // Initialize geomagnetic reference (async, uses default until browser location available)
        this._initGeomagneticReference();

        // Callbacks
        this.onProcessed = options.onProcessed || null;
        this.onOrientationUpdate = options.onOrientationUpdate || null;
        this.onGyroBiasCalibrated = options.onGyroBiasCalibrated || null;
        this.onLog = options.onLog || null;

        // Log startup calibration state
        this._logStartupState();
    }

    /**
     * Log diagnostic message via callback (or console as fallback)
     */
    private _logDiagnostic(message: string): void {
        if (this.onLog) {
            this.onLog(message);
        } else {
            console.log(message);
        }
    }

    /**
     * Log startup calibration state for debugging
     */
    private _logStartupState(): void {
        const calState = this.magCalibration.getState();
        const hasIron = this.magCalibration.hasIronCalibration();

        this._logDiagnostic(`[MagDiag] === STARTUP STATE ===`);
        this._logDiagnostic(`[MagDiag] useMagnetometer: ${this.useMagnetometer}, magTrust: ${this.magTrust}`);
        this._logDiagnostic(`[MagDiag] Iron calibration loaded: ${hasIron}`);

        if (hasIron) {
            // Get saved calibration data
            const calJson = this.magCalibration.toJSON();
            this._logDiagnostic(`[MagDiag] Hard iron offset: [${calJson.hardIronOffset.x.toFixed(1)}, ${calJson.hardIronOffset.y.toFixed(1)}, ${calJson.hardIronOffset.z.toFixed(1)}] µT`);
        }

        this._logDiagnostic(`[MagDiag] Earth field estimate: ${calState.earthMagnitude.toFixed(1)} µT`);
        this._logDiagnostic(`[MagDiag] Cal ready: ${calState.ready}, confidence: ${(calState.confidence * 100).toFixed(0)}%`);
    }

    /**
     * Get diagnostic summary for UI display
     */
    getDiagnosticSummary(): {
        useMagnetometer: boolean;
        magTrust: number;
        hasIronCal: boolean;
        geomagRef: GeomagneticRef | null;
        lastResidual: number;
        lastYaw: number;
        sampleCount: number;
    } {
        const lastHist = this._magResidualHistory[this._magResidualHistory.length - 1];
        return {
            useMagnetometer: this.useMagnetometer,
            magTrust: this.magTrust,
            hasIronCal: this.magCalibration.hasIronCalibration(),
            geomagRef: this.geomagneticRef,
            lastResidual: lastHist?.mag ?? 0,
            lastYaw: lastHist?.yaw ?? 0,
            sampleCount: this._sampleCount
        };
    }

    /**
     * Get magnet detector instance
     */
    getMagnetDetector(): MagnetDetector {
        return this.magnetDetector;
    }

    /**
     * Get current magnet detection state
     */
    getMagnetState(): MagnetDetectorState {
        return this.magnetDetector.getState();
    }

    /**
     * Get mag calibration instance
     */
    getMagCalibration(): UnifiedMagCalibration {
        return this.magCalibration;
    }

    /**
     * Reset mag calibration
     */
    resetMagCalibration(): void {
        this.magCalibration.reset();
        console.log('[TelemetryProcessor] Mag calibration reset');
    }

    /**
     * Initialize geomagnetic reference from location
     */
    private async _initGeomagneticReference(): Promise<void> {
        // Start with default location immediately
        const defaultLoc = getDefaultLocation();
        if (defaultLoc) {
            this._setGeomagneticRef(defaultLoc);
            this._logDiagnostic(`[MagDiag] GeomagRef (default): ${defaultLoc.city} H=${defaultLoc.horizontal.toFixed(1)}µT V=${defaultLoc.vertical.toFixed(1)}µT D=${defaultLoc.declination.toFixed(1)}°`);
        }

        // Try to get browser location (async, updates if successful)
        try {
            const browserLoc = await getBrowserLocation({ timeout: 5000 });
            this._setGeomagneticRef(browserLoc.selected);
            this._logDiagnostic(`[MagDiag] GeomagRef (browser): ${browserLoc.selected.city} H=${browserLoc.selected.horizontal.toFixed(1)}µT V=${browserLoc.selected.vertical.toFixed(1)}µT`);
        } catch (e) {
            // Browser location not available - this is expected in many contexts
            console.debug('[MagDiag] Browser location unavailable:', (e as Error).message);
        }
    }

    /**
     * Set geomagnetic reference on AHRS and calibration module
     */
    private _setGeomagneticRef(location: GeomagneticLocation | null): void {
        if (location) {
            this.geomagneticRef = {
                horizontal: location.horizontal,
                vertical: location.vertical,
                declination: location.declination
            };
            this.imuFusion.setGeomagneticReference(this.geomagneticRef);
            // Also set on calibration module for auto hard iron estimation
            this.magCalibration.setGeomagneticReference(this.geomagneticRef);
        }
    }

    /**
     * Set magnetometer trust factor
     */
    setMagTrust(trust: number): void {
        this.magTrust = Math.max(0, Math.min(1, trust));
        this.imuFusion.setMagTrust(this.magTrust);
    }

    /**
     * Enable/disable magnetometer fusion
     */
    setMagnetometerEnabled(enabled: boolean): void {
        this.useMagnetometer = enabled;
    }

    /**
     * Reload iron calibration from localStorage
     */
    reloadCalibration(): void {
        this.magCalibration.load('gambit_calibration');
        // Reset Earth estimation since iron calibration may have changed
        this.magCalibration.resetEarthEstimation();
    }

    /**
     * Process a telemetry sample through the full pipeline
     */
    process(raw: RawTelemetry): DecoratedTelemetry {
        // IMPORTANT: Preserve raw data, only DECORATE with processed fields
        const decorated: DecoratedTelemetry = { ...raw };

        // Calculate time step
        const now = performance.now();
        const dt = this.lastTimestamp ? (now - this.lastTimestamp) / 1000 : 0.02;
        this.lastTimestamp = now;

        // Store dt for external use
        decorated.dt = dt;

        // ===== Step 1: Unit Conversion =====
        const ax_g = accelLsbToG(raw.ax || 0);
        const ay_g = accelLsbToG(raw.ay || 0);
        const az_g = accelLsbToG(raw.az || 0);

        const gx_dps = gyroLsbToDps(raw.gx || 0);
        const gy_dps = gyroLsbToDps(raw.gy || 0);
        const gz_dps = gyroLsbToDps(raw.gz || 0);

        decorated.ax_g = ax_g;
        decorated.ay_g = ay_g;
        decorated.az_g = az_g;
        decorated.gx_dps = gx_dps;
        decorated.gy_dps = gy_dps;
        decorated.gz_dps = gz_dps;

        // Convert magnetometer to µT (raw sensor frame)
        const mx_ut_raw = magLsbToMicroTesla(raw.mx || 0);
        const my_ut_raw = magLsbToMicroTesla(raw.my || 0);
        const mz_ut_raw = magLsbToMicroTesla(raw.mz || 0);

        // ===== Magnetometer Axis Alignment =====
        // Puck.js has different axis orientation for magnetometer vs accel/gyro:
        //   Accel/Gyro: X→aerial, Y→IR LEDs, Z→into PCB
        //   Magnetometer: X→IR LEDs, Y→aerial, Z→into PCB
        // Swap X and Y to align magnetometer to accel/gyro frame
        // Additionally, negate Y to match accelerometer sign convention
        // (verified by correlation analysis: ay vs my should be positive)
        const mx_ut = my_ut_raw;   // Mag Y (aerial) -> aligned X (aerial)
        const my_ut = -mx_ut_raw;  // Mag X (IR LEDs) -> aligned Y (IR LEDs), NEGATED
        const mz_ut = mz_ut_raw;   // Z unchanged

        decorated.mx_ut = mx_ut;
        decorated.my_ut = my_ut;
        decorated.mz_ut = mz_ut;

        // ===== Step 2: Motion Detection =====
        const motionState = this.motionDetector.update(
            raw.ax || 0, raw.ay || 0, raw.az || 0,
            raw.gx || 0, raw.gy || 0, raw.gz || 0
        );
        decorated.isMoving = motionState.isMoving;
        decorated.accelStd = motionState.accelStd;
        decorated.gyroStd = motionState.gyroStd;

        // ===== Step 3: Gyroscope Bias Calibration =====
        if (!motionState.isMoving) {
            this.gyroBiasState.stationaryCount++;

            if (this.gyroBiasState.stationaryCount > STATIONARY_SAMPLES_FOR_CALIBRATION) {
                this.imuFusion.updateGyroBias(gx_dps, gy_dps, gz_dps, true);

                if (!this.gyroBiasState.calibrated) {
                    this.gyroBiasState.calibrated = true;
                    
                    // Log the actual bias values
                    const bias = (this.imuFusion as any).getGyroBiasDegrees?.() || 
                                 (this.imuFusion as any).gyroBias;
                    if (bias) {
                        const bx = typeof bias.x === 'number' ? bias.x : 0;
                        const by = typeof bias.y === 'number' ? bias.y : 0;
                        const bz = typeof bias.z === 'number' ? bias.z : 0;
                        // Convert from radians if needed (check magnitude)
                        const scale = Math.abs(bx) < 0.1 && Math.abs(by) < 0.1 && Math.abs(bz) < 0.1 ? 180/Math.PI : 1;
                        this._logDiagnostic(`[MagDiag] Gyro bias calibrated: [${(bx*scale).toFixed(3)}, ${(by*scale).toFixed(3)}, ${(bz*scale).toFixed(3)}] °/s`);
                    }
                    console.log('[TelemetryProcessor] Gyroscope bias calibration complete');

                    if (this.onGyroBiasCalibrated) {
                        this.onGyroBiasCalibrated();
                    }
                }
            }
        } else {
            this.gyroBiasState.stationaryCount = 0;
        }
        decorated.gyroBiasCalibrated = this.gyroBiasState.calibrated;

        // ===== Step 4: IMU Sensor Fusion =====
        if (!this.imuInitialized) {
            const accelMag = Math.abs(raw.ax || 0) + Math.abs(raw.ay || 0) + Math.abs(raw.az || 0);
            if (accelMag > 0.5) {
                this.imuFusion.initFromAccelerometer(raw.ax, raw.ay, raw.az);
                this.imuInitialized = true;
                console.log('[TelemetryProcessor] IMU initialized from accelerometer');
            }
        }

        if (this.imuInitialized) {
            const magDataValid = mx_ut !== 0 || my_ut !== 0 || mz_ut !== 0;

            // Progressive 9-DOF fusion: use magnetometer from the start with scaled trust
            // As calibration progresses, trust increases. This provides yaw stability even during warmup.
            if (this.useMagnetometer && magDataValid && this.geomagneticRef) {
                // Get calibration progress (0.0 to 1.0)
                const calProgress = this.magCalibration.getAutoHardIronProgress();
                const isWizardCal = this.magCalibration.hardIronCalibrated;

                // Scale mag trust by calibration progress
                // Wizard cal = full trust, Auto cal = progressive trust
                const effectiveMagTrust = isWizardCal ? this.magTrust : this.magTrust * calProgress;

                // Apply progressive iron correction (uses current estimate even if not "ready")
                const ironCorrected = this.magCalibration.applyProgressiveIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });
                const corrMag = Math.sqrt(ironCorrected.x**2 + ironCorrected.y**2 + ironCorrected.z**2);

                // Set dynamic mag trust on AHRS
                this.imuFusion.setMagTrust(effectiveMagTrust);

                // 9-DOF fusion with magnetometer (always, with scaled trust)
                this.imuFusion.updateWithMag(
                    ax_g, ay_g, az_g,
                    gx_dps, gy_dps, gz_dps,
                    ironCorrected.x, ironCorrected.y, ironCorrected.z,
                    dt, true, false
                );

                // === COMPREHENSIVE DIAGNOSTIC LOGGING ===
                const calState = this.magCalibration.getState();
                const ranges = calState.autoHardIronRanges;
                const autoEst = this.magCalibration.getCurrentAutoHardIronEstimate();
                const rawMag = Math.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2);

                // Log state transition: first time or significant change
                if (!this._loggedMagFusion) {
                    this._logDiagnostic(`[MagDiag] 🚀 Progressive 9-DOF fusion ENABLED from start`);
                    this._logDiagnostic(`[MagDiag] Strategy: scale magTrust by calibration progress`);
                    this._logDiagnostic(`[MagDiag] Base magTrust: ${this.magTrust}, effective: ${effectiveMagTrust.toFixed(3)}`);
                    this._loggedMagFusion = true;
                }

                // Log progress periodically (every 50 samples during calibration)
                if (!calState.autoHardIronReady && calState.autoHardIronSampleCount % 50 === 0 && calState.autoHardIronSampleCount > 0) {
                    const coveragePct = (calProgress * 100).toFixed(0);
                    const autoEstMag = Math.sqrt(autoEst.x**2 + autoEst.y**2 + autoEst.z**2);
                    this._logDiagnostic(`[MagDiag] Cal progress: ${coveragePct}% | trust: ${effectiveMagTrust.toFixed(2)} | ranges: [${ranges.x.toFixed(0)}, ${ranges.y.toFixed(0)}, ${ranges.z.toFixed(0)}] µT`);
                    this._logDiagnostic(`[MagDiag]   offset: [${autoEst.x.toFixed(1)}, ${autoEst.y.toFixed(1)}, ${autoEst.z.toFixed(1)}] µT (|${autoEstMag.toFixed(1)}|) | corrMag: ${corrMag.toFixed(1)} µT | rawMag: ${rawMag.toFixed(1)} µT`);
                }

                // Log when calibration reaches 100%
                if (calState.autoHardIronReady && this._loggedMagFusionDisabled) {
                    const autoEstMag = Math.sqrt(autoEst.x**2 + autoEst.y**2 + autoEst.z**2);
                    const expectedMag = this.geomagneticRef
                        ? Math.sqrt(this.geomagneticRef.horizontal**2 + this.geomagneticRef.vertical**2)
                        : 50;
                    this._logDiagnostic(`[MagDiag] ✅ Calibration COMPLETE - full trust now active`);
                    this._logDiagnostic(`[MagDiag]   Final offset: [${autoEst.x.toFixed(1)}, ${autoEst.y.toFixed(1)}, ${autoEst.z.toFixed(1)}] µT (|${autoEstMag.toFixed(1)}|)`);
                    this._logDiagnostic(`[MagDiag]   Final ranges: [${ranges.x.toFixed(1)}, ${ranges.y.toFixed(1)}, ${ranges.z.toFixed(1)}] µT`);

                    // Soft iron correction analysis - use actual scale from calibration state
                    const softIronScale = calState.autoSoftIronScale;
                    const avgRange = (ranges.x + ranges.y + ranges.z) / 3;
                    const sphericity = Math.min(ranges.x, ranges.y, ranges.z) / Math.max(ranges.x, ranges.y, ranges.z);
                    this._logDiagnostic(`[MagDiag]   Soft iron scale: [${softIronScale.x.toFixed(3)}, ${softIronScale.y.toFixed(3)}, ${softIronScale.z.toFixed(3)}]`);
                    this._logDiagnostic(`[MagDiag]   Sphericity: ${sphericity.toFixed(2)} (${sphericity > 0.7 ? 'good' : sphericity > 0.5 ? 'fair' : 'poor'})`);

                    // Axis asymmetry analysis
                    const xDev = ((ranges.x - avgRange) / avgRange * 100).toFixed(1);
                    const yDev = ((ranges.y - avgRange) / avgRange * 100).toFixed(1);
                    const zDev = ((ranges.z - avgRange) / avgRange * 100).toFixed(1);
                    this._logDiagnostic(`[MagDiag]   Axis deviation: X=${xDev}%, Y=${yDev}%, Z=${zDev}%`);

                    // Re-compute corrected magnitude WITH soft iron applied (since corrMag above may not have it yet)
                    // This gives accurate diagnostic of final calibration quality
                    const hardIronCorrected = {
                        x: mx_ut - autoEst.x,
                        y: my_ut - autoEst.y,
                        z: mz_ut - autoEst.z
                    };
                    const fullyCorrected = {
                        x: hardIronCorrected.x * softIronScale.x,
                        y: hardIronCorrected.y * softIronScale.y,
                        z: hardIronCorrected.z * softIronScale.z
                    };
                    const fullyCorrectedMag = Math.sqrt(fullyCorrected.x**2 + fullyCorrected.y**2 + fullyCorrected.z**2);

                    // Final corrected magnitude check
                    const magError = Math.abs(fullyCorrectedMag - expectedMag);
                    const magErrorPct = (magError / expectedMag * 100).toFixed(1);
                    this._logDiagnostic(`[MagDiag]   Iron-corrected mag: ${fullyCorrectedMag.toFixed(1)} µT (expected ~${expectedMag.toFixed(1)} µT, error: ${magErrorPct}%) ${magError < 5 ? '✓' : '⚠️'}`);

                    // H/V component analysis (critical for finger detection accuracy)
                    // Uses accelerometer to compute roll/pitch, then tilt-compensates magnetometer
                    const accelMag = Math.sqrt(ax_g**2 + ay_g**2 + az_g**2);
                    if (accelMag > 0.5) {
                        const roll = Math.atan2(ay_g, az_g);
                        const pitch = Math.atan2(-ax_g, Math.sqrt(ay_g**2 + az_g**2));
                        const cr = Math.cos(roll), sr = Math.sin(roll);
                        const cp = Math.cos(pitch), sp = Math.sin(pitch);

                        // Tilt-compensate to get H and V components
                        const mx_h = fullyCorrected.x * cp + fullyCorrected.y * sr * sp + fullyCorrected.z * cr * sp;
                        const my_h = fullyCorrected.y * cr - fullyCorrected.z * sr;
                        const mz_h = -fullyCorrected.x * sp + fullyCorrected.y * cr * sp + fullyCorrected.z * cr * cp;
                        const h_mag = Math.sqrt(mx_h**2 + my_h**2);
                        const v_mag = mz_h;

                        const expectedH = this.geomagneticRef?.horizontal ?? 16.0;
                        const expectedV = this.geomagneticRef?.vertical ?? 47.8;
                        const expectedHV = expectedH / expectedV;
                        const actualHV = Math.abs(v_mag) > 1 ? h_mag / Math.abs(v_mag) : 0;

                        this._logDiagnostic(`[MagDiag]   H/V components: H=${h_mag.toFixed(1)} µT (exp ${expectedH.toFixed(1)}), V=${v_mag.toFixed(1)} µT (exp ${expectedV.toFixed(1)})`);
                        this._logDiagnostic(`[MagDiag]   H/V ratio: ${actualHV.toFixed(2)} (expected ${expectedHV.toFixed(2)}) ${Math.abs(actualHV - expectedHV) < 0.2 ? '✓' : '⚠️ DIRECTION ERROR'}`);
                    }

                    this._loggedMagFusionDisabled = false;
                }

                // Track that we're in calibration phase (for detecting completion)
                if (!calState.autoHardIronReady && !this._loggedMagFusionDisabled) {
                    this._loggedMagFusionDisabled = true;
                }
            } else {
                // 6-DOF fusion fallback (no mag data or no geomag ref)
                this.imuFusion.update(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, dt, true);

                // Log once why mag fusion is skipped
                if (this.useMagnetometer && !this._loggedMagFusionDisabled) {
                    if (!magDataValid) {
                        this._logDiagnostic(`[MagDiag] ⚠️ Mag fusion DISABLED - no mag data`);
                    } else if (!this.geomagneticRef) {
                        this._logDiagnostic(`[MagDiag] ⚠️ Mag fusion DISABLED - waiting for geomagnetic reference`);
                    }
                    this._loggedMagFusionDisabled = true;
                }
            }
        }

        // Get current orientation
        const orientation = this.imuInitialized ? this.imuFusion.getQuaternion() : null;
        const euler = this.imuInitialized ? this.imuFusion.getEulerAngles() : null;

        if (orientation) {
            decorated.orientation_w = orientation.w;
            decorated.orientation_x = orientation.x;
            decorated.orientation_y = orientation.y;
            decorated.orientation_z = orientation.z;
        }
        if (euler) {
            decorated.euler_roll = euler.roll;
            decorated.euler_pitch = euler.pitch;
            decorated.euler_yaw = euler.yaw;
        }

        // Add magnetometer residual from AHRS
        if (this.useMagnetometer && this.imuInitialized) {
            const magResidual = this.imuFusion.getMagResidual();
            if (magResidual) {
                decorated.ahrs_mag_residual_x = magResidual.x;
                decorated.ahrs_mag_residual_y = magResidual.y;
                decorated.ahrs_mag_residual_z = magResidual.z;
                decorated.ahrs_mag_residual_magnitude = this.imuFusion.getMagResidualMagnitude();

                // Diagnostic logging for magnetometer drift analysis
                this._sampleCount++;
                const yaw = euler?.yaw ?? 0;
                const residualMag = decorated.ahrs_mag_residual_magnitude;

                // Track residual history (keep last 500 samples = ~10s at 50Hz)
                this._magResidualHistory.push({
                    x: magResidual.x,
                    y: magResidual.y,
                    z: magResidual.z,
                    mag: residualMag,
                    yaw
                });
                if (this._magResidualHistory.length > 500) {
                    this._magResidualHistory.shift();
                }

                // Log diagnostics every 2 seconds (100 samples at 50Hz)
                const now = performance.now();
                if (now - this._lastDiagnosticLog > 2000 && this._magResidualHistory.length >= 50) {
                    this._lastDiagnosticLog = now;

                    // Compute drift metrics
                    const recent = this._magResidualHistory.slice(-50); // last 1 second
                    const older = this._magResidualHistory.slice(0, 50); // first 1 second in window

                    const avgRecentMag = recent.reduce((s, r) => s + r.mag, 0) / recent.length;
                    const avgOlderMag = older.length > 0 ? older.reduce((s, r) => s + r.mag, 0) / older.length : avgRecentMag;
                    const avgRecentYaw = recent.reduce((s, r) => s + r.yaw, 0) / recent.length;
                    const avgOlderYaw = older.length > 0 ? older.reduce((s, r) => s + r.yaw, 0) / older.length : avgRecentYaw;

                    const magDrift = avgRecentMag - avgOlderMag;
                    const yawDrift = avgRecentYaw - avgOlderYaw;
                    const isStationary = !motionState.isMoving;

                    // Get calibration state for diagnostics
                    const calProgress = this.magCalibration.getAutoHardIronProgress();
                    const calState = this.magCalibration.getState();
                    const effectiveTrust = this.magCalibration.hardIronCalibrated
                        ? this.magTrust
                        : this.magTrust * calProgress;
                    const ironCorrected = this.magCalibration.applyProgressiveIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });
                    const corrMag = Math.sqrt(ironCorrected.x**2 + ironCorrected.y**2 + ironCorrected.z**2);

                    // Get Earth-subtracted residual (the one that matters for finger detection)
                    // This should be ~0 µT without finger magnets if calibration is correct
                    let earthResidualMag = 0;
                    if (orientation && calState.ready) {
                        const earthResidual = this.magCalibration.getResidual(mx_ut, my_ut, mz_ut, orientation);
                        if (earthResidual) {
                            earthResidualMag = earthResidual.magnitude;
                        }
                    }

                    this._logDiagnostic(
                        `[MagDiag] ` +
                        `earthRes=${earthResidualMag.toFixed(1)}µT corrMag=${corrMag.toFixed(1)}µT | ` +
                        `yaw=${yaw.toFixed(1)}° | ` +
                        `cal=${(calProgress*100).toFixed(0)}% trust=${effectiveTrust.toFixed(2)} | ` +
                        `Δ: mag=${magDrift.toFixed(1)} yaw=${yawDrift.toFixed(1)}° | ` +
                        `${isStationary ? 'STILL' : 'moving'}`
                    );
                }
            }
        }

        // Notify orientation update
        if (this.onOrientationUpdate && euler && orientation) {
            this.onOrientationUpdate(euler, orientation);
        }

        // ===== Step 5: Magnetometer Calibration (Unified) =====
        // Always use progressive iron correction (includes soft iron scaling when available)
        // This ensures consistency with AHRS fusion and residual calculation
        const ironCorrected = this.magCalibration.applyProgressiveIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

        // Collect samples for orientation-aware calibration (uses accelerometer for direction constraint)
        // This enables full 3x3 soft iron matrix optimization for 90% residual reduction
        this.magCalibration.collectOrientationAwareSample(mx_ut, my_ut, mz_ut, ax_g, ay_g, az_g);
        decorated.iron_mx = ironCorrected.x;
        decorated.iron_my = ironCorrected.y;
        decorated.iron_mz = ironCorrected.z;

        if (orientation) {
            this.magCalibration.update(mx_ut, my_ut, mz_ut, orientation);

            const calState = this.magCalibration.getState();
            decorated.mag_cal_ready = calState.ready;
            decorated.mag_cal_confidence = calState.confidence;
            decorated.mag_cal_mean_residual = calState.meanResidual;
            decorated.mag_cal_earth_magnitude = calState.earthMagnitude;
            decorated.mag_cal_hard_iron = calState.hardIronCalibrated;
            decorated.mag_cal_soft_iron = calState.softIronCalibrated;

            if (calState.ready) {
                const residual = this.magCalibration.getResidual(mx_ut, my_ut, mz_ut, orientation);
                if (residual) {
                    decorated.residual_mx = residual.x;
                    decorated.residual_my = residual.y;
                    decorated.residual_mz = residual.z;
                    decorated.residual_magnitude = residual.magnitude;

                    const magnetState = this.magnetDetector.update(residual.magnitude);
                    decorated.magnet_status = magnetState.status;
                    decorated.magnet_confidence = magnetState.confidence;
                    decorated.magnet_detected = magnetState.detected;
                    decorated.magnet_baseline_established = magnetState.baselineEstablished;
                    decorated.magnet_baseline_residual = magnetState.baselineResidual;
                    decorated.magnet_deviation = magnetState.deviationFromBaseline;

                    if (magnetState.detected && !this._loggedMagnetDetection) {
                        console.log('[TelemetryProcessor] Finger magnets detected! Status:', magnetState.status,
                                    'Confidence:', (magnetState.confidence * 100).toFixed(0) + '%',
                                    'Residual:', magnetState.avgResidual.toFixed(1), 'uT');
                        this._loggedMagnetDetection = true;
                    }
                }
            }
        } else {
            if (!this._loggedCalibrationMissing) {
                console.debug('[TelemetryProcessor] Orientation not available - using raw mag values');
                this._loggedCalibrationMissing = true;
            }
            decorated.residual_magnitude = Math.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2);
        }

        // ===== Step 6: Kalman Filtering =====
        try {
            const magInput = {
                x: decorated.residual_mx ?? mx_ut,
                y: decorated.residual_my ?? my_ut,
                z: decorated.residual_mz ?? mz_ut
            };
            const filteredMag = this.magFilter.update(magInput);
            decorated.filtered_mx = filteredMag.x;
            decorated.filtered_my = filteredMag.y;
            decorated.filtered_mz = filteredMag.z;
        } catch (e) {
            // Filtering failed, skip decoration
        }

        // ===== Step 7: Device Context (v0.4.0) =====
        // Map mode code to full name
        if (raw.mode) {
            const modeMap: Record<string, 'LOW_POWER' | 'NORMAL' | 'HIGH_RES' | 'BURST'> = {
                'L': 'LOW_POWER',
                'N': 'NORMAL',
                'H': 'HIGH_RES',
                'B': 'BURST',
            };
            decorated.modeName = modeMap[raw.mode] || undefined;
        }

        // Map context code to full name
        if (raw.ctx) {
            const ctxMap: Record<string, 'unknown' | 'stored' | 'held' | 'active' | 'table'> = {
                'u': 'unknown',
                's': 'stored',
                'h': 'held',
                'a': 'active',
                't': 'table',
            };
            decorated.contextName = ctxMap[raw.ctx] || undefined;
        }

        // Map grip state
        if (raw.grip !== null && raw.grip !== undefined) {
            decorated.isGripped = raw.grip === 1;
        }

        // Notify processed telemetry
        if (this.onProcessed) {
            this.onProcessed(decorated);
        }

        return decorated;
    }

    /**
     * Reset the processor state
     */
    reset(): void {
        this.imuFusion.reset();
        this.magFilter.reset();
        this.motionDetector.reset();
        this.gyroBiasState = createGyroBiasState();
        this.imuInitialized = false;
        this.lastTimestamp = null;

        this._loggedCalibrationMissing = false;
        this._loggedMagFusion = false;
        this._loggedMagnetDetection = false;

        // Reset diagnostic tracking
        this._sampleCount = 0;
        this._magResidualHistory = [];
        this._lastDiagnosticLog = 0;

        this.magCalibration.reset();
        this.magnetDetector.reset();

        console.log('[TelemetryProcessor] Reset complete');
    }

    /**
     * Get current orientation as Euler angles
     */
    getEulerAngles(): EulerAngles | null {
        return this.imuInitialized ? this.imuFusion.getEulerAngles() : null;
    }

    /**
     * Get current orientation as quaternion
     */
    getQuaternion(): Quaternion | null {
        return this.imuInitialized ? this.imuFusion.getQuaternion() : null;
    }

    /**
     * Get gyroscope bias calibration state
     */
    getGyroBiasState(): GyroBiasState {
        return { ...this.gyroBiasState };
    }

    /**
     * Check if IMU is initialized
     */
    isIMUInitialized(): boolean {
        return this.imuInitialized;
    }

    /**
     * Get motion state
     */
    getMotionState(): MotionState {
        return this.motionDetector.getState();
    }
}

// ===== Factory Function =====

/**
 * Create a TelemetryProcessor instance with standard configuration
 */
export function createTelemetryProcessor(options: TelemetryProcessorOptions = {}): TelemetryProcessor {
    return new TelemetryProcessor(options);
}

// ===== Default Export =====

export default {
    TelemetryProcessor,
    createTelemetryProcessor
};
