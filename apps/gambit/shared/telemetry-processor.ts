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

import type { EulerAngles, Quaternion } from '@core/types';
import type { GeomagneticLocation } from './geomagnetic-field';
import type { MagnetDetectorState } from './magnet-detector';

// ===== Type Definitions =====

export interface RawTelemetry {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx: number;
  my: number;
  mz: number;
  t: number;
}

export interface DecoratedTelemetry extends RawTelemetry {
  dt?: number;
  ax_g?: number;
  ay_g?: number;
  az_g?: number;
  gx_dps?: number;
  gy_dps?: number;
  gz_dps?: number;
  mx_ut?: number;
  my_ut?: number;
  mz_ut?: number;
  isMoving?: boolean;
  accelStd?: number;
  gyroStd?: number;
  gyroBiasCalibrated?: boolean;
  orientation_w?: number;
  orientation_x?: number;
  orientation_y?: number;
  orientation_z?: number;
  euler_roll?: number;
  euler_pitch?: number;
  euler_yaw?: number;
  ahrs_mag_residual_x?: number;
  ahrs_mag_residual_y?: number;
  ahrs_mag_residual_z?: number;
  ahrs_mag_residual_magnitude?: number;
  iron_mx?: number;
  iron_my?: number;
  iron_mz?: number;
  mag_cal_ready?: boolean;
  mag_cal_confidence?: number;
  mag_cal_mean_residual?: number;
  mag_cal_earth_magnitude?: number;
  mag_cal_hard_iron?: boolean;
  mag_cal_soft_iron?: boolean;
  residual_mx?: number;
  residual_my?: number;
  residual_mz?: number;
  residual_magnitude?: number;
  magnet_status?: string;
  magnet_confidence?: number;
  magnet_detected?: boolean;
  magnet_baseline_established?: boolean;
  magnet_baseline_residual?: number;
  magnet_deviation?: number;
  filtered_mx?: number;
  filtered_my?: number;
  filtered_mz?: number;
}

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

interface GeomagneticRef {
  horizontal: number;
  vertical: number;
  declination: number;
}

export interface MotionState {
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}

export interface GyroBiasState {
  calibrated: boolean;
  stationaryCount: number;
}

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
            debug: options.magCalibrationDebug || false
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
        const mx_ut = my_ut_raw;  // Mag Y (aerial) -> aligned X (aerial)
        const my_ut = mx_ut_raw;  // Mag X (IR LEDs) -> aligned Y (IR LEDs)
        const mz_ut = mz_ut_raw;  // Z unchanged

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
            const hasIronCal = this.magCalibration.hasIronCalibration();

            // Only use magnetometer in fusion if iron calibration is available
            // Without iron calibration, the hard iron bias causes massive residual error
            if (this.useMagnetometer && magDataValid && this.geomagneticRef && hasIronCal) {
                // 9-DOF fusion with magnetometer for absolute yaw reference
                const ironCorrected = this.magCalibration.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });

                this.imuFusion.updateWithMag(
                    ax_g, ay_g, az_g,
                    gx_dps, gy_dps, gz_dps,
                    ironCorrected.x, ironCorrected.y, ironCorrected.z,
                    dt, true, false
                );

                // Log when transitioning to 9-DOF (or first time)
                if (!this._loggedMagFusion || this._loggedMagFusionDisabled) {
                    const calType = this.magCalibration.isUsingAutoHardIron() ? 'auto' : 'wizard';
                    const autoEst = this.magCalibration.getAutoHardIronEstimate();
                    if (this._loggedMagFusionDisabled) {
                        this._logDiagnostic(`[MagDiag] ✅ Auto hard iron calibration complete! Enabling 9-DOF fusion...`);
                    }
                    this._logDiagnostic(`[MagDiag] Using 9-DOF fusion with axis-aligned, iron-corrected magnetometer (trust: ${this.magTrust})`);
                    this._logDiagnostic(`[MagDiag] Iron calibration: ${calType.toUpperCase()}`);
                    if (calType === 'auto' && autoEst) {
                        this._logDiagnostic(`[MagDiag] Auto hard iron: [${autoEst.x.toFixed(1)}, ${autoEst.y.toFixed(1)}, ${autoEst.z.toFixed(1)}] µT`);
                    }
                    this._loggedMagFusion = true;
                    this._loggedMagFusionDisabled = false;
                }
            } else {
                // 6-DOF fusion (gyro + accel only)
                this.imuFusion.update(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, dt, true);

                // Log once why mag fusion is skipped
                if (this.useMagnetometer && !this._loggedMagFusionDisabled && !hasIronCal) {
                    this._logDiagnostic(`[MagDiag] ⚠️ Mag fusion DISABLED - no iron calibration yet (auto calibration building...)`);
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

                    // Also log raw mag reading for comparison
                    const rawMagMag = Math.sqrt(mx_ut ** 2 + my_ut ** 2 + mz_ut ** 2);

                    this._logDiagnostic(
                        `[MagDiag] ` +
                        `res=${residualMag.toFixed(1)}µT raw=${rawMagMag.toFixed(1)}µT | ` +
                        `yaw=${yaw.toFixed(1)}° | ` +
                        `Δ(${(this._magResidualHistory.length/50).toFixed(0)}s): mag=${magDrift.toFixed(1)} yaw=${yawDrift.toFixed(1)}° | ` +
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
        if (this.magCalibration.hasIronCalibration()) {
            const ironCorrected = this.magCalibration.applyIronCorrection({ x: mx_ut, y: my_ut, z: mz_ut });
            decorated.iron_mx = ironCorrected.x;
            decorated.iron_my = ironCorrected.y;
            decorated.iron_mz = ironCorrected.z;
        }

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
