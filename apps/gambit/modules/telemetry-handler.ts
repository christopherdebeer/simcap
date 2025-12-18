/**
 * Telemetry Data Handler
 *
 * Processes incoming sensor data using the shared TelemetryProcessor.
 * Handles session recording, calibration wizard, and UI updates.
 *
 * This module wraps the shared TelemetryProcessor and adds collector-specific
 * functionality like session storage, calibration wizard integration, and
 * pose estimation updates.
 */

import { state } from './state.js';
import { TelemetryProcessor } from '../shared/telemetry-processor.js';
import type { EulerAngles, Quaternion } from '@core/types';
import type { UnifiedMagCalibration } from '../shared/unified-mag-calibration';

// ===== Type Definitions =====

interface MagField {
  x: number;
  y: number;
  z: number;
}

interface PoseUpdateData {
  magField: MagField;
  orientation: Quaternion | null;
  euler: EulerAngles | null;
  sample: DecoratedTelemetry;
}

interface ThreeHandSkeleton {
  updateOrientation: (euler: EulerAngles) => void;
}

interface WizardState {
  active: boolean;
  phase: string | null;
  currentStep: number;
  steps: Array<{ id: string }>;
}

interface PoseState {
  enabled: boolean;
}

interface CalibrationBuffers {
  [key: string]: Array<{ mx: number; my: number; mz: number }>;
}

interface Dependencies {
  calibrationInstance: UnifiedMagCalibration | null;
  wizard: WizardState | null;
  calibrationBuffers: CalibrationBuffers | null;
  poseState: PoseState | null;
  updatePoseEstimation: ((data: PoseUpdateData) => void) | null;
  updateMagTrajectory: ((data: { fused_mx: number; fused_my: number; fused_mz: number }) => void) | null;
  updateUI: (() => void) | null;
  $: ((id: string) => HTMLElement | null) | null;
  threeHandSkeleton: ThreeHandSkeleton | (() => ThreeHandSkeleton | null) | null;
}

interface RawTelemetry {
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

interface DecoratedTelemetry extends RawTelemetry {
  mx_ut?: number;
  my_ut?: number;
  mz_ut?: number;
  calibrated_mx?: number;
  calibrated_my?: number;
  calibrated_mz?: number;
  filtered_mx?: number;
  filtered_my?: number;
  filtered_mz?: number;
  residual_mx?: number;
  residual_my?: number;
  residual_mz?: number;
  residual_magnitude?: number;
  [key: string]: any;
}

interface MotionState {
  isMoving: boolean;
  accelStd: number;
  gyroStd: number;
}

interface GyroBiasState {
  calibrated: boolean;
  stationaryCount: number;
}

// ===== Module State =====

let deps: Dependencies = {
    calibrationInstance: null,
    wizard: null,
    calibrationBuffers: null,
    poseState: null,
    updatePoseEstimation: null,
    updateMagTrajectory: null,
    updateUI: null,
    $: null,
    threeHandSkeleton: null
};

// Shared telemetry processor instance
let telemetryProcessor: TelemetryProcessor | null = null;

// ===== Dependency Management =====

/**
 * Set module dependencies
 * @param dependencies - Required dependencies
 */
export function setDependencies(dependencies: Partial<Dependencies>): void {
    deps = { ...deps, ...dependencies };

    // Update calibration in processor if it exists
    if (telemetryProcessor && deps.calibrationInstance) {
        // telemetryProcessor.setCalibration(deps.calibrationInstance);
    }
}

// ===== Processor Management =====

/**
 * Initialize the telemetry processor
 * Call this after dependencies are set
 */
export function initProcessor(): void {
    telemetryProcessor = new TelemetryProcessor({
        useMagnetometer: false,
        calibration: deps.calibrationInstance,
        magCalibrationDebug: true, // Enable debug logging for mag calibration
        onOrientationUpdate: (euler: EulerAngles, quaternion: Quaternion) => {
            // Update Three.js hand skeleton if available
            if (euler) {
                const threeSkeleton = typeof deps.threeHandSkeleton === 'function'
                    ? deps.threeHandSkeleton()
                    : deps.threeHandSkeleton;
                if (threeSkeleton) {
                    threeSkeleton.updateOrientation(euler);
                }
            }
        },
        onGyroBiasCalibrated: () => {
            console.log('[TelemetryHandler] Gyroscope bias calibration complete');
        }
    });
}

/**
 * Get mag calibration instance
 * @returns UnifiedMagCalibration or null
 */
export function getMagCalibration(): UnifiedMagCalibration | null {
    return telemetryProcessor?.getMagCalibration() ?? null;
}

/**
 * Reset telemetry processor state
 * Call this when starting a new session or after disconnection
 */
export function resetProcessor(): void {
    if (telemetryProcessor) {
        telemetryProcessor.reset();
    }
}

/**
 * Reset IMU state (alias for resetProcessor for backward compatibility)
 */
export function resetIMU(): void {
    resetProcessor();
}

// ===== Telemetry Processing =====

/**
 * Main telemetry handler
 * Processes incoming sensor data and decorates with processed fields
 * @param telemetry - Raw telemetry data from device
 */
export function onTelemetry(telemetry: RawTelemetry): void {
    // Initialize processor if needed (always, even when not recording)
    if (!telemetryProcessor) {
        initProcessor();
    }

    // Track whether we should store this sample (only when recording and not paused)
    const shouldStore = state.recording && !state.paused;

    // Update calibration instance if changed
    if (deps.calibrationInstance && (telemetryProcessor as any).calibration !== deps.calibrationInstance) {
        // telemetryProcessor.setCalibration(deps.calibrationInstance);
    }

    // Process telemetry through the shared pipeline
    // This handles: unit conversion, IMU fusion, gyro bias, calibration, filtering
    const decoratedTelemetry = telemetryProcessor!.process(telemetry) as DecoratedTelemetry;

    // Get orientation for pose estimation
    const orientation = telemetryProcessor!.getQuaternion();
    const euler = telemetryProcessor!.getEulerAngles();

    // Update pose estimation with filtered magnetic field + orientation context
    if (deps.poseState?.enabled && deps.updatePoseEstimation && decoratedTelemetry.filtered_mx !== undefined) {
        deps.updatePoseEstimation({
            magField: {
                x: decoratedTelemetry.filtered_mx,
                y: decoratedTelemetry.filtered_my!,
                z: decoratedTelemetry.filtered_mz!
            },
            orientation: orientation,
            euler: euler,
            sample: decoratedTelemetry
        });
    }

    // Store decorated telemetry (includes raw + processed fields) - skip if paused
    if (shouldStore) {
        state.sessionData.push(decoratedTelemetry as any);
    }

    // Collect samples for calibration buffers during wizard
    // IMPORTANT: Use converted µT values, not raw LSB!
    if (deps.wizard?.active && deps.wizard.phase === 'hold') {
        const currentStep = deps.wizard.steps[deps.wizard.currentStep];
        if (currentStep && deps.calibrationBuffers?.[currentStep.id]) {
            deps.calibrationBuffers[currentStep.id].push({
                mx: decoratedTelemetry.mx_ut!,  // Use µT, not raw LSB
                my: decoratedTelemetry.my_ut!,
                mz: decoratedTelemetry.mz_ut!
            });
        }
    }

    // Update live display
    if (deps.$) {
        updateLiveDisplay(telemetry, decoratedTelemetry);
    }

    // Update sample count (throttled)
    if (state.sessionData.length % 10 === 0 && deps.updateUI) {
        deps.updateUI();
    }
}

/**
 * Update live sensor display
 * @param raw - Raw telemetry
 * @param decorated - Decorated telemetry with processed fields
 */
function updateLiveDisplay(raw: RawTelemetry, decorated: DecoratedTelemetry): void {
    const $ = deps.$;
    if (!$) return;

    // Raw IMU values
    const axEl = $('ax');
    const ayEl = $('ay');
    const azEl = $('az');
    const gxEl = $('gx');
    const gyEl = $('gy');
    const gzEl = $('gz');
    if (axEl) axEl.textContent = String(raw.ax);
    if (ayEl) ayEl.textContent = String(raw.ay);
    if (azEl) azEl.textContent = String(raw.az);
    if (gxEl) gxEl.textContent = String(raw.gx);
    if (gyEl) gyEl.textContent = String(raw.gy);
    if (gzEl) gzEl.textContent = String(raw.gz);

    // Calibrated magnetometer (show calibrated if available, otherwise raw)
    const mxEl = $('mx');
    const myEl = $('my');
    const mzEl = $('mz');
    if (mxEl) mxEl.textContent = (decorated.calibrated_mx ?? raw.mx).toFixed(2);
    if (myEl) myEl.textContent = (decorated.calibrated_my ?? raw.my).toFixed(2);
    if (mzEl) mzEl.textContent = (decorated.calibrated_mz ?? raw.mz).toFixed(2);

    // Residual magnetic field display (finger magnet signals)
    // TelemetryProcessor outputs residual_mx/my/mz (Earth field subtracted)
    const fusedMxEl = $('fused_mx');
    const fusedMyEl = $('fused_my');
    const fusedMzEl = $('fused_mz');
    const residualMagEl = $('residual_magnitude');

    if (decorated.residual_mx !== undefined) {
        if (fusedMxEl) fusedMxEl.textContent = decorated.residual_mx.toFixed(2);
        if (fusedMyEl) fusedMyEl.textContent = decorated.residual_my!.toFixed(2);
        if (fusedMzEl) fusedMzEl.textContent = decorated.residual_mz!.toFixed(2);

        // Display residual magnitude
        const residualMag = decorated.residual_magnitude ?? Math.sqrt(
            decorated.residual_mx ** 2 +
            decorated.residual_my! ** 2 +
            decorated.residual_mz! ** 2
        );
        if (residualMagEl) residualMagEl.textContent = residualMag.toFixed(2) + ' μT';

        // Update 3D magnetic trajectory visualization
        if (deps.updateMagTrajectory) {
            deps.updateMagTrajectory({
                fused_mx: decorated.residual_mx,
                fused_my: decorated.residual_my!,
                fused_mz: decorated.residual_mz!
            });
        }
    } else {
        if (fusedMxEl) fusedMxEl.textContent = '-';
        if (fusedMyEl) fusedMyEl.textContent = '-';
        if (fusedMzEl) fusedMzEl.textContent = '-';
        if (residualMagEl) residualMagEl.textContent = '-';
    }
}

// ===== Accessor Functions =====

/**
 * Get the telemetry processor instance
 * @returns TelemetryProcessor or null
 */
export function getProcessor(): TelemetryProcessor | null {
    return telemetryProcessor;
}

/**
 * Get current orientation as Euler angles
 * @returns {roll, pitch, yaw} in degrees or null
 */
export function getEulerAngles(): EulerAngles | null {
    return telemetryProcessor?.getEulerAngles() ?? null;
}

/**
 * Get current orientation as quaternion
 * @returns {w, x, y, z} or null
 */
export function getQuaternion(): Quaternion | null {
    return telemetryProcessor?.getQuaternion() ?? null;
}

/**
 * Check if gyroscope bias is calibrated
 * @returns true if calibrated
 */
export function isGyroBiasCalibrated(): boolean {
    return telemetryProcessor?.getGyroBiasState().calibrated ?? false;
}

/**
 * Get motion state
 * @returns {isMoving, accelStd, gyroStd}
 */
export function getMotionState(): MotionState {
    return telemetryProcessor?.getMotionState() ?? { isMoving: false, accelStd: 0, gyroStd: 0 };
}

// ===== Default Export =====

export default {
    setDependencies,
    initProcessor,
    getMagCalibration,
    resetProcessor,
    resetIMU,
    onTelemetry,
    getProcessor,
    getEulerAngles,
    getQuaternion,
    isGyroBiasCalibrated,
    getMotionState
};
