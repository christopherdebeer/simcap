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

// Module dependencies (initialized by setDependencies)
let deps = {
    calibrationInstance: null,
    wizard: null,
    calibrationBuffers: null,
    poseState: null,
    updatePoseEstimation: null,
    updateMagTrajectory: null,
    updateUI: null,
    $: null
};

// Shared telemetry processor instance
let telemetryProcessor = null;

/**
 * Set module dependencies
 * @param {Object} dependencies - Required dependencies
 */
export function setDependencies(dependencies) {
    deps = { ...deps, ...dependencies };
    
    // Update calibration in processor if it exists
    if (telemetryProcessor && deps.calibrationInstance) {
        // telemetryProcessor.setCalibration(deps.calibrationInstance);
    }
}

/**
 * Initialize the telemetry processor
 * Call this after dependencies are set
 */
export function initProcessor() {
    telemetryProcessor = new TelemetryProcessor({
        calibration: deps.calibrationInstance,
        magCalibrationDebug: true, // Enable debug logging for mag calibration
        onOrientationUpdate: (euler, quaternion) => {
            // Update Three.js hand skeleton if available
            if (euler) {
                const threeSkeleton = typeof deps.threeHandSkeleton === 'function' ? deps.threeHandSkeleton() : deps.threeHandSkeleton;
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
 * @returns {UnifiedMagCalibration|null}
 */
export function getMagCalibration() {
    return telemetryProcessor?.getMagCalibration() ?? null;
}

/**
 * Reset telemetry processor state
 * Call this when starting a new session or after disconnection
 */
export function resetProcessor() {
    if (telemetryProcessor) {
        telemetryProcessor.reset();
    }
}

/**
 * Reset IMU state (alias for resetProcessor for backward compatibility)
 */
export function resetIMU() {
    resetProcessor();
}

/**
 * Main telemetry handler
 * Processes incoming sensor data and decorates with processed fields
 * @param {Object} telemetry - Raw telemetry data from device
 */
export function onTelemetry(telemetry) {
    // Initialize processor if needed (always, even when not recording)
    if (!telemetryProcessor) {
        initProcessor();
    }

    // Track whether we should store this sample (only when recording and not paused)
    const shouldStore = state.recording && !state.paused;
    
    // Update calibration instance if changed
    if (deps.calibrationInstance && telemetryProcessor.calibration !== deps.calibrationInstance) {
        // telemetryProcessor.setCalibration(deps.calibrationInstance);
    }

    // Process telemetry through the shared pipeline
    // This handles: unit conversion, IMU fusion, gyro bias, calibration, filtering
    const decoratedTelemetry = telemetryProcessor.process(telemetry);

    // Get orientation for pose estimation
    const orientation = telemetryProcessor.getQuaternion();
    const euler = telemetryProcessor.getEulerAngles();

    // Update pose estimation with filtered magnetic field + orientation context
    if (deps.poseState?.enabled && deps.updatePoseEstimation && decoratedTelemetry.filtered_mx !== undefined) {
        deps.updatePoseEstimation({
            magField: {
                x: decoratedTelemetry.filtered_mx,
                y: decoratedTelemetry.filtered_my,
                z: decoratedTelemetry.filtered_mz
            },
            orientation: orientation,
            euler: euler,
            sample: decoratedTelemetry
        });
    }

    // Store decorated telemetry (includes raw + processed fields) - skip if paused
    if (shouldStore) {
        state.sessionData.push(decoratedTelemetry);
    }

    // Collect samples for calibration buffers during wizard
    // IMPORTANT: Use converted µT values, not raw LSB!
    if (deps.wizard?.active && deps.wizard.phase === 'hold') {
        const currentStep = deps.wizard.steps[deps.wizard.currentStep];
        if (currentStep && deps.calibrationBuffers?.[currentStep.id]) {
            deps.calibrationBuffers[currentStep.id].push({
                mx: decoratedTelemetry.mx_ut,  // Use µT, not raw LSB
                my: decoratedTelemetry.my_ut,
                mz: decoratedTelemetry.mz_ut
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
 * @param {Object} raw - Raw telemetry
 * @param {Object} decorated - Decorated telemetry with processed fields
 */
function updateLiveDisplay(raw, decorated) {
    const $ = deps.$;
    
    // Raw IMU values
    $('ax').textContent = raw.ax;
    $('ay').textContent = raw.ay;
    $('az').textContent = raw.az;
    $('gx').textContent = raw.gx;
    $('gy').textContent = raw.gy;
    $('gz').textContent = raw.gz;
    
    // Calibrated magnetometer (show calibrated if available, otherwise raw)
    $('mx').textContent = (decorated.calibrated_mx ?? raw.mx).toFixed(2);
    $('my').textContent = (decorated.calibrated_my ?? raw.my).toFixed(2);
    $('mz').textContent = (decorated.calibrated_mz ?? raw.mz).toFixed(2);

    // Residual magnetic field display (finger magnet signals)
    // TelemetryProcessor outputs residual_mx/my/mz (Earth field subtracted)
    if (decorated.residual_mx !== undefined) {
        $('fused_mx').textContent = decorated.residual_mx.toFixed(2);
        $('fused_my').textContent = decorated.residual_my.toFixed(2);
        $('fused_mz').textContent = decorated.residual_mz.toFixed(2);

        // Display residual magnitude
        const residualMag = decorated.residual_magnitude ?? Math.sqrt(
            decorated.residual_mx ** 2 +
            decorated.residual_my ** 2 +
            decorated.residual_mz ** 2
        );
        $('residual_magnitude').textContent = residualMag.toFixed(2) + ' μT';

        // Update 3D magnetic trajectory visualization
        if (deps.updateMagTrajectory) {
            deps.updateMagTrajectory({
                fused_mx: decorated.residual_mx,
                fused_my: decorated.residual_my,
                fused_mz: decorated.residual_mz
            });
        }
    } else {
        $('fused_mx').textContent = '-';
        $('fused_my').textContent = '-';
        $('fused_mz').textContent = '-';
        $('residual_magnitude').textContent = '-';
    }
}

/**
 * Get the telemetry processor instance
 * @returns {TelemetryProcessor|null}
 */
export function getProcessor() {
    return telemetryProcessor;
}

/**
 * Get current orientation as Euler angles
 * @returns {Object|null} {roll, pitch, yaw} in degrees
 */
export function getEulerAngles() {
    return telemetryProcessor?.getEulerAngles() ?? null;
}

/**
 * Get current orientation as quaternion
 * @returns {Object|null} {w, x, y, z}
 */
export function getQuaternion() {
    return telemetryProcessor?.getQuaternion() ?? null;
}

/**
 * Check if gyroscope bias is calibrated
 * @returns {boolean}
 */
export function isGyroBiasCalibrated() {
    return telemetryProcessor?.getGyroBiasState().calibrated ?? false;
}

/**
 * Get motion state
 * @returns {Object} {isMoving, accelStd, gyroStd}
 */
export function getMotionState() {
    return telemetryProcessor?.getMotionState() ?? { isMoving: false, accelStd: 0, gyroStd: 0 };
}
