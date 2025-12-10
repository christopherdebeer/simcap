/**
 * Telemetry Data Handler
 * Processes incoming sensor data with calibration, filtering, and pose estimation
 */

import { state } from './state.js';

// Module dependencies (initialized by setDependencies)
let deps = {
    calibrationInstance: null,
    magFilter: null,
    imuFusion: null,
    wizard: null,
    calibrationBuffers: null,
    poseState: null,
    updatePoseEstimation: null,
    updateUI: null,
    $: null
};

let lastTelemetryTime = null;
let imuInitialized = false;

/**
 * Set module dependencies
 * @param {Object} dependencies - Required dependencies
 */
export function setDependencies(dependencies) {
    deps = { ...deps, ...dependencies };
}

/**
 * Reset IMU state
 */
export function resetIMU() {
    imuInitialized = false;
    lastTelemetryTime = null;
}

/**
 * Main telemetry handler
 * Processes incoming sensor data and decorates with processed fields
 * @param {Object} telemetry - Raw telemetry data from device
 */
export function onTelemetry(telemetry) {
    if (!state.recording) return;

    // IMPORTANT: Preserve raw data, only DECORATE with processed fields
    // Create a decorated copy of telemetry with additional processed fields
    const decoratedTelemetry = {...telemetry};

    // Calculate time step for IMU fusion
    const now = performance.now();
    const dt = lastTelemetryTime ? (now - lastTelemetryTime) / 1000 : 0.02; // Default 50Hz
    lastTelemetryTime = now;

    // Update IMU sensor fusion to estimate device orientation
    // Uses accelerometer + gyroscope (NOT magnetometer - it's our measurement target)
    if (deps.imuFusion && telemetry.ax !== undefined && telemetry.gx !== undefined) {
        if (!imuInitialized && Math.abs(telemetry.ax) + Math.abs(telemetry.ay) + Math.abs(telemetry.az) > 0.5) {
            // Initialize orientation from accelerometer (assumes stationary)
            deps.imuFusion.initFromAccelerometer(telemetry.ax, telemetry.ay, telemetry.az);
            imuInitialized = true;
        }
        // Update orientation estimate
        deps.imuFusion.update(
            telemetry.ax, telemetry.ay, telemetry.az,  // Accelerometer
            telemetry.gx, telemetry.gy, telemetry.gz,  // Gyroscope
            dt,                                         // Time step
            true                                        // Gyro is in deg/s
        );
    }

    // Get current orientation for Earth field subtraction
    const orientation = (deps.imuFusion && imuInitialized) ? deps.imuFusion.getQuaternion() : null;
    const euler = (deps.imuFusion && imuInitialized) ? deps.imuFusion.getEulerAngles() : null;

    // Add orientation to telemetry
    if (orientation) {
        decoratedTelemetry.orientation_w = orientation.w;
        decoratedTelemetry.orientation_x = orientation.x;
        decoratedTelemetry.orientation_y = orientation.y;
        decoratedTelemetry.orientation_z = orientation.z;
        decoratedTelemetry.euler_roll = euler.roll;
        decoratedTelemetry.euler_pitch = euler.pitch;
        decoratedTelemetry.euler_yaw = euler.yaw;
    }

    // Apply calibration correction (adds calibrated_ fields - iron correction only)
    if (deps.calibrationInstance &&
        deps.calibrationInstance.hardIronCalibrated &&
        deps.calibrationInstance.softIronCalibrated) {
        try {
            // Iron correction only (no Earth field subtraction yet)
            const ironCorrected = deps.calibrationInstance.correctIronOnly({
                x: telemetry.mx,
                y: telemetry.my,
                z: telemetry.mz
            });
            decoratedTelemetry.calibrated_mx = ironCorrected.x;
            decoratedTelemetry.calibrated_my = ironCorrected.y;
            decoratedTelemetry.calibrated_mz = ironCorrected.z;

            // Full correction with Earth field subtraction (requires orientation)
            if (deps.calibrationInstance.earthFieldCalibrated && orientation) {
                // Create Quaternion object for calibration.correct()
                const quatOrientation = new Quaternion(
                    orientation.w, orientation.x, orientation.y, orientation.z
                );
                const fused = deps.calibrationInstance.correct(
                    { x: telemetry.mx, y: telemetry.my, z: telemetry.mz },
                    quatOrientation
                );
                decoratedTelemetry.fused_mx = fused.x;
                decoratedTelemetry.fused_my = fused.y;
                decoratedTelemetry.fused_mz = fused.z;
            }
        } catch (e) {
            // Calibration failed, skip decoration
            console.debug('[Calibration] Correction failed:', e.message);
        }
    }

    // Apply Kalman filtering (adds filtered_ fields)
    // Use best available source: fused > calibrated > raw
    if (deps.magFilter) {
        try {
            const magInput = {
                x: decoratedTelemetry.fused_mx || decoratedTelemetry.calibrated_mx || telemetry.mx,
                y: decoratedTelemetry.fused_my || decoratedTelemetry.calibrated_my || telemetry.my,
                z: decoratedTelemetry.fused_mz || decoratedTelemetry.calibrated_mz || telemetry.mz
            };
            const filteredMag = deps.magFilter.update(magInput);
            decoratedTelemetry.filtered_mx = filteredMag.x;
            decoratedTelemetry.filtered_my = filteredMag.y;
            decoratedTelemetry.filtered_mz = filteredMag.z;

            // Update pose estimation with filtered magnetic field
            if (deps.poseState?.enabled && deps.updatePoseEstimation) {
                deps.updatePoseEstimation({
                    x: filteredMag.x,
                    y: filteredMag.y,
                    z: filteredMag.z
                });
            }
        } catch (e) {
            // Filtering failed, skip decoration
        }
    }

    // Store decorated telemetry (includes raw + processed fields)
    state.sessionData.push(decoratedTelemetry);

    // Collect samples for calibration buffers during wizard
    if (deps.wizard?.active && deps.wizard.phase === 'hold') {
        const currentStep = deps.wizard.steps[deps.wizard.currentStep];
        if (currentStep && deps.calibrationBuffers?.[currentStep.id]) {
            deps.calibrationBuffers[currentStep.id].push({
                mx: telemetry.mx,
                my: telemetry.my,
                mz: telemetry.mz
            });
        }
    }

    // Update live display (show calibrated values if available, otherwise raw)
    if (deps.$) {
        const $ = deps.$;
        $('ax').textContent = telemetry.ax;
        $('ay').textContent = telemetry.ay;
        $('az').textContent = telemetry.az;
        $('gx').textContent = telemetry.gx;
        $('gy').textContent = telemetry.gy;
        $('gz').textContent = telemetry.gz;
        $('mx').textContent = (decoratedTelemetry.calibrated_mx || telemetry.mx).toFixed(2);
        $('my').textContent = (decoratedTelemetry.calibrated_my || telemetry.my).toFixed(2);
        $('mz').textContent = (decoratedTelemetry.calibrated_mz || telemetry.mz).toFixed(2);
    }

    // Update sample count (throttled)
    if (state.sessionData.length % 10 === 0 && deps.updateUI) {
        deps.updateUI();
    }
}
