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

import {
    magLsbToMicroTesla,
    getSensorUnitMetadata
} from './sensor-units.js';

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
    /**
     * Create a TelemetryProcessor instance
     * @param {Object} options - Configuration options
     * @param {Object} [options.calibration] - EnvironmentalCalibration instance
     * @param {Function} [options.onProcessed] - Callback for processed telemetry
     * @param {Function} [options.onOrientationUpdate] - Callback for orientation updates
     * @param {Function} [options.onGyroBiasCalibrated] - Callback when gyro bias is calibrated
     */
    constructor(options = {}) {
        this.options = options;
        
        // Calibration instance (external, for magnetometer correction)
        this.calibration = options.calibration || null;
        
        // Create signal processing components
        this.imuFusion = createMadgwickAHRS();
        this.magFilter = createKalmanFilter3D();
        this.motionDetector = createMotionDetector();
        
        // Gyroscope bias calibration state
        this.gyroBiasState = createGyroBiasState();
        
        // IMU initialization state
        this.imuInitialized = false;
        
        // Timing
        this.lastTimestamp = null;
        
        // Callbacks
        this.onProcessed = options.onProcessed || null;
        this.onOrientationUpdate = options.onOrientationUpdate || null;
        this.onGyroBiasCalibrated = options.onGyroBiasCalibrated || null;
        
        // Debug logging flags (to avoid spam)
        this._loggedCalibrationMissing = false;
        this._loggedOrientationMissing = false;
        this._loggedEarthCalibrationMissing = false;
    }
    
    /**
     * Set the calibration instance
     * @param {Object} calibration - EnvironmentalCalibration instance
     */
    setCalibration(calibration) {
        this.calibration = calibration;
        // Reset logging flags when calibration changes
        this._loggedCalibrationMissing = false;
        this._loggedOrientationMissing = false;
        this._loggedEarthCalibrationMissing = false;
    }
    
    /**
     * Process a telemetry sample through the full pipeline
     * 
     * @param {Object} raw - Raw telemetry from device
     * @param {number} raw.ax - Accelerometer X (LSB)
     * @param {number} raw.ay - Accelerometer Y (LSB)
     * @param {number} raw.az - Accelerometer Z (LSB)
     * @param {number} raw.gx - Gyroscope X (LSB)
     * @param {number} raw.gy - Gyroscope Y (LSB)
     * @param {number} raw.gz - Gyroscope Z (LSB)
     * @param {number} raw.mx - Magnetometer X (LSB, raw sensor units)
     * @param {number} raw.my - Magnetometer Y (LSB, raw sensor units)
     * @param {number} raw.mz - Magnetometer Z (LSB, raw sensor units)
     * @returns {Object} Decorated telemetry with processed fields
     */
    process(raw) {
        // IMPORTANT: Preserve raw data, only DECORATE with processed fields
        const decorated = { ...raw };
        
        // Calculate time step
        const now = performance.now();
        const dt = this.lastTimestamp ? (now - this.lastTimestamp) / 1000 : 0.02;
        this.lastTimestamp = now;
        
        // Store dt for external use
        decorated.dt = dt;
        
        // ===== Step 1: Unit Conversion =====
        // Convert accelerometer from LSB to g's
        const ax_g = accelLsbToG(raw.ax || 0);
        const ay_g = accelLsbToG(raw.ay || 0);
        const az_g = accelLsbToG(raw.az || 0);
        
        // Convert gyroscope from LSB to deg/s
        const gx_dps = gyroLsbToDps(raw.gx || 0);
        const gy_dps = gyroLsbToDps(raw.gy || 0);
        const gz_dps = gyroLsbToDps(raw.gz || 0);
        
        // Store converted values (DECORATION - raw values preserved)
        decorated.ax_g = ax_g;
        decorated.ay_g = ay_g;
        decorated.az_g = az_g;
        decorated.gx_dps = gx_dps;
        decorated.gy_dps = gy_dps;
        decorated.gz_dps = gz_dps;

        // Convert magnetometer from LSB to µT (CRITICAL FIX)
        const mx_ut = magLsbToMicroTesla(raw.mx || 0);
        const my_ut = magLsbToMicroTesla(raw.my || 0);
        const mz_ut = magLsbToMicroTesla(raw.mz || 0);

        // Store converted magnetometer values (DECORATION - raw preserved)
        decorated.mx_ut = mx_ut;
        decorated.my_ut = my_ut;
        decorated.mz_ut = mz_ut;
        
        // ===== Step 2: Motion Detection =====
        // Use RAW LSB values for motion detection (thresholds are in LSB)
        const motionState = this.motionDetector.update(
            raw.ax || 0, raw.ay || 0, raw.az || 0,
            raw.gx || 0, raw.gy || 0, raw.gz || 0
        );
        decorated.isMoving = motionState.isMoving;
        decorated.accelStd = motionState.accelStd;
        decorated.gyroStd = motionState.gyroStd;
        
        // ===== Step 3: Gyroscope Bias Calibration =====
        // When device is stationary, estimate and correct gyro bias
        if (!motionState.isMoving) {
            this.gyroBiasState.stationaryCount++;
            
            if (this.gyroBiasState.stationaryCount > STATIONARY_SAMPLES_FOR_CALIBRATION) {
                // Update gyro bias estimate (pass deg/s values)
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
        // Initialize from accelerometer if not yet initialized
        if (!this.imuInitialized) {
            const accelMag = Math.abs(raw.ax || 0) + Math.abs(raw.ay || 0) + Math.abs(raw.az || 0);
            if (accelMag > 0.5) {
                this.imuFusion.initFromAccelerometer(raw.ax, raw.ay, raw.az);
                this.imuInitialized = true;
                console.log('[TelemetryProcessor] IMU initialized from accelerometer');
            }
        }
        
        // Update orientation estimate (pass g's and deg/s)
        if (this.imuInitialized) {
            this.imuFusion.update(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, dt, true);
        }
        
        // Get current orientation
        const orientation = this.imuInitialized ? this.imuFusion.getQuaternion() : null;
        const euler = this.imuInitialized ? this.imuFusion.getEulerAngles() : null;
        
        // Add orientation to decorated telemetry
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
        
        // Notify orientation update
        if (this.onOrientationUpdate && euler) {
            this.onOrientationUpdate(euler, orientation);
        }
        
        // ===== Step 5: Magnetometer Calibration =====
        if (this.calibration &&
            this.calibration.hardIronCalibrated &&
            this.calibration.softIronCalibrated) {
            
            try {
                // Iron correction only (no Earth field subtraction yet)
                // IMPORTANT: Use converted µT values, not raw LSB
                const ironCorrected = this.calibration.correctIronOnly({
                    x: mx_ut,
                    y: my_ut,
                    z: mz_ut
                });
                decorated.calibrated_mx = ironCorrected.x;
                decorated.calibrated_my = ironCorrected.y;
                decorated.calibrated_mz = ironCorrected.z;
                
                // Full correction with Earth field subtraction (requires orientation)
                if (this.calibration.earthFieldCalibrated && orientation) {
                    // Create Quaternion object for calibration.correct()
                    // Quaternion is expected to be globally available from calibration.js
                    const quatOrientation = new Quaternion(
                        orientation.w, orientation.x, orientation.y, orientation.z
                    );
                    // IMPORTANT: Use converted µT values, not raw LSB
                    const fused = this.calibration.correct(
                        { x: mx_ut, y: my_ut, z: mz_ut },
                        quatOrientation
                    );
                    decorated.fused_mx = fused.x;
                    decorated.fused_my = fused.y;
                    decorated.fused_mz = fused.z;
                    
                    // Calculate residual magnitude (useful for finger magnet detection)
                    decorated.residual_magnitude = Math.sqrt(
                        fused.x ** 2 + fused.y ** 2 + fused.z ** 2
                    );
                } else if (!this.calibration.earthFieldCalibrated) {
                    if (!this._loggedEarthCalibrationMissing) {
                        console.debug('[TelemetryProcessor] Earth field not calibrated - skipping fused fields');
                        this._loggedEarthCalibrationMissing = true;
                    }
                } else if (!orientation) {
                    if (!this._loggedOrientationMissing) {
                        console.debug('[TelemetryProcessor] Orientation not available - skipping fused fields');
                        this._loggedOrientationMissing = true;
                    }
                }
            } catch (e) {
                console.error('[TelemetryProcessor] Calibration correction failed:', e);
            }
        } else {
            if (!this._loggedCalibrationMissing) {
                console.debug('[TelemetryProcessor] Calibration status:', {
                    hasInstance: !!this.calibration,
                    hardIronCalibrated: this.calibration?.hardIronCalibrated,
                    softIronCalibrated: this.calibration?.softIronCalibrated
                });
                this._loggedCalibrationMissing = true;
            }
        }
        
        // ===== Step 6: Kalman Filtering =====
        // Use best available source: fused > calibrated > converted µT
        // IMPORTANT: Use µT values, not raw LSB
        try {
            const magInput = {
                x: decorated.fused_mx ?? decorated.calibrated_mx ?? mx_ut,
                y: decorated.fused_my ?? decorated.calibrated_my ?? my_ut,
                z: decorated.fused_mz ?? decorated.calibrated_mz ?? mz_ut
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
     * Call this when starting a new session or after disconnection
     */
    reset() {
        this.imuFusion.reset();
        this.magFilter.reset();
        this.motionDetector.reset();
        this.gyroBiasState = createGyroBiasState();
        this.imuInitialized = false;
        this.lastTimestamp = null;
        
        // Reset logging flags
        this._loggedCalibrationMissing = false;
        this._loggedOrientationMissing = false;
        this._loggedEarthCalibrationMissing = false;
        
        console.log('[TelemetryProcessor] Reset complete');
    }
    
    /**
     * Get current orientation as Euler angles
     * @returns {Object|null} {roll, pitch, yaw} in degrees, or null if not initialized
     */
    getEulerAngles() {
        return this.imuInitialized ? this.imuFusion.getEulerAngles() : null;
    }
    
    /**
     * Get current orientation as quaternion
     * @returns {Object|null} {w, x, y, z} quaternion, or null if not initialized
     */
    getQuaternion() {
        return this.imuInitialized ? this.imuFusion.getQuaternion() : null;
    }
    
    /**
     * Get gyroscope bias calibration state
     * @returns {Object} {calibrated, stationaryCount}
     */
    getGyroBiasState() {
        return { ...this.gyroBiasState };
    }
    
    /**
     * Check if IMU is initialized
     * @returns {boolean}
     */
    isIMUInitialized() {
        return this.imuInitialized;
    }
    
    /**
     * Get motion state
     * @returns {Object} {isMoving, accelStd, gyroStd}
     */
    getMotionState() {
        return this.motionDetector.getState();
    }
}

/**
 * Create a TelemetryProcessor instance with standard configuration
 * @param {Object} options - Configuration options
 * @returns {TelemetryProcessor}
 */
export function createTelemetryProcessor(options = {}) {
    return new TelemetryProcessor(options);
}

// ===== Default Export =====

export default {
    TelemetryProcessor,
    createTelemetryProcessor
};
