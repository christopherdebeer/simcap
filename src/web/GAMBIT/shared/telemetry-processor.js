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
     * @param {Object} [options.calibration] - EnvironmentalCalibration instance (optional, for hard/soft iron)
     * @param {Function} [options.onProcessed] - Callback for processed telemetry
     * @param {Function} [options.onOrientationUpdate] - Callback for orientation updates
     * @param {Function} [options.onGyroBiasCalibrated] - Callback when gyro bias is calibrated
     * @param {boolean} [options.useMagnetometer=true] - Enable 9-DOF fusion with magnetometer
     * @param {number} [options.magTrust=0.5] - Magnetometer trust factor (0-1)
     * @param {boolean} [options.magCalibrationDebug=false] - Enable debug logging for mag calibration
     */
    constructor(options = {}) {
        this.options = options;

        // External calibration instance (optional, for hard/soft iron from wizard)
        this.calibration = options.calibration || null;

        // Unified magnetometer calibration (live Earth field estimation)
        this.magCalibration = new UnifiedMagCalibration({
            windowSize: 200,  // Optimal based on investigation
            minSamples: 50,   // ~1 second at 50Hz
            debug: options.magCalibrationDebug || false
        });
        
        // Magnet detector (detects finger magnet presence from residual magnitude)
        this.magnetDetector = createMagnetDetector({
            onStatusChange: options.onMagnetStatusChange || null
        });

        // Create signal processing components
        this.imuFusion = createMadgwickAHRS();
        this.magFilter = createKalmanFilter3D();
        this.motionDetector = createMotionDetector();

        // Gyroscope bias calibration state
        this.gyroBiasState = createGyroBiasState();

        // IMU initialization state
        this.imuInitialized = false;

        // 9-DOF magnetometer fusion configuration
        this.useMagnetometer = options.useMagnetometer !== false; // Default: enabled
        this.magTrust = options.magTrust ?? 0.5; // Default: moderate trust
        this.geomagneticRef = null;

        // Initialize magnetometer trust on AHRS
        this.imuFusion.setMagTrust(this.magTrust);

        // Initialize geomagnetic reference (async, uses default until browser location available)
        this._initGeomagneticReference();

        // Timing
        this.lastTimestamp = null;

        // Callbacks
        this.onProcessed = options.onProcessed || null;
        this.onOrientationUpdate = options.onOrientationUpdate || null;
        this.onGyroBiasCalibrated = options.onGyroBiasCalibrated || null;

        // Debug logging flags (to avoid spam)
        this._loggedCalibrationMissing = false;
        this._loggedMagFusion = false;
        this._loggedMagnetDetection = false;
    }
    
    /**
     * Get magnet detector instance
     * @returns {MagnetDetector}
     */
    getMagnetDetector() {
        return this.magnetDetector;
    }
    
    /**
     * Get current magnet detection state
     * @returns {Object} Detection state with status, confidence, avgResidual
     */
    getMagnetState() {
        return this.magnetDetector.getState();
    }
    
    /**
     * Get mag calibration instance
     * @returns {UnifiedMagCalibration}
     */
    getMagCalibration() {
        return this.magCalibration;
    }

    /**
     * Reset mag calibration
     */
    resetMagCalibration() {
        this.magCalibration.reset();
        console.log('[TelemetryProcessor] Mag calibration reset');
    }

    /**
     * Initialize geomagnetic reference from location
     * Uses browser geolocation if available, falls back to default
     */
    async _initGeomagneticReference() {
        // Start with default location immediately
        const defaultLoc = getDefaultLocation();
        this._setGeomagneticRef(defaultLoc);
        console.log('[TelemetryProcessor] Using default geomagnetic reference:', defaultLoc.city);

        // Try to get browser location (async, updates if successful)
        try {
            const browserLoc = await getBrowserLocation({ timeout: 5000 });
            this._setGeomagneticRef(browserLoc);
            console.log('[TelemetryProcessor] Updated geomagnetic reference from browser location:', browserLoc.city);
        } catch (e) {
            console.debug('[TelemetryProcessor] Browser location unavailable, using default:', e.message);
        }
    }

    /**
     * Set geomagnetic reference on AHRS
     * @param {Object} location - Location with horizontal, vertical, declination fields
     */
    _setGeomagneticRef(location) {
        if (location) {
            this.geomagneticRef = {
                horizontal: location.horizontal,
                vertical: location.vertical,
                declination: location.declination
            };
            this.imuFusion.setGeomagneticReference(this.geomagneticRef);
        }
    }

    /**
     * Set magnetometer trust factor
     * @param {number} trust - 0.0 (ignore mag) to 1.0 (full trust)
     */
    setMagTrust(trust) {
        this.magTrust = Math.max(0, Math.min(1, trust));
        this.imuFusion.setMagTrust(this.magTrust);
    }

    /**
     * Enable/disable magnetometer fusion
     * @param {boolean} enabled
     */
    setMagnetometerEnabled(enabled) {
        this.useMagnetometer = enabled;
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

        // Update orientation estimate
        if (this.imuInitialized) {
            // Check if we should use 9-DOF magnetometer fusion
            const magDataValid = mx_ut !== 0 || my_ut !== 0 || mz_ut !== 0;

            if (this.useMagnetometer && magDataValid && this.geomagneticRef) {
                // 9-DOF fusion with magnetometer for absolute yaw reference
                // Pass hard iron offset from calibration if available
                if (this.calibration?.hardIronCalibrated && this.calibration.hardIronOffset) {
                    this.imuFusion.setHardIronOffset(this.calibration.hardIronOffset);
                }

                // Use updateWithMag for 9-DOF fusion
                this.imuFusion.updateWithMag(
                    ax_g, ay_g, az_g,
                    gx_dps, gy_dps, gz_dps,
                    mx_ut, my_ut, mz_ut,
                    dt, true, true // gyroInDegrees=true, applyHardIron=true
                );

                if (!this._loggedMagFusion) {
                    console.log('[TelemetryProcessor] Using 9-DOF fusion with magnetometer (trust:', this.magTrust, ')');
                    this._loggedMagFusion = true;
                }
            } else {
                // 6-DOF fusion (gyro + accel only)
                this.imuFusion.update(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, dt, true);
            }
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

        // Add magnetometer residual from AHRS (for finger magnet sensing prep)
        if (this.useMagnetometer && this.imuInitialized) {
            const magResidual = this.imuFusion.getMagResidual();
            if (magResidual) {
                decorated.ahrs_mag_residual_x = magResidual.x;
                decorated.ahrs_mag_residual_y = magResidual.y;
                decorated.ahrs_mag_residual_z = magResidual.z;
                decorated.ahrs_mag_residual_magnitude = this.imuFusion.getMagResidualMagnitude();
            }
        }

        // Notify orientation update
        if (this.onOrientationUpdate && euler) {
            this.onOrientationUpdate(euler, orientation);
        }
        
        // ===== Step 5: Magnetometer Calibration (Unified) =====
        // Use UnifiedMagCalibration for real-time Earth field estimation and subtraction
        // Optionally apply hard iron from stored calibration if available

        // Apply hard iron offset from stored calibration if available
        if (this.calibration?.hardIronCalibrated && this.calibration.hardIronOffset) {
            // Set hard iron on unified calibration (only sets once if unchanged)
            const currentOffset = this.magCalibration.getHardIronOffset();
            const storedOffset = this.calibration.hardIronOffset;
            if (currentOffset.x !== storedOffset.x ||
                currentOffset.y !== storedOffset.y ||
                currentOffset.z !== storedOffset.z) {
                this.magCalibration.setHardIronOffset(storedOffset);
            }
        }

        // Update unified calibration with current sample
        if (orientation) {
            this.magCalibration.update(mx_ut, my_ut, mz_ut, orientation);

            // Add calibration metrics to telemetry
            const calState = this.magCalibration.getState();
            decorated.mag_cal_ready = calState.ready;
            decorated.mag_cal_confidence = calState.confidence;
            decorated.mag_cal_mean_residual = calState.meanResidual;
            decorated.mag_cal_earth_magnitude = calState.earthMagnitude;

            // Compute residual if Earth field has been estimated
            if (calState.ready) {
                const residual = this.magCalibration.getResidual(mx_ut, my_ut, mz_ut, orientation);
                if (residual) {
                    decorated.residual_mx = residual.x;
                    decorated.residual_my = residual.y;
                    decorated.residual_mz = residual.z;
                    decorated.residual_magnitude = residual.magnitude;

                    // Update magnet detector with residual
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
            // No orientation - fall back to raw magnitude
            if (!this._loggedCalibrationMissing) {
                console.debug('[TelemetryProcessor] Orientation not available - using raw mag values');
                this._loggedCalibrationMissing = true;
            }
            decorated.residual_magnitude = Math.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2);
        }
        
        // ===== Step 6: Kalman Filtering =====
        // Apply Kalman filter to residual (Earth-subtracted) values for noise reduction
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
        this._loggedMagFusion = false;
        this._loggedMagnetDetection = false;

        // Reset mag calibration
        this.magCalibration.reset();

        // Reset magnet detector
        this.magnetDetector.reset();

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
