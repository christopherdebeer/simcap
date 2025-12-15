/**
 * Sensor Configuration and Factory Functions
 * 
 * Provides standardized sensor parameters and factory functions for
 * creating consistently configured signal processing components.
 * 
 * Reference: index.html sensor configuration (canonical implementation)
 * 
 * @module shared/sensor-config
 */

// ===== Sensor Scale Constants =====
// Puck.js returns RAW LSB values from the LSM6DS3 sensor
// These constants convert to physical units for AHRS filter

/**
 * Accelerometer scale factor: LSB per g
 * LSM6DS3 at ±2g range, 16-bit resolution
 * @see https://learn.adafruit.com/adafruit-sensorlab-gyroscope-calibration
 */
export const ACCEL_SCALE = 8192;

/**
 * Gyroscope scale factor: LSB per deg/s
 * LSM6DS3 at 245dps range
 */
export const GYRO_SCALE = 114.28;

/**
 * Default sample frequency for GAMBIT firmware
 * Matches firmware accelOn rate (26Hz)
 */
export const DEFAULT_SAMPLE_FREQ = 26;

/**
 * Magnetometer scale factor: LSB to μT
 * LIS3MDL: 6842 LSB/gauss @ ±4 gauss, 1 gauss = 100 μT
 */
export const MAG_SCALE_LSB_TO_UT = 100 / 6842;

// ===== Unit Conversion Functions =====

/**
 * Convert accelerometer reading from LSB to g's
 * @param {number} lsb - Raw LSB value
 * @returns {number} Value in g's
 */
export function accelLsbToG(lsb) {
    return lsb / ACCEL_SCALE;
}

/**
 * Convert gyroscope reading from LSB to deg/s
 * @param {number} lsb - Raw LSB value
 * @returns {number} Value in deg/s
 */
export function gyroLsbToDps(lsb) {
    return lsb / GYRO_SCALE;
}

/**
 * Convert accelerometer vector from LSB to g's
 * @param {Object} raw - {ax, ay, az} in LSB
 * @returns {Object} {ax, ay, az} in g's
 */
export function convertAccelToG(raw) {
    return {
        ax: accelLsbToG(raw.ax || 0),
        ay: accelLsbToG(raw.ay || 0),
        az: accelLsbToG(raw.az || 0)
    };
}

/**
 * Convert gyroscope vector from LSB to deg/s
 * @param {Object} raw - {gx, gy, gz} in LSB
 * @returns {Object} {gx, gy, gz} in deg/s
 */
export function convertGyroToDps(raw) {
    return {
        gx: gyroLsbToDps(raw.gx || 0),
        gy: gyroLsbToDps(raw.gy || 0),
        gz: gyroLsbToDps(raw.gz || 0)
    };
}

// ===== Factory Functions =====

/**
 * Create a MadgwickAHRS instance with standard GAMBIT configuration
 * 
 * NOTE: With stable sensor data (Puck.accelOn fix in firmware v0.3.6),
 * we use standard AHRS parameters instead of jitter-compensating values.
 * 
 * @param {Object} options - Override options
 * @param {number} [options.sampleFreq=26] - Sample frequency in Hz
 * @param {number} [options.beta=0.05] - Filter gain (higher = faster convergence, more noise)
 * @returns {MadgwickAHRS} Configured AHRS instance
 */
export function createMadgwickAHRS(options = {}) {
    const config = {
        sampleFreq: options.sampleFreq || DEFAULT_SAMPLE_FREQ,
        beta: options.beta || 0.05  // Standard Madgwick gain
    };
    
    // MadgwickAHRS is expected to be globally available from filters.js
    if (typeof MadgwickAHRS === 'undefined') {
        throw new Error('MadgwickAHRS not found. Ensure filters.js is loaded.');
    }
    
    return new MadgwickAHRS(config);
}

/**
 * Create a KalmanFilter3D instance with standard GAMBIT configuration
 * 
 * @param {Object} options - Override options
 * @param {number} [options.processNoise=0.1] - Process noise (Q)
 * @param {number} [options.measurementNoise=1.0] - Measurement noise (R)
 * @returns {KalmanFilter3D} Configured filter instance
 */
export function createKalmanFilter3D(options = {}) {
    const config = {
        processNoise: options.processNoise || 0.1,
        measurementNoise: options.measurementNoise || 1.0
    };
    
    // KalmanFilter3D is expected to be globally available from filters.js
    if (typeof KalmanFilter3D === 'undefined') {
        throw new Error('KalmanFilter3D not found. Ensure filters.js is loaded.');
    }
    
    return new KalmanFilter3D(config);
}

/**
 * Create a MotionDetector instance with standard GAMBIT configuration
 * 
 * Uses tight thresholds for accurate stationary detection,
 * which is important for gyroscope bias calibration.
 * 
 * @param {Object} options - Override options
 * @param {number} [options.accelThreshold=200] - Accel std dev threshold (LSB)
 * @param {number} [options.gyroThreshold=300] - Gyro std dev threshold (LSB)
 * @param {number} [options.windowSize=10] - Samples in moving window (~0.5s at 20Hz)
 * @returns {MotionDetector} Configured detector instance
 */
export function createMotionDetector(options = {}) {
    const config = {
        accelThreshold: options.accelThreshold || 200,   // ~0.025g std dev when stationary
        gyroThreshold: options.gyroThreshold || 300,     // ~2.6 deg/s std dev when stationary
        windowSize: options.windowSize || 10
    };
    
    // MotionDetector is expected to be globally available from filters.js
    if (typeof MotionDetector === 'undefined') {
        throw new Error('MotionDetector not found. Ensure filters.js is loaded.');
    }
    
    return new MotionDetector(config);
}

/**
 * Create a 1D KalmanFilter instance for single-axis filtering
 * 
 * @param {Object} options - Override options
 * @param {number} [options.R=0.01] - Process noise
 * @param {number} [options.Q=3] - Measurement noise
 * @returns {KalmanFilter} Configured filter instance
 */
export function createKalmanFilter1D(options = {}) {
    const config = {
        R: options.R || 0.01,
        Q: options.Q || 3
    };
    
    // KalmanFilter is expected to be globally available from kalman.js
    if (typeof KalmanFilter === 'undefined') {
        throw new Error('KalmanFilter not found. Ensure kalman.js is loaded.');
    }
    
    return new KalmanFilter(config);
}

/**
 * Simple low-pass filter for smoothing sensor data
 * Formula: output = alpha * newValue + (1 - alpha) * previousValue
 */
export class LowPassFilter {
    constructor(alpha = 0.3) {
        this.alpha = alpha;  // 0-1, lower = smoother but more lag
        this.value = null;
    }

    filter(newValue) {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }

    reset() {
        this.value = null;
    }

    setValue(value) {
        this.value = value;
    }
}

/**
 * Create a LowPassFilter instance for smoothing
 *
 * @param {number} [alpha=0.3] - Filter coefficient (0-1, lower = smoother but more lag)
 * @returns {LowPassFilter} Configured filter instance
 */
export function createLowPassFilter(alpha = 0.3) {
    return new LowPassFilter(alpha);
}

// ===== Gyroscope Bias Calibration Configuration =====

/**
 * Number of stationary samples required before gyro bias calibration
 * At 20Hz, this is approximately 1 second
 */
export const STATIONARY_SAMPLES_FOR_CALIBRATION = 20;

/**
 * Create gyroscope bias calibration state object
 * @returns {Object} Initial bias calibration state
 */
export function createGyroBiasState() {
    return {
        calibrated: false,
        stationaryCount: 0,
        bias: { x: 0, y: 0, z: 0 }
    };
}

// ===== Cube Visualization Filter Configuration =====

/**
 * Create filter set for 3D cube visualization
 * Uses LowPassFilter for smooth display updates
 * 
 * NOTE: With stable sensor data, we can use higher alpha for responsiveness
 * 
 * @param {Object} options - Override alpha values
 * @returns {Object} Filter set {acc, gyro, mag} each with {x, y, z} filters
 */
export function createCubeFilters(options = {}) {
    const accAlpha = options.accAlpha || 0.4;
    const gyroAlpha = options.gyroAlpha || 0.3;
    const magAlpha = options.magAlpha || 0.3;
    
    return {
        acc: {
            x: createLowPassFilter(accAlpha),
            y: createLowPassFilter(accAlpha),
            z: createLowPassFilter(accAlpha)
        },
        gyro: {
            x: createLowPassFilter(gyroAlpha),
            y: createLowPassFilter(gyroAlpha),
            z: createLowPassFilter(gyroAlpha)
        },
        mag: {
            x: createLowPassFilter(magAlpha),
            y: createLowPassFilter(magAlpha),
            z: createLowPassFilter(magAlpha)
        }
    };
}

// ===== Default Export =====

export default {
    // Constants
    ACCEL_SCALE,
    GYRO_SCALE,
    DEFAULT_SAMPLE_FREQ,
    MAG_SCALE_LSB_TO_UT,
    STATIONARY_SAMPLES_FOR_CALIBRATION,
    
    // Conversion functions
    accelLsbToG,
    gyroLsbToDps,
    convertAccelToG,
    convertGyroToDps,
    
    // Factory functions
    createMadgwickAHRS,
    createKalmanFilter3D,
    createMotionDetector,
    createKalmanFilter1D,
    createLowPassFilter,
    createGyroBiasState,
    createCubeFilters
};
