/**
 * GAMBIT Shared Modules - Central Export Point
 * 
 * This module re-exports all shared functionality for easy importing.
 * 
 * @module shared
 * 
 * @example
 * // Import everything
 * import * as GAMBIT from './shared/index.js';
 * 
 * @example
 * // Import specific items
 * import { TelemetryProcessor, ACCEL_SCALE, createSessionPlayback } from './shared/index.js';
 */

// ===== Sensor Configuration =====
export {
    // Constants
    ACCEL_SCALE,
    GYRO_SCALE,
    DEFAULT_SAMPLE_FREQ,
    STATIONARY_SAMPLES_FOR_CALIBRATION,
    
    // Unit conversion functions
    accelLsbToG,
    gyroLsbToDps,
    
    // Factory functions
    createMadgwickAHRS,
    createKalmanFilter3D,
    createMotionDetector,
    createGyroBiasState,
    createLowPassFilter
} from './sensor-config.js';

// ===== Telemetry Processing =====
export { TelemetryProcessor } from './telemetry-processor.js';

// ===== Session Playback =====
export {
    SessionPlayback,
    createSessionPlayback,
    formatTime,
    formatSessionDisplay
} from '../modules/session-playback.js';

// ===== Gesture Inference =====
export {
    createGesture,
    createFingerTracking,
    createGestureUI,
    isGestureInferenceAvailable,
    isFingerTrackingAvailable,
    getGestureModels,
    getFingerModels,
    getGestureInferenceClass,
    getFingerTrackingInferenceClass,
    GESTURE_LABELS_V1,
    FINGER_NAMES,
    FINGER_STATES,
    DEFAULT_STATS
} from '../modules/gesture-inference-module.js';

// ===== Convenience: Create all processors =====

/**
 * Create a complete GAMBIT processing pipeline
 * @param {Object} options - Configuration options
 * @param {Object} [options.calibration] - Calibration instance
 * @param {Function} [options.onGyroBiasCalibrated] - Gyro bias callback
 * @param {Function} [options.onSample] - Sample callback for playback
 * @param {Function} [options.onPrediction] - Gesture prediction callback
 * @returns {Object} Pipeline components
 */
export function createGambitPipeline(options = {}) {
    const { TelemetryProcessor } = require('./telemetry-processor.js');
    const { SessionPlayback } = require('../modules/session-playback.js');
    
    const telemetry = new TelemetryProcessor({
        calibration: options.calibration,
        onGyroBiasCalibrated: options.onGyroBiasCalibrated
    });
    
    const playback = new SessionPlayback({
        onSample: options.onSample
    });
    
    // Gesture inference requires globals from gesture-inference.js
    let gesture = null;
    if (typeof createGestureInference !== 'undefined') {
        gesture = createGestureInference('v1', {
            onPrediction: options.onPrediction
        });
    }
    
    return {
        telemetry,
        playback,
        gesture,
        
        /**
         * Process a raw telemetry sample
         * @param {Object} sample - Raw sensor data
         * @returns {Object} Processed sample
         */
        process(sample) {
            const processed = telemetry.process(sample);
            if (gesture && gesture.isReady) {
                gesture.addSample(sample);
            }
            return processed;
        },
        
        /**
         * Get current orientation
         * @returns {Object} Euler angles
         */
        getOrientation() {
            return telemetry.getEulerAngles();
        },
        
        /**
         * Dispose all components
         */
        dispose() {
            if (gesture) gesture.dispose();
            playback.dispose();
        }
    };
}
