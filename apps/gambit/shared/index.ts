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

// ===== Type Exports =====
export type { EulerAngles, Quaternion, TelemetrySample } from '@core/types';
export type { TelemetryProcessorOptions, DecoratedTelemetry, MotionState, GyroBiasState } from './telemetry-processor.js';
export type { SessionMetadata, PlaybackState, SessionPlaybackOptions, LoadedSession } from '../modules/session-playback.js';
export type {
    GestureInferenceOptions,
    FingerTrackingOptions,
    GesturePredictionResult,
    GestureUIElements,
    GestureUIController
} from '../modules/gesture-inference-module.js';

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
 * @param options - Configuration options
 * @returns Pipeline components
 */
export async function createGambitPipeline(options: {
    calibration?: any;
    onGyroBiasCalibrated?: () => void;
    onSample?: (sample: any, index: number, total: number) => void;
    onPrediction?: (result: any) => void;
} = {}): Promise<{
    telemetry: InstanceType<typeof import('./telemetry-processor.js').TelemetryProcessor>;
    playback: InstanceType<typeof import('../modules/session-playback.js').SessionPlayback>;
    gesture: any;
    process: (sample: any) => any;
    getOrientation: () => any;
    dispose: () => void;
}> {
    // Dynamic imports for ES modules
    const { TelemetryProcessor } = await import('./telemetry-processor.js');
    const { SessionPlayback } = await import('../modules/session-playback.js');

    const telemetry = new TelemetryProcessor({
        calibration: options.calibration,
        onGyroBiasCalibrated: options.onGyroBiasCalibrated
    });

    const playback = new SessionPlayback({
        onSample: options.onSample
    });

    // Gesture inference requires globals from gesture-inference.js
    let gesture: any = null;
    if (typeof (globalThis as any).createGestureInference !== 'undefined') {
        gesture = (globalThis as any).createGestureInference('v1', {
            onPrediction: options.onPrediction
        });
    }

    return {
        telemetry,
        playback,
        gesture,

        /**
         * Process a raw telemetry sample
         * @param sample - Raw sensor data
         * @returns Processed sample
         */
        process(sample: any) {
            const processed = telemetry.process(sample);
            if (gesture && gesture.isReady) {
                gesture.addSample(sample);
            }
            return processed;
        },

        /**
         * Get current orientation
         * @returns Euler angles
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
