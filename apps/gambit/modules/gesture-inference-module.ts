/**
 * GAMBIT Gesture Inference ES Module Wrapper
 *
 * This module provides ES module exports for the gesture inference classes
 * that are loaded as globals from gesture-inference.js
 *
 * The original gesture-inference.js must be loaded first via <script> tag
 * to make TensorFlow.js work properly (it needs to be a global).
 *
 * @module gesture-inference-module
 */

// ===== Type Definitions =====

declare const GestureInference: any;
declare const FingerTrackingInference: any;
declare const MagneticFingerInference: any;
declare const createGestureInference: (version?: string, options?: GestureInferenceOptions) => any;
declare const createFingerTrackingInference: (version?: string, options?: FingerTrackingOptions) => any;
declare const createMagneticFingerInference: (options?: any) => any;
declare const GESTURE_MODELS: Record<string, any>;
declare const FINGER_MODELS: Record<string, any>;

export interface GestureInferenceOptions {
  windowSize?: number;
  stride?: number;
  confidenceThreshold?: number;
  onPrediction?: (result: GesturePredictionResult) => void;
  onReady?: () => void;
  onError?: (error: Error) => void;
}

export interface FingerTrackingOptions {
  windowSize?: number;
  stride?: number;
  onPrediction?: (result: any) => void;
  onReady?: () => void;
  onError?: (error: Error) => void;
}

export interface GesturePredictionResult {
  gesture: string;
  confidence: number;
  inferenceTime: number;
  probabilities?: Record<string, number>;
}

export interface GestureUIElements {
  statusEl?: HTMLElement | null;
  nameEl?: HTMLElement | null;
  confidenceEl?: HTMLElement | null;
  timeEl?: HTMLElement | null;
  probabilitiesEl?: HTMLElement | null;
  displayEl?: HTMLElement | null;
}

export interface GestureUIController {
  setStatus: (status: string, message?: string) => void;
  updatePrediction: (result: GesturePredictionResult) => void;
  initProbabilityBars: (labels: string[]) => void;
  reset: () => void;
}

// ===== Availability Checks =====

/**
 * Check if gesture inference globals are available
 */
export function isGestureInferenceAvailable(): boolean {
    return typeof GestureInference !== 'undefined' &&
           typeof createGestureInference !== 'undefined';
}

/**
 * Check if finger tracking globals are available
 */
export function isFingerTrackingAvailable(): boolean {
    return typeof FingerTrackingInference !== 'undefined' &&
           typeof createFingerTrackingInference !== 'undefined';
}

// ===== Class Getters =====

/**
 * Get the GestureInference class
 * @throws If gesture-inference.js not loaded
 */
export function getGestureInferenceClass(): any {
    if (typeof GestureInference === 'undefined') {
        throw new Error('GestureInference not available. Load gesture-inference.js first.');
    }
    return GestureInference;
}

/**
 * Get the FingerTrackingInference class
 * @throws If gesture-inference.js not loaded
 */
export function getFingerTrackingInferenceClass(): any {
    if (typeof FingerTrackingInference === 'undefined') {
        throw new Error('FingerTrackingInference not available. Load gesture-inference.js first.');
    }
    return FingerTrackingInference;
}

/**
 * Get the MagneticFingerInference class
 * @throws If gesture-inference.js not loaded
 */
export function getMagneticFingerInferenceClass(): any {
    if (typeof MagneticFingerInference === 'undefined') {
        throw new Error('MagneticFingerInference not available. Load gesture-inference.js first.');
    }
    return MagneticFingerInference;
}

/**
 * Check if magnetic finger inference is available
 */
export function isMagneticFingerInferenceAvailable(): boolean {
    return typeof MagneticFingerInference !== 'undefined' &&
           typeof createMagneticFingerInference !== 'undefined';
}

// ===== Factory Functions =====

/**
 * Create a gesture inference instance
 * Wrapper around the global createGestureInference function
 */
export function createGesture(version: string = 'v1', options: GestureInferenceOptions = {}): any {
    if (typeof createGestureInference === 'undefined') {
        throw new Error('createGestureInference not available. Load gesture-inference.js first.');
    }
    return createGestureInference(version, options);
}

/**
 * Create a finger tracking inference instance
 * Wrapper around the global createFingerTrackingInference function
 */
export function createFingerTracking(version: string = 'v1', options: FingerTrackingOptions = {}): any {
    if (typeof createFingerTrackingInference === 'undefined') {
        throw new Error('createFingerTrackingInference not available. Load gesture-inference.js first.');
    }
    return createFingerTrackingInference(version, options);
}

/**
 * Create a magnetic finger inference instance
 * Uses contrastive pre-trained model for single-sample inference
 */
export function createMagneticFinger(options: any = {}): any {
    if (typeof createMagneticFingerInference === 'undefined') {
        throw new Error('createMagneticFingerInference not available. Load gesture-inference.js first.');
    }
    return createMagneticFingerInference(options);
}

// ===== Model Registries =====

/**
 * Get available gesture model versions
 */
export function getGestureModels(): Record<string, any> {
    if (typeof GESTURE_MODELS === 'undefined') {
        return {};
    }
    return GESTURE_MODELS;
}

/**
 * Get available finger tracking model versions
 */
export function getFingerModels(): Record<string, any> {
    if (typeof FINGER_MODELS === 'undefined') {
        return {};
    }
    return FINGER_MODELS;
}

// ===== Constants =====

/**
 * Gesture labels for v1 model
 */
export const GESTURE_LABELS_V1: string[] = [
    'rest', 'fist', 'open_palm', 'index_up', 'peace',
    'thumbs_up', 'ok_sign', 'pinch', 'grab', 'wave'
];

/**
 * Finger names
 */
export const FINGER_NAMES: string[] = ['thumb', 'index', 'middle', 'ring', 'pinky'];

/**
 * Finger state names
 */
export const FINGER_STATES: string[] = ['extended', 'partial', 'flexed'];

/**
 * Default normalization stats from training
 */
export const DEFAULT_STATS = {
    mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
    std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
};

// ===== UI Helper =====

/**
 * UI Helper: Initialize gesture display elements
 */
export function createGestureUI(elements: GestureUIElements): GestureUIController {
    const { statusEl, nameEl, confidenceEl, timeEl, probabilitiesEl, displayEl } = elements;

    return {
        /**
         * Update status indicator
         */
        setStatus(status: string, message?: string): void {
            if (!statusEl) return;

            statusEl.classList.remove('ready', 'error', 'loading');
            statusEl.classList.add(status);

            const textEl = statusEl.querySelector('span:last-child');
            if (textEl && message) {
                textEl.textContent = message;
            }
        },

        /**
         * Update gesture display with prediction result
         */
        updatePrediction(result: GesturePredictionResult): void {
            if (nameEl) {
                nameEl.textContent = result.gesture.replace('_', ' ');
            }

            if (confidenceEl) {
                confidenceEl.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
            }

            if (timeEl) {
                timeEl.textContent = `${result.inferenceTime.toFixed(1)}ms inference`;
            }

            if (displayEl) {
                if (result.confidence > 0.7) {
                    displayEl.classList.add('active');
                } else {
                    displayEl.classList.remove('active');
                }
            }

            if (probabilitiesEl && result.probabilities) {
                for (const [label, prob] of Object.entries(result.probabilities)) {
                    const bar = probabilitiesEl.querySelector(`#prob-${label}`) as HTMLElement | null;
                    if (bar) {
                        bar.style.width = `${prob * 100}%`;
                    }
                }
            }
        },

        /**
         * Initialize probability bars
         */
        initProbabilityBars(labels: string[]): void {
            if (!probabilitiesEl) return;

            probabilitiesEl.innerHTML = labels.map(label => `
                <div class="prob-item">
                    <div class="prob-label">${label.replace('_', ' ')}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" id="prob-${label}" style="width: 0%"></div>
                    </div>
                </div>
            `).join('');
        },

        /**
         * Reset display to initial state
         */
        reset(): void {
            if (nameEl) nameEl.textContent = '--';
            if (confidenceEl) confidenceEl.textContent = 'Waiting for data...';
            if (timeEl) timeEl.textContent = '';
            if (displayEl) displayEl.classList.remove('active');
        }
    };
}

// ===== Default Export =====

export default {
    createGesture,
    createFingerTracking,
    createMagneticFinger,
    createGestureUI,
    isGestureInferenceAvailable,
    isFingerTrackingAvailable,
    isMagneticFingerInferenceAvailable,
    getGestureModels,
    getFingerModels,
    GESTURE_LABELS_V1,
    FINGER_NAMES,
    FINGER_STATES,
    DEFAULT_STATS
};
