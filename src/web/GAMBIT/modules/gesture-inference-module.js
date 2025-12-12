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

// Re-export globals as ES module exports
// These are defined in gesture-inference.js which must be loaded first

/**
 * Check if gesture inference globals are available
 * @returns {boolean}
 */
export function isGestureInferenceAvailable() {
    return typeof GestureInference !== 'undefined' && 
           typeof createGestureInference !== 'undefined';
}

/**
 * Check if finger tracking globals are available
 * @returns {boolean}
 */
export function isFingerTrackingAvailable() {
    return typeof FingerTrackingInference !== 'undefined' && 
           typeof createFingerTrackingInference !== 'undefined';
}

/**
 * Get the GestureInference class
 * @returns {Function} GestureInference class
 * @throws {Error} If gesture-inference.js not loaded
 */
export function getGestureInferenceClass() {
    if (typeof GestureInference === 'undefined') {
        throw new Error('GestureInference not available. Load gesture-inference.js first.');
    }
    return GestureInference;
}

/**
 * Get the FingerTrackingInference class
 * @returns {Function} FingerTrackingInference class
 * @throws {Error} If gesture-inference.js not loaded
 */
export function getFingerTrackingInferenceClass() {
    if (typeof FingerTrackingInference === 'undefined') {
        throw new Error('FingerTrackingInference not available. Load gesture-inference.js first.');
    }
    return FingerTrackingInference;
}

/**
 * Create a gesture inference instance
 * Wrapper around the global createGestureInference function
 * 
 * @param {string} [version='v1'] - Model version
 * @param {Object} [options={}] - Configuration options
 * @param {number} [options.windowSize=50] - Window size in samples
 * @param {number} [options.stride=25] - Stride between inferences
 * @param {number} [options.confidenceThreshold=0.5] - Minimum confidence
 * @param {Function} [options.onPrediction] - Prediction callback
 * @param {Function} [options.onReady] - Ready callback
 * @param {Function} [options.onError] - Error callback
 * @returns {Object} GestureInference instance
 */
export function createGesture(version = 'v1', options = {}) {
    if (typeof createGestureInference === 'undefined') {
        throw new Error('createGestureInference not available. Load gesture-inference.js first.');
    }
    return createGestureInference(version, options);
}

/**
 * Create a finger tracking inference instance
 * Wrapper around the global createFingerTrackingInference function
 * 
 * @param {string} [version='v1'] - Model version
 * @param {Object} [options={}] - Configuration options
 * @returns {Object} FingerTrackingInference instance
 */
export function createFingerTracking(version = 'v1', options = {}) {
    if (typeof createFingerTrackingInference === 'undefined') {
        throw new Error('createFingerTrackingInference not available. Load gesture-inference.js first.');
    }
    return createFingerTrackingInference(version, options);
}

/**
 * Get available gesture model versions
 * @returns {Object} Model registry
 */
export function getGestureModels() {
    if (typeof GESTURE_MODELS === 'undefined') {
        return {};
    }
    return GESTURE_MODELS;
}

/**
 * Get available finger tracking model versions
 * @returns {Object} Model registry
 */
export function getFingerModels() {
    if (typeof FINGER_MODELS === 'undefined') {
        return {};
    }
    return FINGER_MODELS;
}

/**
 * Gesture labels for v1 model
 */
export const GESTURE_LABELS_V1 = [
    'rest', 'fist', 'open_palm', 'index_up', 'peace',
    'thumbs_up', 'ok_sign', 'pinch', 'grab', 'wave'
];

/**
 * Finger names
 */
export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'];

/**
 * Finger state names
 */
export const FINGER_STATES = ['extended', 'partial', 'flexed'];

/**
 * Default normalization stats from training
 */
export const DEFAULT_STATS = {
    mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
    std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
};

/**
 * UI Helper: Initialize gesture display elements
 * @param {Object} elements - DOM element references
 * @param {HTMLElement} elements.statusEl - Status indicator element
 * @param {HTMLElement} elements.nameEl - Gesture name element
 * @param {HTMLElement} elements.confidenceEl - Confidence display element
 * @param {HTMLElement} elements.timeEl - Inference time element
 * @param {HTMLElement} elements.probabilitiesEl - Probabilities container
 * @returns {Object} UI controller
 */
export function createGestureUI(elements) {
    const { statusEl, nameEl, confidenceEl, timeEl, probabilitiesEl, displayEl } = elements;
    
    return {
        /**
         * Update status indicator
         * @param {string} status - 'loading', 'ready', 'error'
         * @param {string} [message] - Status message
         */
        setStatus(status, message) {
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
         * @param {Object} result - Prediction result
         */
        updatePrediction(result) {
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
            
            // Update probability bars
            if (probabilitiesEl && result.probabilities) {
                for (const [label, prob] of Object.entries(result.probabilities)) {
                    const bar = probabilitiesEl.querySelector(`#prob-${label}`);
                    if (bar) {
                        bar.style.width = `${prob * 100}%`;
                    }
                }
            }
        },
        
        /**
         * Initialize probability bars
         * @param {Array} labels - Gesture labels
         */
        initProbabilityBars(labels) {
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
        reset() {
            if (nameEl) nameEl.textContent = '--';
            if (confidenceEl) confidenceEl.textContent = 'Waiting for data...';
            if (timeEl) timeEl.textContent = '';
            if (displayEl) displayEl.classList.remove('active');
        }
    };
}

// Default export
export default {
    createGesture,
    createFingerTracking,
    createGestureUI,
    isGestureInferenceAvailable,
    isFingerTrackingAvailable,
    getGestureModels,
    getFingerModels,
    GESTURE_LABELS_V1,
    FINGER_NAMES,
    FINGER_STATES,
    DEFAULT_STATS
};
