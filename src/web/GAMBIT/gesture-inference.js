/**
 * GAMBIT Gesture Inference Module
 * 
 * Real-time gesture classification using TensorFlow.js
 * Runs entirely client-side in the browser
 * 
 * Model versions are stored in models/ directory
 */

class GestureInference {
    constructor(options = {}) {
        this.modelPath = options.modelPath || 'models/gesture_v1/model.json';
        this.windowSize = options.windowSize || 50;  // 1 second at 50Hz
        this.stride = options.stride || 25;  // 50% overlap
        this.confidenceThreshold = options.confidenceThreshold || 0.5;
        
        this.model = null;
        this.isReady = false;
        this.buffer = [];
        
        // Normalization stats from training (from dataset_stats.npz)
        this.stats = {
            mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
            std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
        };
        
        // Gesture labels (must match training)
        this.labels = [
            'rest', 'fist', 'open_palm', 'index_up', 'peace',
            'thumbs_up', 'ok_sign', 'pinch', 'grab', 'wave'
        ];
        
        // Callbacks
        this.onPrediction = options.onPrediction || null;
        this.onReady = options.onReady || null;
        this.onError = options.onError || null;
        
        // Performance tracking
        this.lastInferenceTime = 0;
        this.inferenceCount = 0;
    }
    
    /**
     * Load the TensorFlow.js model
     */
    async load() {
        try {
            // Check if TensorFlow.js is available
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded. Include tf.min.js first.');
            }
            
            console.log(`[GestureInference] Loading model from ${this.modelPath}...`);
            
            // Try loading as layers model first (Keras format)
            try {
                this.model = await tf.loadLayersModel(this.modelPath);
                console.log('[GestureInference] Loaded as layers model');
            } catch (e) {
                // Try loading as graph model (SavedModel format)
                this.model = await tf.loadGraphModel(this.modelPath);
                console.log('[GestureInference] Loaded as graph model');
            }
            
            // Warm up the model with a dummy prediction
            const dummyInput = tf.zeros([1, this.windowSize, 9]);
            const warmup = this.model.predict(dummyInput);
            warmup.dispose();
            dummyInput.dispose();
            
            this.isReady = true;
            console.log('[GestureInference] Model ready');
            
            if (this.onReady) {
                this.onReady();
            }
            
            return true;
        } catch (error) {
            console.error('[GestureInference] Failed to load model:', error);
            if (this.onError) {
                this.onError(error);
            }
            return false;
        }
    }
    
    /**
     * Add a sensor sample to the buffer
     * @param {Object} sample - Raw sensor data {ax, ay, az, gx, gy, gz, mx, my, mz}
     */
    addSample(sample) {
        // Extract 9-DoF IMU features
        const features = [
            sample.ax, sample.ay, sample.az,
            sample.gx, sample.gy, sample.gz,
            sample.mx, sample.my, sample.mz
        ];
        
        this.buffer.push(features);
        
        // Keep buffer at window size
        if (this.buffer.length > this.windowSize) {
            this.buffer.shift();
        }
        
        // Run inference when we have enough samples
        if (this.buffer.length === this.windowSize && this.isReady) {
            this.inferenceCount++;
            
            // Run inference every stride samples
            if (this.inferenceCount % this.stride === 0) {
                this.runInference();
            }
        }
    }
    
    /**
     * Normalize a window of data using training statistics
     */
    normalize(window) {
        return window.map(sample => {
            return sample.map((val, i) => {
                return (val - this.stats.mean[i]) / this.stats.std[i];
            });
        });
    }
    
    /**
     * Run inference on the current buffer
     */
    async runInference() {
        if (!this.isReady || this.buffer.length < this.windowSize) {
            return null;
        }
        
        const startTime = performance.now();
        
        try {
            // Normalize the window
            const normalizedWindow = this.normalize(this.buffer);
            
            // Create tensor [1, windowSize, 9]
            const inputTensor = tf.tensor3d([normalizedWindow]);
            
            // Run prediction
            const prediction = this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();
            
            // Find best prediction
            let maxProb = 0;
            let maxIdx = 0;
            for (let i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIdx = i;
                }
            }
            
            const result = {
                gesture: this.labels[maxIdx],
                confidence: maxProb,
                probabilities: Object.fromEntries(
                    this.labels.map((label, i) => [label, probabilities[i]])
                ),
                inferenceTime: performance.now() - startTime
            };
            
            this.lastInferenceTime = result.inferenceTime;
            
            // Only trigger callback if confidence exceeds threshold
            if (this.onPrediction && maxProb >= this.confidenceThreshold) {
                this.onPrediction(result);
            }
            
            return result;
        } catch (error) {
            console.error('[GestureInference] Inference error:', error);
            if (this.onError) {
                this.onError(error);
            }
            return null;
        }
    }
    
    /**
     * Get current prediction without waiting for stride
     */
    async getCurrentPrediction() {
        return this.runInference();
    }
    
    /**
     * Clear the sample buffer
     */
    clearBuffer() {
        this.buffer = [];
        this.inferenceCount = 0;
    }
    
    /**
     * Update normalization stats (e.g., from a different model version)
     */
    setStats(mean, std) {
        this.stats.mean = mean;
        this.stats.std = std;
    }
    
    /**
     * Update gesture labels (e.g., for a different model version)
     */
    setLabels(labels) {
        this.labels = labels;
    }
    
    /**
     * Get model info
     */
    getInfo() {
        return {
            modelPath: this.modelPath,
            isReady: this.isReady,
            windowSize: this.windowSize,
            stride: this.stride,
            labels: this.labels,
            bufferSize: this.buffer.length,
            lastInferenceTime: this.lastInferenceTime
        };
    }
    
    /**
     * Dispose of the model to free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isReady = false;
        this.buffer = [];
    }
}

// Model version registry
const GESTURE_MODELS = {
    'v1': {
        path: 'models/gesture_v1/model.json',
        labels: ['rest', 'fist', 'open_palm', 'index_up', 'peace', 'thumbs_up', 'ok_sign', 'pinch', 'grab', 'wave'],
        stats: {
            mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
            std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
        },
        description: 'Initial model trained on cluster-derived labels',
        date: '2025-12-09'
    }
};

/**
 * Create a gesture inference instance with a specific model version
 */
function createGestureInference(version = 'v1', options = {}) {
    const modelConfig = GESTURE_MODELS[version];
    if (!modelConfig) {
        throw new Error(`Unknown model version: ${version}. Available: ${Object.keys(GESTURE_MODELS).join(', ')}`);
    }
    
    const inference = new GestureInference({
        modelPath: modelConfig.path,
        ...options
    });
    
    inference.setStats(modelConfig.stats.mean, modelConfig.stats.std);
    inference.setLabels(modelConfig.labels);
    
    return inference;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GestureInference, GESTURE_MODELS, createGestureInference };
}
