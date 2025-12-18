/**
 * GAMBIT Gesture Inference Module
 *
 * Real-time gesture classification using TensorFlow.js
 * Runs entirely client-side in the browser
 *
 * Model versions are stored in models/ directory
 */

// ===== Type Definitions =====

export interface GestureInferenceOptions {
  modelPath?: string;
  windowSize?: number;
  stride?: number;
  confidenceThreshold?: number;
  onPrediction?: ((result: GesturePrediction) => void) | null;
  onReady?: (() => void) | null;
  onError?: ((error: Error) => void) | null;
}

export interface GesturePrediction {
  gesture: string;
  confidence: number;
  probabilities: Record<string, number>;
  inferenceTime: number;
}

export interface NormalizationStats {
  mean: number[];
  std: number[];
}

export interface SensorSample {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  mx: number;
  my: number;
  mz: number;
}

export interface ModelConfig {
  path: string;
  labels: string[];
  stats: NormalizationStats;
  description: string;
  date: string;
}

export interface FingerTrackingOptions {
  modelPath?: string;
  windowSize?: number;
  stride?: number;
  confidenceThreshold?: number;
  smoothingAlpha?: number;
  onPrediction?: ((result: FingerPrediction) => void) | null;
  onReady?: (() => void) | null;
  onError?: ((error: Error) => void) | null;
}

export interface FingerPrediction {
  fingers: Record<string, string>;
  states: Record<string, number>;
  confidences: Record<string, number>;
  probabilities: Record<string, number[]>;
  timestamp: number;
  overallConfidence: number;
  binaryString: string;
  inferenceTime?: number;
}

export interface FingerModelConfig {
  path: string;
  stats: NormalizationStats;
  description: string;
  date: string;
}

export interface FingerPose {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
}

// TensorFlow.js types (external library)
interface TFTensor {
  shape: number[];
  data(): Promise<Float32Array>;
  dispose(): void;
}

interface TFModel {
  predict(input: TFTensor): TFTensor | TFTensor[];
  dispose(): void;
  summary?: () => void;
}

interface TFStatic {
  version: { tfjs: string };
  zeros(shape: number[]): TFTensor;
  tensor3d(data: number[][][]): TFTensor;
  loadLayersModel(path: string): Promise<TFModel>;
  loadGraphModel(path: string): Promise<TFModel>;
}

declare const tf: TFStatic;

// ===== Gesture Inference Class =====

export class GestureInference {
  private modelPath: string;
  private windowSize: number;
  private stride: number;
  private confidenceThreshold: number;
  private model: TFModel | null = null;
  private isReady: boolean = false;
  private buffer: number[][] = [];
  private stats: NormalizationStats;
  private labels: string[];
  private onPrediction: ((result: GesturePrediction) => void) | null;
  private onReady: (() => void) | null;
  private onError: ((error: Error) => void) | null;
  private lastInferenceTime: number = 0;
  private inferenceCount: number = 0;

  constructor(options: GestureInferenceOptions = {}) {
    this.modelPath = options.modelPath || 'models/gesture_v1/model.json';
    this.windowSize = options.windowSize || 50;
    this.stride = options.stride || 25;
    this.confidenceThreshold = options.confidenceThreshold || 0.5;

    // Normalization stats from training
    this.stats = {
      mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
      std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
    };

    // Gesture labels
    this.labels = [
      'rest', 'fist', 'open_palm', 'index_up', 'peace',
      'thumbs_up', 'ok_sign', 'pinch', 'grab', 'wave'
    ];

    this.onPrediction = options.onPrediction || null;
    this.onReady = options.onReady || null;
    this.onError = options.onError || null;

    console.log('[GestureInference] Initialized with config:', {
      modelPath: this.modelPath,
      windowSize: this.windowSize,
      stride: this.stride,
      confidenceThreshold: this.confidenceThreshold,
      labels: this.labels
    });
  }

  async load(): Promise<boolean> {
    try {
      console.log('[GestureInference] Checking TensorFlow.js availability...');
      if (typeof tf === 'undefined') {
        const error = new Error('TensorFlow.js not loaded. Include tf.min.js script before gesture-inference.js');
        console.error('[GestureInference] TensorFlow.js is not defined!');
        throw error;
      }
      console.log('[GestureInference] TensorFlow.js version:', tf.version);

      const absoluteUrl = new URL(this.modelPath, window.location.href).href;
      console.log(`[GestureInference] Loading model from path: ${this.modelPath}`);
      console.log(`[GestureInference] Resolved absolute URL: ${absoluteUrl}`);

      console.log('[GestureInference] Attempting to load as Keras layers model...');
      try {
        this.model = await tf.loadLayersModel(this.modelPath);
        console.log('[GestureInference] ✓ Successfully loaded as layers model');
      } catch (e) {
        const layersError = e as Error;
        console.warn('[GestureInference] Failed to load as layers model:', layersError.message);
        console.log('[GestureInference] Attempting to load as graph model (SavedModel format)...');
        try {
          this.model = await tf.loadGraphModel(this.modelPath);
          console.log('[GestureInference] ✓ Successfully loaded as graph model');
        } catch (e2) {
          const graphError = e2 as Error;
          console.error('[GestureInference] Failed to load as graph model:', graphError.message);
          throw new Error(`Failed to load model in both formats. Layers error: ${layersError.message}, Graph error: ${graphError.message}`);
        }
      }

      console.log('[GestureInference] Warming up model with dummy input...');
      const dummyInput = tf.zeros([1, this.windowSize, 9]);
      console.log('[GestureInference] Dummy input shape:', dummyInput.shape);
      const warmup = this.model.predict(dummyInput);
      const warmupTensor = Array.isArray(warmup) ? warmup[0] : warmup;
      console.log('[GestureInference] Warmup output shape:', warmupTensor.shape);
      if (Array.isArray(warmup)) {
        warmup.forEach(t => t.dispose());
      } else {
        warmup.dispose();
      }
      dummyInput.dispose();

      this.isReady = true;
      console.log('[GestureInference] ✓ Model ready for inference');

      if (this.onReady) {
        this.onReady();
      }

      return true;
    } catch (error) {
      const err = error as Error;
      console.error('[GestureInference] ✗ FAILED TO LOAD MODEL');
      console.error('[GestureInference] Error:', err.message);

      if (this.onError) {
        this.onError(err);
      }
      return false;
    }
  }

  addSample(sample: SensorSample): void {
    const features = [
      sample.ax, sample.ay, sample.az,
      sample.gx, sample.gy, sample.gz,
      sample.mx, sample.my, sample.mz
    ];

    if (this.buffer.length === 0) {
      console.log('[GestureInference] First sample received:', features);
    }

    this.buffer.push(features);

    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    if (this.buffer.length === this.windowSize && this.isReady) {
      this.inferenceCount++;

      if (this.inferenceCount === 1) {
        console.log(`[GestureInference] Buffer filled (${this.buffer.length}/${this.windowSize}), starting inference`);
      }

      if (this.inferenceCount % this.stride === 0) {
        this.runInference();
      }
    }
  }

  private normalize(window: number[][]): number[][] {
    return window.map(sample => {
      return sample.map((val, i) => {
        return (val - this.stats.mean[i]) / this.stats.std[i];
      });
    });
  }

  async runInference(): Promise<GesturePrediction | null> {
    if (!this.isReady || this.buffer.length < this.windowSize || !this.model) {
      return null;
    }

    const startTime = performance.now();

    try {
      const normalizedWindow = this.normalize(this.buffer);
      const inputTensor = tf.tensor3d([normalizedWindow]);
      const prediction = this.model.predict(inputTensor);
      const predTensor = Array.isArray(prediction) ? prediction[0] : prediction;
      const probabilities = await predTensor.data();

      inputTensor.dispose();
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose());
      } else {
        prediction.dispose();
      }

      let maxProb = 0;
      let maxIdx = 0;
      for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIdx = i;
        }
      }

      const result: GesturePrediction = {
        gesture: this.labels[maxIdx],
        confidence: maxProb,
        probabilities: Object.fromEntries(
          this.labels.map((label, i) => [label, probabilities[i]])
        ),
        inferenceTime: performance.now() - startTime
      };

      this.lastInferenceTime = result.inferenceTime;

      if (this.onPrediction && maxProb >= this.confidenceThreshold) {
        this.onPrediction(result);
      }

      return result;
    } catch (error) {
      const err = error as Error;
      console.error('[GestureInference] Inference error:', err.message);
      if (this.onError) {
        this.onError(err);
      }
      return null;
    }
  }

  async getCurrentPrediction(): Promise<GesturePrediction | null> {
    return this.runInference();
  }

  clearBuffer(): void {
    this.buffer = [];
    this.inferenceCount = 0;
  }

  setStats(mean: number[], std: number[]): void {
    this.stats.mean = mean;
    this.stats.std = std;
  }

  setLabels(labels: string[]): void {
    this.labels = labels;
  }

  getInfo(): {
    modelPath: string;
    isReady: boolean;
    windowSize: number;
    stride: number;
    labels: string[];
    bufferSize: number;
    lastInferenceTime: number;
  } {
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

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.isReady = false;
    this.buffer = [];
  }
}

// ===== Model Registry =====

export const GESTURE_MODELS: Record<string, ModelConfig> = {
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

// ===== Finger Tracking Inference Class =====

export class FingerTrackingInference {
  private modelPath: string;
  private windowSize: number;
  private stride: number;
  private confidenceThreshold: number;
  private model: TFModel | null = null;
  private isReady: boolean = false;
  private buffer: number[][] = [];
  private stats: NormalizationStats;
  private fingerNames: string[];
  private stateNames: string[];
  private onPrediction: ((result: FingerPrediction) => void) | null;
  private onReady: (() => void) | null;
  private onError: ((error: Error) => void) | null;
  private lastInferenceTime: number = 0;
  private inferenceCount: number = 0;
  private smoothingAlpha: number;
  private lastPrediction: FingerPrediction | null = null;

  constructor(options: FingerTrackingOptions = {}) {
    this.modelPath = options.modelPath || 'models/finger_v1/model.json';
    this.windowSize = options.windowSize || 50;
    this.stride = options.stride || 10;
    this.confidenceThreshold = options.confidenceThreshold || 0.5;
    this.smoothingAlpha = options.smoothingAlpha || 0.3;

    this.stats = {
      mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
      std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
    };

    this.fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    this.stateNames = ['extended', 'partial', 'flexed'];

    this.onPrediction = options.onPrediction || null;
    this.onReady = options.onReady || null;
    this.onError = options.onError || null;

    console.log('[FingerTracking] Initialized with config:', {
      modelPath: this.modelPath,
      windowSize: this.windowSize,
      stride: this.stride,
      fingerNames: this.fingerNames,
      stateNames: this.stateNames
    });
  }

  async load(): Promise<boolean> {
    try {
      console.log('[FingerTracking] Checking TensorFlow.js availability...');
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js not loaded');
      }
      console.log('[FingerTracking] TensorFlow.js version:', tf.version);

      const absoluteUrl = new URL(this.modelPath, window.location.href).href;
      console.log(`[FingerTracking] Loading model from: ${absoluteUrl}`);

      try {
        this.model = await tf.loadLayersModel(this.modelPath);
        console.log('[FingerTracking] ✓ Loaded as layers model');
      } catch (e) {
        console.warn('[FingerTracking] Layers model failed, trying graph model...');
        this.model = await tf.loadGraphModel(this.modelPath);
        console.log('[FingerTracking] ✓ Loaded as graph model');
      }

      console.log('[FingerTracking] Warming up model...');
      const dummyInput = tf.zeros([1, this.windowSize, 9]);
      const warmup = this.model.predict(dummyInput);

      if (Array.isArray(warmup)) {
        console.log('[FingerTracking] Multi-output model detected, outputs:', warmup.length);
        warmup.forEach(t => t.dispose());
      } else {
        console.log('[FingerTracking] Single output model, shape:', warmup.shape);
        warmup.dispose();
      }
      dummyInput.dispose();

      this.isReady = true;
      console.log('[FingerTracking] ✓ Model ready for inference');

      if (this.onReady) {
        this.onReady();
      }

      return true;
    } catch (error) {
      const err = error as Error;
      console.error('[FingerTracking] ✗ Failed to load model:', err);
      if (this.onError) {
        this.onError(err);
      }
      return false;
    }
  }

  addSample(sample: SensorSample): void {
    const features = [
      sample.ax, sample.ay, sample.az,
      sample.gx, sample.gy, sample.gz,
      sample.mx, sample.my, sample.mz
    ];

    this.buffer.push(features);

    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    if (this.buffer.length === this.windowSize && this.isReady) {
      this.inferenceCount++;
      if (this.inferenceCount % this.stride === 0) {
        this.runInference();
      }
    }
  }

  private normalize(window: number[][]): number[][] {
    return window.map(sample => {
      return sample.map((val, i) => {
        return (val - this.stats.mean[i]) / this.stats.std[i];
      });
    });
  }

  async runInference(): Promise<FingerPrediction | null> {
    if (!this.isReady || this.buffer.length < this.windowSize || !this.model) {
      return null;
    }

    const startTime = performance.now();

    try {
      const normalizedWindow = this.normalize(this.buffer);
      const inputTensor = tf.tensor3d([normalizedWindow]);
      const outputs = this.model.predict(inputTensor);

      const prediction = await this.parseOutputs(outputs);
      prediction.inferenceTime = performance.now() - startTime;

      inputTensor.dispose();
      if (Array.isArray(outputs)) {
        outputs.forEach(t => t.dispose());
      } else {
        outputs.dispose();
      }

      const smoothed = this.smoothPrediction(prediction);
      this.lastPrediction = smoothed;
      this.lastInferenceTime = smoothed.inferenceTime || 0;

      if (this.onPrediction) {
        this.onPrediction(smoothed);
      }

      return smoothed;
    } catch (error) {
      const err = error as Error;
      console.error('[FingerTracking] Inference error:', err);
      if (this.onError) {
        this.onError(err);
      }
      return null;
    }
  }

  private async parseOutputs(outputs: TFTensor | TFTensor[]): Promise<FingerPrediction> {
    const prediction: FingerPrediction = {
      fingers: {},
      states: {},
      confidences: {},
      probabilities: {},
      timestamp: Date.now(),
      overallConfidence: 0,
      binaryString: ''
    };

    if (Array.isArray(outputs)) {
      for (let i = 0; i < this.fingerNames.length; i++) {
        const finger = this.fingerNames[i];
        const probs = await outputs[i].data();

        const state = this.argmax(Array.from(probs));
        prediction.fingers[finger] = this.stateNames[state];
        prediction.states[finger] = state;
        prediction.confidences[finger] = probs[state];
        prediction.probabilities[finger] = Array.from(probs);
      }
    } else {
      const data = await outputs.data();

      for (let i = 0; i < this.fingerNames.length; i++) {
        const finger = this.fingerNames[i];
        const probs = Array.from(data.slice(i * 3, (i + 1) * 3));

        const state = this.argmax(probs);
        prediction.fingers[finger] = this.stateNames[state];
        prediction.states[finger] = state;
        prediction.confidences[finger] = probs[state];
        prediction.probabilities[finger] = probs;
      }
    }

    const confidences = Object.values(prediction.confidences);
    prediction.overallConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
    prediction.binaryString = this.fingerNames.map(f => prediction.states[f]).join('');

    return prediction;
  }

  private argmax(arr: number[]): number {
    let maxIdx = 0;
    let maxVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  private smoothPrediction(prediction: FingerPrediction): FingerPrediction {
    if (!this.lastPrediction) {
      return prediction;
    }

    const smoothed: FingerPrediction = {
      ...prediction,
      states: { ...prediction.states },
      confidences: { ...prediction.confidences },
      fingers: { ...prediction.fingers }
    };

    for (const finger of this.fingerNames) {
      const newConf = prediction.confidences[finger];
      const oldConf = this.lastPrediction.confidences[finger] || newConf;
      smoothed.confidences[finger] = this.smoothingAlpha * newConf + (1 - this.smoothingAlpha) * oldConf;

      if (smoothed.confidences[finger] > this.confidenceThreshold) {
        smoothed.states[finger] = prediction.states[finger];
        smoothed.fingers[finger] = prediction.fingers[finger];
      } else if (this.lastPrediction.states[finger] !== undefined) {
        smoothed.states[finger] = this.lastPrediction.states[finger];
        smoothed.fingers[finger] = this.lastPrediction.fingers[finger];
      }
    }

    smoothed.binaryString = this.fingerNames.map(f => smoothed.states[f]).join('');
    return smoothed;
  }

  getCurrentPrediction(): FingerPrediction | null {
    return this.lastPrediction;
  }

  toPoseFormat(prediction: FingerPrediction | null = null): FingerPose {
    const pred = prediction || this.lastPrediction;
    if (!pred) {
      return { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
    }
    return {
      thumb: pred.states['thumb'] || 0,
      index: pred.states['index'] || 0,
      middle: pred.states['middle'] || 0,
      ring: pred.states['ring'] || 0,
      pinky: pred.states['pinky'] || 0
    };
  }

  clearBuffer(): void {
    this.buffer = [];
    this.inferenceCount = 0;
    this.lastPrediction = null;
  }

  setStats(mean: number[], std: number[]): void {
    this.stats.mean = mean;
    this.stats.std = std;
  }

  getInfo(): {
    modelPath: string;
    isReady: boolean;
    windowSize: number;
    stride: number;
    fingerNames: string[];
    stateNames: string[];
    bufferSize: number;
    lastInferenceTime: number;
  } {
    return {
      modelPath: this.modelPath,
      isReady: this.isReady,
      windowSize: this.windowSize,
      stride: this.stride,
      fingerNames: this.fingerNames,
      stateNames: this.stateNames,
      bufferSize: this.buffer.length,
      lastInferenceTime: this.lastInferenceTime
    };
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.isReady = false;
    this.buffer = [];
    this.lastPrediction = null;
  }
}

// ===== Finger Model Registry =====

export const FINGER_MODELS: Record<string, FingerModelConfig> = {
  'v1': {
    path: 'models/finger_v1/model.json',
    stats: {
      mean: [-1106.31, -3629.05, -2285.71, 2740.34, -14231.48, -19574.75, 509.62, 909.94, -558.86],
      std: [3468.31, 5655.28, 4552.77, 1781.28, 3627.35, 1845.62, 380.11, 318.77, 409.51]
    },
    description: 'Multi-output finger tracking model (5 fingers × 3 states)',
    date: '2025-01-12'
  }
};

// ===== Factory Functions =====

export function createFingerTrackingInference(version: string = 'v1', options: FingerTrackingOptions = {}): FingerTrackingInference {
  const modelConfig = FINGER_MODELS[version];
  if (!modelConfig) {
    throw new Error(`Unknown finger model version: ${version}. Available: ${Object.keys(FINGER_MODELS).join(', ')}`);
  }

  const inference = new FingerTrackingInference({
    modelPath: modelConfig.path,
    ...options
  });

  inference.setStats(modelConfig.stats.mean, modelConfig.stats.std);

  return inference;
}

export function createGestureInference(version: string = 'v1', options: GestureInferenceOptions = {}): GestureInference {
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

// Export as globals for backward compatibility
declare global {
  interface Window {
    GestureInference: typeof GestureInference;
    FingerTrackingInference: typeof FingerTrackingInference;
    GESTURE_MODELS: typeof GESTURE_MODELS;
    FINGER_MODELS: typeof FINGER_MODELS;
    createGestureInference: typeof createGestureInference;
    createFingerTrackingInference: typeof createFingerTrackingInference;
  }
}

if (typeof window !== 'undefined') {
  window.GestureInference = GestureInference;
  window.FingerTrackingInference = FingerTrackingInference;
  window.GESTURE_MODELS = GESTURE_MODELS;
  window.FINGER_MODELS = FINGER_MODELS;
  window.createGestureInference = createGestureInference;
  window.createFingerTrackingInference = createFingerTrackingInference;
}
