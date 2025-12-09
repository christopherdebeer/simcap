# SIMCAP Inference Deployment Guide

This document covers deploying trained gesture models to various targets.

## Build Pipeline

### Quick Start

```bash
# Full pipeline: train + convert to all formats
python -m ml.build all --data-dir data/GAMBIT --version v2 --epochs 50

# Just train
python -m ml.build train --data-dir data/GAMBIT --epochs 50

# Just convert existing model
python -m ml.build convert --model ml/models/gesture_model.keras --version v2
```

### Output Formats

| Format | File | Size | Target |
|--------|------|------|--------|
| Keras | `gesture_model.keras` | ~150KB | Python inference |
| TFLite | `gesture_model.tflite` | ~150KB | Mobile, Edge TPU |
| TFLite Quantized | `gesture_model_quant.tflite` | ~75KB | TinyML, ESP32 |
| TensorFlow.js | `model.json` + `.bin` | ~162KB | Browser |
| C Header | `gesture_model.h` | ~200KB | Arduino, ESP-IDF |

---

## Deployment Targets

### 1. Browser (TensorFlow.js)

**Location:** `src/web/GAMBIT/models/gesture_v1/`

**Integration:**
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js"></script>
<script src="gesture-inference.js"></script>
<script>
  const inference = createGestureInference('v1', {
    confidenceThreshold: 0.5,
    onPrediction: (result) => console.log(result.gesture)
  });
  await inference.load();
  
  // Feed sensor data
  inference.addSample({ax, ay, az, gx, gy, gz, mx, my, mz});
</script>
```

**Performance:** ~5-15ms inference on modern browsers

---

### 2. ESP32 (Native C/C++)

**Recommended for:** Production devices, low latency, battery efficiency

**Requirements:**
- ESP-IDF 4.4+ or Arduino ESP32 core
- TensorFlow Lite Micro library

**Setup:**
```bash
# Install TFLite Micro for ESP32
git clone https://github.com/espressif/esp-tflite-micro.git
```

**Code Example:**
```cpp
#include "gesture_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Tensor arena (adjust size based on model)
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Model and interpreter
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Sliding window buffer
float window_buffer[GESTURE_MODEL_WINDOW_SIZE][GESTURE_MODEL_NUM_FEATURES];
int buffer_index = 0;

void setup_model() {
    model = tflite::GetModel(gesture_model_tflite);
    
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void add_sample(float ax, float ay, float az, 
                float gx, float gy, float gz,
                float mx, float my, float mz) {
    // Normalize using training statistics
    window_buffer[buffer_index][0] = (ax - GESTURE_MEAN[0]) / GESTURE_STD[0];
    window_buffer[buffer_index][1] = (ay - GESTURE_MEAN[1]) / GESTURE_STD[1];
    window_buffer[buffer_index][2] = (az - GESTURE_MEAN[2]) / GESTURE_STD[2];
    window_buffer[buffer_index][3] = (gx - GESTURE_MEAN[3]) / GESTURE_STD[3];
    window_buffer[buffer_index][4] = (gy - GESTURE_MEAN[4]) / GESTURE_STD[4];
    window_buffer[buffer_index][5] = (gz - GESTURE_MEAN[5]) / GESTURE_STD[5];
    window_buffer[buffer_index][6] = (mx - GESTURE_MEAN[6]) / GESTURE_STD[6];
    window_buffer[buffer_index][7] = (my - GESTURE_MEAN[7]) / GESTURE_STD[7];
    window_buffer[buffer_index][8] = (mz - GESTURE_MEAN[8]) / GESTURE_STD[8];
    
    buffer_index = (buffer_index + 1) % GESTURE_MODEL_WINDOW_SIZE;
}

int run_inference() {
    // Copy window to input tensor
    for (int i = 0; i < GESTURE_MODEL_WINDOW_SIZE; i++) {
        int idx = (buffer_index + i) % GESTURE_MODEL_WINDOW_SIZE;
        for (int j = 0; j < GESTURE_MODEL_NUM_FEATURES; j++) {
            input->data.f[i * GESTURE_MODEL_NUM_FEATURES + j] = window_buffer[idx][j];
        }
    }
    
    // Run inference
    interpreter->Invoke();
    
    // Find max probability
    int max_idx = 0;
    float max_prob = output->data.f[0];
    for (int i = 1; i < GESTURE_MODEL_NUM_CLASSES; i++) {
        if (output->data.f[i] > max_prob) {
            max_prob = output->data.f[i];
            max_idx = i;
        }
    }
    
    return max_idx;  // Returns gesture index
}
```

**Memory Requirements:**
- Model: ~75KB (quantized)
- Tensor Arena: ~32KB
- Window Buffer: ~2KB
- **Total: ~110KB RAM**

---

### 3. Puck.js (Espruino JavaScript)

**Challenge:** Puck.js has only 64KB RAM and runs interpreted JavaScript.

**Options:**

#### Option A: Offload to Phone/Browser (Recommended)
The Puck.js sends raw sensor data via BLE, and inference runs on the connected device.

```javascript
// Puck.js - Send raw data
var SAMPLE_RATE = 50;
var buffer = [];

function readSensors() {
  var a = Puck.accel();
  var m = Puck.mag();
  
  // Send via BLE characteristic
  Bluetooth.println(JSON.stringify({
    ax: a.x, ay: a.y, az: a.z,
    gx: a.gx || 0, gy: a.gy || 0, gz: a.gz || 0,
    mx: m.x, my: m.y, mz: m.z,
    t: Date.now()
  }));
}

setInterval(readSensors, 1000 / SAMPLE_RATE);
```

#### Option B: Simple On-Device Classifier
For Puck.js, use a simplified decision tree or threshold-based classifier:

```javascript
// Puck.js - Simple gesture detection (no ML)
var THRESHOLDS = {
  FIST: { accel_mag: 1.2, gyro_mag: 0.5 },
  WAVE: { gyro_x_var: 500 },
  REST: { accel_mag: 0.3, gyro_mag: 0.1 }
};

var history = [];
var WINDOW = 50;

function detectGesture() {
  if (history.length < WINDOW) return "unknown";
  
  // Calculate statistics
  var accel_mag = 0, gyro_mag = 0;
  for (var i = 0; i < WINDOW; i++) {
    var h = history[i];
    accel_mag += Math.sqrt(h.ax*h.ax + h.ay*h.ay + h.az*h.az);
    gyro_mag += Math.sqrt(h.gx*h.gx + h.gy*h.gy + h.gz*h.gz);
  }
  accel_mag /= WINDOW;
  gyro_mag /= WINDOW;
  
  // Simple threshold classification
  if (gyro_mag > THRESHOLDS.WAVE.gyro_x_var) return "wave";
  if (accel_mag < THRESHOLDS.REST.accel_mag) return "rest";
  if (accel_mag > THRESHOLDS.FIST.accel_mag) return "fist";
  
  return "unknown";
}

function addSample(sample) {
  history.push(sample);
  if (history.length > WINDOW) history.shift();
}
```

#### Option C: Quantized Lookup Table
Pre-compute cluster centroids and use nearest-neighbor:

```javascript
// Puck.js - Cluster centroid matching
var CENTROIDS = [
  // Pre-computed from clustering (normalized)
  [-0.30, 0.56, 0.11, 0.14, -0.02, -0.11],  // Cluster 0
  [0.96, 0.32, 1.18, -0.05, 0.29, 0.34],    // Cluster 1
  // ... more centroids
];

var CLUSTER_TO_GESTURE = [
  "fist", "thumbs_up", "peace", "rest", "open_palm",
  "index_up", "grab", "pinch", "ok_sign", "rest"
];

function classifyWindow(features) {
  var minDist = Infinity;
  var bestCluster = 0;
  
  for (var i = 0; i < CENTROIDS.length; i++) {
    var dist = 0;
    for (var j = 0; j < features.length; j++) {
      var d = features[j] - CENTROIDS[i][j];
      dist += d * d;
    }
    if (dist < minDist) {
      minDist = dist;
      bestCluster = i;
    }
  }
  
  return CLUSTER_TO_GESTURE[bestCluster];
}
```

---

### 4. Hybrid Architecture (Recommended)

```
┌─────────────────┐     BLE      ┌─────────────────┐
│    Puck.js      │ ──────────▶  │  Phone/Browser  │
│  (Sensor Hub)   │              │  (ML Inference) │
│                 │  ◀──────────  │                 │
│  - Read IMU     │   Commands   │  - TensorFlow.js│
│  - Send data    │              │  - Display UI   │
│  - Execute cmds │              │  - Store data   │
└─────────────────┘              └─────────────────┘
```

**Benefits:**
- Puck.js battery life preserved (no heavy computation)
- Full ML model runs on phone/browser
- Easy model updates (no firmware flash)
- Real-time visualization

---

## Model Versioning

Models are versioned in `src/web/GAMBIT/gesture-inference.js`:

```javascript
const GESTURE_MODELS = {
    'v1': {
        path: 'models/gesture_v1/model.json',
        labels: ['rest', 'fist', ...],
        stats: { mean: [...], std: [...] },
        date: '2025-12-09'
    },
    'v2': {
        path: 'models/gesture_v2/model.json',
        // ... updated config
    }
};
```

To use a specific version:
```javascript
const inference = createGestureInference('v2');
```

---

## Performance Benchmarks

| Platform | Model | Inference Time | Memory |
|----------|-------|----------------|--------|
| Chrome (M1 Mac) | TF.js | 5-8ms | ~50MB |
| Safari (iPhone 14) | TF.js | 10-15ms | ~40MB |
| ESP32-S3 | TFLite Micro | 15-25ms | ~110KB |
| ESP32-C3 | TFLite Micro | 30-50ms | ~110KB |
| Puck.js | Centroid | 2-5ms | ~5KB |

---

## Troubleshooting

### TensorFlow.js Model Won't Load
- Check CORS headers if serving from different origin
- Verify `model.json` and `.bin` files are accessible
- Check browser console for specific errors

### ESP32 Out of Memory
- Use quantized model (`gesture_model_quant.tflite`)
- Reduce tensor arena size
- Use ESP32-S3 (512KB SRAM vs 320KB)

### Puck.js Too Slow
- Use simplified classifier (Option B or C)
- Reduce sample rate to 25Hz
- Offload to phone (Option A)
