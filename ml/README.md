# SIMCAP Machine Learning Pipeline

End-to-end ML pipeline for gesture classification from 9-DoF IMU data.

## Quick Start

```bash
# Install dependencies
pip install -r ml/requirements.txt

# Full pipeline: train + convert to all deployment formats
python -m ml.build all --data-dir data/GAMBIT --version v1 --epochs 50

# Or step by step:
python -m ml.build train --data-dir data/GAMBIT --epochs 50
python -m ml.build convert --model ml/models/gesture_model.keras --version v1
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIMCAP ML Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────────────────┐│
│  │  Data   │───▶│ Cluster  │───▶│  Train  │───▶│       Deploy             ││
│  │ Collect │    │ (unsup.) │    │ (super.)│    │                          ││
│  └─────────┘    └──────────┘    └─────────┘    │  ┌─────────────────────┐ ││
│       │              │               │         │  │ TensorFlow.js       │ ││
│       ▼              ▼               ▼         │  │ (Browser)           │ ││
│  data/GAMBIT/   ml/models/      ml/models/     │  └─────────────────────┘ ││
│  *.json         cluster_*.json  gesture_*.keras│  ┌─────────────────────┐ ││
│  *.meta.json    label_templates/               │  │ TFLite Micro        │ ││
│                                                │  │ (ESP32)             │ ││
│                                                │  └─────────────────────┘ ││
│                                                │  ┌─────────────────────┐ ││
│                                                │  │ Centroid Classifier │ ││
│                                                │  │ (Puck.js)           │ ││
│                                                │  └─────────────────────┘ ││
│                                                └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ml/
├── __init__.py           # Package init
├── build.py              # Unified build pipeline
├── train.py              # Training script
├── cluster.py            # Unsupervised clustering
├── visualize.py          # Data visualization
├── generate_explorer.py  # Interactive explorer
├── data_loader.py        # Dataset loading
├── model.py              # Model architectures
├── schema.py             # Data schemas & gestures
├── filters.py            # Signal processing
├── calibration.py        # Sensor calibration
├── label.py              # Labeling utilities
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── CLUSTERING.md         # Clustering documentation
└── models/               # Output directory
    ├── gesture_model.keras       # Keras model
    ├── gesture_model.tflite      # TFLite model
    ├── gesture_model_quant.tflite# Quantized TFLite
    ├── gesture_model.h           # C header for embedded
    ├── training_results.json     # Training metrics
    ├── clustering_results.json   # Cluster analysis
    ├── cluster_analysis.json     # Detailed cluster info
    └── label_templates/          # Auto-generated labels
```

## Data Format

### Session Data (`data/GAMBIT/*.json`)

```json
[
  {
    "t": 1700845458479,
    "ax": -1234, "ay": 5678, "az": -9012,
    "gx": 123, "gy": -456, "gz": 789,
    "mx": 1000, "my": -2000, "mz": 3000,
    "b": 85
  },
  ...
]
```

### Metadata (`data/GAMBIT/*.meta.json`)

```json
{
  "timestamp": "2025-12-09T15:23:14.877Z",
  "subject_id": "user_001",
  "environment": "home",
  "hand": "right",
  "split": "train",
  "labels": [
    {
      "start_sample": 0,
      "end_sample": 50,
      "gesture": "fist",
      "confidence": "high"
    }
  ],
  "calibration_markers": [...],
  "finger_states": [...]
}
```

## Gestures

| ID | Name | Description |
|----|------|-------------|
| 0 | rest | Hand relaxed, neutral position |
| 1 | fist | Closed fist |
| 2 | open_palm | All fingers extended |
| 3 | index_up | Index finger pointing up |
| 4 | peace | Index and middle fingers up |
| 5 | thumbs_up | Thumb extended upward |
| 6 | ok_sign | Thumb and index forming circle |
| 7 | pinch | Thumb and index touching |
| 8 | grab | Fingers curled as if grabbing |
| 9 | wave | Hand waving motion |

## Commands

### Training

```bash
# Train with default settings
python -m ml.train --data-dir data/GAMBIT

# Train with custom parameters
python -m ml.train \
  --data-dir data/GAMBIT \
  --epochs 100 \
  --batch-size 64 \
  --window-size 50 \
  --stride 25 \
  --val-ratio 0.2

# Summary only (no training)
python -m ml.train --data-dir data/GAMBIT --summary-only
```

### Clustering (Unsupervised)

```bash
# K-means clustering
python -m ml.train --data-dir data/GAMBIT --cluster-only \
  --n-clusters 10 --visualize-clusters --create-templates

# DBSCAN clustering
python -m ml.train --data-dir data/GAMBIT --cluster-only \
  --cluster-method dbscan --dbscan-eps 0.5 --dbscan-min-samples 5
```

### Visualization

```bash
# Generate visualizations for all sessions
python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations

# Generate interactive explorer
python -m ml.generate_explorer --data-dir data/GAMBIT
```

### Build Pipeline

```bash
# Full pipeline
python -m ml.build all --data-dir data/GAMBIT --version v2 --epochs 50

# Train only
python -m ml.build train --data-dir data/GAMBIT --epochs 50

# Convert only
python -m ml.build convert --model ml/models/gesture_model.keras --version v2
```

## Model Architecture

```
Input: (batch, 50, 9) - 1 second window @ 50Hz, 9 features

Conv1D(32, kernel=5, padding=same)
BatchNormalization
ReLU
MaxPooling1D(2)
Dropout(0.3)

Conv1D(64, kernel=5, padding=same)
BatchNormalization
ReLU
MaxPooling1D(2)
Dropout(0.3)

Conv1D(64, kernel=5, padding=same)
BatchNormalization
ReLU
GlobalAveragePooling1D

Dense(64, activation=relu)
Dropout(0.3)
Dense(10, activation=softmax)

Output: (batch, 10) - gesture probabilities
```

**Parameters:** ~37K trainable
**Size:** ~150KB (Keras), ~75KB (quantized TFLite)

## Deployment Targets

### Browser (TensorFlow.js)

```javascript
const inference = createGestureInference('v1', {
  confidenceThreshold: 0.5,
  onPrediction: (result) => console.log(result.gesture)
});
await inference.load();
inference.addSample({ax, ay, az, gx, gy, gz, mx, my, mz});
```

### ESP32 (TFLite Micro)

```cpp
#include "gesture_model.h"
// See src/device/ESP32/gesture_inference.ino
```

### Puck.js (Centroid Classifier)

```javascript
// Lightweight nearest-centroid classification
// See docs/INFERENCE_DEPLOYMENT.md
```

## Output Files

After running `python -m ml.build all`:

| File | Format | Size | Use |
|------|--------|------|-----|
| `gesture_model.keras` | Keras | ~150KB | Python inference |
| `gesture_model.tflite` | TFLite | ~150KB | Mobile/Edge |
| `gesture_model_quant.tflite` | TFLite | ~75KB | TinyML/ESP32 |
| `gesture_model.h` | C Header | ~200KB | Arduino/ESP-IDF |
| `models/gesture_v1/model.json` | TF.js | ~11KB | Browser |
| `models/gesture_v1/*.bin` | TF.js | ~151KB | Browser |
| `training_results.json` | JSON | - | Metrics/history |
| `build_manifest.json` | JSON | - | Build metadata |

## Workflow

### 1. Collect Data

Use the GAMBIT web collector (`src/web/GAMBIT/collector.html`) to record sessions.

### 2. Cluster Unlabeled Data

```bash
python -m ml.train --data-dir data/GAMBIT --cluster-only \
  --visualize-clusters --create-templates
```

### 3. Review & Label

1. Check `ml/models/label_templates/`
2. Assign gesture names to clusters
3. Move `.meta.json` files to `data/GAMBIT/`

### 4. Train Model

```bash
python -m ml.build all --data-dir data/GAMBIT --version v1
```

### 5. Deploy

- **Browser:** Model auto-copied to `src/web/GAMBIT/models/`
- **ESP32:** Copy `gesture_model.h` to firmware directory
- **Puck.js:** Use centroid classifier from clustering results

## Performance

| Metric | Value |
|--------|-------|
| Training accuracy | ~90% |
| Validation accuracy | ~55% (limited data) |
| Inference time (browser) | 5-15ms |
| Inference time (ESP32) | 15-50ms |
| Model size (quantized) | ~75KB |

## Troubleshooting

### "No labeled data found"
- Ensure `.meta.json` files exist in `data/GAMBIT/`
- Run clustering first to generate label templates

### "TensorFlow not found"
```bash
pip install tensorflow tensorflowjs
```

### "Model won't load in browser"
- Check CORS headers
- Verify `model.json` and `.bin` files are accessible
- Check browser console for errors

### "ESP32 out of memory"
- Use quantized model (`gesture_model_quant.tflite`)
- Reduce tensor arena size
- Use ESP32-S3 (more RAM)

## References

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Espruino Puck.js](https://www.espruino.com/Puck.js)
- [SIMCAP Inference Deployment Guide](../docs/inference-deployment-guide.md)
