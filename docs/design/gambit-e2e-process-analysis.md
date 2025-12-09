# GAMBIT End-to-End Process Analysis

## Executive Summary

This document analyzes the GAMBIT calibration and data collection system within the context of the ML pipeline and web UI inference. It identifies gaps in the current implementation and proposes an ideal early-phase end-to-end process focused on gathering quality data and training an initial model to assess accuracy.

---

## Current State Assessment

### What Exists (Strengths)

#### 1. Data Collection Infrastructure
- **Web Collector** (`src/web/GAMBIT/collector.html`): Full-featured multi-label data collection UI
  - Multi-label support (pose, finger states, motion, calibration markers, custom labels)
  - Calibration wizard with earth field, hard iron, soft iron calibration
  - Real-time sensor visualization
  - Hand state preview with manual labels
  - GitHub upload integration
  - Session metadata capture (subject, environment, hand, split)

#### 2. Calibration System
- **JavaScript Calibration** (`src/web/GAMBIT/calibration.js`): Environmental calibration
  - Earth field subtraction
  - Hard iron offset correction
  - Soft iron matrix transformation
  - Quality metrics (sphericity, coverage)
  - localStorage persistence

- **Python Calibration** (`ml/calibration.py`): ML pipeline calibration
  - Mirrors JavaScript implementation
  - Integrates with data loader

#### 3. Filtering Pipeline
- **Kalman Filter** (`src/web/GAMBIT/filters.js`, `ml/filters.py`): 3D magnetometer filtering
  - Noise reduction (8x improvement documented)
  - Velocity estimation
  - Configurable process/measurement noise

#### 4. ML Pipeline
- **Data Loader** (`ml/data_loader.py`): Comprehensive data loading
  - V1 (single-label) and V2 (multi-label) format support
  - Automatic calibration/filtering decoration
  - Windowing with configurable size/stride
  - Normalization (standardize/minmax)

- **Model Architecture** (`ml/model.py`): 1D CNN for gesture classification
  - 3 conv blocks with batch norm, ReLU, max pooling
  - ~37K parameters, ~75KB quantized
  - Keras and PyTorch implementations

- **Training** (`ml/train.py`): Full training pipeline
  - Supervised training with early stopping
  - Unsupervised clustering for label discovery
  - Visualization generation

- **Build Pipeline** (`ml/build.py`): Multi-format export
  - TensorFlow.js for browser
  - TFLite for ESP32
  - C header for embedded

#### 5. Inference
- **Browser Inference** (`src/web/GAMBIT/gesture-inference.js`): TensorFlow.js inference
- **ESP32 Inference** (`src/device/ESP32/gesture_inference.ino`): TFLite Micro template

### What's Missing (Gaps)

#### Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| **No labeled training data** | Cannot train supervised model | 🔴 Critical |
| **No finger magnets** | Cannot do magnetic finger tracking | 🟡 Medium |
| **No accuracy baseline** | Cannot measure improvement | 🔴 Critical |
| **No validation protocol** | Cannot assess real-world performance | 🔴 Critical |

#### Process Gaps

1. **Data Quality Validation**
   - No automated quality checks on collected data
   - No outlier detection
   - No sensor drift monitoring

2. **Labeling Workflow**
   - Clustering generates templates but requires manual gesture assignment
   - No visual review of cluster representatives
   - No inter-rater reliability for labels

3. **Model Evaluation**
   - No held-out test set protocol
   - No cross-validation
   - No confusion matrix analysis workflow
   - No per-gesture accuracy tracking

4. **Calibration Validation**
   - No automated calibration quality assessment
   - No recalibration triggers
   - No calibration drift detection

---

## Ideal Early-Phase End-to-End Process

### Phase 0: Environment Setup (1 day)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT SETUP                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Install Python dependencies                                 │
│     pip install -r ml/requirements.txt                          │
│                                                                 │
│  2. Verify Puck.js device connectivity                          │
│     - Flash GAMBIT firmware                                     │
│     - Test BLE connection via collector.html                    │
│                                                                 │
│  3. Prepare collection environment                              │
│     - Clear metal objects from 1m radius                        │
│     - Consistent lighting for any video reference               │
│     - Quiet environment for focused collection                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Calibration (30 minutes per session)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION PROTOCOL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Earth Field Calibration (5 seconds)                    │
│  ├── Hold device still, away from magnets                       │
│  ├── Capture ambient magnetic field baseline                    │
│  └── Quality check: std < 1.0 μT = excellent                    │
│                                                                 │
│  Step 2: Hard Iron Calibration (8 seconds)                      │
│  ├── Rotate device in figure-8 pattern                          │
│  ├── Cover all 3D orientations                                  │
│  └── Quality check: sphericity > 0.7                            │
│                                                                 │
│  Step 3: Soft Iron Calibration (8 seconds)                      │
│  ├── Continue rotating to capture distortion                    │
│  └── Quality check: eigenvalue ratio > 0.5                      │
│                                                                 │
│  Step 4: Reference Pose (5 seconds)                             │
│  ├── Palm down, fingers extended                                │
│  └── Establishes coordinate frame baseline                      │
│                                                                 │
│  Validation:                                                    │
│  ├── Check calibration quality metrics in wizard                │
│  ├── If any metric fails, repeat that step                      │
│  └── Save calibration to localStorage                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Initial Data Collection (2-4 hours)

#### Target: Minimum Viable Dataset

| Gesture | Target Samples | Duration | Sessions |
|---------|---------------|----------|----------|
| rest | 500 windows | 10 min | 3 |
| fist | 500 windows | 10 min | 3 |
| open_palm | 500 windows | 10 min | 3 |
| index_up | 500 windows | 10 min | 3 |
| peace | 500 windows | 10 min | 3 |
| **Total** | **2,500 windows** | **50 min** | **15** |

#### Collection Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                 DATA COLLECTION PROTOCOL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each gesture (5 gestures × 3 sessions each):               │
│                                                                 │
│  1. SETUP (30 seconds)                                          │
│     ├── Connect device via collector.html                       │
│     ├── Verify calibration is loaded                            │
│     ├── Set metadata: subject_id, environment, hand, split      │
│     └── Select target gesture in UI                             │
│                                                                 │
│  2. COLLECTION (3-4 minutes per gesture)                        │
│     ├── Start recording                                         │
│     ├── Hold gesture steady for 3 seconds                       │
│     ├── Brief rest (1 second)                                   │
│     ├── Repeat 30-40 times                                      │
│     └── Stop recording                                          │
│                                                                 │
│  3. VALIDATION (1 minute)                                       │
│     ├── Review sample count (target: 150-200 samples/gesture)   │
│     ├── Check labels list for correct segmentation              │
│     └── Export data + metadata                                  │
│                                                                 │
│  4. SPLIT ASSIGNMENT                                            │
│     ├── Sessions 1-2: train (80%)                               │
│     ├── Session 3: validation (20%)                             │
│     └── Later: collect separate test set                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Data Quality Checklist

- [ ] Each gesture has ≥500 windows after windowing
- [ ] Train/validation split is ~80/20
- [ ] No single session dominates any gesture class
- [ ] Calibration was performed before each collection session
- [ ] Metadata includes subject_id, environment, hand

### Phase 3: Data Validation & Exploration (1-2 hours)

```bash
# 1. Generate visualizations for all sessions
python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations

# 2. Generate interactive explorer
python -m ml.generate_explorer --data-dir data/GAMBIT

# 3. Run clustering to validate label separability
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --n-clusters 5 \
    --visualize-clusters

# 4. Check dataset summary
python -m ml.data_loader data/GAMBIT
```

#### Validation Criteria

| Metric | Target | Action if Failed |
|--------|--------|------------------|
| Silhouette score | > 0.3 | Review gesture distinctiveness |
| Cluster purity | > 0.7 | Check for mislabeled data |
| Class balance | < 2:1 ratio | Collect more of minority class |
| Window count | ≥ 2000 total | Collect more data |

### Phase 4: Initial Model Training (30 minutes)

```bash
# Train initial model with conservative settings
python -m ml.build all \
    --data-dir data/GAMBIT \
    --version v0.1 \
    --epochs 50 \
    --batch-size 32 \
    --window-size 50 \
    --stride 25
```

#### Expected Outputs

```
ml/models/
├── gesture_model.keras           # Keras model
├── gesture_model.tflite          # TFLite model
├── gesture_model_quant.tflite    # Quantized TFLite
├── gesture_model.h               # C header for ESP32
├── training_results.json         # Training metrics
└── gesture_v0.1/                 # TensorFlow.js model
    ├── model.json
    └── group1-shard1of1.bin
```

### Phase 5: Accuracy Assessment (1-2 hours)

#### 5.1 Quantitative Metrics

```python
# Load training results
import json
with open('ml/models/training_results.json') as f:
    results = json.load(f)

# Key metrics to assess
print(f"Training Accuracy: {results['history']['accuracy'][-1]:.2%}")
print(f"Validation Accuracy: {results['history']['val_accuracy'][-1]:.2%}")
print(f"Validation Loss: {results['history']['val_loss'][-1]:.4f}")

# Per-class accuracy
for gesture, acc in results['metrics']['per_class_accuracy'].items():
    if acc is not None:
        print(f"  {gesture}: {acc:.2%}")
```

#### 5.2 Confusion Matrix Analysis

```python
import numpy as np
from ml.schema import Gesture

confusion = np.array(results['metrics']['confusion_matrix'])
class_names = results['metrics']['class_names']

# Identify problem pairs (high confusion)
for i, name_i in enumerate(class_names):
    for j, name_j in enumerate(class_names):
        if i != j and confusion[i, j] > 5:  # More than 5 misclassifications
            print(f"Confusion: {name_i} → {name_j}: {confusion[i, j]} samples")
```

#### 5.3 Real-Time Inference Test

1. Open `src/web/GAMBIT/index.html`
2. Connect device
3. Load trained model (v0.1)
4. Perform each gesture 10 times
5. Record recognition rate

| Gesture | Attempts | Correct | Accuracy |
|---------|----------|---------|----------|
| rest | 10 | ? | ?% |
| fist | 10 | ? | ?% |
| open_palm | 10 | ? | ?% |
| index_up | 10 | ? | ?% |
| peace | 10 | ? | ?% |
| **Total** | **50** | **?** | **?%** |

#### 5.4 Accuracy Targets

| Phase | Target Accuracy | Notes |
|-------|-----------------|-------|
| Initial (v0.1) | > 60% | Baseline with 5 gestures |
| Improved (v0.2) | > 75% | After data augmentation |
| Production (v1.0) | > 85% | With full dataset |

### Phase 6: Iteration & Improvement

Based on accuracy assessment, iterate:

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMPROVEMENT CYCLE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IF accuracy < 60%:                                             │
│  ├── Review confusion matrix for problem gestures               │
│  ├── Collect more data for confused pairs                       │
│  ├── Check calibration quality                                  │
│  └── Verify label correctness                                   │
│                                                                 │
│  IF accuracy 60-75%:                                            │
│  ├── Add data augmentation (noise, time warping)                │
│  ├── Increase training epochs                                   │
│  ├── Try different window sizes (25, 75, 100)                   │
│  └── Experiment with model architecture                         │
│                                                                 │
│  IF accuracy > 75%:                                             │
│  ├── Add more gesture classes                                   │
│  ├── Collect data from additional subjects                      │
│  ├── Test in different environments                             │
│  └── Prepare for production deployment                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Non-Generative Modeling Considerations

### Why Non-Generative for Initial Phase

1. **Simpler to validate**: Classification accuracy is straightforward to measure
2. **Less data required**: Discriminative models need less data than generative
3. **Faster iteration**: Quick training cycles enable rapid experimentation
4. **Interpretable**: Confusion matrices show exactly where model fails

### Model Options (Non-Generative)

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **1D CNN** (current) | Fast, small, proven | Limited temporal modeling | Static poses |
| **Random Forest** | Interpretable, no GPU | Less accurate | Baseline comparison |
| **SVM** | Good with small data | Doesn't scale | Quick prototyping |
| **1D CNN + LSTM** | Better temporal | More complex | Dynamic gestures |
| **Transformer** | State-of-art | Needs more data | Future enhancement |

### Recommended Initial Model

**1D CNN (current architecture)** is appropriate for initial phase because:
- Proven architecture for time-series classification
- Small enough for embedded deployment (~75KB quantized)
- Fast inference (5-15ms browser, 15-50ms ESP32)
- Sufficient for static pose classification

---

## Filtering Considerations

### Current Filtering Pipeline

```
Raw Magnetometer → Calibration → Kalman Filter → Model Input
     (mx,my,mz)      (offset,     (noise         (normalized
                      matrix)      reduction)     features)
```

### Filter Tuning for Accuracy

| Parameter | Default | For Accuracy | Notes |
|-----------|---------|--------------|-------|
| `processNoise` | 0.1 | 0.05-0.1 | Lower = smoother, more lag |
| `measurementNoise` | 1.0 | 0.5-1.0 | Lower = trust sensor more |
| Window size | 50 | 50-100 | Larger = more context |
| Stride | 25 | 10-25 | Smaller = more overlap |

### Filtering Best Practices

1. **Always apply calibration before filtering**
2. **Use same filter settings for training and inference**
3. **Log filter parameters in training metadata**
4. **Test with filtering disabled to measure impact**

---

## Environment Calibration Considerations

### Per-Environment Calibration

Different environments have different magnetic signatures:

| Environment | Typical Interference | Calibration Frequency |
|-------------|---------------------|----------------------|
| Home | Appliances, wiring | Once per location |
| Office | Computers, metal desks | Once per desk |
| Outdoor | Minimal | Once per session |
| Lab | Equipment | Before each session |

### Calibration Quality Thresholds

| Metric | Excellent | Good | Poor | Action |
|--------|-----------|------|------|--------|
| Earth field std | < 1.0 μT | < 3.0 μT | > 5.0 μT | Recalibrate |
| Sphericity | > 0.9 | > 0.7 | < 0.5 | Check for metal |
| Coverage | > 0.9 | > 0.7 | < 0.5 | Rotate more |
| Eigenvalue ratio | > 0.8 | > 0.5 | < 0.3 | High distortion |

### Calibration Drift Detection

```javascript
// Implement in collector.html
function checkCalibrationDrift(currentSample, savedCalibration) {
    const drift = Math.sqrt(
        Math.pow(currentSample.mx - savedCalibration.earth_field.x, 2) +
        Math.pow(currentSample.my - savedCalibration.earth_field.y, 2) +
        Math.pow(currentSample.mz - savedCalibration.earth_field.z, 2)
    );
    
    if (drift > 5.0) {
        console.warn('Calibration drift detected:', drift, 'μT');
        return true;
    }
    return false;
}
```

---

## Recommended Immediate Actions

### Week 1: Data Collection Sprint

| Day | Task | Output |
|-----|------|--------|
| 1 | Environment setup, calibration | Working collector |
| 2-3 | Collect 5 gestures × 3 sessions | 15 labeled sessions |
| 4 | Validate data, run clustering | Quality report |
| 5 | Train initial model | v0.1 model |

### Week 2: Accuracy Assessment

| Day | Task | Output |
|-----|------|--------|
| 1 | Quantitative evaluation | Metrics report |
| 2 | Real-time inference testing | User test results |
| 3 | Identify improvement areas | Action plan |
| 4-5 | Iterate on data/model | v0.2 model |

### Success Criteria for Initial Phase

- [ ] ≥ 2,500 labeled windows across 5 gestures
- [ ] Validation accuracy ≥ 60%
- [ ] Real-time inference working in browser
- [ ] Documented accuracy per gesture
- [ ] Clear path to improvement identified

---

## Appendix: Quick Reference Commands

```bash
# Calibration check
python -c "from ml.calibration import EnvironmentalCalibration; c = EnvironmentalCalibration(); c.load('data/GAMBIT/gambit_calibration.json'); print(c.get_quality_report())"

# Dataset summary
python -m ml.data_loader data/GAMBIT

# Clustering exploration
python -m ml.train --data-dir data/GAMBIT --cluster-only --n-clusters 5 --visualize-clusters

# Training
python -m ml.build all --data-dir data/GAMBIT --version v0.1 --epochs 50

# Visualization
python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations
python -m ml.generate_explorer --data-dir data/GAMBIT
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-09  
**Author**: Analysis based on SIMCAP codebase review

---

<link rel="stylesheet" href="../../src/simcap.css">
