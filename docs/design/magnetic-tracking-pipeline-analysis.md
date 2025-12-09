# Magnetic Finger Tracking Pipeline: Calibration, Filtering & Model Training

## Executive Summary

This document provides a deep analysis of the GAMBIT system's magnetic finger tracking capabilities, focusing on the **calibration, filtering, and non-generative modeling pipeline** for magnetic dipole-based hand pose estimation. It examines the physical constraints, current implementation status, and identifies critical gaps between the implemented infrastructure and the requirements for magnetic finger tracking described in `magnetic-finger-tracking-analysis.md`.

**Key Finding**: The system has **excellent calibration and filtering infrastructure** (80% complete) but lacks **magnetic-specific data collection workflows** and **SNR characterization tools** needed to validate the magnetic finger tracking approach before scaling to full gesture recognition.

---

## 1. Physical Foundation: Magnetic Dipole Tracking

### 1.1 Signal Characteristics

Per `magnetic-finger-tracking-analysis.md`:

| Magnet Size | Distance | Expected Signal | LIS3MDL Noise | SNR |
|------------|----------|-----------------|---------------|-----|
| 6mm × 3mm N48 | 50mm (flexed) | 141 μT | 1.0 μT | 141:1 |
| 6mm × 3mm N48 | 80mm (extended) | 34 μT | 1.0 μT | 34:1 |
| 5mm × 2mm N42 | 80mm (extended) | 14 μT | 1.0 μT | 14:1 |

**Critical Observation**: Signal strength falls as **1/r³**, making distance sensitivity extreme:
- Flexed finger (50mm): 141 μT → High SNR, easy detection
- Extended finger (80mm): 34 μT → Marginal SNR, requires excellent calibration
- Full extension (100mm): 18 μT → Challenging, Earth field dominates

**Dominant Noise Source**: Earth's magnetic field variation (~10 μT from orientation changes) exceeds sensor noise (~1 μT), making **calibration essential, not optional**.

### 1.2 Multi-Finger Superposition Challenge

Magnetic fields superpose linearly:
```
B_total = B_earth + B_thumb + B_index + B_middle + B_ring + B_pinky + B_noise
```

**Problem**: With same polarity, individual finger contributions blend:
- All extended: 5 × 7 μT ≈ 35 μT aggregate
- One finger flexes: Change masked by 4 extended fingers

**Solution** (per analysis): **Alternating polarity**
```
Index:  N toward palm (+)
Middle: N away from palm (-)
Ring:   N toward palm (+)
Pinky:  N away from palm (-)
Thumb:  N toward palm (+)
```

This creates unique vector signatures for each finger, enabling the 3D magnetometer to distinguish individual finger movements.

---

## 2. Current Implementation Assessment

### 2.1 Calibration System (✅ COMPLETE)

#### JavaScript Implementation (`src/web/GAMBIT/calibration.js`)

**Classes Implemented:**
- `Matrix3`: 3×3 matrix operations
- `Quaternion`: Orientation tracking
- `EnvironmentalCalibration`: Full calibration pipeline

**Calibration Types Supported:**

| Type | Purpose | Algorithm | Quality Metric |
|------|---------|-----------|----------------|
| **Earth Field** | Subtract constant background | `mean(samples)` | std < 1.0 μT |
| **Hard Iron** | Remove offset from metal | `(max + min) / 2` | sphericity > 0.7 |
| **Soft Iron** | Correct field distortion | Eigenvalue decomposition | ratio > 0.5 |

**Data Preservation Philosophy**: ✅ Correct
```javascript
// Raw data NEVER modified
{
  mx: 45.2, my: -12.3, mz: 88.1,  // RAW (preserved)
  calibrated_mx: 42.1,             // DECORATED
  filtered_mx: 42.3                // DECORATED
}
```

#### Python Implementation (`ml/calibration.py`)

**Functionality**: Mirrors JavaScript implementation for ML pipeline consistency
- `EnvironmentalCalibration` class with identical algorithms
- `decorate_telemetry_with_calibration()` adds calibrated fields
- Integration with `ml/data_loader.py` for automatic decoration

**Status**: ✅ Complete and consistent between web and Python

### 2.2 Filtering System (✅ COMPLETE)

#### Kalman Filter (`ml/filters.py:16-174`)

**Architecture:**
```python
State: [x, y, z, vx, vy, vz]  # 6D per finger
Measurement: [x, y, z]          # Position only
Motion Model: Constant velocity
```

**Performance** (documented in `calibration-filtering-guide.md:240-247`):
- Noise reduction: **8× improvement** (3.2 μT → 0.4 μT RMS)
- SNR improvement: **+14 dB** (8 dB → 22 dB)
- Latency: ~20ms at 50Hz (acceptable)

**Tuning Parameters:**
- `processNoise: 0.1` - Default works for most gestures
- `measurementNoise: 1.0` - Matches LIS3MDL noise floor
- Recommendation: Lower to 0.5 if in clean environment

#### Particle Filter (`ml/filters.py:176-317`)

**Purpose**: Multi-hypothesis tracking for ambiguous poses
- 500 particles tracking full 5-finger pose
- Systematic resampling when effective sample size drops
- `magnetic_dipole_field()` implements dipole physics equation
- `magnetic_likelihood()` computes probability of measurement given pose

**Status**: ✅ Implemented but **not yet integrated** with data collection workflow

### 2.3 Data Pipeline (✅ MOSTLY COMPLETE)

#### Data Loader (`ml/data_loader.py`)

**Automatic Decoration Pipeline** (lines 48-124):
```python
load_session_data(json_path, 
                  apply_calibration=True,   # Decorates with calibrated_mx
                  apply_filtering=True)      # Decorates with filtered_mx
```

**Search Order** for calibration files:
1. `{data_dir}/gambit_calibration.json` (per-dataset)
2. `~/.gambit/calibration.json` (user-global)
3. No calibration (uses raw data)

**Feature Priority** (line 113-115):
```python
mx = sample.get('filtered_mx', 
     sample.get('calibrated_mx', 
     sample.get('mx', 0)))  # Best → Raw fallback
```

**Multi-Label Support** (V2 format):
- `load_multilabel_sessions()` - Load multiple label columns
- `load_finger_tracking_sessions()` - Convenience for 5-finger tracking
- `labels_from_segments_v2()` - Convert segments to label matrix

**Status**: ✅ Complete for standard gesture classification

### 2.4 Model Architecture (`ml/model.py`)

**Current Model**: 1D CNN for gesture classification
```
Input: (50 timesteps, 9 features)
├── Conv1D(32) → BatchNorm → ReLU → MaxPool → Dropout
├── Conv1D(64) → BatchNorm → ReLU → MaxPool → Dropout
├── Conv1D(64) → BatchNorm → ReLU
├── GlobalAvgPool1D
├── Dense(64) → ReLU → Dropout
└── Dense(10) → Softmax
```

**Parameters**: ~37K trainable
**Size**: ~75KB quantized (TFLite)
**Inference**: 5-15ms browser, 15-50ms ESP32

**Status**: ✅ Excellent for static pose classification

### 2.5 Web Inference (`gesture-inference.js`)

**Features**:
- TensorFlow.js model loading
- Real-time windowing (50 samples)
- Z-score normalization using training stats
- Confidence thresholding
- Performance tracking

**Status**: ✅ Complete for deployed models

---

## 3. Critical Gaps for Magnetic Finger Tracking

### 3.1 Missing: Calibration Wizard UI Integration

**Status**: Algorithms exist but **no UI workflow**

**What Exists**:
- ✅ `src/web/GAMBIT/calibration.js` - Full calibration algorithms
- ✅ `src/web/GAMBIT/collector.html` - Data collection UI

**What's Missing**:
- ❌ Calibration wizard button in collector.html
- ❌ Step-by-step calibration instructions
- ❌ Real-time quality feedback (sphericity, coverage)
- ❌ Visual feedback for rotation coverage (3D scatter)
- ❌ localStorage → file export workflow

**Impact**: Users cannot perform calibration before data collection, resulting in poor SNR and Earth field interference.

**Recommended Implementation**:
```html
<!-- Add to collector.html -->
<section id="calibration-wizard" class="wizard-modal">
  <h2>Calibration Wizard</h2>
  
  <div id="step-1" class="wizard-step">
    <h3>Step 1: Earth Field Calibration</h3>
    <p>Hold device still, away from magnets (>50cm)</p>
    <button id="start-earth-cal">Start (5 seconds)</button>
    <div id="earth-quality">Quality: <span></span></div>
  </div>
  
  <div id="step-2" class="wizard-step">
    <h3>Step 2: Hard Iron Calibration</h3>
    <p>Rotate device in figure-8 pattern covering all orientations</p>
    <canvas id="coverage-viz"></canvas>
    <button id="start-hard-iron">Start (10 seconds)</button>
    <div id="hard-quality">Sphericity: <span></span>, Coverage: <span></span></div>
  </div>
  
  <!-- Similar for soft iron -->
</section>
```

### 3.2 Missing: SNR Characterization Tools

**Purpose**: Validate magnetic signal detectability before full data collection

**What's Needed**:
```python
# ml/analyze_snr.py (NEW FILE)
def analyze_magnetic_snr(data_dir, finger='index'):
    """
    Compute SNR for finger magnet signals.
    
    Returns:
        {
            'signal_extended': float,  # Field magnitude at extension
            'signal_flexed': float,    # Field magnitude at flexion
            'signal_delta': float,     # Change in signal
            'noise_floor': float,      # RMS noise during static
            'snr_extended': float,     # Signal/noise at extension
            'snr_flexed': float,       # Signal/noise at flexion
        }
    """
    dataset = GambitDataset(data_dir)
    
    # Filter for SNR test sessions (custom label: "snr_test")
    # Extract segments with motion="static"
    # Compute |B| = sqrt(calibrated_mx² + calibrated_my² + calibrated_mz²)
    # Separate extended vs flexed segments
    # Calculate statistics
```

**Usage**:
```bash
python -m ml.analyze_snr --data-dir data/GAMBIT --finger index --plot
# Output:
#   Signal (extended): 32.4 μT
#   Signal (flexed): 138.7 μT
#   Signal delta: 106.3 μT
#   Noise floor: 0.8 μT
#   SNR (extended): 40.5:1 ✓ GOOD
#   SNR (flexed): 173.4:1 ✓ EXCELLENT
```

**Impact**: Without this, cannot validate physics predictions or determine if magnet size is adequate.

### 3.3 Missing: Magnet Configuration Procedure

**What's Needed**: Documentation for:
1. **Magnet Specifications**
   - Recommended: 6mm × 3mm N48 neodymium disc
   - Minimum viable: 5mm × 2mm N42
   - Purchase links

2. **Attachment Method**
   - Ring mounting (preferred)
   - Adhesive mounting (prototyping)
   - Safety warnings

3. **Polarity Assignment**
   - Per-finger polarity table (see Section 1.2)
   - Polarity testing procedure (using compass or sensor)
   - Recording in session metadata

4. **Calibration Protocol**
   - Magnet baseline (no finger movement)
   - Per-finger range of motion test
   - Full 5-finger superposition test

**Impact**: Without standardized procedure, data collection will be inconsistent across sessions.

### 3.4 Missing: Magnetic-Specific Data Loader Stats

**Current Issue**: `ml/data_loader.py:143-171` computes global normalization stats across **all** data

**Problem for Magnetic Tracking**:
- Stats computed **without finger magnets** (calibration sessions)
- Stats computed **with finger magnets** (tracking sessions)
- **Different magnitude scales** → Poor normalization

**Recommended Solution**:
```python
# ml/data_loader.py enhancement
def compute_dataset_stats(data_dir: Path, with_magnets: bool = False):
    """
    with_magnets=False: Calibration sessions only
    with_magnets=True: Finger magnet tracking sessions
    """
    for json_path in data_dir.glob('*.json'):
        meta = load_session_metadata(json_path)
        
        # Filter based on magnet config in metadata
        if with_magnets:
            if not meta or not meta.magnet_config:
                continue  # Skip non-magnet sessions
        else:
            if meta and meta.magnet_config:
                continue  # Skip magnet sessions
        
        # Compute stats...
```

**Impact**: Model sees incorrectly normalized data, reducing accuracy.

### 3.5 Missing: Multi-Output Finger Tracking Model

**Current Model**: Single output (10 gesture classes)

**Needed for Magnetic Tracking**: Multi-output (5 fingers × 3 states each)

```python
# ml/model.py - Add new model type
def create_finger_tracking_model_keras(
    window_size: int = 50,
    num_features: int = 9,
    num_states: int = 3  # extended, partial, flexed
) -> keras.Model:
    """
    Multi-output model for per-finger state prediction.
    
    Architecture:
        Input: (50, 9)
        ├── Shared CNN feature extraction
        └── 5 output heads (one per finger)
            ├── thumb_state: 3 classes
            ├── index_state: 3 classes
            ├── middle_state: 3 classes
            ├── ring_state: 3 classes
            └── pinky_state: 3 classes
    """
    inputs = keras.Input(shape=(window_size, num_features))
    
    # Shared feature extraction
    x = layers.Conv1D(32, 5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Per-finger output heads
    outputs = {}
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        out = layers.Dense(16, activation='relu')(x)
        out = layers.Dropout(0.3)(out)
        out = layers.Dense(num_states, activation='softmax', name=f'{finger}_state')(out)
        outputs[finger] = out
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Multi-output loss
    model.compile(
        optimizer='adam',
        loss={f'{finger}_state': 'sparse_categorical_crossentropy' 
              for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']},
        metrics={f'{finger}_state': 'accuracy' 
                 for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']}
    )
    
    return model
```

**Training**:
```python
# ml/train.py - Add multi-output training
X, y_multi = dataset.load_finger_tracking_sessions()
# y_multi shape: (num_windows, 5) - one column per finger

# Reshape for Keras multi-output
y_dict = {
    'thumb_state': y_multi[:, 0],
    'index_state': y_multi[:, 1],
    'middle_state': y_multi[:, 2],
    'ring_state': y_multi[:, 3],
    'pinky_state': y_multi[:, 4]
}

model.fit(X, y_dict, epochs=50, validation_split=0.2)
```

**Impact**: Cannot train per-finger models without this architecture.

---

## 4. Recommended Magnetic Tracking Pipeline

### Phase 0: Pre-Flight Validation (NEW)

**Goal**: Validate physics predictions before full data collection

#### Step 1: Hardware Setup
```
Magnet: 6mm × 3mm N48 neodymium disc
Attachment: Ring on index finger
Orientation: N pole toward palm
Device: Puck.js with GAMBIT firmware
```

#### Step 2: Environment Calibration
```
1. Run calibration wizard (MISSING - TO BE IMPLEMENTED)
2. Verify quality metrics:
   - Earth field std < 1.0 μT
   - Sphericity > 0.7
   - Coverage > 0.7
3. Export calibration to data/GAMBIT/gambit_calibration.json
```

#### Step 3: SNR Characterization Session
```javascript
// In collector.html
Session Type: "Phase 0 - SNR Test"
Labels:
  - calibration: "finger_range"
  - motion: "static"
  - fingers: index="extended", others="unknown"
  - custom: ["snr_test", "index_only"]

Protocol:
1. Index extended (far) - hold 5 seconds
2. Index flexed (near) - hold 5 seconds
3. Repeat 10 times
4. Export session
```

#### Step 4: SNR Analysis
```bash
python -m ml.analyze_snr --data-dir data/GAMBIT --finger index --plot
# Expected output:
#   SNR (extended): 20-40:1
#   SNR (flexed): 100-200:1
#   Signal delta: > 50 μT
```

**Success Criteria**:
- SNR > 10:1 at extended position → Proceed to Phase 1
- SNR < 10:1 at extended → Use larger magnet or improve calibration

### Phase 1: Single-Finger Validation

**Goal**: Prove magnetic tracking concept with one finger

**Data Collection**:
- 10 sessions × 3 minutes each
- Labels: `index_extended`, `index_flexed`, `index_partial`
- 30 windows per session → 300 total windows

**Model Training**:
```bash
python -m ml.train --data-dir data/GAMBIT --model-type classification --epochs 30
```

**Expected Accuracy**: > 90% (3-class problem is easy with good SNR)

### Phase 2: Multi-Finger Pose Classification

**Goal**: 16-32 static poses with 5 fingers

**Magnet Configuration**:
```
ALL 5 FINGERS with alternating polarity:
  Thumb: N → palm (+)
  Index: N → palm (+)
  Middle: N ← away (-)
  Ring: N → palm (+)
  Pinky: N ← away (-)
```

**Data Collection**:
```javascript
Predefined poses:
  pose_00000 (all extended)
  pose_22222 (all flexed)
  pose_10000 (thumb only)
  // ... 32 total

Per pose:
  - 3 sessions × 10 reps × 3 seconds
  - motion: "static"
  - fingers: individual states
```

**Model Training**:
```bash
python -m ml.train --data-dir data/GAMBIT --model-type finger_tracking --epochs 50
```

**Expected Accuracy**: 70-85% per-finger state prediction

### Phase 3: Filtered Real-Time Tracking

**Integration**:
```javascript
// gesture-inference.js
class MagneticFingerInference extends GestureInference {
    constructor(options) {
        super(options);
        this.calibration = loadCalibration();
        this.kalmanFilter = new KalmanFilter3D();
    }
    
    addSample(sample) {
        // 1. Apply calibration
        const calibrated = this.calibration.correct({
            x: sample.mx, y: sample.my, z: sample.mz
        });
        
        // 2. Apply Kalman filter
        const filtered = this.kalmanFilter.update(calibrated);
        
        // 3. Override sample with filtered values
        sample.mx = filtered.x;
        sample.my = filtered.y;
        sample.mz = filtered.z;
        
        // 4. Run inference
        super.addSample(sample);
    }
}
```

---

## 5. Key Contrasts: Standard Gestures vs. Magnetic Tracking

| Aspect | Standard Gesture Recognition | Magnetic Finger Tracking |
|--------|----------------------------|-------------------------|
| **Sensors Used** | Accel + Gyro (6-DoF) | Accel + Gyro + **Mag (9-DoF)** |
| **Signal Source** | Hand motion dynamics | **Magnetic dipole field** |
| **Calibration Importance** | Optional (improves accuracy) | **Critical (enables detection)** |
| **Dominant Noise** | Motion blur (~1g RMS) | **Earth field (~25-65 μT)** |
| **SNR Challenge** | High (10-100:1) | **Moderate (10-40:1 extended)** |
| **Distance Sensitivity** | Linear (1/r) | **Cubic (1/r³)** |
| **Multi-Finger Sensing** | Indirect (joint kinematics) | **Direct (superposition)** |
| **Polarity Matters** | N/A | **Essential (alternating)** |
| **Filtering Impact** | 2× improvement | **8× improvement** |
| **Model Architecture** | 1D CNN (current) | Multi-output CNN (needed) |
| **Validation Method** | Confusion matrix | **SNR analysis + confusion** |

---

## 6. Non-Generative Modeling Justification

### Why Non-Generative for Magnetic Tracking?

**Generative Models** (VAE, Diffusion, Flow-based):
- Learn p(x, y) - joint distribution of data and labels
- Can generate synthetic data
- Require large datasets (10K+ samples)
- Slower training and inference

**Discriminative Models** (CNN, Random Forest, SVM):
- Learn p(y|x) - direct mapping from features to labels
- Simpler, faster, more data-efficient
- Better for low-data regimes

**Magnetic Tracking Constraints**:
1. **Limited data** (hundreds to thousands of samples, not millions)
2. **Real-time inference required** (< 50ms latency)
3. **Embedded deployment target** (ESP32 with 520KB RAM)
4. **Clear decision boundaries** (pose classes are well-separated in magnetic field space)

**Recommendation**: Discriminative (non-generative) approach is **optimal** for:
- Phase 1-2: Static pose classification
- Phase 3: Per-finger state prediction
- Phase 4: Consider temporal models (LSTM, Transformer) but still discriminative

---

## 7. Filtering Strategy: Kalman vs. Particle Filter

### When to Use Kalman Filter

**Best For**:
- Single-finger tracking (Phase 1)
- Low-ambiguity poses (Phase 2)
- Real-time inference with tight latency

**Advantages**:
- Fast (< 1ms per filter update)
- Optimal for Gaussian noise
- Simple to tune

**Implementation**:
```python
from ml.filters import KalmanFilter3D

mag_filter = KalmanFilter3D(
    process_noise=0.1,      # Finger movement variability
    measurement_noise=1.0,   # Sensor noise floor
    dt=0.02                 # 50 Hz sampling
)

for telemetry in stream:
    filtered = mag_filter.update({
        'x': telemetry['calibrated_mx'],
        'y': telemetry['calibrated_my'],
        'z': telemetry['calibrated_mz']
    })
```

### When to Use Particle Filter

**Best For**:
- Multi-finger tracking (Phase 3-4)
- Ambiguous poses (similar field patterns)
- Non-Gaussian noise (environmental interference)

**Advantages**:
- Handles multimodal distributions
- Incorporates physics model (dipole likelihood)
- More robust to outliers

**Disadvantages**:
- Slower (500 particles × likelihood = 5-10ms)
- Needs magnet configuration
- More complex to tune

**Implementation**:
```python
from ml.filters import ParticleFilter, magnetic_likelihood

pf = ParticleFilter(num_particles=500)

# Define magnet configuration with alternating polarity
magnet_config = {
    'thumb': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},   # N → palm
    'index': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},   # N → palm
    'middle': {'moment': {'x': 0, 'y': 0, 'z': -0.01}}, # N ← away
    'ring': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},    # N → palm
    'pinky': {'moment': {'x': 0, 'y': 0, 'z': -0.01}}   # N ← away
}

pf.initialize(initial_pose={...})

for telemetry in stream:
    pf.predict(dt=0.02)
    pf.update(
        measurement={'x': telemetry['calibrated_mx'], ...},
        likelihood_fn=lambda particle, meas: 
            magnetic_likelihood(particle, meas, magnet_config)
    )
    estimated_pose = pf.estimate()
```

**Recommendation**:
- **Phase 1-2**: Kalman filter (simpler, faster, sufficient)
- **Phase 3-4**: Experiment with both, compare accuracy/latency

---

## 8. Critical Path Summary

### Infrastructure Status

| Component | Status | Blocking? | Action |
|-----------|--------|-----------|--------|
| Calibration algorithms | ✅ Complete | No | - |
| Calibration UI | ❌ Missing | **YES** | Implement wizard |
| Filtering algorithms | ✅ Complete | No | - |
| Data loader | ✅ Complete | No | - |
| Single-output model | ✅ Complete | No | - |
| Multi-output model | ❌ Missing | **YES** | Implement finger tracking |
| SNR analysis tool | ❌ Missing | **YES** | Create analyze_snr.py |
| Magnet procedure docs | ❌ Missing | **YES** | Document setup |
| Web inference | ✅ Complete | No | Enhance with filtering |

### Immediate Priorities

**Week 1 (Unblock Data Collection)**:
1. ✅ Calibration algorithms exist → ❌ **Add wizard UI to collector.html**
2. ❌ **Document magnet attachment procedure** (sizes, polarities, safety)
3. ❌ **Create SNR analysis script** (`ml/analyze_snr.py`)

**Week 2 (Validate Physics)**:
4. Collect Phase 0 SNR test data (1 finger, 10 sessions)
5. Run SNR analysis, verify 10-40:1 SNR at extension
6. If adequate, proceed to Phase 1; if not, debug calibration

**Week 3-4 (Scale to Multi-Finger)**:
7. Implement multi-output finger tracking model
8. Collect Phase 2 data (5 fingers, 32 poses)
9. Train and evaluate per-finger accuracy

### Success Metrics

| Phase | Metric | Target | Indicates |
|-------|--------|--------|-----------|
| Phase 0 | SNR (extended) | > 10:1 | Magnetic tracking viable |
| Phase 1 | Single-finger accuracy | > 90% | Good signal quality |
| Phase 2 | Per-finger accuracy | > 70% | Multi-finger distinguishable |
| Phase 3 | Real-time latency | < 50ms | Production-ready |

---

## 9. Comparison with gambit-e2e-process-analysis.md

### Complementary Focus Areas

| Document | Focus | Audience | Detail Level |
|----------|-------|----------|--------------|
| **gambit-e2e-process-analysis.md** | Standard gesture recognition workflow | Users/collectors | Operational |
| **This document** | Magnetic tracking physics & pipeline | Developers/researchers | Technical |

### Key Differences

#### 1. Calibration Emphasis

**E2E Process** (gambit-e2e-process-analysis.md:124-155):
- Calibration is Step 1 of data collection
- Focus on procedure and quality thresholds
- **Assumes calibration wizard exists**

**This Document**:
- Calibration is **prerequisite** for magnetic tracking
- Deep dive into algorithms and physics
- **Identifies missing wizard UI as critical gap**

#### 2. Data Collection Scope

**E2E Process** (lines 159-203):
- 5 standard gestures (rest, fist, open_palm, index_up, peace)
- Target: 500 windows per gesture
- Focus: Getting labeled data quickly

**This Document**:
- **Phase 0**: SNR validation (new concept)
- **Phase 1**: Single-finger proof-of-concept
- **Phase 2**: 32 multi-finger poses with polarity assignments
- Focus: Validating magnetic tracking feasibility

#### 3. Model Requirements

**E2E Process** (lines 242-267):
- Uses existing 1D CNN (10-class classifier)
- TensorFlow.js + TFLite export
- Standard training pipeline

**This Document**:
- **New requirement**: Multi-output model for per-finger prediction
- Alternating polarity magnet configuration
- Magnetic-specific normalization stats

#### 4. Validation Approach

**E2E Process** (lines 269-321):
- Confusion matrix
- Per-class accuracy
- Real-time inference testing

**This Document**:
- **SNR analysis** (new tool needed)
- Per-finger state accuracy
- Magnetic signal characterization
- Filter impact measurement

---

## 10. Recommendations

### For Standard Gesture Recognition Users
→ **Follow gambit-e2e-process-analysis.md**
- No magnets needed
- Focus on IMU motion patterns (accel + gyro)
- Calibration optional but recommended
- Use existing 10-class model

### For Magnetic Finger Tracking Developers
→ **Follow this document**
1. Implement missing infrastructure (wizard, SNR tool, multi-output model)
2. Validate physics with Phase 0 SNR tests
3. Scale progressively: 1 finger → 5 fingers → dynamic tracking
4. Use alternating polarity magnet configuration
5. Apply Kalman filter for smoothing

### Hybrid Approach (Recommended)
**Combine both techniques**:
- Use standard gestures (IMU) for coarse-grained recognition
- Use magnetic tracking for fine-grained finger pose estimation
- Sensor fusion: accel/gyro for orientation, magnetometer for finger states

Example:
```
IMU detects: "Hand is in pointing gesture"
Magnetometer refines: "Index extended, others flexed"
```

---

## Appendix A: Missing Files Checklist

### High Priority
- [ ] `src/web/GAMBIT/collector.html` - Calibration wizard UI integration
- [ ] `ml/analyze_snr.py` - SNR characterization tool
- [ ] `ml/model.py` - Multi-output finger tracking model
- [ ] `docs/procedures/magnet-attachment-guide.md` - Hardware setup documentation

### Medium Priority
- [ ] `ml/data_loader.py` - Magnetic-specific normalization stats
- [ ] `src/web/GAMBIT/gesture-inference.js` - Filtering integration
- [ ] `ml/visualize_magnetic_field.py` - Dipole field visualization

### Low Priority
- [ ] `ml/validate_polarity.py` - Magnet polarity testing tool
- [ ] `src/web/GAMBIT/hand-visualizer.html` - 3D hand pose visualization

---

## Appendix B: Physics Validation Checklist

Before collecting multi-finger data, validate:

- [ ] **SNR adequate**: > 10:1 at extended position (80mm)
- [ ] **Signal delta measurable**: > 20 μT between extended/flexed
- [ ] **Calibration quality**: All metrics in "good" or "excellent" range
- [ ] **Magnet attachment secure**: No movement during gestures
- [ ] **Polarity verified**: Compass test or sensor reading confirms orientation
- [ ] **Superposition observed**: Adding/removing fingers changes field vector

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-09  
**Author**: Deep analysis of SIMCAP magnetic tracking infrastructure  
**Related Documents**:
- `magnetic-finger-tracking-analysis.md` - Physics foundation
- `gambit-e2e-process-analysis.md` - Standard workflow
- `calibration-filtering-guide.md` - Technical reference
- `gambit-workflow-review.md` - Implementation status

---

<link rel="stylesheet" href="../../src/simcap.css">
