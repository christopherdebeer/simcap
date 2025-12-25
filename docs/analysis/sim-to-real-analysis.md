# Simulation-to-Real Data Analysis Report

## Executive Summary

Analysis of 17 real sessions (9,050 samples) compared with physics-based synthetic data reveals actionable gaps in both data collection and model design. The primary finding is that **all real data lacks finger-level labels**, making supervised learning impossible without annotation.

## Data Inventory

| Metric | Value |
|--------|-------|
| Total Sessions | 17 |
| Total Samples | 9,050 |
| Samples with labels | 0 (all "unknown") |
| Calibration confidence > 50% | 18.9% |
| Calibration confidence > 80% | 4.5% |
| Motion (moving) samples | 81.3% |

## Key Findings

### 1. Missing Ground Truth Labels

**Critical Issue**: All 17 sessions have `fingers: "unknown"` for all five fingers.

```
THUMB: unknown: 15 sessions
INDEX: unknown: 15 sessions
MIDDLE: unknown: 15 sessions
RING: unknown: 15 sessions
PINKY: unknown: 15 sessions
```

**Impact**: Cannot train supervised finger-state classifier without labels.

**Recommendation**:
- Add labeled data collection mode with prompted gestures
- Consider video-based annotation pipeline
- Explore semi-supervised learning approaches

### 2. Sim-to-Real Field Distribution Gap

| Metric | Real | Synthetic | Gap |
|--------|------|-----------|-----|
| Mean (μT) | 70.1 | 53.4 | -24% |
| Std (μT) | 25.8 | 10.3 | -60% |
| Min (μT) | 3.8 | 20.3 | +433% |
| Max (μT) | 202.5 | 84.7 | -58% |
| CV | 37% | 19% | -49% |

**Key Observations**:
1. Real data has 2.5x more variability than synthetic
2. Synthetic data missing extreme values (close-range poses)
3. Real X-axis has +31 μT offset (hard iron bias)

### 3. Hard Iron Bias

Per-axis analysis reveals significant offsets in real data:

| Axis | Real Mean | Synthetic Mean | Offset |
|------|-----------|----------------|--------|
| X | +31.0 μT | +24.4 μT | +6.6 μT |
| Y | -4.8 μT | -11.6 μT | +6.8 μT |
| Z | +0.1 μT | +33.3 μT | -33.2 μT |

The X-axis shows persistent positive bias, indicating hard iron contamination from device components.

### 4. Calibration Quality

```
Confidence Distribution:
  0.00-0.25: 46.7% (poor)
  0.25-0.50: 28.6% (marginal)
  0.50-0.75: 19.5% (acceptable)
  0.75-1.00:  5.3% (good)
```

**Only 4.5% of samples have high-quality calibration.**

Earth field magnitude from calibration: **52.7 μT** (expected: ~50 μT for Edinburgh) ✓

### 5. Temporal Dynamics

Real data shows high temporal variability:

| Gradient | Mean | Std | Range |
|----------|------|-----|-------|
| ΔBx | 0.04 | 6.22 | [-31.6, 39.5] μT/sample |
| ΔBy | 0.02 | 7.95 | [-53.6, 62.0] μT/sample |
| ΔBz | 0.02 | 12.20 | [-58.9, 92.0] μT/sample |

Mean rate of change: **3.63 μT/sample** indicates active finger movement during data collection.

---

## Recommendations

### Data Collection Improvements

#### Priority 1: Add Labeled Data Collection (Critical)

1. **Prompted Gesture Mode**
   - Display target gesture on screen
   - Record 3-5 seconds of static pose
   - Minimum: 10 repetitions per gesture × 8 gestures × 3 subjects = 240 labeled sequences

2. **Gesture Set for Training**
   ```
   Static poses (3 states per finger):
   - open_palm (all extended)
   - fist (all flexed)
   - pointing (index extended, others flexed)
   - peace (index+middle extended)
   - thumbs_up (thumb extended)
   - ok_sign (thumb+index touching)
   ```

3. **Label Format Enhancement**
   ```json
   {
     "fingers": {
       "thumb": "extended|partial|flexed",
       "index": "extended|partial|flexed",
       ...
     },
     "pose_name": "pointing",
     "confidence": 0.95
   }
   ```

#### Priority 2: Improve Calibration Quality

1. **Require calibration before data collection**
   - Block recording until `mag_cal_confidence > 0.7`
   - Show calibration progress indicator

2. **Record hard iron offset vector**
   - Store calibrated `hard_iron: [x, y, z]` in session metadata
   - Use for simulation domain randomization

3. **Calibration procedure guidance**
   - Prompt user to rotate device in figure-8 pattern
   - Visual feedback for coverage quality

#### Priority 3: Increase Domain Diversity

1. **Multiple subjects** (currently appears single-subject)
2. **Multiple device orientations** (current: wide range ✓)
3. **Multiple environments** (indoor/outdoor)
4. **Multiple hand sizes** (small/medium/large)

### Simulation Improvements

#### 1. Increase Field Variability

Current simulation produces narrow distribution (std=10.3 vs real std=25.8).

**Changes needed in `sensor_model.py`**:
```python
# Increase hard iron bias range
hard_iron_range = (20, 60)  # was (10, 40)

# Add soft iron distortion
soft_iron_scale = np.random.uniform(0.8, 1.2)

# Increase noise
noise_range = (3, 12)  # was (2, 8)
```

#### 2. Add Close-Range Poses

Real data shows magnitudes up to 202 μT, but simulation maxes at 85 μT.

**Add poses with fingers closer to sensor**:
- `fist_tight`: fingers curled tight against palm
- `finger_tap`: single finger touching sensor position
- `pinch_close`: thumb-index pinch near wrist

#### 3. Add Temporal Dynamics

Current simulation generates static samples. Real data has rapid changes.

**Add to `generator.py`**:
```python
def generate_motion_sequence(self, start_pose, end_pose, duration_samples=50):
    """Generate smooth transition with realistic motion dynamics."""
    # Add acceleration/deceleration profile
    # Add small random perturbations during motion
```

### Model Architecture Improvements

#### 1. Add Gradient Features

Include first-order derivatives as input features:

```python
features = [
    mx, my, mz,           # Field values
    dmx, dmy, dmz,        # First derivatives
    accel_std, gyro_std,  # Motion indicators
]
```

#### 2. Consider Temporal Models

Given high temporal variability (3.63 μT/sample mean change):

- **LSTM/GRU**: Better for sequence modeling
- **1D CNN**: Good for local temporal patterns
- **Transformer**: Best for long-range dependencies

#### 3. Domain Adaptation Layer

Add domain-adversarial training to bridge sim-to-real gap:

```python
class DomainAdaptiveModel:
    def __init__(self):
        self.feature_extractor = ...  # Shared
        self.finger_classifier = ...   # Task head
        self.domain_classifier = ...   # Domain discriminator
```

#### 4. Self-Supervised Pre-training

Use unlabeled real data for pre-training:
- **Contrastive learning**: Learn representations from temporal consistency
- **Masked prediction**: Predict masked sensor values
- **Next-sample prediction**: Predict future field values

---

## Immediate Action Items

1. **Create labeled dataset** (2-3 hours of data collection)
   - 8 gestures × 20 repetitions × 50 samples = 8,000 labeled samples

2. **Update simulation parameters** to increase variability
   - Hard iron: 20-60 μT
   - Noise: 3-12 μT
   - Add close-range poses

3. **Train baseline model** on synthetic data, evaluate on real
   - Measure zero-shot transfer accuracy
   - Identify systematic failure modes

4. **Implement fine-tuning pipeline** for labeled real data
   - Pre-train on synthetic → Fine-tune on real

---

## Unsupervised Learning Results

Given the absence of labeled real data, we evaluated three approaches for leveraging labeled synthetic data:

### Approach Comparison

| Approach | Entropy | Diversity | Domain Distance | Status |
|----------|---------|-----------|-----------------|--------|
| Zero-shot transfer | 0.18 | Low | 419.0 | ✗ Failed |
| Contrastive pre-training | 0.58 | High | 10.2 | ✓ **Best** |
| Domain adaptation | 0.36 | Medium | 4.2 | ○ Moderate |

### 1. Zero-shot Transfer (Baseline)

**Method**: Train directly on labeled synthetic data, evaluate on real data.

**Result**: Complete failure - model collapses to single prediction ("flexed" for all fingers).
- Prediction entropy: 0.18 (near minimum diversity)
- Domain distance: 419 (massive embedding gap)
- All fingers predicted as 100% flexed

**Conclusion**: The sim-to-real gap is too large for direct transfer.

### 2. Contrastive Pre-training (Best Approach)

**Method**: SimCLR-style self-supervised pre-training on unlabeled real data, then fine-tune on synthetic labels.

**Result**: Significant improvement in prediction diversity and domain alignment.
- Prediction entropy: 0.58 (3.2x improvement)
- Domain distance: 10.2 (40x reduction)
- Diverse predictions across states:
  ```
  thumb : ext:44%, part:51%, flex:4%
  index : ext:1%, part:42%, flex:57%
  middle: part:2%, flex:98%
  ring  : part:4%, flex:95%
  pinky : ext:38%, part:18%, flex:44%
  ```

**Why it works**: Contrastive learning discovers the natural structure of the real data distribution, creating representations that transfer better from synthetic labels.

### 3. Domain Adaptation

**Method**: Adversarial training with domain discriminator to learn domain-invariant features.

**Result**: Good domain alignment but partial mode collapse.
- Prediction entropy: 0.36 (2x improvement)
- Domain distance: 4.2 (100x reduction)
- Some fingers still show collapse (ring: 99% flexed)

**Conclusion**: Effective for domain alignment but less stable than contrastive pre-training.

### Recommended Pipeline

Based on these results, the recommended training pipeline is:

```python
from ml.unsupervised_learning import ContrastiveLearning, load_real_data, generate_synthetic_data

# 1. Load data
real_samples, _ = load_real_data()
syn_samples, syn_labels, _ = generate_synthetic_data(n_sessions=30)

# 2. Contrastive pre-training on real data
cl = ContrastiveLearning(latent_dim=32, projection_dim=16)
cl.pretrain(real_samples, epochs=100)

# 3. Fine-tune classifier on synthetic labels
cl.finetune_on_synthetic(syn_samples, syn_labels, epochs=30)

# 4. Predict on real data
predictions = cl.predict(real_samples)
```

### Next Steps

1. **Collect labeled validation set** (100-200 samples with known gestures)
   - Required to measure actual accuracy, not just diversity

2. **Ensemble approaches** - Combine contrastive + domain adaptation

3. **Temporal modeling** - Current model is sample-independent
   - Add LSTM/Transformer for sequence context

4. **Active learning** - Use model uncertainty to select samples for labeling

---

## Appendix: Field Distribution Comparison

![Sim-to-Real Comparison](../../images/sim_to_real_comparison.png)

See `images/sim_to_real_comparison.png` for visual comparison of field distributions.
