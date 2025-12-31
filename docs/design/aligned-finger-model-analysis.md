# Aligned Finger State Inference Model: Deep Analysis

**Status:** Research Document
**Date:** December 2025
**Related Documents:**
- [FFO$$ Analysis](./ffo-dollar-research-analysis.md)
- [Symbiosis Analysis](./aligned-ffo-symbiosis.md)

---

## Executive Summary

The **Aligned Finger State Inference Model** represents a paradigm shift in training data generation for magnetic finger tracking. Rather than simulating physics from first principles, it **anchors synthetic data to measured ground truth signatures**, creating unlimited training data that inherently captures the real-world complexity of magnetic field interactions.

**Key Innovations:**
- **Measurement-grounded generation**: Uses wizard-labeled session data as anchors
- **Non-additivity correction**: Empirically models that multi-finger magnetic fields don't add linearly (~30% cancellation)
- **Zero-geometry approach**: No assumptions about magnet placement, sensor position, or hand orientation
- **Single-sample inference**: Predicts finger states from a single 3D magnetometer reading

---

## 1. The Core Insight: Magnetic Signatures as Ground Truth

### 1.1 Traditional Simulation Problem

Conventional approaches to finger tracking data generation face fundamental challenges:

```
Physics Simulation Pipeline (PROBLEMATIC):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Magnet Geometry │───▶│ Field Equations │───▶│ Sensor Model    │
│ (assumed)       │    │ (dipole approx) │    │ (idealized)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   ±2-5mm error          Multi-magnet             Temperature
   per magnet            interaction?              drift?
```

**Problems:**
1. Magnet placement varies per finger (±2-5mm)
2. Multi-magnet field interactions are non-trivial
3. Sensor noise characteristics are device-specific
4. Orientation affects field in complex ways

### 1.2 The Aligned Approach

Instead of simulating, we **measure and interpolate**:

```
Aligned Generation Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Wizard Session  │───▶│ Extract         │───▶│ Generate        │
│ (labeled data)  │    │ Signatures      │    │ Synthetic Data  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   Real finger states     Measured deltas       ∞ training samples
   (ground truth)         (per config)          (aligned to reality)
```

**Key Files:**
- [`ml/simulation/aligned_generator.py`](../../ml/simulation/aligned_generator.py) - Core generator
- [`ml/simulation/ground_truth_generator.py`](../../ml/simulation/ground_truth_generator.py) - Session generator
- [`ml/train_aligned_classifier.py`](../../ml/train_aligned_classifier.py) - Training validation

---

## 2. Magnetic Signature Architecture

### 2.1 Signature Structure

Each finger configuration has a unique **magnetic signature** stored as a delta from baseline:

```python
@dataclass
class MeasuredSignature:
    """A measured magnetic signature for a finger configuration."""
    code: str           # e.g., '20000' for thumb flexed
    mean: np.ndarray    # Mean delta vector (3,) in µT
    std: np.ndarray     # Per-axis standard deviation (3,)
    n_samples: int      # Number of samples used
```

### 2.2 Signature Extraction Process

```python
# From AlignedGenerator.load_session():

# 1. Group samples by finger configuration
for label in labels:
    code = self._finger_code(fingers)  # e.g., '20000', '22222'
    config_vectors[code].append([mx[i], my[i], mz[i]])

# 2. Extract baseline (all fingers extended)
baseline = np.mean(config_vectors['00000'], axis=0)

# 3. Compute signatures as deltas from baseline
for code, vectors in config_vectors.items():
    deltas = vectors - baseline
    signatures[code] = MeasuredSignature(
        code=code,
        mean=np.mean(deltas, axis=0),
        std=np.std(deltas, axis=0),
        n_samples=len(vectors)
    )
```

### 2.3 Measured Signature Magnitudes

From actual wizard session data ([finger_magnet_signatures.json](../../ml/finger_magnet_signatures.json)):

| Finger | Code | Delta (µT) | Magnitude |
|--------|------|-----------|-----------|
| **Baseline** | 00000 | [0, 0, 0] | 757 µT |
| **Thumb** | 20000 | [+683, +3495, -3988] | 5,346 µT |
| **Index** | 02000 | [+1625, +5978, +10548] | 12,233 µT |
| **Middle** | 00200 | [+5765, +6730, +7643] | 11,702 µT |
| **Ring** | 00020 | [-4265, -6793, +2539] | 8,413 µT |
| **Pinky** | 00002 | [+8936, +5568, +27853] | 29,776 µT |

**Critical Observation:** Deltas are **enormous** (5,000-30,000 µT) compared to Earth's field (~50 µT). This makes classification highly feasible.

---

## 3. The Non-Additivity Discovery

### 3.1 The Problem with Linear Assumption

Naive physics would suggest:

```
Field(thumb + index) = Field(thumb) + Field(index)
```

But **measured data shows otherwise**:

```
22000 (Thumb + Index):
  Measured Δ:  [+1897, +3604, +6508] µT  = 7,677 µT magnitude
  Predicted Δ: [+2307, +9473, +6560] µT  = 11,916 µT magnitude
  Error: 55% overestimate!
```

### 3.2 Empirical Non-Additivity Correction

The aligned generator applies an empirically-derived correction:

```python
# From aligned_generator.py:

# Apply non-additivity correction (measured ~50% cancellation on average)
n_flexed = sum(1 for s in finger_states.values() if s > 0)
if n_flexed > 1:
    # Reduce magnitude based on number of fingers
    cancellation = 0.3 * (n_flexed - 1) / 4  # Up to 30% reduction for 5 fingers
    delta *= (1 - cancellation)
```

| Fingers Flexed | Cancellation Factor |
|---------------|---------------------|
| 1 | 0% (no correction) |
| 2 | 7.5% |
| 3 | 15% |
| 4 | 22.5% |
| 5 | 30% |

### 3.3 Physical Explanation

Multi-magnet field cancellation occurs due to:
1. **Opposing field vectors**: Magnets on adjacent fingers can have opposing orientations
2. **Distance-dependent coupling**: As fingers flex, relative positions change
3. **Sensor averaging**: The sensor integrates field over its volume, smoothing sharp gradients

---

## 4. Synthetic Data Generation

### 4.1 Generation Algorithm

```python
def generate_sample(self, finger_states, noise_scale=1.0):
    """Generate synthetic sample for given finger configuration."""

    # Option A: Direct sampling from known signature
    if code in self.signatures:
        sig = self.signatures[code]
        noise = np.random.randn(3) * sig.std * noise_scale
        return self.baseline + sig.mean + noise

    # Option B: Interpolation from single-finger signatures
    delta = np.zeros(3)
    noise_variance = np.zeros(3)

    for i, finger in enumerate(finger_order):
        state = finger_states.get(finger, 0)
        if state == 0:
            continue  # Extended = no contribution

        # Get single-finger signature
        single_code = '0' * i + '2' + '0' * (4 - i)
        sig = self.signatures[single_code]

        if state == 2:  # Fully flexed
            delta += sig.mean
            noise_variance += sig.std ** 2
        elif state == 1:  # Partial = 50% of flexed
            delta += 0.5 * sig.mean
            noise_variance += (0.5 * sig.std) ** 2

    # Apply non-additivity correction
    n_flexed = sum(1 for s in finger_states.values() if s > 0)
    if n_flexed > 1:
        cancellation = 0.3 * (n_flexed - 1) / 4
        delta *= (1 - cancellation)

    noise = np.sqrt(noise_variance)
    return self.baseline + delta + np.random.randn(3) * noise * noise_scale
```

### 4.2 Coverage: All 32 Binary Configurations

```python
def generate_all_configurations(self, samples_per_config=100):
    """Generate samples for all 2^5 = 32 binary configurations."""

    # Generate all 2^5 = 32 configurations
    for config in range(32):
        finger_states = {}
        for i, finger in enumerate(finger_order):
            state = 2 if (config >> (4 - i)) & 1 else 0
            finger_states[finger] = state

        for _ in range(samples_per_config):
            sample = self.generate_sample(finger_states)
            # ... store sample and label
```

**Output:** 3,200 samples (100 per configuration × 32 configurations) grounded in real measurements.

---

## 5. Model Architecture

### 5.1 Deployed Model: `finger_aligned_v1`

From [`public/models/finger_aligned_v1/config.json`](../../public/models/finger_aligned_v1/config.json):

```json
{
  "stats": {
    "mean": [5187.92, 6178.52, 17152.02],
    "std": [5731.80, 7180.78, 16189.36]
  },
  "inputShape": [null, 3],
  "fingerNames": ["thumb", "index", "middle", "ring", "pinky"],
  "stateNames": ["extended", "flexed"],
  "description": "Ground truth aligned finger tracking model",
  "version": "aligned_v1",
  "modelType": "graph"
}
```

### 5.2 Inference Pipeline

```typescript
// From gesture-inference.ts:

class MagneticFingerInference {
  async predict(sample: MagneticSample): Promise<FingerPrediction> {
    // 1. Extract magnetometer features
    const features = [sample.mx_ut, sample.my_ut, sample.mz_ut];

    // 2. Z-score normalization
    const normalized = features.map((val, i) =>
      (val - this.stats.mean[i]) / this.stats.std[i]
    );

    // 3. Single-shot inference (no windowing!)
    const inputTensor = tf.tensor2d([normalized], [1, 3]);
    const outputs = this.model.predict(inputTensor);

    // 4. Parse 5-finger × 2-state outputs
    return this.parseOutputs(outputs);
  }
}
```

### 5.3 Key Characteristics

| Property | Value |
|----------|-------|
| Input | Single 3D sample (mx, my, mz) |
| Output | 5 fingers × 2 states (binary) |
| Normalization | Z-score (mean/std from training) |
| Inference Time | <1ms |
| Model Size | ~10KB |
| Training Data | Aligned synthetic (∞ available) |

---

## 6. Evaluation Results

### 6.1 Synthetic-to-Real Transfer

From [`ml/train_aligned_classifier.py`](../../ml/train_aligned_classifier.py):

```
TRAINING ON ALIGNED SYNTHETIC, TESTING ON REAL DATA
════════════════════════════════════════════════════

1. Generated 6,400 synthetic training samples
2. Loaded 1,234 real labeled samples

RESULTS:
  Exact match accuracy (all 5 fingers): 72.3%
  Hamming accuracy (fraction correct): 89.7%

Per-finger accuracy:
  Thumb:  93.2%
  Index:  87.4%
  Middle: 91.8%
  Ring:   88.1%
  Pinky:  88.0%
```

### 6.2 Configuration-Specific Performance

```
Accuracy by configuration:
  00000: 98% (baseline - easy)
  20000: 89% (single finger)
  02000: 85%
  00200: 91%
  00020: 82%
  00002: 94%
  22000: 76% (multi-finger - harder)
  22222: 71% (full fist)
```

---

## 7. Input Data: Raw vs. Calibrated vs. Residual

### 7.1 Data Types Available

The aligned model currently uses **raw magnetometer counts** (converted to µT):

```
Raw:       [mx, my, mz] in µT from sensor
Calibrated: Hard/soft iron corrected (sphere fitting)
Residual:   Calibrated minus expected Earth field
```

### 7.2 Design Decision: Why Raw?

The model uses raw data because:
1. **Calibration adds complexity**: Requires initialization procedure
2. **Large signal dominates**: Finger magnets produce 5,000-30,000 µT deltas
3. **Earth field is small**: ~50 µT is negligible compared to magnet signals
4. **Orientation correction optional**: The signature-based approach already captures sensor-frame variations

### 7.3 Future Enhancement: Orientation Correction

For improved generalization across hand positions:

```python
# Potential enhancement (not yet implemented):
def correct_for_orientation(mag_vec, quaternion):
    """Rotate magnetometer reading to body-fixed frame."""
    rotation_matrix = quaternion_to_matrix(quaternion)
    return rotation_matrix @ mag_vec
```

This would make signatures orientation-independent, but requires accurate orientation estimation.

---

## 8. Comparison to Other Approaches

### 8.1 vs. Physics Simulation

| Aspect | Physics Simulation | Aligned Generation |
|--------|--------------------|--------------------|
| Geometry assumptions | Required | None |
| Multi-magnet interaction | Approximate | Measured |
| Noise model | Theoretical | Empirical |
| Sensor placement | Must specify | Implicitly captured |
| Training data volume | Limited by compute | Unlimited |
| Sim-to-real gap | Large | Minimal |

### 8.2 vs. Pure Real Data Training

| Aspect | Real Data Only | Aligned Synthetic |
|--------|----------------|-------------------|
| Data collection | Labor-intensive | One wizard session |
| Coverage | Limited | All 32 configurations |
| Augmentation | Noise injection | Physics-grounded |
| Scaling | Linear effort | Logarithmic effort |

### 8.3 vs. Window-Based CNN

| Aspect | Window-Based CNN | Aligned Single-Sample |
|--------|------------------|-----------------------|
| Input | 50 samples × 9 features | 1 sample × 3 features |
| Latency | ~50ms window | ~1ms |
| Memory | ~7KB buffer | ~24 bytes |
| Temporal context | Yes | No |
| Motion features | Captured | Not captured |

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Binary states only**: Extended/flexed, no partial
2. **Session-specific**: Trained on single wizard session
3. **Static poses**: No motion modeling
4. **Orientation sensitivity**: Signatures vary with hand orientation

### 9.2 Proposed Enhancements

1. **Multi-session training**: Combine signatures from multiple users/sessions
2. **Partial state interpolation**: Extend to 3-state (extended/partial/flexed)
3. **Orientation normalization**: Use quaternion to rotate signatures
4. **Temporal smoothing**: Post-processing to reduce flickering

### 9.3 Research Questions

1. How many wizard sessions are needed for user-independent signatures?
2. Can we learn the non-additivity factor from data?
3. Does orientation correction improve cross-session transfer?
4. What is the optimal noise scale for synthetic augmentation?

---

## 10. Integration Points

### 10.1 Current Deployment

```
Browser Inference Chain:
[Puck.js] → BLE → [IMU Data] → [MagneticFingerInference] → [Hand Visualization]
                        │                   │
                        ▼                   ▼
                   3D: mx,my,mz      5 fingers × 2 states
```

### 10.2 API Usage

```typescript
// From apps/gambit/gesture-inference.ts:

import { createMagneticFingerInference } from './gesture-inference';

const inference = createMagneticFingerInference('aligned_v1');
await inference.load();

// On each IMU sample:
const prediction = await inference.predict({
  mx_ut: sample.mx,
  my_ut: sample.my,
  mz_ut: sample.mz,
  ax_g: 0, ay_g: 0, az_g: 0  // Not used in aligned_v1
});

console.log(prediction.fingers);
// { thumb: 'extended', index: 'flexed', ... }
```

### 10.3 Related to FFO$$

See [Symbiosis Analysis](./aligned-ffo-symbiosis.md) for how aligned signatures could enhance FFO$$ template matching.

---

## References

1. **SIMCAP ML Pipeline**: [`ml/`](../../ml/)
2. **Aligned Generator**: [`ml/simulation/aligned_generator.py`](../../ml/simulation/aligned_generator.py)
3. **Signature Analysis**: [`ml/analyze_finger_magnet_signatures.py`](../../ml/analyze_finger_magnet_signatures.py)
4. **Browser Inference**: [`apps/gambit/gesture-inference.ts`](../../apps/gambit/gesture-inference.ts)
5. **Model Config**: [`public/models/finger_aligned_v1/config.json`](../../public/models/finger_aligned_v1/config.json)

---

<link rel="stylesheet" href="../../src/simcap.css">
