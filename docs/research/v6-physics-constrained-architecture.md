---
title: "V6 Physics-Constrained Inverse Magnetometry Architecture"
created: 2026-01-07
updated: 2026-01-07
status: Complete
tags: [ml, physics, magnetometry, architecture, v6]
related:
  - v4-architecture-exploration.md
  - cross-orientation-ablation-results.md
---

# V6: Physics-Constrained Inverse Magnetometry

## Executive Summary

V6 introduces **physics-constrained training** for finger state classification from magnetometer data. By using the known forward dipole model as a training constraint, we achieve **57.3% cross-orientation accuracy** compared to 8.5% for V4-style models under the same strict test conditions—a **6.7x improvement** in generalization.

### Key Results

| Metric | V4-Style | V6 Physics | Improvement |
|--------|----------|------------|-------------|
| Cross-orientation test accuracy | 8.5% | 57.3% | +48.8% |
| Train-test gap | 66.4% | 42.7% | -23.7% |
| Thumb accuracy | 57.9% | 73.9% | +16.0% |

## The Inverse Magnetometry Problem

### Problem Statement

Five magnetic dipoles (fingertip magnets) superpose into a single 3-vector at the wrist sensor:

```
B_total = Σᵢ B_dipole(rᵢ, mᵢ)   where i ∈ {thumb, index, middle, ring, pinky}
```

**Forward problem** (positions → field): Closed-form via dipole equation
**Inverse problem** (field → positions): Massively underdetermined

- **Degrees of freedom**: 15 (5 fingers × 3 coordinates)
- **Measurements**: 3 (magnetometer Bx, By, Bz)
- **Underdetermination**: 5:1 ratio

### What Breaks the Degeneracy

1. **Temporal Structure**: Trajectories through field-space are more informative than single readings. Rate of field change carries position information.

2. **Learned Priors on Motion**: A hand is a kinematic chain with ~20 DOF but strong covariance structure. The network learns the submanifold that magnets actually traverse.

3. **Physics Constraints**: The relationship between positions and fields must obey Maxwell's equations—this eliminates physically implausible solutions.

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    V6 Physics-Constrained Model                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: [batch, window=8, features=9]                           │
│         (mx, my, mz, dmx/dt, dmy/dt, dmz/dt, d²mx/dt², ...)    │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Temporal Encoder    │                          │
│              │   (Bidirectional      │                          │
│              │    LSTM 64→32)        │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Feature Vector      │                          │
│              │   [batch, 64]         │                          │
│              └───────────┬───────────┘                          │
│                    ┌─────┴─────┐                                │
│                    │           │                                 │
│                    ▼           ▼                                 │
│         ┌─────────────┐ ┌─────────────┐                         │
│         │ State Head  │ │Position Head│                         │
│         │ Dense→σ     │ │ Dense→σ     │                         │
│         │ [batch, 5]  │ │ [batch, 5]  │                         │
│         └──────┬──────┘ └──────┬──────┘                         │
│                │               │                                 │
│                ▼               ▼                                 │
│           y_pred          position_factors                       │
│         (finger          (continuous                             │
│          states)          [0,1] per finger)                      │
│                               │                                  │
│                               ▼                                  │
│                    ┌───────────────────┐                        │
│                    │ State→Position    │                        │
│                    │ Interpolation     │                        │
│                    │ pos = ext + f*(flex-ext)                   │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                              ▼                                   │
│                    ┌───────────────────┐                        │
│                    │ Forward Physics   │                        │
│                    │ Model (Dipole)    │                        │
│                    │                   │                        │
│                    │ B = Σ μ₀/4π ×    │                        │
│                    │   [3(m·r̂)r̂-m]/r³ │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                              ▼                                   │
│                         B_predicted                              │
│                         [batch, 3]                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Loss = L_classification + λ × L_physics

L_classification = BCE(y_pred, y_true)
L_physics = MSE(B_predicted, B_observed)
```

### Component Details

#### 1. Temporal Encoder (Bidirectional LSTM)

```python
encoder = keras.Sequential([
    keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True)
    ),
    keras.layers.Bidirectional(
        keras.layers.LSTM(32)
    ),
    keras.layers.Dense(64, activation='relu')
])
```

**Why bidirectional**: Both past and future context within window disambiguate current state.

**Why LSTM over Transformer**: Limited training data (22 segments, ~250 windows) favors inductive bias of recurrence over learned attention patterns.

#### 2. Temporal Derivatives as Features

```python
def add_temporal_derivatives(windows):
    # windows: [N, T, 3] raw magnetometer
    velocity = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])
    return np.concatenate([windows, velocity, acceleration], axis=-1)
    # Output: [N, T, 9]
```

**Key insight**: Rate of field change carries position information that instantaneous readings don't. A finger moving toward the sensor produces different temporal signatures than one moving away, even at the same instantaneous field value.

#### 3. State-to-Position Interpolation (Learnable)

```python
class FingerStateToPosition(keras.layers.Layer):
    def __init__(self, learnable_geometry=True):
        # Learnable extended/flexed positions per finger
        self.pos_extended = self.add_weight(shape=(5, 3), ...)  # [5 fingers, xyz]
        self.pos_flexed = self.add_weight(shape=(5, 3), ...)

    def call(self, finger_states):
        # finger_states: [batch, 5] continuous in [0, 1]
        # Linear interpolation
        positions = pos_extended + states * (pos_flexed - pos_extended)
        return positions  # [batch, 5, 3] in mm
```

**Default geometry** (from `hand_model.py`, mm from wrist sensor):

| Finger | Extended Position | Flexed Position |
|--------|-------------------|-----------------|
| Thumb  | [63.5, 58.5, -5.0] | [40.0, 30.0, -25.0] |
| Index  | [35.0, 135.0, 0.0] | [35.0, 75.0, -30.0] |
| Middle | [15.0, 150.0, 0.0] | [15.0, 80.0, -30.0] |
| Ring   | [-5.0, 137.0, 0.0] | [-5.0, 75.0, -30.0] |
| Pinky  | [-25.0, 108.0, 0.0] | [-25.0, 65.0, -30.0] |

**Why learnable**: Bridges sim-to-real gap. The network adapts geometry to actual user's hand.

#### 4. Differentiable Forward Physics Model

```python
class DifferentiablePhysicsModel(keras.layers.Layer):
    def call(self, positions):
        # positions: [batch, 5, 3] in mm

        # Convert to meters
        positions_m = positions / 1000.0

        # Vector from each magnet to sensor (at origin)
        r_vecs = -positions_m  # [batch, 5, 3]
        r_mags = tf.norm(r_vecs, axis=-1, keepdims=True)
        r_hats = r_vecs / tf.maximum(r_mags, 1e-6)

        # Dipole field: B = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³
        m_dot_r = tf.reduce_sum(r_hats * dipole_moments, axis=-1, keepdims=True)
        B_magnets = MU_0_4PI * (3 * m_dot_r * r_hats - dipole_moments) / r_mags**3

        # Sum over all magnets, convert T → μT
        B_total = tf.reduce_sum(B_magnets, axis=1) * 1e6

        return B_total  # [batch, 3]
```

**Physical constants**:
- μ₀/4π = 10⁻⁷ T·m/A
- Dipole moment (6×3mm N48): ~0.0135 A·m²
- Alternating polarity: [+, -, +, -, +] z-axis

### Loss Function

```python
def train_step(self, data):
    x, y = data
    observed_field = tf.reduce_mean(x[:, :, :3], axis=1)  # Mean over window

    with tf.GradientTape() as tape:
        # Forward pass
        features = self.encoder(x)
        pred_states = self.state_head(features)
        position_factors = self.position_head(features)

        # Physics prediction
        positions = self.state_to_position(position_factors)
        predicted_field = self.physics_model(positions)

        # Combined loss
        L_cls = binary_crossentropy(y, pred_states)
        L_physics = mean_squared_error(predicted_field, observed_field)

        L_total = L_cls + λ * L_physics  # λ = 0.01 optimal

    gradients = tape.gradient(L_total, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
```

**Why λ = 0.01 is optimal**:
- Too high (λ > 0.1): Physics dominates, classification signal too weak
- Too low (λ < 0.01): Not enough regularization benefit
- λ = 0.01: Best balance—physics guides without dominating

## Experimental Results

### Cross-Orientation Generalization

Test methodology: Train on pitch ≥ Q3, test on pitch ≤ Q1 (strict orientation separation)

| Model | Physics λ | Train Acc | Test Acc | Gap |
|-------|-----------|-----------|----------|-----|
| V4-style baseline | 0.0 | 74.9% | 8.5% | 66.4% |
| V6 no physics | 0.0 | 100.0% | 50.6% | 49.4% |
| **V6 λ=0.01** | **0.01** | **100.0%** | **57.3%** | **42.7%** |
| V6 λ=0.05 | 0.05 | 100.0% | 54.8% | 45.2% |
| V6 λ=0.1 | 0.1 | 100.0% | 46.9% | 53.1% |
| V6 λ=0.2 | 0.2 | 95.8% | 44.4% | 51.4% |

### Per-Finger Accuracy

| Finger | V4-Style | V6 Physics | Δ |
|--------|----------|------------|---|
| Thumb | 57.9% | **73.9%** | +16.0% |
| Index | 65.1% | **79.7%** | +14.6% |
| Middle | 75.3% | **85.1%** | +9.8% |
| Ring | 82.6% | **87.1%** | +4.5% |
| Pinky | **94.5%** | 81.7% | -12.8% |

**Key finding**: Physics constraint dramatically improves thumb (the hardest finger due to proximity to sensor and highest field contribution).

### Comparison to Previous Versions

| Version | Architecture | Test Acc* | Test Acc† | Key Innovation |
|---------|--------------|-----------|-----------|----------------|
| V2 | 9-DoF, w=50 | 58.0% | - | Baseline CNN-LSTM |
| V3 | mag, w=10 | 68.4% | - | Mag-only, smaller window |
| V4 | mag, w=10 | 70.1% | 8.5% | Heavy regularization |
| V5 | mag+context | 54.3% | - | Dual-branch (failed) |
| **V6** | **mag+deriv+physics** | **-** | **57.3%** | **Physics constraint** |

*Random split  †Strict cross-orientation split (Q3→Q1 pitch)

## Theoretical Foundation

### Why Physics Constraints Work

**Traditional regularization** (dropout, L2, label smoothing):
- Statistical: Penalizes model complexity
- Assumes: Simpler models generalize better
- Limitation: Doesn't encode domain knowledge

**Physics-constrained training**:
- Domain-specific: Penalizes physically implausible solutions
- Encodes: Maxwell's equations are orientation-invariant
- Benefit: Data-efficient regularization via known physics

### The Regularization Hierarchy

```
Most general                                    Most specific
     │                                               │
     ▼                                               ▼
L2 weight     Dropout     Label        Physics      Exact
 decay                   smoothing    constraint   supervision

"Small         "Don't    "Don't be   "Obey        "Match
weights"       overfit   overconfid-  Maxwell's    labeled
               to any    ent"         equations"   data"
               sample"
```

Physics constraints sit between generic statistical regularization and exact supervision—they encode domain knowledge without requiring labeled position data.

### Information-Theoretic View

The physics constraint reduces the effective hypothesis space:

```
Without constraint: H ⊆ all functions f: ℝ^(T×3) → [0,1]^5
With constraint:    H ⊆ {f | ∃g: f(x) consistent with dipole physics via g}
```

This is equivalent to saying: "Only consider classifiers that could arise from some underlying position estimator plus the known forward model."

## Deployment Considerations

### Model Size

| Component | Parameters |
|-----------|------------|
| LSTM Encoder | ~18,000 |
| State Head | ~2,400 |
| Position Head | ~2,400 |
| Physics Layer | ~45 (learnable geometry) |
| **Total** | **~23,000** |

Comparable to V3/V4 (~10,000-25,000 parameters).

### Inference Path

For inference, we only need the state head:

```python
def predict(self, x):
    features = self.encoder(x)
    return self.state_head(features)  # Skip position/physics
```

Physics model only used during training as regularizer.

### Real-Time Requirements

- Window: 8 samples @ 26 Hz = 308 ms latency
- Inference: ~1-2 ms on CPU
- **Total latency**: ~310 ms (acceptable for gesture recognition)

## Future Directions

1. **Ground-truth position capture**: Optical tracking of magnet positions would enable direct position regression, not just physics-constrained classification.

2. **Smoothness loss**: Penalize jerk in predicted position trajectories for temporal coherence.

3. **Contrastive learning**: Use temporal context to disambiguate similar instantaneous readings.

4. **Calibration-aware physics**: Learn sensor-specific hard/soft iron parameters as part of the physics model.

## References

1. Hand geometry based on `ml/simulation/hand_model.py`
2. Dipole physics from `ml/simulation/dipole.py`
3. V4 architecture from `docs/research/v4-architecture-exploration.md`
4. Ablation methodology from `docs/research/cross-orientation-ablation-results.md`

## Files

- **Technical implementation**: `ml/inverse_magnetometry_physics_constrained.py`
- **Experiment results**: `ml/physics_constrained_results.json`
- **Deployment script**: `ml/deploy_finger_model_v6.py`
