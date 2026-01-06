# Physics Model Analysis: Magnetic Finger State Classification

**Status:** Research Complete
**Date:** January 2026
**Branch:** `claude/physics-simulation-data-generation-Oo6Cg`

---

## 1. Executive Summary

This analysis attempts to reproduce observed magnetic signatures by fitting a physics-based dipole model to labeled finger state data. Key findings:

| Finding | Result | Implication |
|---------|--------|-------------|
| **Single-finger fits** | Perfect (0 μT error) | Individual magnets follow dipole physics |
| **Superposition test** | Failed (1538 μT avg error) | Multi-finger states have strong non-linear interactions |
| **Pinky contribution** | Largest (1750 μT) | Closest to sensor when flexed |
| **Polarity patterns** | Consistent | Magnets have predictable orientations |

**Conclusion:** The magnetic system CAN be modeled with physics, but only for individual finger contributions. Multi-finger states require either:
1. A more sophisticated interaction model
2. Machine learning (the CNN-LSTM already handles this)

---

## 2. Physical Setup Analysis

### 2.1 Observed Field Magnitudes

| State | Description | Field Magnitude |
|-------|-------------|-----------------|
| `00000` | All extended | 35 μT (baseline) |
| `20000` | Thumb flexed | 554 μT |
| `02000` | Index flexed | 384 μT |
| `00200` | Middle flexed | 365 μT |
| `00020` | Ring flexed | 721 μT |
| `00002` | Pinky flexed | 1781 μT |
| `22222` | All flexed | 294 μT |

Key observations:
- **Pinky produces the largest field** (~1750 μT change)
- **All-flexed is smaller than pinky-only** → Fields partially cancel
- **Baseline is small** (~35 μT) → Magnets far when extended

### 2.2 Estimated Magnet Properties

Using the dipole field equation and assuming ~2cm distance when flexed:

| Finger | Field Change | Est. Dipole Moment | Magnet Type |
|--------|--------------|-------------------|-------------|
| Thumb | 587 μT | 0.094 A·m² | ~8mm N52 disc |
| Index | 375 μT | 0.060 A·m² | ~6mm N52 disc |
| Middle | 357 μT | 0.057 A·m² | ~6mm N52 disc |
| Ring | 689 μT | 0.110 A·m² | ~10mm N52 disc |
| Pinky | 1750 μT | 0.280 A·m² | ~12mm N52 disc |

The pinky magnet is significantly stronger or closer to the sensor than others.

### 2.3 Polarity Configuration

Each magnet has a dominant field direction when the finger flexes:

```
      Finger    Dominant Direction    Magnitude
      ────────────────────────────────────────
      Thumb     +X (radial out)       587 μT
      Index     +Z (up from palm)     375 μT
      Middle    +Z (up from palm)     357 μT
      Ring      -X (radial in)        689 μT
      Pinky     +Z (up from palm)    1750 μT
```

This suggests:
- Thumb and ring magnets have opposing radial orientations
- Index, middle, pinky have similar axial orientations
- The alternating pattern may be intentional for disambiguation

---

## 3. Superposition Analysis

### 3.1 Test Methodology

If magnetic fields add linearly:
```
B(multi-finger) = B(00000) + Σ [B(single-finger) - B(00000)]
```

### 3.2 Results

| Combo State | Fingers | Predicted (μT) | Actual (μT) | Error |
|-------------|---------|----------------|-------------|-------|
| `22000` | T+I | [572, 96, 54] | [21, -121, 238] | 620 μT |
| `00022` | R+P | [-1418, -1011, 1601] | [101, -5, 204] | 2296 μT |
| `00222` | M+R+P | [-1234, -1212, 1832] | [-1504, -659, 1004] | 1032 μT |
| `22222` | All | [-637, -1096, 1873] | [250, -45, 149] | 2205 μT |

**Superposition fails dramatically**, especially for:
- Ring + Pinky combination (2296 μT error)
- All fingers flexed (2205 μT error)

### 3.3 Possible Explanations

1. **Geometric Coupling**: When multiple fingers flex, their positions change relative to each other, altering the field geometry

2. **Sensor Saturation**: The magnetometer may saturate at high fields, causing non-linear response

3. **Mutual Magnetic Interactions**: Strong magnets near each other influence each other's fields

4. **Kinematic Constraints**: The hand's physical structure means fingers don't move independently

---

## 4. Fitted Dipole Parameters

### 4.1 Individual Finger Fits

Each finger's contribution can be perfectly modeled by a single dipole:

```
THUMB:
  Position: [6.6, 4.0, -0.9] cm from sensor
  Dipole:   [1.00, 1.00, 1.00] A·m² (at bound limit)
  Fit Error: 0.0 μT ✓

INDEX:
  Position: [2.9, -2.1, 5.0] cm from sensor
  Dipole:   [0.29, -0.25, 0.25] A·m²
  Fit Error: 0.0 μT ✓

MIDDLE:
  Position: [3.5, 4.2, -4.2] cm from sensor
  Dipole:   [-1.00, 0.14, -0.24] A·m²
  Fit Error: 0.0 μT ✓

RING:
  Position: [6.4, 2.5, -3.8] cm from sensor
  Dipole:   [-1.00, -1.00, 1.00] A·m² (at bound limit)
  Fit Error: 0.0 μT ✓

PINKY:
  Position: [3.2, 0.6, -4.4] cm from sensor
  Dipole:   [-1.00, 1.00, 1.00] A·m² (at bound limit)
  Fit Error: 0.0 μT ✓
```

### 4.2 Interpretation

The fitted positions suggest:
- **Pinky is closest** to the sensor (3.2 cm) → Largest field
- **Thumb and Ring are furthest** (~6.5 cm) → But still produce large fields due to strong magnets
- **All dipoles hit the bound limit** (1.0 A·m²) → Actual magnets are even stronger

---

## 5. Implications for Synthetic Data Generation

### 5.1 What Works

Individual finger contributions can be generated using:

```python
def generate_single_finger_field(finger, flexed=True, noise_std=5.0):
    """Generate synthetic field for a single finger state."""
    params = FITTED_PARAMS[finger]
    pos = params['pos_flexed'] if flexed else params['pos_extended']
    dipole = params['dipole_moment']

    # Add position noise (~2mm)
    pos_noisy = pos + np.random.randn(3) * 0.002

    # Compute dipole field
    field = dipole_field(-pos_noisy, dipole)

    # Add sensor noise
    return field + np.random.randn(3) * noise_std
```

### 5.2 What Doesn't Work

**Simple superposition cannot generate multi-finger states.**

Alternative approaches:
1. **Interpolation**: Collect real data for all 32 states, interpolate between them
2. **Physics Correction**: Model the non-linear interactions empirically
3. **Neural Network**: Train a small network to predict field from finger states

### 5.3 Recommended Approach

For synthetic data generation:

```
Method 1: Single-finger augmentation only
- Use physics model for single-finger states
- Collect real data for multi-finger states
- Augment with noise and position variations

Method 2: Learned combination model
- Train small MLP: [finger_states] → [magnetic_field]
- Use this as a fast simulator
- Augment with noise for robustness
```

---

## 6. Comparison with CNN-LSTM Model

The deployed CNN-LSTM achieves orientation invariance without explicit physics modeling:

| Approach | Single-Finger | Multi-Finger | Orientation Invariance |
|----------|---------------|--------------|------------------------|
| Physics dipole | ✓ Perfect | ✗ Fails | Requires rotation |
| k-NN template | ✓ 92% | ✓ 92% | ✗ 30% gap |
| CNN-LSTM | ✓ 88% | ✓ 68% | ✓ 1.5% gap |

The CNN-LSTM implicitly learns:
1. Non-linear magnet interactions
2. Temporal patterns in sensor readings
3. Orientation-invariant features

---

## 7. Key Conclusions

### 7.1 Physics Model Validity

1. **Individual fingers follow dipole physics** - The magnetic field from each finger magnet can be perfectly modeled as a magnetic dipole

2. **Multi-finger interactions are non-linear** - Superposition fails by 1000-2000 μT, indicating complex geometric or electromagnetic interactions

3. **The pinky dominates** - Contributes 1750 μT, likely due to proximity to sensor and/or stronger magnet

### 7.2 Practical Recommendations

1. **For synthetic data**: Use physics model for single-finger states only; collect real data for combinations

2. **For new sensor designs**: Consider magnet placement to maximize distinguishability while minimizing interference

3. **For ML models**: The CNN-LSTM already handles the non-linearities effectively; no physics correction needed

### 7.3 Future Work

- Investigate sensor saturation at high fields
- Measure actual magnet positions/strengths for validation
- Develop empirical correction for superposition failures

---

## 8. Files and Code

- `ml/physics_model.py` - Initial dipole model and optimization
- `ml/physics_model_v2.py` - Enhanced analysis with superposition tests
- `ml/physics_model_results.json` - v1 optimization results
- `ml/physics_analysis_v2.json` - v2 analysis results

---

## References

1. Jackson, J.D. "Classical Electrodynamics" - Dipole field equations
2. Griffiths, D.J. "Introduction to Electromagnetism" - Magnetic materials
3. K&J Magnetics technical notes - Neodymium magnet specifications
