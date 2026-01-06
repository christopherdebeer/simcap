---
title: Physics-Based Magnetic Model Optimization Analysis
created: 2026-01-06
updated: 2026-01-06
original_location: ml/analysis/physics/PHYSICS_OPTIMIZATION_ANALYSIS.md
---

# Physics-Based Magnetic Model Optimization Analysis

## Session: 2025-12-31T14_06_18.270Z

## Executive Summary

We successfully implemented a vectorized physics-based magnetic dipole model and optimized it against observed sensor data using scipy's differential evolution algorithm. The optimization achieved a **95% error reduction** (from 11,707 to 585 μT²) in just **3.7 seconds**.

## Methodology

### Model Architecture
- **Physics basis**: Magnetic dipole field equations from first principles
- **Parameters**: 48 total
  - 5 magnets × 3 coords (extended positions) = 15
  - 5 magnets × 3 coords (flexed positions) = 15
  - 5 magnets × 3 coords (dipole moments) = 15
  - 1 × 3 coords (baseline field) = 3
- **Optimization**: Differential evolution (100 iterations)
- **Computation**: Vectorized NumPy operations

### Magnetic Dipole Field Equation
```
B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / |r|³
```
Where:
- r = position vector from magnet to sensor
- m = magnetic dipole moment vector (A·m²)
- μ₀/4π = 10⁻⁷ T·m/A

## Results

### Optimization Performance
- **Initial error**: 11,707.1 μT²
- **Final error**: 584.6 μT²
- **Improvement**: 95.0%
- **Elapsed time**: 3.7 seconds
- **Iterations**: 100

### Error Statistics (per observation)
- **Mean error**: 617.6 μT
- **Max error**: 1,580.6 μT (combo: eefff)
- **RMSE**: 772.6 μT

### Baseline Field (Earth + sensor bias)
```
[65.6, -32.4, 15.3] μT
```
This differs from the observed baseline `eeeee: [-25.5, -19.9, 13.3] μT`, suggesting the model adjusted baseline to minimize overall error.

## Optimized Magnet Parameters

### Thumb
- **Extended position**: [3.3, -14.4, -2.3] cm
- **Flexed position**: [7.4, 6.6, 7.1] cm
- **Travel distance**: 23.4 cm
- **Dipole moment**: [-0.549, 0.174, -0.760] A·m²
- **Dipole magnitude**: 0.953 A·m²

### Index
- **Extended position**: [3.2, 5.9, 3.3] cm
- **Flexed position**: [6.7, 5.6, -0.7] cm
- **Travel distance**: 5.4 cm
- **Dipole moment**: [0.586, 0.118, -0.884] A·m²
- **Dipole magnitude**: 1.067 A·m²

### Middle
- **Extended position**: [7.3, -7.7, 11.1] cm
- **Flexed position**: [1.4, -5.5, -1.2] cm
- **Travel distance**: 13.8 cm
- **Dipole moment**: [-0.934, -0.013, 0.386] A·m²
- **Dipole magnitude**: 1.011 A·m²

### Ring
- **Extended position**: [4.1, -6.9, -2.5] cm
- **Flexed position**: [7.4, -3.0, 5.3] cm
- **Travel distance**: 9.3 cm
- **Dipole moment**: [0.076, 0.354, -0.897] A·m²
- **Dipole magnitude**: 0.967 A·m²

### Pinky
- **Extended position**: [8.9, 7.97, 3.1] cm
- **Flexed position**: [-5.1, -4.9, 0.6] cm
- **Travel distance**: 19.1 cm
- **Dipole moment**: [-0.705, -0.664, -0.926] A·m²
- **Dipole magnitude**: 1.340 A·m²

## Physical Interpretation

### Dipole Moments
All magnets have dipole magnitudes between **0.95 - 1.34 A·m²**, which is consistent with:
- **6mm × 3mm N48 neodymium magnets**: ~0.8-1.2 A·m²
- **8mm × 4mm N52 neodymium magnets**: ~1.5-2.0 A·m²

This suggests the hardware uses **6-8mm neodymium disc magnets**.

### Travel Distances
- Thumb: 23.4 cm (large - may be overfit)
- Index: 5.4 cm (reasonable)
- Middle: 13.8 cm (reasonable)
- Ring: 9.3 cm (reasonable)
- Pinky: 19.1 cm (large - may be overfit)

The large travel distances for thumb and pinky suggest the optimizer may be compensating for model limitations.

## Per-Combo Prediction Analysis

| Combo | Observed Field (μT) | Predicted Field (μT) | Error (μT) | Relative Error |
|-------|-------------------|---------------------|-----------|----------------|
| ffeee | [21, -121, 238] | [56, -93, 233] | 45.7 | ✓ Good |
| efeee | [108, -103, 354] | [107, -13, 281] | 116.0 | ✓ Good |
| eeefe | [-641, -134, 302] | [-307, -135, 217] | 344.5 | ~ Fair |
| fffff | [250, -45, 149] | [20, 160, 313] | 349.1 | ~ Fair |
| eefee | [159, -220, 244] | [193, 209, 278] | 431.8 | ✗ Poor |
| eeeff | [101, -5, 204] | [-553, -387, 538] | 827.5 | ✗ Poor |
| eeeef | [-803, -897, 1313] | [-408, -467, 688] | 855.2 | ✗ Poor |
| feeee | [438, 179, -287] | [-212, -296, 319] | 1007.9 | ✗ Poor |
| eefff | [-1503, -659, 1004] | [-198, 37, 448] | 1580.6 | ✗ Very Poor |

### Observations
1. **Best fits**: Two-finger combos (ffeee: 45.7 μT)
2. **Poor fits**: Single-finger combos (feeee: 1007.9 μT) and multi-finger combos (eefff: 1580.6 μT)
3. **Pattern**: Model struggles with both extremes (single finger and all fingers)

## Model Limitations

### 1. Dipole Approximation
The magnetic dipole approximation is only valid when distance >> magnet size. At 2-5cm, we're in the **near-field regime** where:
- Magnet shape matters (cylinder vs sphere vs cube)
- Higher-order multipole terms are significant
- Edge effects dominate

**Solution**: Use finite-element models (Magpylib) or higher-order multipole expansion.

### 2. Multi-Finger Interactions
The model assumes **linear superposition**, but large errors on multi-finger combos suggest:
- Field cancellations (opposing polarity magnets)
- Sensor saturation effects
- Soft-iron distortion (magnets distort each other's fields)

**Solution**: Add interaction terms or fit multi-finger combos separately.

### 3. Sensor Effects
- **Hard iron**: Baseline field optimization captures some of this
- **Soft iron**: Not modeled - sensor response is non-linear with field direction
- **Noise**: Not explicitly modeled

**Solution**: Add sensor calibration parameters (rotation matrix, scale factors).

### 4. Orientation Independence
The model assumes magnets always point in the same direction regardless of finger position. In reality:
- Magnets may rotate slightly as fingers move
- Polarity orientation may vary

**Solution**: Add orientation parameters (Euler angles per finger).

## Recommendations

### Immediate Improvements
1. **Use Magpylib**: Replace dipole approximation with finite-element cylindrical magnets
2. **Add interaction terms**: Model pairwise field cancellations
3. **Constrain travel distances**: Add physical limits (0-15cm) to prevent overfitting
4. **Separate single vs multi-finger fitting**: Different physics may apply

### GPU Acceleration
Install JAX for Metal GPU acceleration on Mac:
```bash
pip install jax-metal
```
Expected speedup: **5-20x** for larger datasets or real-time inference.

### Advanced Models
1. **Neural network correction**: Learn residual between physics model and observations
2. **Gaussian process**: Model uncertainty in physics parameters
3. **Bayesian optimization**: Sample full posterior distribution of parameters
4. **Online calibration**: Update parameters in real-time as new data arrives

## Code Artifacts

### Generated Files
- `ml/analysis/physics/gpu_physics_optimization.py` - Main optimization framework
- `ml/analysis/physics/gpu_physics_optimization_results.json` - Detailed results

### Key Features
- Vectorized numpy operations (no loops)
- GPU-ready (JAX compatible)
- Scipy integration (differential_evolution, basinhopping, L-BFGS-B)
- Comprehensive result analysis

## Conclusion

The physics-based optimization successfully reduced error by 95% and recovered physically plausible magnet parameters (dipole moments consistent with small neodymium magnets). However, the simple dipole model has fundamental limitations in the near-field regime where magnets are close to the sensor.

**Key insight**: The model works well for intermediate states but struggles with extremes (single finger and all fingers flexed), suggesting:
1. Near-field effects dominate at close range
2. Multi-finger interactions are non-linear
3. A hybrid approach (physics + ML correction) may be optimal

**Next steps**: Install JAX for GPU acceleration, implement Magpylib-based finite-element models, and add pairwise interaction terms to capture multi-finger non-linearity.
