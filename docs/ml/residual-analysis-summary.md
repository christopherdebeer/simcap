# Magnetic Residual Training Analysis

## Summary

This analysis explores using **magnetic residual** (raw_mag - baseline) instead of raw magnetometer values for finger state inference. Key finding: **Single-sample residual achieves 97.3% accuracy** on observed combos with just 3 features.

## Key Findings

### 1. Residual vs Raw Features (Windowed)
- 3-feature residual (97.8%) ≈ 9-feature raw (97.2%)
- **66% fewer features** with no accuracy loss
- Baseline: [46.0, -45.8, 31.3] μT (open palm)

### 2. Single-Sample Performance
| Approach | Test Accuracy | Training Data |
|----------|--------------|---------------|
| Real Data Only | **97.3%** | 1,728 samples |
| Real + 0.5x Synthetic | 95.2% | 3,928 samples |
| Real + 1x Synthetic | 89.5% | 6,128 samples |
| Real + 2x Synthetic | 86.5% | 10,528 samples |

### 3. Why Additivity Fails
The additive synthetic model assumes `thumb + index = ffeee`, but actual error is **76%**:
- Predicted: [+925, -225, +641] μT
- Actual:    [+352, -185, +636] μT

**Physical reason**: When multiple fingers flex together, magnets move into different positions relative to each other, creating field interactions that don't exist when measured separately.

### 4. Per-Finger SNR Analysis
| Finger | SNR | Notes |
|--------|-----|-------|
| Pinky | 111.0 | Highest - furthest from sensor |
| Middle | 84.9 | |
| Ring | 67.8 | |
| Index | 67.3 | |
| Thumb | 19.9 | Lowest - closest to sensor |

## Recommendations

### For Observed Combos (10/32)
Use **real data only** with 3-feature residual:
- 97.3% accuracy
- No synthetic noise
- Fast single-sample inference

### For All 32 Combos
Use **nearest-neighbor interpolation** (not additive):
- Accept ~5% accuracy drop on observed combos
- Gains coverage for 22 missing combos
- Better than 76% error from additive model

### To Improve Further
1. Collect more real finger combinations (especially adjacent pairs like fefee, effee)
2. Model finger interaction terms explicitly
3. Consider per-combo calibration

## Files Created
- `analyze_finger_interactions.py` - Additivity analysis
- `improved_synthetic_generator.py` - NN interpolation generator
- `train_hybrid_residual.py` - Hybrid training experiments
- `train_single_sample_residual.py` - Single-sample baseline
- `residual_model_stats.json` - Saved statistics
- `nn_generator_results.json` - NN generator results
- `hybrid_training_results.json` - Experiment results
