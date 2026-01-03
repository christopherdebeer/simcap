# Physics Simulation Fitting Findings

**Date:** 2026-01-02
**Status:** Research Complete

## Executive Summary

Attempts to fit a physics simulation to observed magnetic residuals have revealed fundamental challenges. The alternating magnet polarity creates complex non-linear interactions that cannot be accurately modeled with only 4 multi-finger observations.

## Key Findings

### 1. Alternating Magnet Polarity Confirmed

From X-axis signs of single-finger effects:
- Thumb: **+** (X = +341)
- Index: **+** (X = +584)
- Middle: **+** (X = +657)
- Ring: **−** (X = −663)
- Pinky: **+** (X = +544)

Pattern: `[+, +, +, −, +]` - Ring has opposite polarity

### 2. Sub-Additivity is EXTREME

| Combo | Fingers | Obs/Sum Ratio |
|-------|---------|---------------|
| ffeee | thumb+index | 0.65 |
| eeeff | ring+pinky | 0.24 |
| eefff | mid+ring+pinky | 0.44 |
| fffff | all 5 | 0.18 |

The ring+pinky combo (0.24x) shows opposite-polarity cancellation.

### 3. Direction Changes, Not Just Magnitude

For `eeeff` (ring + pinky):
- Sum of singles: `[−120, −456, 2968]`
- Observed: `[+365, +27, +613]`
- X-axis **reverses sign**!

This cannot be explained by simple scaling.

### 4. Model Fitting Results

| Model | Mean Error | Notes |
|-------|------------|-------|
| Additive (sum of singles) | 117% | Baseline |
| Global interaction scaling | 39% | Previous work |
| Pairwise + global fitting | 16-18% | Over-parameterized |
| Polarity-aware coupling | 20% | Under-constrained |

All models fail to capture the `eefff` combo (83% error).

### 5. Why Synthetic Data Hurts

Training experiments:
- Real data only: **97.8%** accuracy
- Hybrid (real + synthetic): **51-92%** accuracy

Synthetic data hurts because:
1. Multi-finger prediction error is 20-80%
2. Wrong predictions create confusing examples
3. Model learns incorrect patterns

## Physics Insights

### Magnetic Coupling Hypothesis

When adjacent magnets with opposite polarity are both present:
1. Their fields partially cancel at the sensor
2. The cancellation depends on exact geometry
3. Non-adjacent pairs have less interaction

### Required Data for Accurate Model

To constrain pairwise interactions:
- 10 pairwise combos (C(5,2))
- Currently have: 1 (ffeee = thumb+index)
- Missing: 9 pairwise combinations

Priority pairings to observe:
1. `eefef` - middle+pinky (opposite polarity)
2. `efefe` - index+ring (opposite polarity)
3. `effee` - index+middle (same polarity)

## Recommendations

### Near-Term: Collect More Real Data

**Priority combos to observe:**
1. **2-finger pairs** (need 9 more):
   - `eefef`, `efefe`, `effee`, `fefee`, `feefe`, `feeef`
   - These constrain pairwise interactions

2. **3-finger combos** (need ~10 more):
   - Different configurations to test 3-way effects

### Medium-Term: Physics Modeling

Once more data is collected:
1. Fit pairwise coupling matrix (5×5 = 25 params)
2. Model 3-body effects if needed
3. Validate on held-out combos

### Long-Term: Alternative Approaches

1. **Prototype-based classification**: Use observed combos as templates
2. **Few-shot learning**: Adapt to new combos with minimal examples
3. **Physics simulation**: Use magpylib once geometry is known

## Files Created

| File | Purpose |
|------|---------|
| `physics_fit_to_observations.py` | Per-finger fitting |
| `synthetic_from_observations.py` | Structured interpolation |
| `physics_alternating_polarity.py` | Polarity-aware model |
| `polarity_model_results.json` | Fitted parameters |

## Conclusion

**Synthetic data generation is not viable with current observations.**

The 10 observed combos (5 single + 4 multi + baseline) do not provide enough constraints to model the complex magnetic interactions created by alternating polarity.

The best path forward is **collecting more real training data**, specifically:
- All 10 pairwise combinations
- Representative 3-finger combinations

Until then, the 97.8% accuracy on observed combos is our ceiling.
