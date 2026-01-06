---
title: DropDim Regularization Experiment Results
created: 2026-01-06
updated: 2026-01-06
status: Complete
tags: [experiment, regularization, dropdim, negative-result]
related:
  - arxiv-literature-review.md
  - v4-architecture-exploration.md
  - v2-v3-benchmark-comparison.md
---

# DropDim Regularization Experiment Results

**Date:** January 6, 2026
**Objective:** Test DropDim regularization (drops entire feature dimensions) against V4-Regularized baseline
**Result:** ❌ **NEGATIVE** - DropDim significantly worsens performance

---

## Executive Summary

DropDim regularization, which drops entire feature dimensions instead of random neurons, **does not improve** our magnetometer-based finger classification model. Test accuracy dropped from **67.6% to 53.1%** (-14.5%), with catastrophic failures on Ring (-62.1%) and Middle (-27.4%) fingers.

**Recommendation:** Continue using V4-Regularized with standard dropout.

---

## Background

### DropDim Technique

From Zhang, H., et al. "DropDim: A Regularization Method for Transformer Networks." arXiv:2304.10321, Apr 2023.

**Key Idea:** Drop entire feature dimensions instead of random neurons, forcing the model to encode information redundantly across dimensions.

**Hypothesis for Our Problem:**
- Force redundancy across magnetometer axes (mx, my, mz)
- Improve robustness when one axis is corrupted/weak (e.g., pinky)
- Better handle orientation variability

### Implementation

**V4-Regularized (Baseline):**
```python
x = keras.layers.Dropout(0.4)(x)  # Random neurons
```

**V4-DropDim:**
```python
x = DropDim(0.3)(x)  # Entire feature dimensions
```

Applied after Conv1D (32 dims) and Dense layer (32 dims), dropping 30% of dimensions.

---

## Experimental Setup

### Data Split (Cross-Orientation)

- **Train:** Q3 pitch angles (high pitch, ≥26.9°) + 50% synthetic - 116 windows
- **Val:** Q3 pitch angles (held-out portion) - 36 windows
- **Test:** Q1 pitch angles (low pitch, ≤-26.2°, COMPLETELY HELD OUT) - 95 windows

### Model Architecture

Both models have identical architecture (~10K parameters):
- Input: (10, 3) - 10 samples × 3 magnetometer features
- Conv1D → BatchNorm → MaxPool → **Regularization Layer** → LSTM → **Regularization Layer** → Dense → Output
- L2 regularization (0.01), Label smoothing (0.1)

**Only difference:** Dropout (baseline) vs DropDim (experiment)

---

## Results

### Overall Accuracy

| Metric | V4-Regularized | V4-DropDim | Change |
|--------|----------------|------------|--------|
| **Validation Accuracy** | 88.3% | 57.2% | -31.1% |
| **Test Accuracy** | 67.6% | 53.1% | **-14.5%** |
| **Generalization Gap** | 20.8% | 4.2% | -16.6% |

### Per-Finger Test Accuracy

| Finger | V4-Regularized | V4-DropDim | Change |
|--------|----------------|------------|--------|
| Thumb | 56.8% | 63.2% | +6.3% ✓ |
| Index | 65.3% | 82.1% | +16.8% ✓ |
| Middle | 78.9% | 51.6% | **-27.4%** ✗ |
| Ring | 87.4% | 25.3% | **-62.1%** ✗ |
| Pinky | 49.5% | 43.2% | -6.3% ✗ |

### Per-Finger Validation Accuracy

| Finger | V4-Regularized | V4-DropDim | Change |
|--------|----------------|------------|--------|
| Thumb | 91.7% | 41.7% | -50.0% |
| Index | 88.9% | 91.7% | +2.8% |
| Middle | 94.4% | 47.2% | -47.2% |
| Ring | 83.3% | 36.1% | -47.2% |
| Pinky | 83.3% | 69.4% | -13.9% |

---

## Analysis

### Why DropDim Failed

#### 1. Feature Space Too Small

**Problem:** Only 3 raw magnetometer features (mx, my, mz) → 32 Conv1D feature maps

**DropDim Impact:** Drops 30% = 10 of 32 feature dimensions completely
- Too aggressive for small feature spaces
- Model can't encode enough information in remaining 22 dimensions
- Magnetometer signals are already sparse (weak pinky signal)

**Contrast with Transformers:** DropDim was designed for transformers with 512-1024 embedding dimensions. Dropping 30% there still leaves 350-700 dimensions. We only have 32.

#### 2. Misleading Generalization Gap Reduction

**Baseline:** 88.3% val → 67.6% test = 20.8% gap
**DropDim:** 57.2% val → 53.1% test = 4.2% gap

**Why this is BAD, not good:**
- Gap decreased because **both** val and test accuracy dropped
- Model is undertrained/underfitted, not better generalized
- The small gap just means "equally bad on both domains"

A good improvement would be: 88.3% val → 75% test = 13% gap (test improves while val stays high)

#### 3. Catastrophic Failures

**Ring finger:** 87.4% → 25.3% (-62.1%)
**Middle finger:** 78.9% → 51.6% (-27.4%)

These fingers had the **strongest baseline performance**, suggesting:
- DropDim destroys learned representations for clear signals
- Model can't recover critical feature dimensions during inference
- Redundancy assumption doesn't hold for magnetometer axes

#### 4. Inconsistent Improvements

**Index finger:** +16.8% improvement (65.3% → 82.1%)
**Thumb finger:** +6.3% improvement (56.8% → 63.2%)

**Why this doesn't validate DropDim:**
- Improvements on 2 fingers don't compensate for catastrophic failures on 3 others
- Overall test accuracy still drops 14.5%
- Suggests DropDim randomly helps some fingers while hurting others (not systematic)

---

## Comparison with Literature

### Why DropDim Works for Transformers

From the original paper (arXiv:2304.10321):
- **Domain:** Speech recognition, machine translation (large embedding spaces)
- **Dimension count:** 512-1024 embedding dimensions
- **Drop rate:** 10-20% (still leaves 400-800 dimensions)
- **Results:** WER 19.1% → 15.1% (significant improvement)

### Why It Doesn't Work Here

| Aspect | Transformers | Our Problem |
|--------|-------------|-------------|
| Input dimensions | 512-1024 | 32 (post-Conv1D) |
| Drop rate | 10-20% | 30% (tried) |
| Remaining dims | 400-800 | 22 |
| Signal strength | Dense text embeddings | Sparse magnetometer signals |
| Redundancy | High (text has context) | Low (3 independent axes) |

**Conclusion:** DropDim requires sufficiently high-dimensional feature spaces to enforce redundancy without destroying critical information. Our 32-dimensional space is too small.

---

## Lessons Learned

### 1. Small Feature Spaces Need Different Regularization

Standard dropout (randomly drop neurons) is more appropriate when:
- Feature space is small (< 64 dimensions)
- Input signals are sparse (magnetometer)
- Every dimension carries unique information

DropDim is appropriate when:
- Feature space is large (> 512 dimensions)
- Input signals are dense (text embeddings)
- Redundancy can be enforced without information loss

### 2. Generalization Gap Alone is Misleading

A small generalization gap doesn't mean better generalization if both val and test accuracy are poor.

**Good progress:** Val stays high, test improves (gap narrows from above)
**Bad progress:** Both drop, gap narrows (DropDim case)

### 3. Negative Results are Valuable

This experiment saved us from deploying a worse model. Literature findings don't always transfer to different domains—empirical testing is essential.

### 4. Not All Literature Techniques Apply

Just because a technique works for transformers/NLP doesn't mean it works for:
- Small CNN-LSTM models
- Time series sensor data
- Sparse signals with limited features

---

## Next Steps

### Immediate: Continue with V4-Regularized

**V4-Regularized remains the best model:**
- 67.6% cross-orientation test accuracy
- 20.8% generalization gap
- Consistent per-finger performance
- Proven architecture from v2-v3-v4 progression

### Priority 1: Context-Aware Adaptation (Chorus)

From arxiv-literature-review.md Priority 1

**Why this is more promising than DropDim:**
- Directly addresses orientation generalization (our core problem)
- Uses orientation features (pitch, roll, yaw) as context signal
- Gated fusion adapts to signal strength (helps pinky)
- Proven on IoT sensor data with orientation shifts

**Expected Impact:**
- Test accuracy: 67.6% → 75-78%
- Generalization gap: 20.8% → 15-18%

**Timeline:** 1 week implementation

### Priority 2: Active Self-Training (ActiveSelfHAR)

From arxiv-literature-review.md Priority 2

**Why this could work:**
- Addresses limited labeled data in Q1 domain
- Uses pseudo-labels + confidence-based filtering
- Minimal manual labeling (20-50 samples)

**Expected Impact:**
- Test accuracy: 67.6% → 75-80%
- Improved Q1 coverage

**Timeline:** 1 week (mostly data collection)

### Priority 3: Explore Alternative Architectures

From motion inference papers:
- **Self-supervised PINN** (SSPINNpose): Use physics constraints as supervision
- **UKF sensor fusion** (UMotion): Uncertainty-driven estimation
- **Cross-view Mamba** (milliMamba): Linear complexity temporal modeling

**Timeline:** 1-2 weeks per technique

---

## Conclusion

**DropDim regularization is NOT effective for our magnetometer-based finger classification task.** Test accuracy dropped 14.5% with catastrophic failures on Ring and Middle fingers.

**Root cause:** Our feature space (32 dimensions) is too small for DropDim's dimension-dropping approach. The technique requires high-dimensional spaces (512+) where redundancy can be enforced without destroying critical information.

**Recommendation:**
1. Continue using V4-Regularized with standard dropout
2. Move to Priority 1: Context-Aware Adaptation (Chorus approach)
3. Consider Active Self-Training if context-aware approach doesn't yield sufficient gains

**Key Takeaway:** Not all literature techniques transfer to different domains. Empirical testing is essential to validate applicability.

---

## References

1. **DropDim:** Zhang, H., et al. "DropDim: A Regularization Method for Transformer Networks." arXiv:2304.10321, Apr 2023.

2. **Chorus:** Zhang, L., et al. "Harmonizing Context and Sensing Signals for Data-Free Model Customization in IoT." arXiv:2512.15206, Dec 2025.

3. **ActiveSelfHAR:** Wei, B., et al. "Incorporating Self Training into Active Learning to Improve Cross-Subject Human Activity Recognition." arXiv:2303.15107, Mar 2023.

4. **Our V4 Baseline:** `docs/research/v4-architecture-exploration.md`

5. **Literature Review:** `docs/research/arxiv-literature-review.md`

---

**Report Generated:** January 6, 2026
**Experiment Code:** `ml/test_dropdim_regularization.py`
**Results File:** `ml/results/dropdim_experiment.json`
