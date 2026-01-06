---
title: V5 Context-Aware Experiment Results
created: 2026-01-06
updated: 2026-01-06
status: Complete
tags: [experiment, context-aware, chorus, gated-fusion, negative-result]
related:
  - arxiv-literature-review.md
  - dropdim-experiment-results.md
  - v4-architecture-exploration.md
---

# V5 Context-Aware Experiment Results

**Date:** January 6, 2026
**Objective:** Test context-aware gated fusion (Chorus-inspired) with orientation features against V4-Regularized baseline
**Result:** ❌ **NEGATIVE** - V5-Context significantly worsens performance

---

## Executive Summary

Context-aware architecture with gated fusion of magnetometer and orientation features **does not improve** our finger classification model. Test accuracy dropped from **72.8% to 54.3%** (-18.5%), with catastrophic failures on Ring (-44.2%), Middle (-41.1%), and Index (-28.4%) fingers.

**Recommendation:** Continue using V4-Regularized. Move to Priority 2: Active Self-Training (addresses data scarcity without architectural complexity).

---

## Background

### Chorus-Inspired Technique

From Zhang, L., et al. "Chorus: Harmonizing Context and Sensing Signals for Data-Free Model Customization in IoT." arXiv:2512.15206, Dec 2025.

**Key Idea:** Use orientation (pitch, roll, yaw) as context signal, dynamically weight magnetometer vs context contributions based on learned gate.

**Hypothesis for Our Problem:**
- Orientation context helps model adapt to unseen pitch angles
- Gated fusion learns when to rely on mag signal vs orientation
- More context weighting for weak signals (pinky)
- Expected: 67.6% → 75-78% test accuracy, gap: 21.8% → 15-18%

### V5 Architecture

**Dual-Branch Design:**
1. **Magnetometer branch:** Conv1D → LSTM → feature embedding (32 dims)
2. **Context branch:** Dense layers → context embedding (32 dims)
3. **Gated fusion:** `output = mag_features * gate + context_embedding * (1 - gate)`
   - Gate learns optimal weighting (sigmoid → [0, 1])
4. **Final prediction:** Dense layer (5 outputs)

**Model size:** 10,934 parameters (vs 9,989 for V4-Regularized)

---

## Experimental Setup

### Data Split (Cross-Orientation)

- **Train:** Q3 pitch angles (high pitch, ≥26.9°) + 50% synthetic - 116 windows
- **Val:** Q3 pitch angles (held-out portion) - 36 windows
- **Test:** Q1 pitch angles (low pitch, ≤-26.2°, COMPLETELY HELD OUT) - 95 windows

### Input Features

**V4-Regularized (Baseline):**
- Magnetometer only: (10, 3) - 10 samples × 3 features (mx, my, mz)

**V5-Context:**
- Magnetometer: (10, 3) - 10 samples × 3 features (mx, my, mz)
- Context: (3,) - aggregated orientation (mean pitch, roll, yaw over window)

---

## Results

### Overall Accuracy

| Metric | V4-Regularized | V5-Context | Change |
|--------|----------------|------------|--------|
| **Validation Accuracy** | 90.6% | 58.3% | -32.3% |
| **Test Accuracy** | 72.8% | 54.3% | **-18.5%** |
| **Generalization Gap** | 17.7% | 4.0% | -13.7% |

### Per-Finger Test Accuracy

| Finger | V4-Regularized | V5-Context | Change | Note |
|--------|----------------|------------|--------|------|
| Ring | **95.8%** | 51.6% | **-44.2%** | Catastrophic |
| Middle | **78.9%** | 37.9% | **-41.1%** | Catastrophic |
| Index | **65.3%** | 36.8% | **-28.4%** | Catastrophic |
| Pinky | 67.4% | **88.4%** | **+21.1%** | ✓ Significant improvement |
| Thumb | 56.8% | 56.8% | 0.0% | Unchanged |

### Per-Finger Validation Accuracy

| Finger | V4-Regularized | V5-Context | Change |
|--------|----------------|------------|--------|
| Middle | **94.4%** | 22.2% | -72.2% |
| Thumb | **91.7%** | 83.3% | -8.4% |
| Ring | **88.9%** | 52.8% | -36.1% |
| Index | **88.9%** | 58.3% | -30.6% |
| Pinky | **88.9%** | 75.0% | -13.9% |

---

## Analysis

### Why V5-Context Failed

#### 1. Insufficient Training Data for Increased Complexity

**Problem:** Only 116 training samples for more complex architecture

**V4-Regularized:** 9,989 parameters
**V5-Context:** 10,934 parameters (+9.5% more parameters)

With limited data, the more complex dual-branch architecture cannot learn effective representations. The gated fusion adds:
- Context encoder: 64 + 272 + 544 = 880 additional parameters
- Gate mechanism: 65 parameters

**Result:** Model underfits - both validation (58.3%) and test (54.3%) accuracy drop significantly.

#### 2. Context Features May Not Be Informative

**Orientation features used:** Pitch, roll, yaw (3 values, aggregated over 10-sample window)

**Issues:**
- Only 3 scalar values vs 30 magnetometer values (10 samples × 3 axes)
- Aggregation (mean) loses temporal dynamics
- Pitch is already captured in train/test split (Q3 vs Q1)
- Roll and yaw might not correlate strongly with finger states

**Evidence:** Validation accuracy on middle finger dropped from 94.4% to 22.2% (-72.2%), suggesting context actively interferes with learning.

#### 3. Gated Fusion Too Complex

**Hypothesis:** Gate learns to ignore one branch entirely

**Possible failure modes:**
- Gate → 1: Ignore context (then why add it?)
- Gate → 0: Ignore magnetometer (catastrophic - throws away signal)
- Gate unstable: Random weighting (inconsistent predictions)

**Without inspecting gate values, we cannot confirm** - but catastrophic failures on 3/5 fingers suggest the gate isn't learning useful weightings.

#### 4. Misleading Generalization Gap Reduction

**V4-Regularized:** 90.6% val → 72.8% test = 17.7% gap
**V5-Context:** 58.3% val → 54.3% test = 4.0% gap

**Why this is BAD:**
- Gap decreased because **both** val and test accuracy dropped
- This is underfitting, not better generalization
- The small gap means "equally bad on both domains"

**Good improvement would be:** 90% val → 80% test = 10% gap (test improves while val stays high)

#### 5. Pinky Improvement Doesn't Compensate

**Pinky:** 67.4% → 88.4% (+21.1%)

**Why this doesn't validate V5:**
- Ring, Middle, Index losses (-44%, -41%, -28%) far outweigh pinky gain
- Overall test accuracy still drops 18.5%
- Suggests gate learned to prioritize context for pinky (weak mag signal) but destroys other fingers in the process
- Not a systematic improvement, more like random rebalancing

---

## Comparison with Literature

### Why Chorus Works for IoT Sensing

From the original paper (arXiv:2512.15206):
- **Domain:** IMU, speech, WiFi sensing (large-scale datasets)
- **Training data:** Thousands of samples across multiple contexts
- **Context signals:** Rich environmental/behavioral context (location, activity, user ID)
- **Results:** +11.3% accuracy improvement in unseen contexts

### Why It Doesn't Work Here

| Aspect | Chorus (IoT) | Our Problem |
|--------|--------------|-------------|
| Training samples | Thousands | 116 |
| Context richness | Multi-modal (location, activity, etc.) | 3 values (pitch, roll, yaw) |
| Context diversity | Many contexts in training | Only Q3 pitch range |
| Test context | Moderate shift | Extreme shift (Q3 → Q1) |
| Architecture capacity | Can afford dual-branch | Limited by data |

**Conclusion:** Chorus requires:
1. Sufficient training data to learn gated fusion
2. Rich context signals with predictive power
3. Diverse contexts during training to learn adaptive weighting

We lack all three.

---

## Lessons Learned

### 1. Limited Data Constrains Architecture Complexity

When training data is scarce (<200 samples), simpler architectures generalize better:
- V4-Regularized (9,989 params): 72.8% test
- V5-Context (10,934 params): 54.3% test

**Rule of thumb:** ~50-100 samples per parameter for good generalization. We have 116 samples for 10K+ parameters - severely data-limited.

### 2. Context Features Must Be Informative

Adding context only helps if:
- Context has strong correlation with output (pitch might not predict finger state well)
- Context is diverse during training (we only train on Q3 pitch)
- Context adds information not already in primary signal (mag already varies with orientation)

**Our context (orientation) may be redundant or confusing the model.**

### 3. Gated Fusion Requires Careful Training

Dual-branch architectures with learned gates are hard to train:
- Risk of gate collapse (ignoring one branch)
- Harder to optimize (two branches + gate)
- Needs more data to learn effective weighting

**Without sufficient data, standard single-branch architecture is more robust.**

### 4. Generalization Gap Can Be Misleading

**Good progress:** Val stays high, test improves (gap narrows from above)
**Bad progress:** Both drop, gap narrows (V5-Context, DropDim)

Always check absolute accuracies, not just the gap.

### 5. Literature Techniques Need Domain Adaptation

Chorus works for IoT with:
- Large datasets
- Rich context
- Moderate context shifts

Our problem has:
- Tiny dataset (116 samples)
- Sparse context (3 values)
- Extreme context shift (Q3 → Q1 pitch)

**Direct application of literature techniques without domain adaptation often fails.**

---

## Pattern: Second Negative Result

**DropDim Experiment:**
- Test accuracy: 67.6% → 53.1% (-14.5%)
- Reason: Feature space too small (32 dims)

**V5-Context Experiment:**
- Test accuracy: 72.8% → 54.3% (-18.5%)
- Reason: Insufficient data for increased complexity

**Common Pattern:**
- Adding complexity (DropDim, dual-branch fusion) without addressing core constraint (limited data)
- Models underfit: both val and test accuracy drop
- Misleading gap reduction (both drop, not generalization improvement)

**Insight:** With 116 training samples, we cannot increase model complexity. We must either:
1. Increase data (Priority 2: Active Self-Training)
2. Simplify model further (unlikely to help - V4 is already small)
3. Use transfer learning / pre-training (Priority 3: Self-supervised PINN)

---

## Next Steps

### ❌ Deprioritize Architecture Changes

Two failed experiments (DropDim, V5-Context) show that **architectural innovations don't help** with our data constraint:
- DropDim: -14.5% accuracy
- V5-Context: -18.5% accuracy

**V4-Regularized remains the best architecture** for our data size (~10K params, strong regularization).

### ✅ Priority 2: Active Self-Training (ActiveSelfHAR)

**Why this is more promising:**
- **Addresses root cause:** Limited labeled data in Q1 test domain
- **No architecture changes:** Uses V4-Regularized as-is
- **Proven technique:** From ActiveSelfHAR paper (arXiv:2303.15107)

**Approach:**
1. Train V4 on Q3 (high pitch) - DONE ✓
2. Generate pseudo-labels for Q1 (low pitch) samples
3. Select 20-50 high-uncertainty Q1 samples for manual labeling
4. Fine-tune on combined: Q3 labeled + Q1 pseudo-labeled + Q1 manually-labeled

**Expected impact:**
- Test accuracy: 72.8% → 80-85% (addressing domain gap directly)
- Minimal implementation effort (no architecture changes)
- Low risk (can always revert to V4)

**Timeline:** 1 week (mostly data collection + manual labeling)

### ✅ Priority 3: Self-Supervised PINN (If Active Self-Training Insufficient)

**Why this could work:**
- Uses physics constraints as supervision
- Doesn't require more labeled data
- Addresses Q1 domain with unlabeled samples + physics model

**Timeline:** 1-2 weeks (requires physics model integration)

---

## Conclusion

**V5 Context-Aware architecture with gated fusion is NOT effective for our magnetometer-based finger classification task.** Test accuracy dropped 18.5% with catastrophic failures on Ring, Middle, and Index fingers.

**Root causes:**
1. Insufficient training data (116 samples) for increased complexity (10,934 params)
2. Context features (orientation) not informative enough
3. Gated fusion too complex to train with limited data

**Recommendation:**
1. **Continue using V4-Regularized** (72.8% test, 17.7% gap)
2. **Move to Priority 2: Active Self-Training** - addresses data scarcity without architectural changes
3. If self-training insufficient, try Priority 3: Self-Supervised PINN

**Key Takeaway:** With limited training data (~116 samples), architectural innovations fail. The solution is to increase effective training data (active self-training, pseudo-labeling, physics-informed supervision), not to add model complexity.

---

## References

1. **Chorus:** Zhang, L., et al. "Harmonizing Context and Sensing Signals for Data-Free Model Customization in IoT." arXiv:2512.15206, Dec 2025.

2. **ActiveSelfHAR:** Wei, B., et al. "Incorporating Self Training into Active Learning to Improve Cross-Subject Human Activity Recognition." arXiv:2303.15107, Mar 2023.

3. **SSPINNpose:** "Self-supervised physics-informed neural network for estimating joint kinematics and kinetics from IMU data." arXiv:2506.11786, Jun 2025.

4. **DropDim Experiment:** `docs/research/dropdim-experiment-results.md`

5. **Our V4 Baseline:** `docs/research/v4-architecture-exploration.md`

6. **Literature Review:** `docs/research/arxiv-literature-review.md`

---

**Report Generated:** January 6, 2026
**Experiment Code:** `ml/test_context_aware_v5.py`
**Results File:** `ml/results/v5_context_experiment.json`
