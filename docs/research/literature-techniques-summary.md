---
title: Summary - Literature Techniques for Small Model Training
created: 2026-01-06
updated: 2026-01-06
status: Complete
tags: [summary, negative-results, lessons-learned, data-constraints]
related:
  - arxiv-literature-review.md
  - dropdim-experiment-results.md
  - v5-context-experiment-results.md
  - v4-architecture-exploration.md
---

# Summary: Literature Techniques for Small Model Training

**Date:** January 6, 2026
**Objective:** Test state-of-the-art techniques from recent literature to improve cross-orientation generalization
**Result:** ❌ **THREE NEGATIVE RESULTS** - All literature techniques failed to improve V4-Regularized baseline

---

## Executive Summary

Tested three promising techniques from recent arXiv papers (2023-2025) to improve our magnetometer-based finger classification model:

1. **DropDim Regularization** → -14.5% accuracy
2. **Context-Aware Gated Fusion (Chorus)** → -18.5% accuracy
3. **Active Self-Training (ActiveSelfHAR)** → -6.9% accuracy

**All three techniques FAILED to improve our V4-Regularized baseline (72.8% cross-orientation test accuracy).**

**Root Cause:** Severe data constraint (~116-152 training samples) prevents any technique that increases model complexity or requires sufficient training data to learn new mechanisms.

**Recommendation:** V4-Regularized remains the best model. Future improvements require collecting more labeled data, not trying additional literature techniques.

---

## Baseline: V4-Regularized

**Performance (from DropDim experiment run):**
- Test Accuracy: **67.6%** (low pitch, Q1)
- Validation Accuracy: 88.3% (high pitch, Q3)
- Generalization Gap: 20.8%
- Architecture: ~10K parameters, strong regularization (dropout 0.4/0.5/0.4 + L2 0.01 + label smoothing)

**Performance (from V5-Context experiment run):**
- Test Accuracy: **72.8%** (low pitch, Q1)
- Validation Accuracy: 90.6% (high pitch, Q3)
- Generalization Gap: 17.7%

**Performance (from Active Self-Training experiment run):**
- Test Accuracy: **72.2%** (low pitch, Q1)

**Note:** Small variations (67.6%-72.8%) due to random initialization, but all baseline runs show V4-Regularized achieves **~70%** cross-orientation test accuracy.

---

## Experiment 1: DropDim Regularization

**Source:** Zhang, H., et al. "DropDim: A Regularization Method for Transformer Networks." arXiv:2304.10321, Apr 2023.

### Hypothesis

Drop entire feature dimensions instead of random neurons, forcing model to encode information redundantly across magnetometer axes (mx, my, mz).

### Implementation

- Applied DropDim (30% drop rate) after Conv1D (32 dims) and Dense layer (32 dims)
- Replaced standard dropout with dimension-wise dropout

### Results

| Metric | V4-Regularized | V4-DropDim | Change |
|--------|----------------|------------|--------|
| **Test Accuracy** | 67.6% | 53.1% | **-14.5%** |
| Generalization Gap | 20.8% | 4.2% | -16.6% |

**Per-Finger:**
- Ring: 87.4% → 25.3% (**-62.1%** catastrophic)
- Middle: 78.9% → 51.6% (**-27.4%**)
- Index: 65.3% → 82.1% (+16.8%)

### Why It Failed

1. **Feature space too small**: Only 32 dimensions post-Conv1D (vs 512-1024 in transformers where DropDim works)
2. **Too aggressive**: Dropping 30% of 32 dims (10 dimensions) destroys critical information
3. **Misleading gap reduction**: Both val and test dropped, not true generalization improvement

**Conclusion:** DropDim requires high-dimensional spaces (512+). Our 32-dimensional space is too small.

**Report:** `docs/research/dropdim-experiment-results.md`

---

## Experiment 2: Context-Aware Gated Fusion (V5)

**Source:** Zhang, L., et al. "Chorus: Harmonizing Context and Sensing Signals for Data-Free Model Customization in IoT." arXiv:2512.15206, Dec 2025.

### Hypothesis

Treat hand orientation (pitch, roll, yaw) as context signal, dynamically weight magnetometer vs context based on learned gate. Expected +5-8% accuracy improvement.

### Implementation

- Dual-branch architecture: Magnetometer branch + Context branch
- Gated fusion: `output = mag_features * gate + context_embedding * (1 - gate)`
- Model size: 10,934 parameters (vs 9,989 for V4)

### Results

| Metric | V4-Regularized | V5-Context | Change |
|--------|----------------|------------|--------|
| **Test Accuracy** | 72.8% | 54.3% | **-18.5%** |
| Generalization Gap | 17.7% | 4.0% | -13.7% |

**Per-Finger:**
- Ring: 95.8% → 51.6% (**-44.2%** catastrophic)
- Middle: 78.9% → 37.9% (**-41.1%**)
- Index: 65.3% → 36.8% (**-28.4%**)
- Pinky: 67.4% → 88.4% (+21.1%)

### Why It Failed

1. **Insufficient data for complexity**: Only 116 training samples for 10,934 params (+945 params)
2. **Context not informative**: Orientation (pitch, roll, yaw) may not predict finger state well
3. **Gated fusion too complex**: Dual-branch + learned gate hard to train with limited data
4. **Underfitting**: Both validation (58.3%) and test (54.3%) dropped significantly

**Conclusion:** Context-aware fusion requires (1) sufficient training data, (2) rich context signals, (3) diverse contexts in training. We lack all three.

**Report:** `docs/research/v5-context-experiment-results.md`

---

## Experiment 3: Active Self-Training

**Source:** Wei, B., et al. "ActiveSelfHAR: Incorporating Self Training into Active Learning to Improve Cross-Subject Human Activity Recognition." arXiv:2303.15107, Mar 2023.

### Hypothesis

Generate pseudo-labels for Q1 (target domain), select high-confidence samples, actively label uncertain samples, fine-tune on combined dataset. Expected +7-13% accuracy improvement.

### Implementation

1. Train V4 on Q3 (source): 121 samples
2. Generate pseudo-labels for Q1 (target): 95 samples
3. Select high-confidence (conf > 0.9): **11 samples (11.6%)**
4. Select uncertain for "manual labeling" (conf < 0.6): **30 samples (31.6%)**
5. Fine-tune on combined: 121 + 11 + 30 = 162 samples

### Results

| Metric | Baseline (Q3 only) | Active Self-Training | Change |
|--------|-------------------|----------------------|--------|
| **Test Accuracy** | 72.2% | 65.3% | **-6.9%** |

**Per-Finger:**
- Ring: 92.6% → 49.5% (**-43.2%** catastrophic)
- Pinky: 67.4% → 83.2% (+15.8%)
- Middle: 78.9% → 75.8% (-3.2%)

### Why It Failed

1. **Very low confidence**: Mean confidence on Q1 only 0.669 - baseline doesn't generalize
2. **Too few pseudo-labels**: Only 11 samples (11.6%) had conf > 0.9 - insufficient for training
3. **Domain shift too extreme**: Q3 → Q1 pitch shift is large, baseline predictions unreliable
4. **Fine-tuning overfitted**: Pinky improved but Ring catastrophically failed (-43.2%)

**Conclusion:** Active self-training requires baseline model to have reasonable target-domain confidence (>0.8 mean). Our baseline only achieves 0.669 mean confidence on Q1, making pseudo-labels unreliable.

**Experiment Code:** `ml/test_active_self_training.py`

---

## Common Failure Pattern

All three techniques failed for the same fundamental reason:

### The Data Constraint

**Training Data:** ~116-152 samples (depending on synthetic ratio and train/val split)
**Model Parameters:** ~10,000
**Samples-to-Parameters Ratio:** ~0.01 (rule of thumb: need 50-100 samples/parameter for good generalization)

**Consequence:** Any technique that increases model complexity OR requires sufficient data to learn new mechanisms (gating, pseudo-labeling, dimension regularization) will fail.

### Pattern Across All Three Experiments

1. **Increased complexity** (DropDim, V5-Context) or **insufficient reliable pseudo-labels** (Active Self-Training)
2. **Underfitting**: Both validation and test accuracy drop
3. **Misleading gap reduction**: Gap decreases because BOTH val and test drop, not true generalization
4. **Catastrophic failures**: 1-3 fingers lose 20-60% accuracy
5. **Occasional finger improvements**: Pinky sometimes improves, but overall accuracy drops

---

## Lessons Learned

### 1. Literature Techniques Don't Always Transfer

**Transformers (512+ dims)** ≠ **Small CNNs (32 dims)**
**Large IoT Datasets (1000s of samples)** ≠ **Our Dataset (100s of samples)**
**Moderate Domain Shift** ≠ **Extreme Orientation Shift (Q3 → Q1)**

**Takeaway:** Always validate literature findings empirically on your specific domain. Domain characteristics (data size, feature dimensionality, shift magnitude) determine technique applicability.

### 2. Data Constraint is Fundamental

With ~116-152 training samples:
- Cannot increase model complexity (DropDim, V5-Context fail)
- Cannot rely on pseudo-labeling (Active Self-Training fails)
- Cannot learn complex mechanisms (gating, redundancy)

**Takeaway:** Architectural and algorithmic innovations cannot overcome fundamental data scarcity. Need to collect more labeled data, not try more techniques.

### 3. Generalization Gap Can Be Misleading

**Bad progress:** Val 90% → 60%, Test 70% → 55%, Gap 20% → 5%
- Gap decreased, but both accuracies dropped (underfitting)

**Good progress:** Val 90% → 90%, Test 70% → 85%, Gap 20% → 5%
- Gap decreased because test improved while val stayed high

**Takeaway:** Always check absolute accuracies, not just the gap. A small gap from universal poor performance is not progress.

### 4. Negative Results are Valuable

These three experiments saved us from:
- Deploying worse models
- Wasting time on similar techniques
- Misunderstanding our core problem

**Takeaway:** Rigorous empirical testing, even when results are negative, clarifies the problem and prevents wasted effort.

### 5. Simpler is Better with Limited Data

**V4-Regularized (9,989 params):** 67-73% test accuracy
**V5-Context (10,934 params):** 54% test accuracy
**V4-DropDim (9,989 params):** 53% test accuracy

**Takeaway:** With limited data, simpler architectures with strong regularization outperform complex architectures. Occam's Razor applies.

---

## What Actually Works: V4-Regularized Progression

While literature techniques failed, **our V2 → V3 → V4 progression succeeded:**

| Model | Test Accuracy | Generalization Gap | Key Improvement |
|-------|---------------|-------------------|-----------------|
| V2 | 58.0% | 39.1% | Baseline |
| V3 | 68.4% | 25.8% | Window size 10, 50% synthetic |
| V4-Regularized | 70.1% | 21.8% | Stronger regularization |

**What worked:**
- Increasing window size (1 → 10)
- Adding tight synthetic data (50% ratio, 1x std)
- Stronger regularization: Dropout (0.4, 0.5, 0.4) + L2 (0.01) + Label smoothing (0.1)

**Why it worked:**
- No architecture complexity increase (kept ~10K params)
- Regularization prevents overfitting with limited data
- Synthetic data increases effective training set without quality degradation

**Takeaway:** Incremental improvements through regularization and careful data augmentation work better than large architectural changes.

---

## Recommendations

### ✅ What to Do

1. **Continue using V4-Regularized** as production model (~70% cross-orientation accuracy)

2. **Collect more labeled data** - This is the ONLY path forward
   - Target: 500-1000 labeled Q1 samples (low pitch)
   - This would enable:
     - Fine-tuning without overfitting
     - Reliable pseudo-labeling (confidence would increase)
     - More complex architectures if needed

3. **Improve synthetic data generation**
   - Use physics-based magnetic field models for more accurate synthetic samples
   - Vary orientation systematically in synthetic data to cover Q1 pitch range

4. **Collect diverse sessions**
   - Multiple users
   - Multiple orientations (Q1, Q2, Q3, Q4 pitch quartiles)
   - Multiple environments (magnetic interference)

### ❌ What NOT to Do

1. **Don't try more literature techniques** without addressing data constraint
   - Self-supervised learning (PINN) - requires unlabeled data quality
   - Meta-learning - requires multiple tasks/domains
   - Transfer learning - requires similar pre-trained models
   - All will likely fail for same reason: insufficient data

2. **Don't increase model complexity**
   - Multi-branch architectures
   - Attention mechanisms
   - Ensemble methods
   - All require more data than we have

3. **Don't rely on pseudo-labeling**
   - Baseline confidence on Q1 is too low (0.669 mean)
   - Only 11.6% of Q1 samples have conf > 0.9
   - Pseudo-labels will be noisy and hurt performance

### 🔬 Experimental Ideas (Lower Priority)

If data collection is not feasible:

1. **Physics-informed synthetic data**
   - Use magnetic dipole model to generate realistic samples across all orientations
   - May improve Q1 generalization if physics model is accurate

2. **Domain randomization**
   - Add noise/perturbations to Q3 samples during training
   - May help model generalize to Q1, but limited evidence

3. **Test-time augmentation**
   - Average predictions across augmented Q1 samples (rotations, noise)
   - May improve accuracy by 1-3% without retraining

---

## Path Forward

### Immediate (Current State)

**Use V4-Regularized in production**
- 67-73% cross-orientation test accuracy (varies by run)
- Best balance of accuracy, model size, and inference speed
- Proven through rigorous V2 → V3 → V4 progression

### Short-Term (1-2 Months)

**Collect more labeled data**
- Priority 1: 200-500 Q1 samples (low pitch)
- Priority 2: 200-500 Q2/Q4 samples (medium pitch)
- Priority 3: Multiple users (cross-subject validation)

**With more data, revisit:**
- Active self-training (needs conf > 0.8 on target domain)
- Context-aware fusion (needs 500+ samples to train gating)

### Long-Term (3-6 Months)

**Physics-informed modeling**
- Develop accurate magnetic dipole simulation
- Generate unlimited synthetic data across all orientations
- Use as pre-training or augmentation

**Multi-session validation**
- Collect 5-10 sessions with diverse conditions
- Validate V4 and future models across all sessions
- Identify systematic failure modes

---

## Conclusion

**We tested three state-of-the-art techniques from recent literature. All three failed to improve our baseline V4-Regularized model.** Test accuracy dropped by 6.9% to 18.5% across all experiments.

**Root cause:** Severe data constraint (~116-152 training samples) prevents any technique that increases complexity or requires sufficient data to learn new mechanisms.

**Key insight:** With limited data, **simpler is better**. V4-Regularized with strong regularization (dropout + L2 + label smoothing) outperforms all complex alternatives (DropDim, context-aware fusion, active self-training).

**Path forward:** The ONLY viable path to significant improvement is **collecting more labeled data**, specifically in the Q1 (low pitch) domain where generalization fails. Trying additional literature techniques without addressing the data constraint will continue to fail.

**Value of negative results:** These experiments clarified our fundamental constraint (data scarcity), prevented deployment of worse models, and saved time on similar approaches. Rigorous empirical testing is essential, even when results are negative.

---

## Final Recommendation

**Production Model:** V4-Regularized (~70% cross-orientation accuracy)

**Next Action:** Collect 200-500 labeled Q1 samples before attempting any further model improvements.

**Long-Term:** Develop physics-based synthetic data generation for unlimited training data across all orientations.

---

## References

### Experiments

1. **DropDim:** `docs/research/dropdim-experiment-results.md` | `ml/test_dropdim_regularization.py`
2. **V5-Context:** `docs/research/v5-context-experiment-results.md` | `ml/test_context_aware_v5.py`
3. **Active Self-Training:** `ml/test_active_self_training.py`

### Literature Sources

1. **DropDim:** Zhang, H., et al. arXiv:2304.10321, Apr 2023.
2. **Chorus:** Zhang, L., et al. arXiv:2512.15206, Dec 2025.
3. **ActiveSelfHAR:** Wei, B., et al. arXiv:2303.15107, Mar 2023.
4. **Literature Review:** `docs/research/arxiv-literature-review.md`

### Related Docs

- **V4 Architecture:** `docs/research/v4-architecture-exploration.md`
- **V2 vs V3 Benchmark:** `docs/research/v2-v3-benchmark-comparison.md`
- **Physics Optimization:** `docs/ml/physics/optimization-report.md`

---

**Report Generated:** January 6, 2026
**Experiments Conducted:** 3 (all negative results)
**Key Lesson:** Data constraint is fundamental - collect more data, don't add complexity
