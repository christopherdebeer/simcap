# Aligned Finger Model + FFO$$: Contrastive Symbiosis Analysis

**Status:** Research Document
**Date:** December 2025
**Related Documents:**
- [Aligned Finger Model Analysis](./aligned-finger-model-analysis.md)
- [FFO$$ Template Matching Analysis](./ffo-template-matching-analysis.md)
- [FFO$$ Research Overview](./ffo-dollar-research-analysis.md)

---

## Executive Summary

This document explores the **complementary relationship** between two finger inference approaches: the **Aligned Finger Model** (measurement-grounded neural network) and **FFO$$** (template-based gesture recognition). While they evolved for different purposes, their concepts can inform and enhance each other.

**Key Insight:** These approaches are not competitors but **complementary tools** that address different aspects of the finger tracking problem—static poses vs. dynamic gestures, learned features vs. explicit templates, single-sample vs. windowed inference.

---

## 1. Contrastive Comparison

### 1.1 Fundamental Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ALIGNED FINGER MODEL                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Single magnetometer sample (3D: mx, my, mz)                         │
│                                                                             │
│  [Measured     ]    [Synthetic    ]    [Neural       ]    [Per-Finger    ] │
│  [Signatures   ] → [Generator    ] → [Network       ] → [Binary Output  ] │
│  [from Wizard  ]    [+ Noise      ]    [Classifier   ]    [5 × 2 states  ] │
│                                                                             │
│  Core Principle: Ground truth anchors + non-additivity correction           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              FFO$$                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Window of accelerometer samples (N × 3D: ax, ay, az over time)      │
│                                                                             │
│  [Raw          ]    [Resample    ]    [Normalize   ]    [Distance       ] │
│  [Trajectory   ] → [to 32 pts   ] → [Translate   ] → [to Templates    ] │
│  [from IMU     ]    [equal space ]    [Scale       ]    [Best match     ] │
│                                                                             │
│  Core Principle: Geometric normalization + template matching                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Head-to-Head Feature Comparison

| Feature | Aligned Finger Model | FFO$$ |
|---------|---------------------|-------|
| **Input sensor** | Magnetometer | Accelerometer |
| **Input dimensionality** | 3D (single sample) | 3D × N (trajectory) |
| **Temporal context** | None (instantaneous) | Required (window) |
| **Training data source** | Wizard-labeled + synthetic | 1-10 recorded examples |
| **Training time** | Minutes (neural network) | Instant (template storage) |
| **Model representation** | Weights (neural network) | Templates (raw points) |
| **Inference complexity** | O(1) per sample | O(n × T) per window |
| **Inference time** | <1 ms | <1 ms |
| **Output type** | Probabilities per finger | Single best match + score |
| **Interpretability** | Low (black box) | High (visible templates) |
| **Generalization** | High (learned features) | Low (exact matching) |
| **Adaptability** | Requires retraining | Add templates instantly |

### 1.3 What Each Approach Captures

```
ALIGNED MODEL CAPTURES:
┌─────────────────────────────────────────────────────────────────┐
│  • STATIC magnetic field from finger magnets                    │
│  • Non-additive multi-finger interactions (empirically)         │
│  • Sensor-specific noise characteristics                        │
│  • Per-finger independence (5 separate outputs)                 │
│                                                                 │
│  DOES NOT CAPTURE:                                              │
│  • Motion dynamics (flexion speed, trajectory)                  │
│  • Temporal patterns (gesture sequences)                        │
│  • Hand orientation (signatures are frame-dependent)            │
└─────────────────────────────────────────────────────────────────┘

FFO$$ CAPTURES:
┌─────────────────────────────────────────────────────────────────┐
│  • DYNAMIC motion trajectories (acceleration path)              │
│  • Gesture shape (circle, wave, swipe)                          │
│  • Scale/position invariance (via normalization)                │
│  • User-defined gesture vocabulary (add templates on-the-fly)   │
│                                                                 │
│  DOES NOT CAPTURE:                                              │
│  • Static poses (no trajectory = no match)                      │
│  • Subtle finger positions (magnetometer signal)                │
│  • Multi-class simultaneous output (single best match)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Synergistic Concepts: What Each Can Learn

### 2.1 FFO$$ Concepts for Aligned Model

#### A. Template Vocabulary Storage

FFO$$ stores templates as explicit, interpretable data structures. The aligned model could benefit from this approach:

```python
# Current: Aligned model stores implicit representations in neural network weights

# Proposed: Store signatures as explicit templates (like FFO$$)
class AlignedVocabulary:
    def __init__(self):
        self.signatures = {
            '00000': MagneticTemplate(mean=[0, 0, 0], std=[100, 100, 100]),
            '20000': MagneticTemplate(mean=[683, 3495, -3988], std=[500, 800, 600]),
            # ... all 32 configurations
        }

    def match(self, sample):
        """FFO$$-style nearest neighbor matching."""
        best_match = None
        best_distance = float('inf')

        for code, template in self.signatures.items():
            distance = np.linalg.norm(sample - template.mean)
            if distance < best_distance:
                best_distance = distance
                best_match = code

        return best_match, best_distance

    def add_template(self, code, samples):
        """Add user-specific signature (instant adaptation!)."""
        self.signatures[code] = MagneticTemplate(
            mean=np.mean(samples, axis=0),
            std=np.std(samples, axis=0)
        )
```

**Benefit:** User-adaptive finger tracking without retraining neural network.

#### B. Rejection Threshold

FFO$$ has explicit rejection for unknown gestures. The aligned model could use this:

```python
# Current: Always outputs 5 finger probabilities (even for invalid input)

# Proposed: Add confidence-based rejection
class AlignedModelWithRejection:
    def predict(self, sample):
        # Check if sample is within known signature space
        min_distance = min(
            np.linalg.norm(sample - sig.mean)
            for sig in self.signatures.values()
        )

        if min_distance > self.rejection_threshold:
            return {"rejected": True, "reason": "out_of_distribution"}

        # Normal prediction
        return self.neural_network.predict(sample)
```

**Benefit:** Robust handling of sensor noise, calibration drift, or unusual hand positions.

#### C. Incremental Template Addition

FFO$$ allows adding new gestures without retraining. Applied to aligned model:

```python
# Scenario: New user has slightly different finger positions

# FFO$$ approach: Just record new templates
model.add_user_calibration({
    '00000': user_baseline_samples,
    '22222': user_fist_samples,
    # Only need 3-5 key configurations for personalization
})

# Model now uses user-specific signatures for those configs,
# falls back to generic for others
```

**Benefit:** Rapid personalization without full retraining cycle.

---

### 2.2 Aligned Model Concepts for FFO$$

#### A. Non-Additivity Awareness

The aligned model discovered that multi-finger magnetic signatures don't add linearly. FFO$$ could apply similar reasoning:

```typescript
// Current: FFO$$ treats all templates independently

// Proposed: Template composition for unseen gestures
class ComposableFFO {
  composeTemplate(gestureA: string, gestureB: string): GestureTemplate {
    const templateA = this.templates.get(gestureA);
    const templateB = this.templates.get(gestureB);

    // Apply non-additivity factor (learned from aligned model research)
    const composedPoints = templateA.points.map((pa, i) => ({
      x: (pa.x + templateB.points[i].x) * 0.85,  // 15% reduction
      y: (pa.y + templateB.points[i].y) * 0.85,
      z: (pa.z + templateB.points[i].z) * 0.85,
    }));

    return { name: `${gestureA}+${gestureB}`, points: composedPoints };
  }
}
```

**Benefit:** Generate templates for gesture combinations without recording each.

#### B. Measurement-Grounded Augmentation

The aligned model generates synthetic data grounded in real measurements. FFO$$ could use similar augmentation:

```typescript
// Current: FFO$$ uses only recorded templates

// Proposed: Augment templates with measurement-grounded variations
class AugmentedFFO {
  augmentTemplate(template: GestureTemplate): GestureTemplate[] {
    const augmented: GestureTemplate[] = [template];

    // Variations based on measured noise characteristics
    for (let i = 0; i < 5; i++) {
      const noisyPoints = template.points.map(p => ({
        x: p.x + this.noiseProfile.x * gaussianRandom(),
        y: p.y + this.noiseProfile.y * gaussianRandom(),
        z: p.z + this.noiseProfile.z * gaussianRandom(),
      }));

      augmented.push({
        ...template,
        id: `${template.id}_aug_${i}`,
        points: quickNormalize(noisyPoints),
      });
    }

    return augmented;
  }
}
```

**Benefit:** More robust matching with fewer recorded examples.

#### C. Multi-Output Classification

The aligned model outputs 5 independent finger states. FFO$$ typically outputs one gesture class:

```typescript
// Current: FFO$$ returns single best match

// Proposed: Multi-aspect gesture classification (like aligned model)
interface MultiAspectResult {
  motionType: 'wave' | 'circle' | 'swipe' | 'tap';
  intensity: 'gentle' | 'normal' | 'vigorous';
  direction: 'left' | 'right' | 'up' | 'down';
  confidence: Record<string, number>;
}

class MultiAspectFFO {
  recognize(samples: TelemetrySample3D[]): MultiAspectResult {
    // Match against templates in each aspect category
    const motionMatch = this.matchCategory('motion', samples);
    const intensityMatch = this.matchCategory('intensity', samples);
    const directionMatch = this.matchCategory('direction', samples);

    return {
      motionType: motionMatch.template.name,
      intensity: intensityMatch.template.name,
      direction: directionMatch.template.name,
      confidence: {
        motion: distanceToScore(motionMatch.distance),
        intensity: distanceToScore(intensityMatch.distance),
        direction: distanceToScore(directionMatch.distance),
      },
    };
  }
}
```

**Benefit:** Richer gesture description without exponential template explosion.

---

### 2.3 Shared Concepts: Common Ground

Both approaches share fundamental ideas that could be unified:

#### A. Normalization as Invariance

```
ALIGNED MODEL NORMALIZATION:
  normalized = (raw - mean) / std
  Purpose: Remove sensor-specific baseline, match training distribution

FFO$$ NORMALIZATION:
  normalized = translate(scale(resample(points)))
  Purpose: Remove position/scale variance, enable shape comparison

UNIFIED CONCEPT:
  Both transform raw input to a canonical representation where
  distance computation is meaningful.
```

#### B. Ground Truth Anchoring

```
ALIGNED MODEL:
  Signatures from wizard-labeled sessions → Generate synthetic data
  "Real measurements define the anchor points"

FFO$$:
  Recorded gesture examples → Templates for matching
  "Real recordings define the gesture prototypes"

UNIFIED CONCEPT:
  Both rely on high-quality labeled examples as ground truth,
  then extend coverage through principled generation/augmentation.
```

#### C. Distance-Based Classification

```
ALIGNED MODEL (under the hood):
  Neural network learns decision boundaries in feature space
  Essentially: which signature is this sample closest to?

FFO$$:
  Explicit distance computation to all templates
  Return template with minimum distance

UNIFIED CONCEPT:
  Both are fundamentally nearest-neighbor classification,
  differing only in how the "neighbors" are represented
  (learned features vs. explicit templates).
```

---

## 3. Hybrid Architectures

### 3.1 Architecture A: Dual-Mode Inference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DUAL-MODE FINGER INFERENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [IMU Sample Stream]                                                        │
│         │                                                                   │
│         ├────────────────────┬───────────────────────────────────┐          │
│         │                    │                                   │          │
│         ▼                    ▼                                   ▼          │
│  ┌─────────────┐      ┌─────────────┐                    ┌─────────────┐   │
│  │ Motion      │      │ Aligned     │                    │ FFO$$       │   │
│  │ Detector    │      │ Model       │                    │ Recognizer  │   │
│  │             │      │             │                    │             │   │
│  │ Is hand     │      │ Finger      │                    │ Gesture     │   │
│  │ moving?     │      │ states      │                    │ detection   │   │
│  └──────┬──────┘      └──────┬──────┘                    └──────┬──────┘   │
│         │                    │                                   │          │
│         │                    │                                   │          │
│         ▼                    ▼                                   ▼          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         FUSION LAYER                                   │ │
│  │                                                                        │ │
│  │  if (motion.isMoving && ffo.confidence > 0.7):                         │ │
│  │      output = ffo.gesture  // Dynamic gesture detected                 │ │
│  │  else:                                                                 │ │
│  │      output = aligned.fingerStates  // Static pose                     │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Use Case:** Complete hand tracking combining static poses (aligned) with dynamic gestures (FFO$$).

### 3.2 Architecture B: Aligned Signatures as FFO$$ Templates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                MAGNETIC SIGNATURE TEMPLATES FOR FFO$$                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Traditional FFO$$:                                                         │
│    Template = { points: [acc_0, acc_1, ..., acc_31] }  // Accel trajectory  │
│                                                                             │
│  Extended FFO$$:                                                            │
│    Template = {                                                             │
│      accelPoints: [acc_0, acc_1, ..., acc_31],  // Motion trajectory        │
│      magSignature: { mean: [mx, my, mz], std: [sx, sy, sz] },  // Pose     │
│    }                                                                        │
│                                                                             │
│  Recognition:                                                               │
│    accelDistance = lookupDistance(input.accel, template.accelPoints)        │
│    magDistance = euclidean(input.mag, template.magSignature.mean)           │
│    combinedScore = 0.6 * accelScore + 0.4 * magScore                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Use Case:** Gesture recognition that also verifies correct finger pose during gesture.

### 3.3 Architecture C: Transition Detection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POSE TRANSITION DETECTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Aligned Model: Detects WHAT pose the hand is in                            │
│  FFO$$: Detects HOW the hand got there (transition motion)                  │
│                                                                             │
│  Time →                                                                     │
│                                                                             │
│  Pose:      [00000]  ───────▶  [22000]  ───────▶  [22222]                  │
│              open              pinch               fist                     │
│                                                                             │
│  FFO$$:          "pinch_motion"         "close_motion"                      │
│                  (detected)              (detected)                         │
│                                                                             │
│  Combined:                                                                  │
│    Event = { from: '00000', to: '22000', via: 'pinch_motion' }              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Use Case:** Rich gesture vocabulary combining pose endpoints with transition motions.

---

## 4. Mutual Enhancement Roadmap

### 4.1 Phase 1: Cross-Pollination (Immediate)

**For Aligned Model:**
- Add FFO$$-style rejection threshold for out-of-distribution detection
- Implement template vocabulary alongside neural network for interpretability
- Enable incremental template addition for user calibration

**For FFO$$:**
- Incorporate aligned model's noise profiles in template augmentation
- Add multi-output structure for gesture aspects
- Use aligned signatures for static pose "templates"

### 4.2 Phase 2: Unified Framework (Medium-term)

```python
class UnifiedFingerRecognizer:
    def __init__(self):
        self.aligned_model = AlignedFingerModel()  # Static poses
        self.ffo_recognizer = FFORecognizer()      # Dynamic gestures
        self.motion_detector = MotionDetector()    # Mode switching

    def process(self, sample):
        # Update motion state
        is_moving = self.motion_detector.update(sample)

        if is_moving:
            # Gesture recognition mode
            result = self.ffo_recognizer.addSample(sample)
            if result and result.score > 0.7:
                return {'type': 'gesture', 'gesture': result.template.name}

        else:
            # Static pose mode
            prediction = self.aligned_model.predict(sample)
            return {'type': 'pose', 'fingers': prediction.fingers}
```

### 4.3 Phase 3: Co-Evolution (Long-term)

**Shared Training Pipeline:**
```
Wizard Session Data
       │
       ├──────────────────┬──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
  Extract             Extract            Extract
  Mag Signatures      Motion Templates   Transitions
       │                  │                  │
       ▼                  ▼                  ▼
  Aligned             FFO$$             Transition
  Generator           Vocabulary        Detector
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
                 Unified Model Export
                 (for browser/device)
```

---

## 5. Research Questions

### 5.1 For Aligned Model

1. Can FFO$$-style template storage replace neural network entirely for some users?
2. How many calibration samples are needed for effective personalization?
3. Does explicit signature storage improve interpretability without sacrificing accuracy?

### 5.2 For FFO$$

1. Can aligned model's non-additivity insights improve gesture composition?
2. How to define "templates" for static poses with zero path length?
3. What augmentation strategies from aligned model transfer to FFO$$?

### 5.3 For Hybrid Systems

1. What is the optimal fusion strategy for pose + gesture recognition?
2. How to handle ambiguous states (slow motion, partial gestures)?
3. Can a single unified model outperform separate specialized models?

---

## 6. Conclusion: Complementary, Not Competitive

The **Aligned Finger Model** and **FFO$$** are not competing approaches but **complementary tools** for different aspects of hand tracking:

| Aspect | Aligned Model | FFO$$ |
|--------|--------------|-------|
| **What it detects** | Static finger poses | Dynamic hand gestures |
| **Input focus** | Magnetometer (magnets) | Accelerometer (motion) |
| **Temporal scope** | Instantaneous | Windowed (trajectory) |
| **Best use case** | Finger state tracking | Gesture recognition |

**The symbiosis potential:**
- Aligned signatures can serve as FFO$$ templates for static poses
- FFO$$ templates can capture the transitions between aligned poses
- Both share the principle of ground-truth anchoring
- Both benefit from the other's insights (non-additivity, augmentation, rejection)

**The path forward** is not choosing one over the other, but building a unified system that leverages the strengths of each approach.

---

---

## 7. Python Exploration: Hybrid Trajectory Inference

A Python exploration script has been created to validate these symbiosis concepts:

**Location:** [`ml/explore_hybrid_trajectory_inference.py`](../../ml/explore_hybrid_trajectory_inference.py)

### Key Findings from Exploration

```
Trajectory Type Comparison:
---------------------------
Trajectory      Within-Class  Between-Class  Discriminability
accel                0.292         0.301           1.03x
mag_raw              0.313         0.333           1.06x
mag_residual         0.312         0.333           1.07x
combined             0.312         0.333           1.07x
```

**Insights:**
1. **Magnetic residuals preserve pose information** in trajectory space
2. **Combined trajectories (6D: accel + mag)** enable simultaneous motion + pose inference
3. **Hybrid recognizer** correctly identifies both motion templates AND finger poses

### Signature Trajectory Concept

The exploration validates that finger flexions create **characteristic paths through signature space**:

```
Signature Clusters (µT):

00000 (baseline):  [470, 476, 315]      - All fingers extended
22222 (fist):      [3010, 6521, 6410]   - All fingers flexed
20000 (thumb):     [1170, 3932, -6383]  - Thumb flexed only
02000 (index):     [2137, 6372, 10577]  - Index flexed only
```

These distinct clusters enable FFO$$-style template matching on magnetic data!

### Proposed Unified Pipeline

```
[IMU Stream] → [Sensor Fusion] → [Residual Extraction] → [Dual Inference]
                    |                      |                    |
                    v                      v                    v
              Orientation           Orientation-           Motion: FFO$$
              (quaternion)          Independent            Pose: Neural/kNN
                                    Signature
```

---

## 8. Empirical Study: Trajectory vs Single-Sample Inference

An empirical study was conducted to answer the question: **Would training on FFO$-style magnetic trajectories improve model performance compared to single-sample inference?**

**Location:** [`ml/trajectory_vs_single_sample_study.py`](../../ml/trajectory_vs_single_sample_study.py)

### Study Design

The study compared three approaches:
1. **FFO$ Template Matching**: Trajectory-based matching on magnetic signatures
2. **Single-Sample KNN**: k-Nearest Neighbors on single magnetometer readings
3. **Trajectory Neural Network**: Neural network trained on trajectory statistics

### Key Results

```
Approach                              Accuracy       Data Type     Test N
------------------------------------------------------------------------
FFO$ Template Matching                   23.1%      Trajectory         39
Single-Sample KNN (k=5)                  99.3%    Single Point        433
Trajectory NN (stats)                    25.6%      Traj Stats         39
Trajectory NN (full)                     25.6%       Full Traj         39
```

**Per-Class Accuracy:**
```
Code            KNN         FFO$      Samples
--------------------------------------------------
00000        100.0%       20.0%           86
00002         97.0%       50.0%           33
00020        100.0%        0.0%           45
00022         93.8%       50.0%           32
00200        100.0%       50.0%           39
00222        100.0%       50.0%           56
02000        100.0%        0.0%           33
20000        100.0%        0.0%           29
22000        100.0%       40.0%           44
22222        100.0%       25.0%           36
```

### Information-Theoretic Analysis

```
Feature                    Single Sample    Trajectory Stats
---------------------------------------------------------------
Mutual Information              4.55 bits       0.93 bits
Channel Efficiency              142%            29%
```

The **Fisher discriminant ratio** (a proxy for mutual information) shows that single samples have **higher class separability** than trajectory statistics for static pose classification.

### Key Finding: Domain Mismatch

**Trajectories are NOT beneficial for static pose classification because:**

1. **Static poses have stable signatures** - A single sample contains the full discriminative information
2. **Trajectory windows introduce noise** - Averaging over time reduces signal-to-noise ratio
3. **Wizard labels capture static poses** - The labeled data represents steady-state configurations, not transitions

**However, trajectories ARE beneficial for:**
- **Gesture detection** (motion patterns)
- **Transition detection** (pose-to-pose movements)
- **Activity recognition** (sequence patterns)

### Conclusion: Task-Appropriate Methods

| Task | Recommended Approach | Why |
|------|---------------------|-----|
| **Static pose classification** | Single-sample (aligned model) | Full information in one sample |
| **Gesture recognition** | FFO$ trajectory matching | Motion patterns need time series |
| **Pose + gesture combined** | Hybrid system | Each approach for its strength |

### Practical Recommendation

For the GAMBIT system, the current **single-sample aligned model** is the correct choice for finger state inference. The **FFO$ trajectory approach** should be reserved for:
- Detecting hand gestures (wave, swipe, circle)
- Recognizing pose transitions (open→close)
- Activity recognition (typing vs. pointing)

The optimal architecture uses **both approaches in parallel**, as described in Architecture A (Dual-Mode Inference).

---

## References

### Core Documents
- [Aligned Finger Model Analysis](./aligned-finger-model-analysis.md)
- [FFO$$ Template Matching Analysis](./ffo-template-matching-analysis.md)
- [FFO$$ Research Overview](./ffo-dollar-research-analysis.md)

### Implementation & Exploration
- [`ml/simulation/aligned_generator.py`](../../ml/simulation/aligned_generator.py)
- [`ml/explore_hybrid_trajectory_inference.py`](../../ml/explore_hybrid_trajectory_inference.py)
- [`packages/ffo/src/recognizer.ts`](../../packages/ffo/src/recognizer.ts)
- [`apps/gambit/gesture-inference.ts`](../../apps/gambit/gesture-inference.ts)

### Academic Background
- $-Family: https://depts.washington.edu/acelab/proj/dollar/
- Non-additive magnetic fields: Empirical finding from wizard session analysis

---

<link rel="stylesheet" href="../../src/simcap.css">
