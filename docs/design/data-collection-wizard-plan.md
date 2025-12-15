# Data Collection Wizard & Multi-Label Model Plan

**Author:** Claude
**Date:** 2025-12-15
**Status:** Proposal
**Related Documents:**
- [Magnetic Finger Tracking Analysis](magnetic-finger-tracking-analysis.md)
- [Orientation & Magnetometer System](../../src/web/GAMBIT/analysis/ORIENTATION_AND_MAGNETOMETER_SYSTEM.md)
- [ML Pipeline README](../../ml/README.md)

---

## Executive Summary

This document outlines a comprehensive plan to enhance the GAMBIT data collection system to support flexible multi-label data collection with auto-labeling capabilities. The goal is to transition from fixed pose classification to a richer model that can output multiple labels simultaneously (poses, finger positions, motion states, etc.).

### Key Objectives

1. **Enhanced Data Collection Wizard**: Guide users through collecting diverse, well-labeled data
2. **Auto-Labeling System**: Reduce manual labeling burden using sensor-driven heuristics and ML inference
3. **Multi-Label ML Pipeline**: Train models that output multiple predictions simultaneously
4. **Iterative Improvement**: Enable users to collect data, train models, and use those models to improve future data collection

---

## Current State Analysis

### What Exists ✅

#### 1. Collector Application
- **Location**: `src/web/GAMBIT/collector.html` + `collector-app.js`
- **Features**:
  - Multi-label UI (poses, per-finger states, motion, calibration markers, custom labels)
  - Manual label selection with active label display
  - Real-time sensor visualization
  - 3D hand visualization with sensor fusion
  - Export to JSON with metadata
  - GitHub upload integration

#### 2. Data Collection Wizard
- **Location**: `src/web/GAMBIT/modules/wizard.js`
- **Current Capabilities**:
  - Guided step-by-step data collection
  - Three modes: Quick, Full, 5-Magnet Finger Tracking
  - Two-phase approach per step: transition (unlabeled) → hold (labeled)
  - Auto-label application based on step ID
  - Fixed wizard templates (hard-coded steps)

**Wizard Modes:**
| Mode | Steps | Duration | Purpose |
|------|-------|----------|---------|
| Quick | 7 steps | ~50s | Reference poses + finger isolation |
| Full | 13 steps | ~80s | Quick + common gestures |
| 5-Magnet | 11 steps | ~85s | All fingers with magnets - full tracking |

#### 3. ML Pipeline
- **Location**: `ml/`
- **Models**:
  - `create_cnn_model_keras()`: Single-output gesture classifier (10 poses)
  - `create_finger_tracking_model_keras()`: **Multi-output model** (5 fingers × 3 states)
- **Training**: Supports both single-label and multi-label scenarios
- **Data Format**: V2.1 JSON with embedded labels and metadata

**Multi-Output Model Architecture** (Already Implemented):
```
Input (50, 9) → Shared CNN → 5 Output Heads
                              ├─ thumb_state (3-class softmax)
                              ├─ index_state (3-class softmax)
                              ├─ middle_state (3-class softmax)
                              ├─ ring_state (3-class softmax)
                              └─ pinky_state (3-class softmax)
```

#### 4. Label Schema
- **Pose labels**: fist, open_palm, pinch, etc. (10 fixed poses)
- **Finger states**: extended(0), partial(1), flexed(2) per finger
- **Motion states**: static, moving, transition
- **Calibration markers**: earth_field, hard_iron, soft_iron, etc.
- **Custom labels**: User-defined tags

### Current Limitations ⚠️

1. **Wizard Inflexibility**:
   - Fixed step definitions hard-coded in `WIZARD_STEPS`
   - Cannot easily add new poses or scenarios
   - No support for dynamic/continuous motion sequences
   - Limited to 2-phase (transition → hold) approach

2. **No Auto-Labeling**:
   - All labels must be manually selected
   - No sensor-driven label suggestions
   - No ML-assisted labeling during collection

3. **Single-Label Training Bias**:
   - Current training scripts assume single pose per window
   - Multi-output model exists but isn't primary workflow
   - Data loader supports multi-label but training defaults to gesture classification

4. **No Iterative Workflow**:
   - Cannot easily use a trained model to assist with labeling new data
   - No "label suggestions" mode during collection

---

## Proposed Enhancements

### Phase 1: Enhanced Wizard System

#### 1.1 Configurable Wizard Templates

**Goal**: Allow custom wizard configurations without code changes.

**Implementation**:
- Create JSON-based wizard template format
- Store templates in `src/web/GAMBIT/wizard-templates/`
- Load templates dynamically at runtime

**Template Format**:
```json
{
  "id": "magnetic_finger_tracking_full",
  "name": "Magnetic Finger Tracking - Full Dataset",
  "description": "Comprehensive collection for finger position tracking",
  "duration_estimate": 120,
  "requires_calibration": true,
  "magnet_config": "alternating",
  "steps": [
    {
      "id": "baseline_reference",
      "type": "static_hold",
      "title": "Baseline Reference",
      "icon": "✋",
      "description": "Palm flat, all fingers extended together",
      "transition_duration": 5,
      "hold_duration": 10,
      "labels": {
        "pose": null,
        "fingers": {
          "thumb": "extended",
          "index": "extended",
          "middle": "extended",
          "ring": "extended",
          "pinky": "extended"
        },
        "motion": "static",
        "custom": ["baseline", "reference_pose"]
      }
    },
    {
      "id": "fist",
      "type": "static_hold",
      "title": "Fist (All Flexed)",
      "icon": "✊",
      "description": "Make a tight fist",
      "transition_duration": 5,
      "hold_duration": 6,
      "labels": {
        "pose": "fist",
        "fingers": {
          "thumb": "flexed",
          "index": "flexed",
          "middle": "flexed",
          "ring": "flexed",
          "pinky": "flexed"
        },
        "motion": "static",
        "custom": ["all_flexed"]
      }
    },
    {
      "id": "thumb_flex_sweep",
      "type": "continuous_motion",
      "title": "Thumb Flex Sweep",
      "icon": "👍",
      "description": "Slowly flex thumb from extended to fully flexed",
      "duration": 8,
      "labels": {
        "pose": null,
        "fingers": {
          "thumb": "dynamic",
          "index": "extended",
          "middle": "extended",
          "ring": "extended",
          "pinky": "extended"
        },
        "motion": "continuous",
        "custom": ["thumb_sweep", "continuous_flex"]
      },
      "auto_label_hints": {
        "use_residual_magnitude": true,
        "finger_to_track": "thumb"
      }
    }
  ]
}
```

**Step Types**:
- `static_hold`: Traditional 2-phase (transition → hold)
- `continuous_motion`: Single phase with dynamic motion
- `multi_pose_sequence`: Multiple holds in sequence without transitions
- `calibration`: Special calibration sequence

#### 1.2 Auto-Label Suggestions

**Goal**: Reduce manual labeling using sensor-driven heuristics.

**Strategies**:

1. **Magnetic Residual-Based**:
   - When magnetometer is calibrated (earth field removed)
   - Track residual magnitude changes over time
   - Suggest finger states based on magnitude thresholds

   ```javascript
   // Auto-label logic example
   function suggestFingerStateFromMag(residual, baseline) {
     const delta = Math.abs(residual - baseline);
     if (delta < 5)  return 'extended';  // Low change
     if (delta < 20) return 'partial';   // Medium change
     return 'flexed';                     // High change
   }
   ```

2. **Motion-Based**:
   - Use gyroscope magnitude to detect static vs moving
   - Suggest `motion: 'static'` when gyro < threshold
   - Suggest `motion: 'moving'` when gyro > threshold

3. **Orientation-Based**:
   - Use IMU orientation (roll/pitch/yaw) to suggest hand orientation labels
   - Example: "palm_up", "palm_down", "vertical"

4. **ML Inference-Based** (Phase 2):
   - Load previously trained model
   - Run real-time inference
   - Suggest labels with confidence scores
   - User can accept/reject suggestions

**UI Changes**:
- Add "Auto-Label" toggle button
- Show suggested labels with confidence badges
- Allow one-click accept or manual override
- Track auto-label vs manual-label usage for analysis

#### 1.3 Enhanced Wizard UI

**Features**:
1. **Progress Timeline**:
   - Visual timeline showing past, current, and upcoming steps
   - Color-coded by label type (pose, finger, motion)

2. **Real-Time Preview**:
   - Show 3D hand visualization during collection
   - Update hand pose based on current labels
   - Show sensor fusion orientation

3. **Label Confidence Indicators**:
   - When auto-labeling is active, show confidence scores
   - Visual feedback: green (confident), yellow (uncertain), red (conflicting)

4. **Flexible Step Control**:
   - Skip step
   - Repeat step
   - Extend hold duration on-the-fly
   - Pause/resume collection

### Phase 2: Multi-Label ML Pipeline

#### 2.1 Enhanced Data Loader

**Current State**: `ml/data_loader.py` supports multi-label but defaults to single-label

**Enhancements**:
1. **Multi-Label Window Extraction**:
   ```python
   def extract_multi_label_windows(session, window_size, stride):
       """
       Extract windows with all label types:
       - pose (optional, 10 classes or null)
       - fingers (5 × 3-class per finger)
       - motion (3 classes: static, moving, transition)
       - custom (multi-hot encoding of active tags)
       """
   ```

2. **Smart Label Aggregation**:
   - For windows spanning multiple labels, choose majority label
   - Option to discard ambiguous windows
   - Option to create "transition" class for ambiguous windows

3. **Data Augmentation**:
   - Time warping (stretch/compress windows slightly)
   - Noise injection (simulate sensor noise)
   - Rotation augmentation (simulate different hand orientations)

#### 2.2 Multi-Output Model Architecture

**Current State**: `create_finger_tracking_model_keras()` exists but is separate from main workflow

**Enhancements**:

1. **Unified Multi-Output Model**:
   ```python
   def create_multi_label_model_keras(
       window_size=50,
       num_features=9,
       outputs_config={
           'pose': {'type': 'categorical', 'num_classes': 10, 'optional': True},
           'fingers': {'type': 'multi_finger', 'num_states': 3},
           'motion': {'type': 'categorical', 'num_classes': 3},
           'custom': {'type': 'multi_hot', 'num_tags': 20}
       }
   ):
       """
       Unified model with flexible output heads.

       Architecture:
         Input → Shared CNN → Output Heads:
                              ├─ pose (10-class softmax, optional)
                              ├─ thumb_state (3-class softmax)
                              ├─ index_state (3-class softmax)
                              ├─ middle_state (3-class softmax)
                              ├─ ring_state (3-class softmax)
                              ├─ pinky_state (3-class softmax)
                              ├─ motion (3-class softmax)
                              └─ custom_tags (20-class sigmoid multi-hot)
       """
   ```

2. **Hierarchical Model** (Advanced):
   - First predict high-level state (fist vs open vs partial)
   - Then predict per-finger details
   - Can improve accuracy by constraining finger combinations

3. **Loss Weighting**:
   - Allow different weights for different outputs
   - Example: prioritize finger states over custom tags
   - Configurable per training run

#### 2.3 Training Pipeline Updates

**New Training Modes**:

1. **Finger-Tracking Mode** (Already exists, enhance):
   ```bash
   python -m ml.train \
     --data-dir data/GAMBIT \
     --model-type finger_tracking \
     --epochs 50
   ```

2. **Multi-Label Mode** (New):
   ```bash
   python -m ml.train \
     --data-dir data/GAMBIT \
     --model-type multi_label \
     --outputs pose,fingers,motion \
     --epochs 50
   ```

3. **Custom Output Mode** (New):
   ```bash
   python -m ml.train \
     --data-dir data/GAMBIT \
     --model-type custom \
     --config custom_model_config.json
   ```

**Evaluation Metrics**:
- Per-output accuracy
- Overall multi-label accuracy (all outputs correct)
- Per-finger confusion matrices
- Pose classification report
- Motion state accuracy

### Phase 3: Iterative Workflow

#### 3.1 Model-Assisted Labeling

**Workflow**:
1. Collect initial dataset with manual labels (wizard-guided)
2. Train first model
3. Export model to TensorFlow.js
4. Load model in collector application
5. Use model to suggest labels for new data collection
6. User reviews and corrects suggestions
7. Retrain model with corrected data
8. Repeat

**UI Integration**:
```javascript
// In collector-app.js
const labelingMode = {
  MANUAL: 'manual',           // User selects all labels
  SUGGEST: 'suggest',          // Model suggests, user confirms
  AUTO: 'auto'                 // Model auto-labels (user can override)
};

// Load trained model
const inferenceModel = await tf.loadLayersModel('models/multi_label_v1/model.json');

// During data collection
function onTelemetrySample(sample) {
  if (labelingMode === 'SUGGEST' || labelingMode === 'AUTO') {
    const prediction = await inferenceModel.predict(windowBuffer);
    const suggestedLabels = decodePrediction(prediction);

    if (labelingMode === 'SUGGEST') {
      // Show suggestions, wait for user confirmation
      showLabelSuggestions(suggestedLabels);
    } else {
      // Auto-apply labels
      applyLabels(suggestedLabels);
    }
  }
}
```

#### 3.2 Active Learning

**Goal**: Collect data that maximally improves the model

**Strategy**:
1. **Uncertainty Sampling**:
   - Model outputs confidence scores
   - When confidence is low, flag for manual review
   - Prioritize collecting data in uncertain regions

2. **Coverage Analysis**:
   - Track which label combinations have been collected
   - Wizard suggests under-represented scenarios
   - Example: "You have 100 samples of 'fist' but only 5 of 'pinch' - collect more pinch data"

3. **Error Analysis**:
   - After training, identify common misclassifications
   - Generate wizard templates targeting those scenarios
   - Example: If model confuses "partial flex" with "extended", create targeted sweep data

---

## Implementation Roadmap

### Sprint 1: Wizard Template System (5-7 days)

**Goal**: Enable custom wizard configurations

**Tasks**:
1. ✅ Design template JSON schema
2. Create template loader module
3. Update wizard.js to accept external templates
4. Create 3-5 example templates:
   - `basic_gestures.json`: Simple hand poses
   - `magnetic_finger_tracking.json`: Full finger tracking
   - `continuous_motion.json`: Dynamic sweeps
   - `calibration_sequence.json`: Mag calibration workflow
   - `active_learning_template.json`: Targeted data collection
5. Add template selection UI in collector
6. Test with existing hardware

**Deliverables**:
- `src/web/GAMBIT/wizard-templates/` directory
- Template schema documentation
- Updated wizard.js with template support
- Template creation guide

### Sprint 2: Auto-Labeling Foundation (5-7 days)

**Goal**: Basic auto-labeling from sensor heuristics

**Tasks**:
1. Implement magnetic residual-based finger state suggestions
2. Implement motion detection (static vs moving)
3. Add auto-label UI toggle and confidence display
4. Create "suggestion mode" where user confirms auto-labels
5. Track auto-label accuracy (compare suggestions to final labels)
6. Add auto-label metadata to exported data

**Deliverables**:
- Auto-labeling module (`auto-labeler.js`)
- Updated collector UI with suggestion mode
- Auto-label metrics in export metadata

### Sprint 3: Multi-Label Training Pipeline (7-10 days)

**Goal**: Train multi-output models end-to-end

**Tasks**:
1. Update `data_loader.py` to extract multi-label windows
2. Create `create_multi_label_model_keras()` in `model.py`
3. Update `train.py` to support `--model-type multi_label`
4. Implement per-output evaluation metrics
5. Test training on existing multi-label data
6. Export to TensorFlow.js format
7. Create deployment guide

**Deliverables**:
- Updated ML pipeline supporting multi-label
- Trained multi-label model (as proof of concept)
- Evaluation report comparing single-label vs multi-label
- TensorFlow.js export

### Sprint 4: ML-Assisted Labeling (7-10 days)

**Goal**: Use trained models to assist data collection

**Tasks**:
1. Create TensorFlow.js inference module for collector
2. Add "Load Model" UI in collector
3. Implement real-time inference during collection
4. Add suggestion UI with confidence scores
5. Add accept/reject controls
6. Create iterative training workflow guide
7. Test full loop: collect → train → assist → retrain

**Deliverables**:
- ML-assisted labeling feature in collector
- Inference performance metrics
- Iterative training guide
- Demo video showing workflow

### Sprint 5: Active Learning & Polish (5-7 days)

**Goal**: Intelligent data collection prioritization

**Tasks**:
1. Implement coverage analysis (track label distribution)
2. Create wizard template generator based on coverage gaps
3. Add uncertainty-based sampling
4. Create dashboard showing data coverage
5. Polish UI/UX based on user testing
6. Write comprehensive documentation
7. Create tutorial videos

**Deliverables**:
- Active learning module
- Data coverage dashboard
- Complete documentation
- Tutorial materials

---

## Technical Considerations

### Performance Constraints

1. **Real-Time Inference**:
   - TensorFlow.js model must run < 50ms per prediction
   - Use quantized models if needed
   - Consider sliding window optimization

2. **Browser Memory**:
   - Large datasets can exhaust browser memory
   - Implement chunked processing for export
   - Clear old samples periodically

3. **Model Size**:
   - Multi-output models are larger (~200KB vs ~150KB)
   - Use compression and quantization for deployment
   - Consider model pruning

### Data Quality

1. **Label Consistency**:
   - Define clear label guidelines
   - Provide visual references for each label
   - Track inter-rater reliability (if multiple labelers)

2. **Transition Handling**:
   - Current wizard uses unlabeled transitions
   - Consider labeling transitions as separate class
   - Or discard transition windows from training

3. **Calibration Dependency**:
   - Magnetic finger tracking requires good calibration
   - Wizard should enforce calibration check before finger tracking
   - Include calibration quality in metadata

### Edge Cases

1. **Missing Labels**:
   - Some windows may have pose but no finger states
   - Model must handle optional outputs
   - Use masked loss for missing labels

2. **Conflicting Labels**:
   - Pose "fist" implies all fingers flexed
   - Validate label consistency before training
   - Auto-correct or flag conflicts

3. **Continuous Motion**:
   - Hard to assign single label to moving windows
   - Consider regression outputs (flex angle 0-1) instead of classification
   - Or use temporal models (LSTM/Transformer)

---

## Success Metrics

### Data Collection

- **Wizard Adoption**: % of data collected via wizard vs manual
- **Auto-Label Accuracy**: % of auto-labels accepted without changes
- **Collection Time**: Time to collect 1000 labeled samples (target: < 10 minutes)
- **Label Coverage**: % of label combinations represented in dataset

### Model Performance

- **Overall Accuracy**: % of windows with all labels correct
- **Per-Output Accuracy**: Accuracy for pose, fingers, motion separately
- **Inference Speed**: Time to run inference in browser (target: < 50ms)
- **Model Size**: Size of deployed model (target: < 250KB quantized)

### User Experience

- **Labeling Efficiency**: Time saved using auto-labeling vs manual
- **Ease of Use**: User survey ratings (1-5 scale)
- **Error Rate**: % of mislabeled data caught in review
- **Iteration Cycle**: Time from data collection to trained model (target: < 30 minutes)

---

## Open Questions

1. **Continuous Motion Modeling**:
   - Should we support regression outputs (flex angle 0-1)?
   - Or stick to classification (extended/partial/flexed)?
   - Consider temporal models (LSTM) for motion sequences?

2. **Calibration Standardization**:
   - Should wizard enforce calibration before finger tracking?
   - How to handle sessions with different calibration quality?
   - Include calibration quality metric in model input?

3. **Multi-User Generalization**:
   - Train per-user models or universal model?
   - How to handle different hand sizes?
   - Use transfer learning from universal to personal model?

4. **Label Granularity**:
   - 3-state finger model (extended/partial/flexed) sufficient?
   - Or need finer granularity (5-7 states)?
   - Trade-off between detail and model complexity?

5. **Deployment Strategy**:
   - Browser-only inference (TensorFlow.js)?
   - Or support ESP32 on-device inference (TFLite)?
   - Hybrid approach (browser for data collection, ESP32 for real-time control)?

---

## Next Steps

### Immediate Actions

1. **Review this plan** with project stakeholders
2. **Prioritize features** - which sprints are most valuable?
3. **Validate technical approach** - any blockers or constraints?
4. **Assign resources** - who will implement which components?
5. **Set up development environment** - ensure all dependencies installed

### Quick Win (1-2 days)

**Goal**: Demonstrate end-to-end multi-label workflow

**Scope**:
1. Create one custom wizard template (JSON file)
2. Collect 100 samples using wizard
3. Train multi-output finger tracking model
4. Export to TensorFlow.js
5. Show demo of real-time inference in collector

**Success**: Proves concept, builds confidence, identifies blockers

---

## Appendix

### A. Template JSON Schema

```typescript
interface WizardTemplate {
  id: string;
  name: string;
  description: string;
  duration_estimate: number;  // seconds
  requires_calibration: boolean;
  magnet_config: 'none' | 'single_index' | 'alternating' | 'custom';
  steps: WizardStep[];
}

interface WizardStep {
  id: string;
  type: 'static_hold' | 'continuous_motion' | 'multi_pose_sequence' | 'calibration';
  title: string;
  icon: string;
  description: string;

  // For static_hold:
  transition_duration?: number;
  hold_duration?: number;

  // For continuous_motion:
  duration?: number;

  // For multi_pose_sequence:
  poses?: Array<{labels: MultiLabel, duration: number}>;

  labels: MultiLabel;
  auto_label_hints?: AutoLabelHints;
}

interface MultiLabel {
  pose?: string | null;
  fingers?: {
    thumb: 'extended' | 'partial' | 'flexed' | 'dynamic' | null;
    index: 'extended' | 'partial' | 'flexed' | 'dynamic' | null;
    middle: 'extended' | 'partial' | 'flexed' | 'dynamic' | null;
    ring: 'extended' | 'partial' | 'flexed' | 'dynamic' | null;
    pinky: 'extended' | 'partial' | 'flexed' | 'dynamic' | null;
  };
  motion?: 'static' | 'moving' | 'continuous' | 'transition';
  custom?: string[];
}

interface AutoLabelHints {
  use_residual_magnitude?: boolean;
  finger_to_track?: 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';
  confidence_threshold?: number;
}
```

### B. Multi-Label Model Output Format

```typescript
interface MultiLabelPrediction {
  pose: {
    class: string;
    confidence: number;
    probabilities: {[pose: string]: number};
  } | null;

  fingers: {
    thumb: {state: 0|1|2, confidence: number, probabilities: [number, number, number]};
    index: {state: 0|1|2, confidence: number, probabilities: [number, number, number]};
    middle: {state: 0|1|2, confidence: number, probabilities: [number, number, number]};
    ring: {state: 0|1|2, confidence: number, probabilities: [number, number, number]};
    pinky: {state: 0|1|2, confidence: number, probabilities: [number, number, number]};
  };

  motion: {
    class: 'static' | 'moving' | 'transition';
    confidence: number;
    probabilities: {[motion: string]: number};
  };

  custom_tags: {
    [tag: string]: {active: boolean, confidence: number};
  };
}
```

### C. Reference Documentation

- [TensorFlow.js Multi-Output Models](https://www.tensorflow.org/js/guide/models_and_layers#multi-output_models)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [Active Learning Survey](https://arxiv.org/abs/2009.00236)
- [Multi-Label Classification Best Practices](https://arxiv.org/abs/1502.05988)

---

**End of Document**
