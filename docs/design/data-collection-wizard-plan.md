# Data Collection Wizard & Multi-Label Model Plan

**Author:** Claude
**Date:** 2025-12-15
**Status:** Proposal (Revised)
**Related Documents:**
- [Magnetic Finger Tracking Analysis](magnetic-finger-tracking-analysis.md)
- [Orientation & Magnetometer System](../../src/web/GAMBIT/analysis/ORIENTATION_AND_MAGNETOMETER_SYSTEM.md)
- [ML Pipeline README](../../ml/README.md)

---

## Executive Summary

This plan outlines a **progressive, wizard-driven data collection system** for training multi-label hand tracking models. The wizard guides users through pose instructions, automatically labels data during recording, and supports tiered complexity progression from simple poses to fine-grained finger tracking.

### Core Concept: Wizard-Driven Auto-Labeling

**Traditional Manual Approach (Current):**
```
User → Manually select labels → Record data → Export
```

**New Wizard Approach:**
```
Wizard → Show instruction "👊 Make a fist"
       → User adopts pose
       → User clicks "Ready - Record"
       → Auto-apply labels {pose: "fist", fingers: {all: "flexed"}}
       → Record 6-10s while user rotates/moves hand
       → Auto-save labeled segment
       → Next instruction
```

### Key Principles

1. **Zero manual labeling during wizard**: Labels are automatically applied based on the instruction given
2. **Movement encouraged**: Users rotate/move hand during recording to capture orientation diversity (prevents overfitting)
3. **Progressive complexity**: Start simple (whole-hand poses) → advance to complex (per-finger control)
4. **Explicit control**: User decides when ready to record each pose
5. **Safe exit**: Can exit wizard anytime without losing collected data
6. **Validation gates**: Train and validate after each tier before advancing

---

## Progressive Complexity Tiers

### Tier 1: Basic Whole-Hand Poses

**Goal**: Train a simple classifier on poses where all fingers are in the same state.

**Poses (5-8 total):**
| Pose | Description | Finger States | Training Target |
|------|-------------|---------------|-----------------|
| Rest | Hand relaxed, palm down | All: unknown | Baseline |
| Fist | Tight fist | All: flexed | Core pose |
| Open Palm | All fingers extended, spread | All: extended | Core pose |
| Thumbs Up | Thumb up, others flexed | Thumb: extended, Others: flexed | Simple variation |
| Point | Index up, others flexed | Index: extended, Others: flexed | Simple variation |

**Collection Strategy:**
- 10 seconds per pose × 5 poses = ~50 seconds of labeled data
- Encourage rotation/movement during each 10-second hold
- Transition time: 5 seconds between poses (unlabeled)

**Training:**
- Model: Simple 5-8 class gesture classifier
- Input: 50-sample windows (1 second @ 50Hz)
- Output: Single pose prediction
- Success: >85% validation accuracy

**Validation Gate:**
- Test on held-out data
- Per-class accuracy > 75%
- If pass → Tier 2, If fail → collect more Tier 1 data

### Tier 2: Single-Finger Isolation

**Goal**: Train multi-output model to predict per-finger states (one finger varies, others fixed).

**Poses (10-12 total):**
| Pose | Description | Finger States |
|------|-------------|---------------|
| Reference | All extended | All: extended |
| Thumb Flex Only | Thumb flexed, others extended | Thumb: flexed, Others: extended |
| Index Flex Only | Index flexed, others extended | Index: flexed, Others: extended |
| Middle Flex Only | ... | ... |
| Ring Flex Only | ... | ... |
| Pinky Flex Only | ... | ... |
| Thumb Extended Only | All flexed except thumb | Thumb: extended, Others: flexed |
| ... | (mirror for other fingers) | ... |

**Collection Strategy:**
- 8 seconds per pose × 10 poses = ~80 seconds
- Focus on magnetic field changes (requires calibrated magnetometer)
- Movement encouraged but less critical than Tier 1

**Training:**
- Model: Multi-output (5 fingers × 3 states each)
- Architecture: Shared CNN → 5 output heads
- Output: `{thumb_state: 0-2, index_state: 0-2, ...}`
- Success: >75% per-finger accuracy

**Validation Gate:**
- Per-finger confusion matrices
- Check for finger independence (index prediction not influenced by thumb state)
- If pass → Tier 3, If fail → collect more single-finger data

### Tier 3: Complex Finger Combinations

**Goal**: Handle arbitrary finger combinations for real-world gestures.

**Poses (15-20 total):**
| Pose | Description | Finger States |
|------|-------------|---------------|
| Pinch | Thumb + index tips touching | Thumb: partial, Index: partial, Others: extended |
| Peace | Index + middle up, others flexed | Index: extended, Middle: extended, Others: flexed |
| OK Sign | Thumb + index circle, others extended | Thumb: partial, Index: partial, Others: extended |
| Shaka | Thumb + pinky extended, others flexed | Thumb: extended, Pinky: extended, Others: flexed |
| ... | Custom combinations | ... |

**Collection Strategy:**
- 6-8 seconds per pose × 15 poses = ~90-120 seconds
- Include partial flexion states (not just 0 or 2, but 1 as well)
- Diverse orientations critical

**Training:**
- Refine Tier 2 model with additional data
- May need increased model capacity (more filters)
- Consider ensemble or hierarchical approach

**Deployment:**
- Export to TensorFlow.js for browser inference
- Real-time pose estimation in collector app
- Use for future data collection assistance

---

## Wizard User Experience

### Three-Phase Per Pose

Each wizard step has three phases:

#### Phase 1: Preview (Instruction Display)
```
┌─────────────────────────────────────┐
│  Step 3 of 8                        │
│                                     │
│  👊 Make a Fist                     │
│                                     │
│  Close all fingers tightly into    │
│  your palm. Rotate and move your   │
│  hand in different orientations    │
│  while recording.                   │
│                                     │
│  [🎯 See Example] [➡️  Next]        │
└─────────────────────────────────────┘
```

**User Actions:**
- Read instruction
- Optionally view example video/image
- Click "Next" when understood

#### Phase 2: Prepare (Adopt Pose)
```
┌─────────────────────────────────────┐
│  👊 Make a Fist                     │
│                                     │
│  🎥 Camera Preview (optional)       │
│  [Hand visualization showing        │
│   current sensor readings]          │
│                                     │
│  Take your time to adopt the pose. │
│  When ready, click Record.          │
│                                     │
│  [⏸️  Pause Wizard] [🔴 Ready - Record] │
└─────────────────────────────────────┘
```

**User Actions:**
- Adopt the pose
- Check preview to verify pose is correct
- Click "Ready - Record" when comfortable
- Or "Pause Wizard" to take a break

#### Phase 3: Record (Collect Data)
```
┌─────────────────────────────────────┐
│  🔴 RECORDING                        │
│  👊 Make a Fist                     │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ 7.3 seconds                   │ │
│  │ ████████████░░░░░░░░░ 73%     │ │
│  └───────────────────────────────┘ │
│                                     │
│  Rotate and move your hand         │
│  to capture different angles       │
│                                     │
│  [⏹️  Stop Recording]               │
└─────────────────────────────────────┘
```

**Auto-Labeling Happens Here:**
- Wizard applies labels: `{pose: "fist", fingers: {all: "flexed"}, motion: "moving"}`
- Records for configured duration (default 10s)
- User can stop early if needed
- Labels are auto-saved with timestamp range

**Movement Encouragement:**
- Visual cue: "Rotate your hand" with directional arrows
- Optional: Gyro feedback showing rotation coverage
- Goal: Capture diverse orientations to prevent overfitting

### Wizard Controls (Always Available)

**Top Bar:**
```
[⏸️  Pause] [❌ Exit] [↩️  Restart Step] [⏭️  Skip]
Progress: ████████░░░░░░░░ Step 3 of 8 (37%)
```

**Safety Features:**
1. **Pause**: Stops wizard, preserves all data collected so far
2. **Exit**: Closes wizard, saves session with partial data
3. **Restart Step**: Discards current step's data, re-shows instruction
4. **Skip**: Skips current pose (useful if physically difficult)

### Data Preservation

**On Exit (any time):**
```javascript
// All collected data is saved to session
{
  version: "2.1",
  timestamp: "2025-12-15T10:30:00Z",
  samples: [...],  // All samples collected before exit
  labels: [...],   // All completed pose segments
  metadata: {
    wizard_session: {
      template_id: "tier1_basic_poses",
      completed_steps: [0, 1, 2],  // Steps 1-3 completed
      incomplete_step: 3,           // Step 4 was in progress
      exit_reason: "user_initiated"
    }
  }
}
```

**User can:**
- Export partial session as JSON
- Resume wizard later (if implemented)
- Use partial data for training (just fewer samples)

---

## Wizard Template Format

### JSON Schema

```json
{
  "id": "tier1_basic_poses",
  "name": "Tier 1: Basic Hand Poses",
  "description": "Simple whole-hand gestures for initial model training",
  "tier": 1,
  "estimated_duration": 80,
  "requires_calibration": false,
  "requirements": {
    "magnet_config": "none",
    "min_samples_per_pose": 250
  },
  "steps": [
    {
      "id": "rest",
      "title": "Rest Position",
      "icon": "✋",
      "instruction": {
        "short": "Hand relaxed, palm down",
        "detailed": "Place your hand in a natural, relaxed position with palm facing down. Fingers should be together but not tense.",
        "example_image": "assets/poses/rest.png",
        "example_video": "assets/poses/rest.mp4"
      },
      "timing": {
        "preview_duration": null,
        "prepare_duration": null,
        "record_duration": 10,
        "transition_duration": 5
      },
      "labels": {
        "pose": "rest",
        "fingers": {
          "thumb": null,
          "index": null,
          "middle": null,
          "ring": null,
          "pinky": null
        },
        "motion": "moving",
        "custom": ["tier1", "baseline", "wizard_guided"]
      },
      "recording_guidance": {
        "encourage_movement": true,
        "movement_instruction": "Slowly rotate your hand to capture different angles - up, down, left, right",
        "show_orientation_coverage": true
      }
    },
    {
      "id": "fist",
      "title": "Fist",
      "icon": "✊",
      "instruction": {
        "short": "Make a tight fist",
        "detailed": "Close all fingers tightly into your palm. Thumb wraps over fingers. Make it as tight as comfortable.",
        "example_image": "assets/poses/fist.png"
      },
      "timing": {
        "record_duration": 10,
        "transition_duration": 5
      },
      "labels": {
        "pose": "fist",
        "fingers": {
          "thumb": "flexed",
          "index": "flexed",
          "middle": "flexed",
          "ring": "flexed",
          "pinky": "flexed"
        },
        "motion": "moving",
        "custom": ["tier1", "all_flexed", "wizard_guided"]
      },
      "recording_guidance": {
        "encourage_movement": true,
        "movement_instruction": "Rotate your fist - show it from all sides"
      }
    }
    // ... more steps
  ],
  "validation": {
    "required_accuracy": 0.85,
    "next_tier": "tier2_single_finger"
  }
}
```

### Template Locations

```
src/web/GAMBIT/wizard-templates/
├── tier1_basic_poses.json           (5-8 poses, ~80s)
├── tier2_single_finger.json         (10-12 poses, ~100s)
├── tier3_complex_combinations.json  (15-20 poses, ~120s)
├── calibration_sequence.json        (mag calibration workflow)
└── custom/
    └── user_defined_template.json
```

---

## Implementation Roadmap

### Phase 1: Template System & UI (Week 1)

**Goal**: Load wizard templates from JSON, implement three-phase UI.

**Tasks:**
1. Create JSON schema and validator
2. Build template loader module
3. Update wizard.js to use external templates:
   - Parse template JSON
   - Render three-phase UI (preview → prepare → record)
   - Auto-apply labels from template
4. Create tier1_basic_poses.json template (5 poses)
5. Add wizard controls (pause, exit, restart, skip)
6. Implement data preservation on exit
7. Test with real data collection

**Deliverables:**
- Template loader: `src/web/GAMBIT/modules/template-loader.js`
- Updated wizard UI with three phases
- Tier 1 template
- User guide: "How to create wizard templates"

**Success Criteria:**
- Can load external templates without code changes
- User can pause/exit/resume wizard
- All data preserved on exit
- Labels auto-applied correctly

### Phase 2: Movement Guidance & Coverage (Week 2)

**Goal**: Encourage diverse orientations during recording.

**Tasks:**
1. Add orientation coverage visualization:
   - Track euler angles (roll, pitch, yaw) during recording
   - Show 3D sphere with visited orientations
   - Real-time feedback: "Rotate left more"
2. Implement movement instruction display during recording
3. Add optional gyro-based movement validation:
   - Warn if user is too still
   - Suggest specific rotations
4. Create Tier 2 template (single-finger isolation)
5. Test with magnetic finger tracking setup

**Deliverables:**
- Orientation coverage module
- Movement guidance UI
- Tier 2 template
- Collection quality metrics

**Success Criteria:**
- Users naturally rotate hand during recording
- Orientation diversity metric > 80%
- Tier 2 template works with magnetometer data

### Phase 3: Multi-Label Training Pipeline (Week 3)

**Goal**: Train multi-output models on wizard-collected data.

**Tasks:**
1. Update `ml/data_loader.py`:
   - Extract windows with multi-label support
   - Handle optional finger states (Tier 1 has null fingers)
   - Aggregate labels across transition zones
2. Create `ml/train.py` workflow:
   ```bash
   # Tier 1: Simple classifier
   python -m ml.train --data-dir data/GAMBIT \
     --model-type gesture \
     --tier 1 \
     --epochs 50

   # Tier 2: Multi-output
   python -m ml.train --data-dir data/GAMBIT \
     --model-type finger_tracking \
     --tier 2 \
     --epochs 50
   ```
3. Implement per-tier validation:
   - Automatic train/val split by session
   - Per-tier accuracy thresholds
   - Generate validation report
4. Export to TensorFlow.js for deployment

**Deliverables:**
- Updated ML pipeline supporting tiers
- Training scripts for Tier 1 and Tier 2
- Validation report generator
- TensorFlow.js export

**Success Criteria:**
- Tier 1 model: >85% accuracy on 5-8 poses
- Tier 2 model: >75% per-finger accuracy
- Models export cleanly to TensorFlow.js

### Phase 4: Tier 3 & Deployment (Week 4)

**Goal**: Complex combinations and real-world deployment.

**Tasks:**
1. Create Tier 3 template (15-20 complex poses)
2. Collect Tier 3 dataset
3. Train refined multi-output model
4. Deploy to collector app:
   - Load TensorFlow.js model
   - Real-time inference display
   - Confidence visualization
5. Create end-to-end tutorial:
   - "Zero to Trained Model in 30 Minutes"
   - Video walkthrough
6. Polish UI/UX based on user testing

**Deliverables:**
- Tier 3 template
- Trained Tier 3 model
- Deployed inference in collector
- Tutorial materials

**Success Criteria:**
- Tier 3 model handles 15-20 poses
- Real-time inference < 50ms
- Tutorial completable in 30 minutes
- User feedback positive (>4/5 rating)

---

## Technical Details

### Auto-Labeling Logic

**During wizard recording phase:**

```javascript
// In wizard.js - startWizardCollection()
async function recordPoseWithAutoLabels(step) {
  const labels = step.labels;  // From template JSON
  const duration = step.timing.record_duration * 1000;  // ms

  // Start recording if not already
  if (!state.recording) {
    await startRecording();
  }

  // Apply labels from template
  applyLabelsFromTemplate(labels);

  // Close previous segment and start new one
  closeCurrentLabel();
  state.currentLabelStart = state.sessionData.length;

  // Show recording UI with countdown
  await showRecordingPhase(step, duration);

  // After recording completes, close the labeled segment
  closeCurrentLabel();

  // Clear labels for transition
  clearLabels();
}

function applyLabelsFromTemplate(labels) {
  if (labels.pose) {
    state.currentLabels.pose = labels.pose;
  }
  if (labels.fingers) {
    state.currentLabels.fingers = {...labels.fingers};
  }
  if (labels.motion) {
    state.currentLabels.motion = labels.motion;
  }
  if (labels.custom) {
    state.currentLabels.custom = [...labels.custom];
  }
}
```

### Movement Encouragement

**Orientation Coverage Tracking:**

```javascript
class OrientationCoverageTracker {
  constructor() {
    this.samples = [];
    this.coverageSphere = this.initSphere();  // Discretized sphere
  }

  addSample(euler) {
    // Map euler angles to sphere bucket
    const bucket = this.eulerToBucket(euler.roll, euler.pitch, euler.yaw);
    this.coverageSphere[bucket] = true;
    this.samples.push(euler);
  }

  getCoveragePercentage() {
    const visited = Object.values(this.coverageSphere).filter(v => v).length;
    const total = Object.keys(this.coverageSphere).length;
    return visited / total;
  }

  getSuggestedMovement() {
    // Analyze which regions are undersampled
    // Return suggestion: "Rotate left", "Tilt forward", etc.
  }
}
```

**Real-time Feedback During Recording:**

```
┌─────────────────────────────────────┐
│  🔴 RECORDING - 7.3s / 10s          │
│                                     │
│  Coverage: 73%  🟢🟢🟢🟡⚪          │
│                                     │
│  💡 Suggestion: Rotate right more  │
└─────────────────────────────────────┘
```

### Data Quality Metrics

**Per-session metadata:**

```json
{
  "session_quality": {
    "orientation_coverage": 0.73,
    "orientation_diversity_score": 0.81,
    "samples_per_pose": {
      "fist": 487,
      "open_palm": 512,
      "rest": 502
    },
    "stillness_warnings": 2,
    "wizard_completion": 1.0
  }
}
```

---

## Training Workflow

### Tier-Based Pipeline

```bash
# Step 1: Collect Tier 1 data (5-8 basic poses)
# Use wizard: tier1_basic_poses.json
# Export to: data/GAMBIT/tier1_session_001.json

# Step 2: Train Tier 1 model
python -m ml.train \
  --data-dir data/GAMBIT \
  --model-type gesture \
  --filter-tier 1 \
  --epochs 50 \
  --output-dir ml/models/tier1

# Step 3: Validate Tier 1
python -m ml.evaluate \
  --model ml/models/tier1/gesture_model.keras \
  --data-dir data/GAMBIT \
  --filter-tier 1

# Output:
# Accuracy: 0.87 (PASS - threshold 0.85)
# Per-class accuracy:
#   rest: 0.89
#   fist: 0.92
#   open_palm: 0.85
#   thumbs_up: 0.81
#   point: 0.88

# Step 4: If validation passes, proceed to Tier 2
# Collect Tier 2 data (single-finger isolation)
# Use wizard: tier2_single_finger.json

# Step 5: Train Tier 2 model
python -m ml.train \
  --data-dir data/GAMBIT \
  --model-type finger_tracking \
  --filter-tier 2 \
  --epochs 50 \
  --output-dir ml/models/tier2

# Step 6: Validate Tier 2
python -m ml.evaluate \
  --model ml/models/tier2/finger_model.keras \
  --data-dir data/GAMBIT \
  --filter-tier 2

# Output:
# Overall accuracy: 0.78 (PASS - threshold 0.75)
# Per-finger accuracy:
#   thumb: 0.81
#   index: 0.79
#   middle: 0.76
#   ring: 0.74
#   pinky: 0.80

# Step 7: Deploy to browser
python -m ml.build convert \
  --model ml/models/tier2/finger_model.keras \
  --output-dir src/web/GAMBIT/models/tier2_v1
```

### Model Architecture Progression

**Tier 1: Simple Classifier**
```
Input (50, 9) → Conv1D(32) → Conv1D(64) → Conv1D(64)
              → GlobalAvgPool → Dense(64) → Dense(5-8)
              → Softmax
Output: Pose prediction
Params: ~37K
```

**Tier 2: Multi-Output**
```
Input (50, 9) → [Shared CNN] → Dense(64)
                              ├─ thumb_state (3-class softmax)
                              ├─ index_state (3-class softmax)
                              ├─ middle_state (3-class softmax)
                              ├─ ring_state (3-class softmax)
                              └─ pinky_state (3-class softmax)
Output: Per-finger states
Params: ~45K
```

**Tier 3: Unified Multi-Output**
```
Input (50, 9) → [Shared CNN] → Dense(128)
                              ├─ pose (20-class softmax)
                              ├─ thumb_state (3-class softmax)
                              ├─ index_state (3-class softmax)
                              ├─ middle_state (3-class softmax)
                              ├─ ring_state (3-class softmax)
                              ├─ pinky_state (3-class softmax)
                              └─ motion (3-class softmax)
Output: Full multi-label prediction
Params: ~55K
```

---

## Success Metrics

### Data Collection

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to collect Tier 1 dataset | < 5 minutes | Wizard duration |
| Orientation coverage per pose | > 70% | Sphere discretization |
| Samples per pose | > 250 (5s @ 50Hz) | Count |
| Wizard completion rate | > 90% | % users who finish |
| User-initiated exits | < 10% | Track exit reasons |

### Model Performance

| Metric | Tier 1 | Tier 2 | Tier 3 |
|--------|--------|--------|--------|
| Overall accuracy | > 85% | > 75% | > 70% |
| Per-class minimum | > 75% | > 65% | > 60% |
| Inference time (browser) | < 20ms | < 30ms | < 50ms |
| Model size (quantized) | < 100KB | < 150KB | < 200KB |

### User Experience

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ease of use rating | > 4/5 | User survey |
| Instructions clarity | > 4/5 | User survey |
| Time to first trained model | < 30 min | End-to-end tutorial |
| Error recovery success | > 95% | Track pause/resume usage |

---

## Open Questions & Design Decisions

### 1. Transition Duration

**Question**: How long should unlabeled transitions be between poses?

**Options:**
- A: 3 seconds (quick transitions, less wasted time)
- B: 5 seconds (comfortable transitions, less data loss)
- C: User-configurable per template

**Recommendation**: Start with 5 seconds, make configurable per step in template.

### 2. Partial Flexion Labeling

**Question**: How to handle "partial" finger states in Tier 1/2?

**Options:**
- A: Tier 1/2 use only extended(0) and flexed(2), no partial(1)
- B: Include partial(1) from the start
- C: Introduce partial in Tier 3 only

**Recommendation**: Option A for simplicity, add partial in Tier 3 when model is more robust.

### 3. Movement Validation

**Question**: Should wizard enforce minimum orientation coverage?

**Options:**
- A: Hard requirement - can't proceed if coverage < 60%
- B: Soft warning - warn user but allow proceed
- C: No enforcement - trust user to follow instructions

**Recommendation**: Option B (soft warning) to avoid frustration.

### 4. Model Retraining

**Question**: When adding Tier 2 data, retrain from scratch or fine-tune Tier 1 model?

**Options:**
- A: Train separate models per tier
- B: Accumulate data and retrain from scratch
- C: Fine-tune previous tier model with new data

**Recommendation**: Option B for simplicity and best accuracy.

### 5. Calibration Requirement

**Question**: Should Tier 2+ require magnetometer calibration?

**Options:**
- A: Strict requirement - wizard checks calibration before starting
- B: Recommended but optional
- C: No requirement

**Recommendation**: Option A - magnetic finger tracking requires calibration for quality data.

---

## Next Steps

### Immediate Actions (This Week)

1. **Review this plan** - validate approach and priorities
2. **Create Tier 1 template** - draft tier1_basic_poses.json
3. **Prototype three-phase UI** - mockup preview → prepare → record flow
4. **Test with existing wizard** - collect sample data to validate approach

### Quick Win Demo (2-3 Days)

**Goal**: Prove the concept end-to-end

**Scope:**
1. Manually create tier1_basic_poses.json (3 poses: rest, fist, open_palm)
2. Update wizard.js to load and execute template
3. Collect 100 samples per pose
4. Train simple 3-class classifier
5. Achieve >85% validation accuracy

**Success**: Demonstrates wizard → training workflow, builds confidence

### Full Implementation (4 Weeks)

Follow the 4-phase roadmap outlined above.

---

## Appendix

### A. Example Tier 1 Template (Minimal)

```json
{
  "id": "tier1_minimal",
  "name": "Tier 1: Minimal (3 poses)",
  "tier": 1,
  "estimated_duration": 45,
  "steps": [
    {
      "id": "rest",
      "title": "Rest",
      "icon": "✋",
      "instruction": {"short": "Hand relaxed, palm down"},
      "timing": {"record_duration": 10, "transition_duration": 5},
      "labels": {
        "pose": "rest",
        "fingers": {"thumb": null, "index": null, "middle": null, "ring": null, "pinky": null},
        "motion": "moving",
        "custom": ["tier1"]
      },
      "recording_guidance": {"encourage_movement": true}
    },
    {
      "id": "fist",
      "title": "Fist",
      "icon": "✊",
      "instruction": {"short": "Make a tight fist"},
      "timing": {"record_duration": 10, "transition_duration": 5},
      "labels": {
        "pose": "fist",
        "fingers": {"thumb": "flexed", "index": "flexed", "middle": "flexed", "ring": "flexed", "pinky": "flexed"},
        "motion": "moving",
        "custom": ["tier1"]
      },
      "recording_guidance": {"encourage_movement": true}
    },
    {
      "id": "open_palm",
      "title": "Open Palm",
      "icon": "🖐️",
      "instruction": {"short": "Spread all fingers wide"},
      "timing": {"record_duration": 10, "transition_duration": 0},
      "labels": {
        "pose": "open_palm",
        "fingers": {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
        "motion": "moving",
        "custom": ["tier1"]
      },
      "recording_guidance": {"encourage_movement": true}
    }
  ]
}
```

### B. Wizard State Management

```javascript
// Wizard state tracking
const wizardState = {
  active: false,
  template: null,           // Loaded template object
  currentStepIndex: 0,
  phase: 'preview',         // 'preview' | 'prepare' | 'record'

  // Data collection
  sessionStartIndex: 0,
  stepsCompleted: [],
  currentStepStartIndex: null,

  // Coverage tracking
  orientationTracker: null,

  // User controls
  paused: false,
  pausedAt: null
};

// Phase transitions
function transitionToPhase(newPhase) {
  wizardState.phase = newPhase;
  renderWizardUI();

  if (newPhase === 'record') {
    startRecordingPhase();
  }
}
```

### C. Template Validation Schema

```javascript
const templateSchema = {
  type: 'object',
  required: ['id', 'name', 'tier', 'steps'],
  properties: {
    id: {type: 'string', pattern: '^[a-z0-9_]+$'},
    name: {type: 'string'},
    tier: {type: 'integer', minimum: 1, maximum: 3},
    estimated_duration: {type: 'integer'},
    steps: {
      type: 'array',
      minItems: 1,
      items: {
        type: 'object',
        required: ['id', 'title', 'labels', 'timing'],
        properties: {
          id: {type: 'string'},
          title: {type: 'string'},
          icon: {type: 'string'},
          instruction: {
            type: 'object',
            required: ['short'],
            properties: {
              short: {type: 'string'},
              detailed: {type: 'string'},
              example_image: {type: 'string'},
              example_video: {type: 'string'}
            }
          },
          timing: {
            type: 'object',
            required: ['record_duration'],
            properties: {
              record_duration: {type: 'number', minimum: 1},
              transition_duration: {type: 'number', minimum: 0}
            }
          },
          labels: {
            type: 'object',
            required: ['motion'],
            properties: {
              pose: {type: ['string', 'null']},
              fingers: {type: 'object'},
              motion: {type: 'string'},
              custom: {type: 'array'}
            }
          }
        }
      }
    }
  }
};
```

---

**End of Document**
