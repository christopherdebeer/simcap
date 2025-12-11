# Data Collection, Model Training, and Geometry Alignment

**Document Version:** 1.0  
**Date:** 2025-01-12  
**Status:** Design Proposal  
**Author:** SIMCAP Development

---

## Executive Summary

This document provides a comprehensive technical analysis of the alignment between:
1. **Data Collection** - What labels can be easily and reliably collected from users
2. **Model Training** - What classification/regression targets are learnable from sensor data
3. **Geometry Mapping** - How model outputs translate to concrete hand joint angles

The goal is to establish a theoretically sound and practically implementable pipeline from raw IMU/magnetometer data to real-time 3D hand visualization.

---

## Table of Contents

1. [Current System Status](#1-current-system-status)
2. [Information-Theoretic Framework](#2-information-theoretic-framework)
3. [Representation Layers](#3-representation-layers)
4. [Anatomical Constraints](#4-anatomical-constraints)
5. [Proposed Architecture](#5-proposed-architecture)
6. [Implementation Specifications](#6-implementation-specifications)
7. [Validation Strategy](#7-validation-strategy)
8. [Roadmap](#8-roadmap)

---

## 1. Current System Status

### 1.1 Implementation Status (2025-01-12)

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Data Collection UI | ✅ Complete | `src/web/GAMBIT/collector.html` | Multi-label support, per-finger states |
| ML Schema | ✅ Complete | `ml/schema.py` | V2 multi-label, FingerLabels class |
| Data Loader | ✅ Complete | `ml/data_loader.py` | Multi-label window creation |
| Hand Renderer | ✅ Complete | `src/web/GAMBIT/hand-3d-renderer.js` | 3D visualization with pose input |
| Calibration Utils | ✅ Complete | `src/web/GAMBIT/calibration.js` | Hard/soft iron, Earth field |
| Filtering | ✅ Complete | `src/web/GAMBIT/filters.js` | Kalman, particle filters |
| Model Training | ⚠️ Partial | `ml/train.py` | Needs multi-label extension |
| Inference Pipeline | ❌ Missing | - | End-to-end not connected |

### 1.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT IMPLEMENTATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   GAMBIT     │    │  Collector   │    │   GitHub     │                  │
│  │   Device     │───▶│     UI       │───▶│   Storage    │                  │
│  │  (Puck.js)   │BLE │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         │ 9-DoF IMU         │ Labels            │ .json + .meta.json       │
│         │ 50Hz              │ (multi-label)     │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    ML TRAINING PIPELINE                       │          │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │          │
│  │  │  Load   │───▶│ Window  │───▶│  Train  │───▶│ Export  │   │          │
│  │  │  Data   │    │ Create  │    │  Model  │    │ TF.js   │   │          │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    INFERENCE PIPELINE                         │          │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │          │
│  │  │  Live   │───▶│ Calibr- │───▶│  Model  │───▶│  Hand   │   │          │
│  │  │  Data   │    │  ation  │    │ Predict │    │ Render  │   │          │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Information-Theoretic Framework

### 2.1 Channel Capacity Analysis

The fundamental question: **How much information about hand pose can be extracted from the sensor suite?**

#### Sensor Information Content

| Sensor | Dimensions | Sample Rate | Bits/Sample | Information Rate |
|--------|------------|-------------|-------------|------------------|
| Accelerometer | 3 | 50 Hz | 16-bit × 3 | 2,400 bits/sec |
| Gyroscope | 3 | 50 Hz | 16-bit × 3 | 2,400 bits/sec |
| Magnetometer | 3 | 10 Hz | 16-bit × 3 | 480 bits/sec |
| **Total** | **9** | **50 Hz** | - | **~5,280 bits/sec** |

#### Target Information Content

| Representation | States | Bits Required | Update Rate | Information Rate |
|----------------|--------|---------------|-------------|------------------|
| 5 fingers × 3 states | 243 | 7.9 bits | 50 Hz | 395 bits/sec |
| 5 fingers × 5 states | 3,125 | 11.6 bits | 50 Hz | 580 bits/sec |
| 5 fingers × continuous | ∞ | ~40 bits (8-bit × 5) | 50 Hz | 2,000 bits/sec |
| 15 joint angles | ∞ | ~120 bits (8-bit × 15) | 50 Hz | 6,000 bits/sec |

**Key Insight:** The sensor channel capacity (~5,280 bits/sec) is sufficient for discrete pose classification but marginal for full continuous joint angle estimation. This justifies a **hierarchical approach**: discrete states first, continuous refinement later.

### 2.2 Mutual Information Analysis

The learnable information depends on the **mutual information** between sensor readings and hand pose:

```
I(Sensors; Pose) = H(Pose) - H(Pose | Sensors)
```

Where:
- `H(Pose)` = entropy of pose distribution (depends on task/vocabulary)
- `H(Pose | Sensors)` = remaining uncertainty after observing sensors

#### Factors Affecting Mutual Information

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Sensor noise | Reduces I(S;P) | Filtering, averaging |
| Environmental interference | Reduces I(S;P) | Calibration, baseline subtraction |
| User variation | Reduces I(S;P) | Per-user calibration, transfer learning |
| Pose ambiguity | Increases H(P\|S) | Temporal context, multi-modal fusion |

### 2.3 Theoretical Bounds

For a **discrete pose classifier** with N classes:

```
Achievable accuracy ≤ 1 - H(P|S) / log₂(N)
```

For **5 fingers × 3 states = 243 classes**:
- If H(P|S) ≈ 1 bit (good sensor signal): accuracy ≤ 87%
- If H(P|S) ≈ 2 bits (moderate noise): accuracy ≤ 75%
- If H(P|S) ≈ 3 bits (high noise): accuracy ≤ 62%

**Practical target:** 80%+ accuracy on 243-class problem, or 95%+ on per-finger 3-class problems.

---

## 3. Representation Layers

### 3.1 Layer 1: Data Collection Labels

The collector UI supports three label granularities:

#### 3.1.1 High-Level Pose Labels (V1 Compatible)

```python
class Gesture(IntEnum):
    REST = 0           # Hand relaxed
    FIST = 1           # All fingers flexed
    OPEN_PALM = 2      # All fingers extended
    INDEX_UP = 3       # Index extended, others flexed
    PEACE = 4          # Index + middle extended
    THUMBS_UP = 5      # Thumb extended upward
    OK_SIGN = 6        # Thumb-index circle
    PINCH = 7          # Thumb-index pinch
    GRAB = 8           # Fingers curled
    WAVE = 9           # Dynamic gesture
```

**Pros:** Easy to collect, intuitive for users  
**Cons:** Coarse granularity, doesn't capture partial states

#### 3.1.2 Per-Finger State Labels (V2)

```python
class FingerState(str, Enum):
    EXTENDED = "extended"    # 0 - Finger fully extended
    PARTIAL = "partial"      # 1 - Finger partially flexed
    FLEXED = "flexed"        # 2 - Finger fully flexed
    UNKNOWN = "unknown"      # ? - State not specified
```

**Binary encoding:** `"02222"` = thumb extended, others flexed (index up)

**Pros:** Fine-grained, compositional, 243 unique poses  
**Cons:** Requires more careful labeling, "partial" is ambiguous

#### 3.1.3 Continuous Labels (Future)

```python
@dataclass
class ContinuousFingerState:
    thumb: float   # 0.0 (extended) to 1.0 (flexed)
    index: float
    middle: float
    ring: float
    pinky: float
```

**Pros:** Maximum expressiveness  
**Cons:** Hard to label accurately without ground truth system

### 3.2 Layer 2: Model Output Representations

#### 3.2.1 Multi-Class Classification

**Output:** Softmax over 243 classes (all finger combinations)

```
Input: [T × 9] sensor window
Output: [243] probability distribution
Loss: Cross-entropy
```

**Pros:** Single prediction, handles correlations  
**Cons:** Sparse training data per class, doesn't generalize to unseen combinations

#### 3.2.2 Multi-Label Classification (Recommended)

**Output:** 5 independent 3-class predictions

```
Input: [T × 9] sensor window
Output: [5 × 3] = [15] logits (5 fingers × 3 states)
Loss: Sum of 5 cross-entropy losses
```

**Pros:** Generalizes to unseen combinations, efficient training  
**Cons:** Ignores inter-finger correlations

#### 3.2.3 Multi-Output Regression

**Output:** 5 continuous values

```
Input: [T × 9] sensor window
Output: [5] values in [0, 1]
Loss: MSE or smooth L1
```

**Pros:** Smooth transitions, continuous control  
**Cons:** Harder to train, requires continuous labels

### 3.3 Layer 3: Geometry Representation

#### 3.3.1 Current Implementation

```javascript
// hand-3d-renderer.js - setFingerPoses()
setFingerPoses(poses) {
    fingerNames.forEach((name, i) => {
        const state = poses[name] || 0;  // 0, 1, or 2
        const intensity = state / 2;      // 0, 0.5, or 1.0
        
        this.joints[i][0] = 0;                    // Spread
        this.joints[i][1] = intensity * 80;       // MCP
        this.joints[i][2] = intensity * 70;       // PIP
        this.joints[i][3] = intensity * 60;       // DIP
    });
}
```

**Mapping table:**

| State | Intensity | MCP | PIP | DIP |
|-------|-----------|-----|-----|-----|
| 0 (Extended) | 0.0 | 0° | 0° | 0° |
| 1 (Partial) | 0.5 | 40° | 35° | 30° |
| 2 (Flexed) | 1.0 | 80° | 70° | 60° |

#### 3.3.2 Anatomical Joint Ranges (Reference)

From anatomical literature:

| Joint | Type | DoF | Flexion Range | Extension Range |
|-------|------|-----|---------------|-----------------|
| Finger MCP | Condyloid | 2 | 0° to 90° | 0° to 40° (hyper) |
| Finger PIP | Hinge | 1 | 0° to 100° | 0° to 10° (hyper) |
| Finger DIP | Hinge | 1 | 0° to 80° | 0° to 5° (hyper) |
| Thumb CMC | Saddle | 2 | 15-25° flex, 25-35° abd | 30-45° ext |
| Thumb MCP | Hinge | 1 | 0° to 60° | - |
| Thumb IP | Hinge | 1 | 0° to 80° | - |

---

## 4. Anatomical Constraints

### 4.1 Thumb Specialization

The thumb requires different treatment due to its unique CMC saddle joint:

```javascript
// Proposed thumb-specific mapping
if (fingerIndex === 0) {  // Thumb
    // CMC joint: opposition movement (across palm)
    const cmcOpposition = intensity * 45;  // 0° to 45°
    
    // MCP joint: flexion
    const mcpFlexion = intensity * 50;     // 0° to 50°
    
    // IP joint: flexion (coupled to MCP)
    const ipFlexion = intensity * 70;      // 0° to 70°
    
    this.joints[0] = [0, cmcOpposition, mcpFlexion, ipFlexion];
}
```

### 4.2 DIP-PIP Coupling

Anatomical constraint: DIP flexion is mechanically coupled to PIP flexion.

```javascript
// Anatomical coupling: DIP ≈ 2/3 × PIP
const pipAngle = intensity * 100;  // Full PIP range
const dipAngle = pipAngle * 0.67;  // Coupled DIP
```

### 4.3 Inter-Finger Coupling

Adjacent fingers share muscle-tendon units:

```javascript
// Ring-pinky coupling (they tend to move together)
const ringState = poses.ring;
const pinkyState = poses.pinky;

// Blend toward average when both moving
if (ringState > 0 && pinkyState > 0) {
    const avgState = (ringState + pinkyState) / 2;
    // Apply slight coupling force
    poses.ring = ringState * 0.8 + avgState * 0.2;
    poses.pinky = pinkyState * 0.8 + avgState * 0.2;
}
```

### 4.4 Hyperextension Limits

Prevent anatomically impossible poses:

```javascript
// Clamp to valid ranges
const mcpAngle = Math.max(-30, Math.min(90, rawMcpAngle));  // Allow slight hyperextension
const pipAngle = Math.max(0, Math.min(100, rawPipAngle));   // No PIP hyperextension
const dipAngle = Math.max(0, Math.min(80, rawDipAngle));    // No DIP hyperextension
```

---

## 5. Proposed Architecture

### 5.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROPOSED ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    SENSOR PREPROCESSING                             │    │
│  │                                                                     │    │
│  │  Raw IMU ──▶ Calibration ──▶ Filtering ──▶ Windowing ──▶ Features  │    │
│  │   9-DoF      (hard/soft      (Kalman)      (32 samples   (T×9)     │    │
│  │   50Hz        iron, Earth)                  @ 50Hz)                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    MODEL INFERENCE                                  │    │
│  │                                                                     │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │    │
│  │  │   Shared    │    │  Per-Finger │    │   Output    │            │    │
│  │  │   Encoder   │───▶│    Heads    │───▶│   Fusion    │            │    │
│  │  │  (1D CNN)   │    │  (5 × FC)   │    │             │            │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘            │    │
│  │                                                                     │    │
│  │  Input: [32 × 9]    Hidden: [128]       Output: [5 × 3] logits    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    GEOMETRY MAPPING                                 │    │
│  │                                                                     │    │
│  │  [5 × 3] logits ──▶ Softmax ──▶ State (0/1/2) ──▶ Joint Angles    │    │
│  │                      or                            (15 values)     │    │
│  │                     Argmax                                         │    │
│  │                                                                     │    │
│  │  Anatomical Constraints:                                           │    │
│  │  • Thumb CMC opposition mapping                                    │    │
│  │  • DIP-PIP coupling (DIP = 0.67 × PIP)                            │    │
│  │  • Inter-finger coupling (ring-pinky)                              │    │
│  │  • Hyperextension limits                                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    HAND RENDERING                                   │    │
│  │                                                                     │    │
│  │  Joint Angles ──▶ Forward Kinematics ──▶ 3D Positions ──▶ Canvas  │    │
│  │  [15 values]      (bone transforms)      (joint coords)   (2D/3D) │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Model Architecture Details

#### 5.2.1 Shared Encoder (1D CNN)

```python
class SharedEncoder(nn.Module):
    def __init__(self, input_channels=9, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch, time, channels] -> [batch, channels, time]
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [batch, hidden_dim]
        return x
```

#### 5.2.2 Per-Finger Heads

```python
class FingerHead(nn.Module):
    def __init__(self, hidden_dim=128, num_states=3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_states)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # [batch, num_states]

class MultiFingerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.heads = nn.ModuleList([FingerHead() for _ in range(5)])
        
    def forward(self, x):
        features = self.encoder(x)
        outputs = [head(features) for head in self.heads]
        return torch.stack(outputs, dim=1)  # [batch, 5, 3]
```

### 5.3 Loss Function

```python
def multi_finger_loss(predictions, targets):
    """
    predictions: [batch, 5, 3] - logits for each finger
    targets: [batch, 5] - integer labels (0, 1, 2) for each finger
    """
    total_loss = 0
    for finger_idx in range(5):
        finger_pred = predictions[:, finger_idx, :]  # [batch, 3]
        finger_target = targets[:, finger_idx]        # [batch]
        total_loss += F.cross_entropy(finger_pred, finger_target)
    return total_loss / 5
```

---

## 6. Implementation Specifications

### 6.1 Geometry Mapping Function

```javascript
/**
 * Map finger states to joint angles with anatomical constraints
 * @param {Object} poses - {thumb: 0-2, index: 0-2, middle: 0-2, ring: 0-2, pinky: 0-2}
 * @returns {Array} joints - [[spread, MCP, PIP, DIP], ...] for 5 fingers
 */
function mapPosesToJoints(poses) {
    const joints = [];
    const fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    
    // Anatomical max angles (degrees)
    const FINGER_ANGLES = {
        MCP_MAX: 90,
        PIP_MAX: 100,
        DIP_MAX: 80,
        DIP_PIP_RATIO: 0.67  // DIP coupled to PIP
    };
    
    const THUMB_ANGLES = {
        CMC_OPPOSITION_MAX: 45,
        MCP_MAX: 50,
        IP_MAX: 70
    };
    
    fingerNames.forEach((name, i) => {
        const state = poses[name] || 0;
        const intensity = state / 2;  // 0, 0.5, or 1.0
        
        if (name === 'thumb') {
            // Thumb: CMC opposition + MCP + IP
            joints[i] = [
                0,  // Spread (handled separately)
                intensity * THUMB_ANGLES.CMC_OPPOSITION_MAX,
                intensity * THUMB_ANGLES.MCP_MAX,
                intensity * THUMB_ANGLES.IP_MAX
            ];
        } else {
            // Other fingers: MCP + PIP + DIP (with coupling)
            const pipAngle = intensity * FINGER_ANGLES.PIP_MAX;
            const dipAngle = pipAngle * FINGER_ANGLES.DIP_PIP_RATIO;
            
            joints[i] = [
                0,  // Spread
                intensity * FINGER_ANGLES.MCP_MAX,
                pipAngle,
                dipAngle
            ];
        }
    });
    
    // Apply inter-finger coupling (ring-pinky)
    applyInterFingerCoupling(joints, poses);
    
    return joints;
}

function applyInterFingerCoupling(joints, poses) {
    const ringIdx = 3;
    const pinkyIdx = 4;
    
    // Blend ring and pinky when both are moving
    if (poses.ring > 0 && poses.pinky > 0) {
        const couplingStrength = 0.15;  // 15% coupling
        
        for (let j = 1; j <= 3; j++) {  // MCP, PIP, DIP
            const avg = (joints[ringIdx][j] + joints[pinkyIdx][j]) / 2;
            joints[ringIdx][j] = joints[ringIdx][j] * (1 - couplingStrength) + avg * couplingStrength;
            joints[pinkyIdx][j] = joints[pinkyIdx][j] * (1 - couplingStrength) + avg * couplingStrength;
        }
    }
}
```

### 6.2 Continuous Output Mapping

For regression models outputting continuous values:

```javascript
/**
 * Map continuous finger values [0, 1] to joint angles
 * @param {Array} values - [thumb, index, middle, ring, pinky] in [0, 1]
 * @returns {Array} joints - [[spread, MCP, PIP, DIP], ...] for 5 fingers
 */
function mapContinuousToJoints(values) {
    const joints = [];
    const fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    
    fingerNames.forEach((name, i) => {
        const intensity = Math.max(0, Math.min(1, values[i]));  // Clamp to [0, 1]
        
        if (name === 'thumb') {
            joints[i] = [
                0,
                intensity * 45,   // CMC opposition
                intensity * 50,   // MCP
                intensity * 70    // IP
            ];
        } else {
            const pipAngle = intensity * 100;
            joints[i] = [
                0,
                intensity * 90,   // MCP
                pipAngle,         // PIP
                pipAngle * 0.67   // DIP (coupled)
            ];
        }
    });
    
    return joints;
}
```

### 6.3 Training Data Format

```python
# Expected data format for multi-label training
{
    "windows": np.array([...]),      # Shape: [N, T, 9] - N windows, T timesteps, 9 features
    "labels": np.array([...]),       # Shape: [N, 5] - N windows, 5 finger states (0/1/2)
    "metadata": {
        "window_size": 32,
        "sample_rate": 50,
        "feature_names": ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"],
        "finger_names": ["thumb", "index", "middle", "ring", "pinky"],
        "state_names": ["extended", "partial", "flexed"]
    }
}
```

---

## 7. Validation Strategy

### 7.1 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Per-finger accuracy | Accuracy on each finger independently | > 90% |
| Exact match accuracy | All 5 fingers correct | > 70% |
| Hamming distance | Average number of incorrect fingers | < 0.5 |
| Confusion matrix | Per-state confusion | Diagonal dominant |

### 7.2 Test Protocol

1. **Hold-out validation:** 80/10/10 train/val/test split by session
2. **Cross-user validation:** Leave-one-user-out for generalization
3. **Cross-environment validation:** Test on unseen environments
4. **Temporal consistency:** Measure prediction stability over time

### 7.3 Visual Validation

The hand renderer provides immediate visual feedback:
- Predicted pose should match user's actual hand
- Transitions should be smooth (no jitter)
- Anatomically impossible poses should not occur

---

## 8. Roadmap

### Phase 1: Discrete Classification (Current Focus)

**Timeline:** 2-4 weeks

- [ ] Collect labeled data with current collector UI
- [ ] Train multi-label classifier (5 × 3 outputs)
- [ ] Integrate model with hand renderer
- [ ] Validate on held-out data

**Success criteria:** 85%+ per-finger accuracy, 70%+ exact match

### Phase 2: Anatomical Refinement

**Timeline:** 2-3 weeks

- [ ] Implement thumb-specific geometry mapping
- [ ] Add DIP-PIP coupling
- [ ] Add inter-finger coupling
- [ ] Tune angle ranges based on visual feedback

**Success criteria:** Visually plausible hand poses

### Phase 3: Continuous Regression

**Timeline:** 4-6 weeks

- [ ] Collect continuous labels (or interpolate from discrete)
- [ ] Train regression model
- [ ] Implement smooth transitions
- [ ] Add temporal filtering

**Success criteria:** Smooth, responsive hand tracking

### Phase 4: Magnetic Finger Tracking

**Timeline:** 6-8 weeks

- [ ] Add finger magnets (hardware)
- [ ] Implement magnetic field analysis
- [ ] Train position regression model
- [ ] Full motion capture demonstration

**Success criteria:** Approximate finger position tracking

---

## Appendix A: Sensor Data Schema

```python
# Full sensor sample structure
@dataclass
class SensorSample:
    # Accelerometer (raw ADC, ±16g range)
    ax: int  # -16384 to 16384
    ay: int
    az: int
    
    # Gyroscope (raw ADC, ±2000°/s range)
    gx: int  # -32768 to 32768
    gy: int
    gz: int
    
    # Magnetometer (raw ADC, ±2048 μT range)
    mx: int  # -2048 to 2048
    my: int
    mz: int
```

---

## Appendix B: Label Encoding Reference

### Binary String Encoding

Per-finger states encoded as 5-character string:

| Position | Finger | Values |
|----------|--------|--------|
| 0 | Thumb | 0=extended, 1=partial, 2=flexed |
| 1 | Index | 0=extended, 1=partial, 2=flexed |
| 2 | Middle | 0=extended, 1=partial, 2=flexed |
| 3 | Ring | 0=extended, 1=partial, 2=flexed |
| 4 | Pinky | 0=extended, 1=partial, 2=flexed |

### Common Pose Encodings

| Pose | Binary | Description |
|------|--------|-------------|
| `00000` | Open palm | All fingers extended |
| `22222` | Fist | All fingers flexed |
| `02222` | Index up | Index extended, others flexed |
| `00222` | Peace | Index + middle extended |
| `20000` | Thumbs up | Thumb extended (flexed toward palm) |
| `21222` | OK sign | Thumb-index circle |
| `22000` | Pinch | Thumb-index pinch |
| `11111` | Relaxed | All fingers partially flexed |

### Numeric Encoding (Base-3)

For ML training, binary strings can be converted to integers:

```python
def binary_to_int(binary_str):
    """Convert '02222' to integer (0*81 + 2*27 + 2*9 + 2*3 + 2*1 = 80)"""
    result = 0
    for i, char in enumerate(binary_str):
        result += int(char) * (3 ** (4 - i))
    return result

def int_to_binary(n):
    """Convert integer back to binary string"""
    result = []
    for _ in range(5):
        result.append(str(n % 3))
        n //= 3
    return ''.join(reversed(result))
```

---

## Appendix C: Anatomical Reference

### Finger Joint Nomenclature

```
                    DIP (Distal Interphalangeal)
                     │
                     ▼
              ┌──────────┐
              │  Distal  │
              │ Phalanx  │
              └────┬─────┘
                   │
                  PIP (Proximal Interphalangeal)
                   │
              ┌────┴─────┐
              │ Middle   │
              │ Phalanx  │
              └────┬─────┘
                   │
                  MCP (Metacarpophalangeal)
                   │
              ┌────┴─────┐
              │Proximal  │
              │ Phalanx  │
              └────┬─────┘
                   │
              ┌────┴─────┐
              │Metacarpal│
              └──────────┘
```

### Thumb Joint Nomenclature

```
                    IP (Interphalangeal)
                     │
                     ▼
              ┌──────────┐
              │  Distal  │
              │ Phalanx  │
              └────┬─────┘
                   │
                  MCP (Metacarpophalangeal)
                   │
              ┌────┴─────┐
              │Proximal  │
              │ Phalanx  │
              └────┬─────┘
                   │
                  CMC (Carpometacarpal) - Saddle Joint
                   │
              ┌────┴─────┐
              │Metacarpal│
              └──────────┘
```

### Range of Motion Summary

| Joint | Flexion | Extension | Abduction | Adduction |
|-------|---------|-----------|-----------|-----------|
| Finger MCP | 90° | 30-40° | 20° | 20° |
| Finger PIP | 100-110° | 0-10° | - | - |
| Finger DIP | 80-90° | 0-5° | - | - |
| Thumb CMC | 15-25° | 30-45° | 25-35° | 25-35° |
| Thumb MCP | 50-60° | 0° | - | - |
| Thumb IP | 80-90° | 0-20° | - | - |

---

## Appendix D: Signal Processing Parameters

### Recommended Filter Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 50 Hz | Nyquist for ~20 Hz hand motion |
| Window size | 32 samples | 640 ms context, captures gesture |
| Window overlap | 50% | Smooth predictions |
| Kalman process noise (Q) | 0.01 | Smooth tracking |
| Kalman measurement noise (R) | 0.1 | Trust measurements |
| Low-pass cutoff | 10 Hz | Remove high-freq noise |

### Calibration Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hard iron samples | 500+ | Rotate through all orientations |
| Soft iron samples | 500+ | Same as hard iron |
| Earth field samples | 100+ | Hold still, multiple orientations |
| Sphericity threshold | 0.9 | Quality metric for calibration |

---

## Appendix E: Model Hyperparameters

### Recommended Training Configuration

```python
config = {
    # Data
    "window_size": 32,
    "window_stride": 16,
    "features": ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"],
    
    # Model
    "encoder_channels": [32, 64, 128],
    "kernel_sizes": [5, 5, 3],
    "head_hidden_dim": 32,
    "dropout": 0.2,
    
    # Training
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "early_stopping_patience": 10,
    
    # Augmentation
    "noise_std": 0.02,
    "time_shift_max": 4,
    "scale_range": [0.9, 1.1]
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-12 | SIMCAP Dev | Initial comprehensive document |

---

<link rel="stylesheet" href="../../src/simcap.css">
