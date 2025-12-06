# Revisiting SIMCAP

## Executive Summary

SIMCAP (Sensor Inferred MOtion CAPture) represents a far richer concept than "gesture recognizer but nerdier." This document reframes the project's theoretical potential and contrasts it with the current implementation to identify the path forward.

---

## Theoretical Foundation: What SIMCAP Could Be

### The Complete Vision

The full SIMCAP concept envisions a sophisticated hand motion capture system comprising:

**Hardware Components:**
- 1× ESP32 + 9-DoF IMU mounted on the palm
  - Gyroscope (angular velocity)
  - Accelerometer (linear acceleration / gravity)
  - Magnetometer (ambient field + finger magnet detection)
- 5× magnets mounted on fingers (as rings)
  - Creating measurable magnetic field distortions
  - Detected as a superposition by the palm magnetometer of:
    - Earth's magnetic field
    - Environmental noise and interference
    - Per-finger magnetic signatures in various configurations

**Intended Capabilities:**
- Finger position inference from magnetic field distortions
- Approximate hand pose reconstruction
- Full gesture recognition with temporal dynamics
- True motion capture in a probabilistic, learned sense

### Information Theory Analysis

The theoretical information available from the sensor suite includes:

1. **Palm Orientation**
   - Complete 3D orientation (quaternion/rotation matrix) over time
   - Reconstructable palm pointing direction
   - Rotation, swing, and motion dynamics

2. **Relative Finger Configuration**
   - Magnetic field distortion patterns encoding:
     - Which fingers are flexed/extended
     - Proximity to palm
     - Relative pose configurations
   - Non-linear field combination requiring learned models rather than closed-form solutions

3. **Temporal Structure**
   - Time-series data enabling:
     - Dynamic gesture recognition (pinch → swipe → release)
     - Motion dynamics (fast flick vs. slow sweep)
     - Sequence pattern classification

The elegance lies in the architecture: one "smart" node (palm sensor) + five "dumb" nodes (passive magnets).

---

## Current Implementation: Where SIMCAP Stands Today

### Hardware Reality

**Current Platform:** Puck.js (Espruino-based wearable)
- 9-DoF IMU sensor ✓
- Accelerometer data (x, y, z) ✓
- Gyroscope data (x, y, z) ✓
- Magnetometer data (x, y, z) ✓
- Additional sensors:
  - Light sensor
  - Capacitive touch
  - Temperature (magnetometer temp)
  - Battery monitoring
- BLE connectivity ✓
- NFC tap-to-pair ✓

**Missing from Vision:**
- No finger-mounted magnets mentioned or implemented
- No infrastructure for finger tracking via magnetic distortion
- Single device focus (dual-hand use conceptualized in JOYPAD but not implemented)

### Software Reality

**GAMBIT Firmware (Current Device Code):**
```javascript
Location: src/device/GAMBIT/app.js
Functionality:
- Raw telemetry collection at 50Hz for 30-second bursts
- Button-triggered data capture
- BLE advertising and console output
- NFC-triggered web UI launch
```

**Current Capabilities:**
- ✓ Raw sensor data streaming (acc, gyro, mag)
- ✓ BLE communication to web UI
- ✓ Basic state machine (on/off toggle)
- ✓ Baseline data collection

**Missing Capabilities:**
- ✗ No machine learning inference
- ✗ No gesture recognition
- ✗ No finger pose estimation
- ✗ No magnetic field distortion analysis
- ✗ No coordinate frame normalization
- ✗ No on-device or off-device ML pipeline
- ✗ No calibration procedures
- ✗ No training data collection workflow

**Web UI (GAMBIT):**
```
Location: src/web/GAMBIT/
Purpose: Baseline data collection and visualization
Features:
- WebBLE connectivity
- Kalman filtering (kalman.js present)
- Real-time data display
- NFC-triggered launch
```

**Status:** Data collection infrastructure exists, but no processing pipeline.

**Other Components:**
- **BAE** (Bluetooth Advertise Everything): Basic BLE advertising experiment
- **JOYPAD**: Conceptual dual-hand game controller prototype (unimplemented)
- **P0**: Earlier prototype versions

---

## The Gap: Vision vs. Reality

### Architectural Gaps

| Theoretical Vision | Current Implementation | Gap Analysis |
|-------------------|------------------------|--------------|
| Finger-mounted magnets as passive markers | No magnets mentioned | **Critical hardware gap** |
| Magnetic field distortion analysis | Raw magnetometer data only | **No analysis pipeline** |
| Multi-tier ML pipeline (poses → gestures → full tracking) | No ML implementation | **Complete ML gap** |
| On-device TinyML inference | Raw data streaming only | **No embedded ML** |
| User-specific calibration | No calibration system | **Missing calibration** |
| Environment baseline adaptation | Static data collection | **No adaptation** |
| Palm-centric coordinate transformation | Raw sensor coordinates | **No coordinate normalization** |
| Chorded input / gesture vocabulary | No gesture system | **No high-level interface** |

### What Works

The current implementation provides a solid foundation:
- ✓ Reliable sensor data acquisition
- ✓ BLE streaming infrastructure
- ✓ User-friendly NFC tap-to-connect
- ✓ Web-based data visualization
- ✓ Button-triggered capture sessions
- ✓ Extensible firmware architecture

### What Doesn't Exist Yet

The vision's core differentiators remain unbuilt:
- ✗ Magnetic finger tracking (hardware + software)
- ✗ Machine learning inference
- ✗ Gesture recognition
- ✗ Motion capture capabilities
- ✗ Practical input modalities

---

## Proposed Roadmap: Closing the Gap

### Tier 1 – Static Finger Pose Classifier

**Goal:** Treat the system as a chorded keyboard made of fingers.

**Prerequisites:**
- Add finger-mounted magnets (rings with embedded neodymium magnets)
- Establish stable palm-centric coordinate frame

**Implementation:**
- Keep hand relatively still during training
- Collect magnetometer + IMU data for discrete finger states:
  - "Index flexed"
  - "Middle flexed"
  - "Ring + little flexed"
  - "Thumb touching index"
  - etc. (10-20 poses total)
- Train small classifier network:
  - Input: windowed sensor data (T × 9 features)
  - Output: discrete pose classification
  - Framework: TensorFlow Lite Micro / Edge Impulse for ESP32
- Prototype pipeline:
  - Phase 1: Stream to laptop, train in Python
  - Phase 2: Quantize and deploy to device

**Success Criteria:**
- Reliable classification of 10+ static poses
- Practical demo: finger patterns as macro input
- Dataset validation of magnetic signature separability

**Current Project Status:** Infrastructure exists for data collection; need magnet hardware and ML pipeline.

### Tier 2 – Dynamic Gesture Recognition

**Goal:** Add temporal dynamics to pose recognition.

**Implementation:**
- Collect gesture sequences:
  - "Index flex → sweep right → release"
  - "Thumb-index pinch → lift upward"
- Use sliding windows over sensor stream
- Train sequence model:
  - 1D CNN + GRU/LSTM
  - Or pure 1D temporal convolutions (more efficient on MCU)
- Vocabulary: 5-15 distinct gestures
- Each gesture anchored to known starting pose

**Success Criteria:**
- Recognize dynamic multi-step gestures
- Real-time inference at 20-50 Hz
- Practical demo: gesture-based UI control

**Current Project Status:** No gesture infrastructure; requires Tier 1 foundation.

### Tier 3 – Approximate Hand Pose Estimation

**Goal:** Probabilistic skeletal hand tracking.

**Implementation:**
- Define low-dimensional pose representation:
  - Per finger: [flexion, abduction] ∈ [0,1]²
  - Total: ~10-dimensional pose vector
- Training approach:
  - Ground truth: camera-based hand tracking or manual annotation
  - Model: regress sensor stream → continuous pose vector
  - Output rate: 50-100 Hz
- Applications:
  - Drive 3D hand model visualization
  - Continuous control signals
  - Full motion capture in XR environments

**Success Criteria:**
- Plausible hand pose reconstruction
- Demonstration of "sensor-inferred motion capture"
- Not precision mocap, but meaningful representation

**Current Project Status:** Requires Tiers 1-2; represents long-term vision.

---

## Design Considerations & Known Challenges

### Challenges Identified in Vision

**Magnetic Interference & Drift:**
- Environmental variation (steel structures, wiring, devices)
- Mitigation:
  - Per-environment baseline calibration
  - Training on diverse environments
  - Adaptive filtering

**User-Specific Calibration:**
- Hand size, finger length, ring position variations
- Mitigation:
  - Short per-user calibration sequence
  - Per-user fine-tuning of final layer
  - Transfer learning from base model

**Coordinate Frame Consistency:**
- Raw IMU data is device-relative
- Solution:
  - Transform to palm-centric frame:
    - x = across palm
    - y = along fingers
    - z = out of palm
  - Simplifies learned representations

**Sampling & Windowing:**
- Recommended parameters:
  - 50-100 Hz sample rate
  - 200-500 ms input windows (10-50 samples)
  - 50% overlapping windows for smooth output

**Development Workflow:**
- MVP: Stream to laptop via BLE/Wi-Fi
- Prototype in Python (PyTorch/JAX)
- Production: Quantize and deploy to ESP32

### Current Implementation Considerations

**What GAMBIT Does Well:**
- 50 Hz sampling ✓
- 30-second capture sessions ✓
- BLE streaming ✓
- Button-triggered control ✓

**What Needs Refinement:**
- Coordinate frame transformation (currently raw data)
- Windowing strategy (currently continuous stream)
- Data labeling workflow (no annotation system)
- Training/inference pipeline (doesn't exist)

---

## Architecture Sketch: Concrete ML Pipeline

### Data Format (Per Timestep t)

```
acc[t]  – 3 floats (ax, ay, az)
gyro[t] – 3 floats (gx, gy, gz)
mag[t]  – 3 floats (mx, my, mz)
Optional: derived features (magnitude, filtered variants)
```

### Window Structure

```
Shape: (T, F) where T = timesteps (e.g., 32), F = features (9+ dims)
```

### Model Architecture (Classification)

```
Input: T × F tensor
  ↓
1D Conv (temporal) + ReLU
  ↓
1D Conv + ReLU
  ↓
Global average pooling over time
  ↓
Dense → Dense → Softmax over N gestures
```

### Model Architecture (Regression to Hand Pose)

```
Same backbone
  ↓
Linear output layer → pose vector (e.g., 10 dims for 5 fingers)
```

**Model Size:** Small enough to:
- Train on laptop
- Deploy to ESP32 via quantization

**Current Implementation:** No model exists; GAMBIT provides raw data streams only.

---

## Why SIMCAP Matters: Conceptual Significance

### The Core Question

> "How much structural information about a complex body (the hand) can be inferred from a tiny, noisy, indirect sensor + some magnets?"

This represents a fascinating exploration of:
- Sensor fusion under constraints
- Learned indirect measurement
- Physically-embedded state inference
- Minimalist hardware with sophisticated software

### The Broader Vision

SIMCAP embodies a "state machine with latent context" paradigm—a physically-embedded version of inferring complex state from minimal observations. The glove becomes a one-sensor state machine for hand dynamics.

### Potential Applications

1. **Chorded Input Device**
   - IDE control via finger combinations
   - Wearable macro keyboard
   - Accessibility interface

2. **Spatial Computing Controller**
   - Memory palace navigation
   - 3D environment manipulation
   - XR interaction paradigm

3. **Cognitive Tool**
   - Physical "algorithm sketching"
   - Gesture-based structure manipulation
   - Embodied computation interface

---

## Current Status Assessment

### What Exists Today (Strengths)

The SIMCAP project has established:
- ✓ Functional hardware platform (Puck.js)
- ✓ Complete 9-DoF sensor suite
- ✓ BLE communication infrastructure
- ✓ Web-based data collection UI
- ✓ NFC tap-to-connect UX
- ✓ Raw telemetry streaming at appropriate rates
- ✓ Basic state machine architecture
- ✓ Extensible firmware design

### What's Missing (Gaps)

The vision requires:
- ✗ Physical magnet hardware on fingers
- ✗ Machine learning pipeline (training + inference)
- ✗ Gesture recognition system
- ✗ Coordinate frame normalization
- ✗ Calibration procedures
- ✗ Data annotation workflow
- ✗ Model deployment infrastructure
- ✗ Real-world application interfaces

### The Path Forward

**Immediate Next Steps (Minimum Viable Extension):**
1. Add finger magnets (hardware modification)
2. Implement palm-centric coordinate transformation
3. Build data collection + labeling UI for poses
4. Train simple static pose classifier offline
5. Deploy to device or laptop for real-time demo

**Medium-Term Goals:**
- Dynamic gesture recognition
- On-device TinyML inference
- Practical application (chorded input, UI control)

**Long-Term Vision:**
- Approximate hand pose estimation
- Dual-hand coordination (JOYPAD concept)
- XR integration
- Cognitive tool applications

---

## Conclusion: The Project Isn't Dead

SIMCAP has been in "a really long compile cycle," not abandoned. The foundation exists; the vision remains compelling; the path is clear.

The current implementation (GAMBIT) provides reliable sensor data streaming—a necessary but insufficient component. The gap between vision and reality is significant but bridgeable through systematic execution of the proposed tier-based roadmap.

The project sits at a critical juncture: **the infrastructure works; the intelligence layer awaits construction.**

With finger magnets, coordinate normalization, and a basic ML pipeline, SIMCAP could rapidly evolve from "interesting data collector" to "functional motion inference system."

---

## References

- Current implementation: `src/device/GAMBIT/app.js`
- Web UI: `src/web/GAMBIT/`
- Project README: `README.md`
- Hardware platform: Espruino Puck.js (puck-js.com)
