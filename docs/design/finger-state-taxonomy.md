# Finger State Taxonomy and Data Collection Strategy

**Status:** Design Document
**Date:** January 2026
**Related:**
- [Aligned-FFO Symbiosis](./aligned-ffo-symbiosis.md)
- [Trajectory Study](../../ml/trajectory_vs_single_sample_study.py)

---

## Executive Summary

This document defines a comprehensive taxonomy for finger states, semantic poses, and motions to improve data collection for the GAMBIT finger tracking system. The taxonomy is organized in layers from coarse to fine-grained, enabling:

1. **Robust coarse classification** (finger states)
2. **Rich semantic recognition** (poses and gestures)
3. **Orientation-independent inference** (via multi-orientation capture)
4. **Noise-robust models** (via erratic/full-range phases)

---

## 1. Finger State Taxonomy

### 1.1 Coarse States (3-Level)

The minimum viable categorization for each finger:

| State | Code | Description | Approximate Angle |
|-------|------|-------------|-------------------|
| **Extended** | `0` | Finger straight, fully extended | 0-30° |
| **Partial** | `1` | Relaxed natural curl, partially flexed | 30-70° |
| **Flexed** | `2` | Finger curled into palm | 70-90° |

**Notation:** A hand pose is encoded as 5 digits, e.g., `00000` (all extended), `22222` (fist), `20000` (thumb flexed).

### 1.2 Fine States (5-Level)

For higher-fidelity tracking when hardware/data supports it:

| State | Code | Description | Approximate Angle |
|-------|------|-------------|-------------------|
| **Hyperextended** | `0` | Extended beyond straight (thumb opposition capable) | <0° |
| **Extended** | `1` | Fully straight | 0-20° |
| **Relaxed** | `2` | Natural slight curl (resting state) | 20-45° |
| **Partial** | `3` | Actively flexed but not closed | 45-70° |
| **Flexed** | `4` | Fully curled into palm | 70-90° |

### 1.3 Thumb Special States

The thumb has additional degrees of freedom:

| State | Description |
|-------|-------------|
| **abducted** | Thumb spread away from palm |
| **adducted** | Thumb alongside palm |
| **opposed** | Thumb rotated to face fingers (for pinch/grip) |
| **tucked** | Thumb curled under fingers (inside fist) |

---

## 2. Semantic Pose Vocabulary

### 2.1 Tier 1: Core Poses (Essential)

These poses appear frequently in everyday interaction and should be prioritized for data collection.

| Pose | Finger Code | Description | Common Use |
|------|-------------|-------------|------------|
| **open_palm** | `00000` | All fingers extended, spread | Stop, wave, high-five |
| **fist** | `22222` | All fingers flexed | Power grip, knock |
| **point** | `02222` | Index extended only | Pointing, selecting |
| **thumbs_up** | `20000` | Thumb extended only | Approval |
| **peace** | `00222` | Index + middle extended | Victory, peace, two |
| **pinch** | `21111` | Thumb + index tips touching | Precision grip |
| **ok** | `21100` | Thumb + index circle, others extended | OK sign |
| **rest** | `11111` | All relaxed natural curl | Neutral resting |

### 2.2 Tier 2: Counting & Numbers

| Pose | Finger Code | Description |
|------|-------------|-------------|
| **one** | `02222` | Index up (same as point) |
| **two** | `00222` | Index + middle up (same as peace) |
| **three** | `00022` | Index + middle + ring up |
| **four** | `00002` | All except thumb extended |
| **five** | `00000` | All extended (same as open_palm) |
| **six** | `20002` | Thumb + pinky (call me / shaka) |
| **seven** | `20022` | Thumb + ring + pinky |
| **eight** | `20222` | Thumb + middle + ring + pinky |
| **nine** | `22220` | Thumb over index, others up |
| **ten** | `00000` × 2 | Both hands open / thumbs up shake |

### 2.3 Tier 3: Grip Poses

| Pose | Description | Use Case |
|------|-------------|----------|
| **power_grip** | Full fist around object | Holding hammer |
| **precision_grip** | Thumb-finger opposition | Holding pen |
| **lateral_grip** | Thumb against index side | Holding key |
| **hook_grip** | Fingers curled, thumb free | Carrying bag |
| **spherical_grip** | All fingers curved around sphere | Holding ball |
| **cylindrical_grip** | Fingers wrap cylinder | Holding bottle |
| **tripod_grip** | Thumb + index + middle | Writing |

### 2.4 Tier 4: Interaction Gestures

| Gesture | Motion Type | Description |
|---------|------------|-------------|
| **grab** | dynamic | Close from open to fist |
| **release** | dynamic | Open from fist to open |
| **tap** | dynamic | Quick flex-extend of index |
| **pinch_zoom** | dynamic | Pinch spreading apart |
| **scroll** | dynamic | Fingers curled, wrist twist |
| **swipe** | dynamic | Flat hand, lateral motion |
| **wave** | dynamic | Open palm, side-to-side |
| **rotate** | dynamic | Wrist rotation (pronation/supination) |
| **flick** | dynamic | Quick finger extension |

### 2.5 Tier 5: Sign Language (ASL Alphabet)

For accessibility and rich vocabulary:

| Letter | Description | Finger Code |
|--------|-------------|-------------|
| A | Fist, thumb beside | `22222` (thumb adducted) |
| B | Flat fingers, thumb tucked | `00002` (thumb across) |
| C | Curved fingers, open C shape | curved |
| D | Index up, others form circle | special |
| ... | (Full ASL alphabet) | ... |

---

## 3. Motion Taxonomy

### 3.1 Motion Types

| Type | Code | Description | Duration |
|------|------|-------------|----------|
| **static** | `S` | No intentional movement | 2-10s |
| **transition** | `T` | Moving between two poses | 0.5-2s |
| **repetitive** | `R` | Same motion repeated | cycles |
| **continuous** | `C` | Ongoing fluid motion | variable |
| **erratic** | `E` | Random/unpredictable | 5-15s |

### 3.2 Transition Matrix

Key transitions to capture (from → to):

| From | To | Name | Priority |
|------|-----|------|----------|
| rest | fist | close_hand | High |
| fist | open_palm | open_hand | High |
| rest | point | point_start | High |
| point | rest | point_end | High |
| open_palm | pinch | pinch_start | Medium |
| pinch | open_palm | pinch_release | Medium |
| rest | thumbs_up | thumbs_gesture | Medium |
| any | rest | relaxation | High |

### 3.3 Repetitive Patterns

| Pattern | Description | Cycles |
|---------|-------------|--------|
| **finger_tap** | Repeated index tap | 5-10 |
| **finger_drum** | Sequential finger taps | 3-5 |
| **fist_pump** | Open-close-open cycle | 3-5 |
| **wave_motion** | Side-to-side wave | 3-5 |
| **pinch_release** | Pinch-open cycle | 5-10 |

---

## 4. Orientation Taxonomy

### 4.1 Palm Orientations (6 Cardinal)

To ensure orientation-independent signatures, capture each pose in multiple orientations:

| Orientation | Description | Rotation |
|-------------|-------------|----------|
| **palm_down** | Palm facing ground | Reference (0°) |
| **palm_up** | Palm facing ceiling | 180° roll |
| **palm_left** | Palm facing left | 90° yaw |
| **palm_right** | Palm facing right | -90° yaw |
| **palm_forward** | Palm facing away from body | 90° pitch |
| **palm_back** | Palm facing toward body | -90° pitch |

### 4.2 Wrist Positions

| Position | Description |
|----------|-------------|
| **neutral** | Wrist straight |
| **flexed** | Wrist bent, palm toward forearm |
| **extended** | Wrist bent, back of hand toward forearm |
| **ulnar** | Wrist tilted toward pinky |
| **radial** | Wrist tilted toward thumb |

### 4.3 Orientation Sweep Protocol

For robust orientation-independence:

1. **Static orientations**: Hold pose in each of 6 cardinal orientations (3s each)
2. **Slow rotation**: Rotate hand 360° over 10s while maintaining pose
3. **Fast rotation**: Rotate hand 360° over 3s while maintaining pose
4. **Random orientation**: Move hand to random orientations (10s total)

---

## 5. Non-Semantic Robustness Phases

These phases don't capture semantic poses but ensure the model handles real-world variability.

### 5.1 Erratic Movement Phase

**Purpose:** Capture sensor noise and transient states

| Phase | Description | Duration |
|-------|-------------|----------|
| **erratic_fingers** | Wiggle all fingers randomly | 10s |
| **erratic_orientation** | Rotate hand randomly | 10s |
| **erratic_combined** | Random fingers + orientation | 15s |

### 5.2 Full Range Phase

**Purpose:** Capture complete motion envelope for each finger

| Phase | Description | Cycles |
|-------|-------------|--------|
| **range_thumb** | Thumb full abduction/adduction + flex/extend | 3 |
| **range_index** | Index full flex/extend | 3 |
| **range_middle** | Middle full flex/extend | 3 |
| **range_ring** | Ring full flex/extend | 3 |
| **range_pinky** | Pinky full flex/extend | 3 |
| **range_all** | All fingers simultaneously | 3 |
| **range_sequential** | Fingers one at a time, wave pattern | 3 |

### 5.3 Calibration Phases

**Purpose:** Establish sensor baselines and environmental corrections

| Phase | Description | Duration |
|-------|-------------|----------|
| **cal_earth_field** | Hold device still, no magnets | 10s |
| **cal_hard_iron** | Rotate slowly (figure-8) for bias | 20s |
| **cal_reference** | Neutral pose, palm down | 10s |
| **cal_per_finger** | Each finger isolated flex | 5s × 5 |

---

## 6. Wizard Session Configurations

### 6.1 Session Types

| Type | Purpose | Duration | Priority |
|------|---------|----------|----------|
| **quick_cal** | Fast calibration only | 2 min | Required |
| **core_poses** | Essential 8 poses | 5 min | Required |
| **full_taxonomy** | All coarse states | 15 min | Recommended |
| **semantic_rich** | All semantic poses | 25 min | Optional |
| **robustness** | Erratic + orientation | 10 min | Recommended |
| **comprehensive** | Everything above | 45 min | Full coverage |

### 6.2 Quick Calibration Session (2 min)

```
1. cal_earth_field (10s)
2. cal_hard_iron (20s)
3. cal_reference (10s)
4. rest baseline (10s)
5. fist baseline (10s)
6. open_palm baseline (10s)
7. Quick finger isolation (5s × 5 = 25s)
```

### 6.3 Core Poses Session (5 min)

```
For each of 8 core poses (open_palm, fist, point, thumbs_up, peace, pinch, ok, rest):
  - Static hold palm_down (5s)
  - Static hold palm_up (5s)
  - Transition to next pose (3s)
```

### 6.4 Full Robustness Session (10 min)

```
1. Orientation sweeps per pose (3 poses × 6 orientations × 3s = 54s each × 3 = 2.7 min)
2. Erratic movement phases (10s × 3 = 30s)
3. Full range phases (10s × 7 = 70s)
4. Random orientation holds (30s × 3 poses = 90s)
```

---

## 7. Data Collection Priorities

### 7.1 Priority Matrix

| Data Type | Importance | Coverage Goal |
|-----------|------------|---------------|
| Coarse finger states (32 configs) | Critical | 100% |
| Core semantic poses (8) | Critical | 100% |
| Multiple orientations per pose | High | ≥3 per pose |
| Erratic/noise phases | High | 1 per session |
| Full range per finger | High | 1 per session |
| Extended semantic poses | Medium | 50% |
| Transitions | Medium | Top 10 |
| ASL alphabet | Low | As needed |

### 7.2 Minimum Viable Dataset

For initial model training:

1. **32 coarse configurations**: All combinations of 5 fingers × 2 states (extended/flexed)
2. **6 orientations per config**: Palm down/up/left/right/forward/back
3. **Hold duration**: 3-5s per orientation
4. **Repetitions**: 2-3 sessions per configuration
5. **Total samples**: ~32 × 6 × 4s × 50Hz × 3 reps = ~115,200 samples

### 7.3 Data Quality Requirements

| Requirement | Threshold | Measurement |
|-------------|-----------|-------------|
| Static stability | σ < 50 μT | Magnetometer std during hold |
| Orientation coverage | Δ > 30° | Quaternion diversity |
| Label accuracy | 100% | Wizard-verified |
| Session completeness | >95% | All steps completed |
| Calibration quality | R² > 0.95 | Earth field fit |

---

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure (Immediate)
- [ ] Update wizard.ts with new session types
- [ ] Add orientation tracking to label schema
- [ ] Implement robustness phases

### Phase 2: Data Collection (Week 1-2)
- [ ] Collect quick_cal + core_poses from 5 users
- [ ] Collect full_taxonomy from 2 users
- [ ] Collect robustness sessions

### Phase 3: Model Update (Week 2-3)
- [ ] Retrain aligned model with orientation-augmented data
- [ ] Evaluate per-orientation accuracy
- [ ] Compare to baseline

### Phase 4: Extended Vocabulary (Future)
- [ ] Add semantic pose recognition
- [ ] Add transition detection
- [ ] Add gesture recognition

---

## Appendix A: Complete Finger State Matrix

All 32 binary (extended/flexed) configurations:

| Code | Thumb | Index | Middle | Ring | Pinky | Common Name |
|------|-------|-------|--------|------|-------|-------------|
| 00000 | E | E | E | E | E | open_palm |
| 00002 | E | E | E | E | F | four |
| 00020 | E | E | E | F | E | - |
| 00022 | E | E | E | F | F | three |
| 00200 | E | E | F | E | E | - |
| 00202 | E | E | F | E | F | - |
| 00220 | E | E | F | F | E | - |
| 00222 | E | E | F | F | F | peace/two |
| 02000 | E | F | E | E | E | - |
| 02002 | E | F | E | E | F | - |
| 02020 | E | F | E | F | E | - |
| 02022 | E | F | E | F | F | - |
| 02200 | E | F | F | E | E | - |
| 02202 | E | F | F | E | F | - |
| 02220 | E | F | F | F | E | - |
| 02222 | E | F | F | F | F | point |
| 20000 | F | E | E | E | E | thumbs_up |
| 20002 | F | E | E | E | F | shaka/six |
| 20020 | F | E | E | F | E | - |
| 20022 | F | E | E | F | F | - |
| 20200 | F | E | F | E | E | - |
| 20202 | F | E | F | E | F | - |
| 20220 | F | E | F | F | E | - |
| 20222 | F | E | F | F | F | - |
| 22000 | F | F | E | E | E | - |
| 22002 | F | F | E | E | F | - |
| 22020 | F | F | E | F | E | - |
| 22022 | F | F | E | F | F | - |
| 22200 | F | F | F | E | E | - |
| 22202 | F | F | F | E | F | - |
| 22220 | F | F | F | F | E | - |
| 22222 | F | F | F | F | F | fist |

---

## Appendix B: Orientation Quaternion Examples

Reference orientations for capture (approximate quaternions):

| Orientation | w | x | y | z | Notes |
|-------------|---|---|---|---|-------|
| palm_down | 1.0 | 0 | 0 | 0 | Reference |
| palm_up | 0 | 1 | 0 | 0 | 180° roll |
| palm_left | 0.707 | 0 | 0 | 0.707 | 90° yaw |
| palm_right | 0.707 | 0 | 0 | -0.707 | -90° yaw |
| palm_forward | 0.707 | 0.707 | 0 | 0 | 90° pitch |
| palm_back | 0.707 | -0.707 | 0 | 0 | -90° pitch |

---

<link rel="stylesheet" href="../../src/simcap.css">
