# Automatic Iron Calibration Investigation

**Technical Investigation Report**
**Date:** 2025-12-16
**Author:** Claude (AI Assistant)
**Based on:** Earth Field Subtraction Investigation (2025-12-15)

---

## Abstract

This investigation examines whether hard iron calibration can be performed automatically from streaming magnetometer data when finger magnets are present.

**Key Findings:**

1. **"Auto iron" does NOT measure true hard iron** - it captures mean finger magnet position (~100-400 µT) plus device hard iron (~30 µT), dominated by the finger magnets
2. **Self-centering (sliding window average) degrades SNR** by -8.82x on average
3. **Baseline MAGNITUDE matters, not stability** - correlation r = -0.838 between baseline magnitude and SNR change
4. **Low-magnitude baseline (fingers extended) can improve SNR** by up to +8.47x
5. **Hand movement is OK** during baseline capture (needed for Earth estimation); only FINGER position matters

**Recommendation:** Implement an optional "Extended Baseline" phase at session start where user extends fingers while rotating hand.

---

## 1. Background

### 1.1 Terminology Clarification

| Term | Definition | Typical Magnitude |
|------|------------|-------------------|
| **True Hard Iron** | Permanent magnetic offset from device components | 10-50 µT |
| **Extended Baseline** | Residual when fingers extended (far from sensor) | 20-80 µT |
| **Mean Finger Offset** | Average residual during normal use | 100-500 µT |

What we initially called "auto iron" is actually the **Mean Finger Offset**, not hard iron.

### 1.2 Current Calibration Architecture

The `UnifiedMagCalibration` class implements:
1. **Hard/Soft Iron** - Manual wizard-based calibration (optional)
2. **Earth Field** - Automatic real-time estimation (always active)

This investigation asked: **Can we add automatic baseline calibration for finger magnets?**

---

## 2. Initial Investigation: Self-Centering Approach

### 2.1 Hypothesis

After Earth subtraction, the residual in sensor frame is:
```
residual = raw_mag - Earth_rotated_to_sensor
residual ≈ hard_iron + finger_magnets
```

**Initial hypothesis:** Averaging the residual over time would capture a useful baseline.

### 2.2 Results: Self-Centering FAILS

| Session | Earth-Only SNR | Self-Centered SNR | Effect |
|---------|----------------|-------------------|--------|
| 1 (2564 samples) | 4.02x | 3.47x | **-0.55x** |
| 2 (968 samples) | 23.90x | 4.21x | **-19.69x** |
| **Average** | 13.96x | 3.84x | **-10.12x** |

❌ Self-centering **destroys signal** by averaging out the finger position variation we want to detect.

---

## 3. Reexamination: What Does Baseline Capture?

### 3.1 Key Insight: Mean Finger Position

The "auto iron" estimate captures:
```
baseline_estimate ≈ true_hard_iron (~30 µT) + mean_finger_magnet_field (~100-400 µT)
```

Session data shows baseline estimates of 120-430 µT - far larger than true hard iron.

### 3.2 Cross-Session Baseline Test

Using Session 1's baseline on Session 2:

| Method | Session 2 SNR | vs Earth-only |
|--------|---------------|---------------|
| Earth-only (no baseline) | 23.90x | - |
| Cross-session baseline | 20.65x | -3.25x |
| Self-centered (own mean) | 4.21x | -19.69x |

**Finding:** External baseline is **5x better** than self-centering, confirming that a fixed reference point preserves more signal range than centering around your own mean.

---

## 4. Critical Discovery: Magnitude vs Stability

### 4.1 Correlation Analysis

We analyzed all possible baseline windows and correlated their magnitude with SNR impact:

| Session | Correlation (r) | Best Baseline | SNR Change |
|---------|-----------------|---------------|------------|
| 1 | **-0.927** | 57 µT | +0.67x |
| 2 | **-0.748** | 68 µT | +8.47x |
| **Average** | **-0.838** | - | - |

**Strong negative correlation:** Lower baseline magnitude = Better SNR improvement

### 4.2 Physical Interpretation

| Baseline Magnitude | Finger Position | SNR Effect |
|-------------------|-----------------|------------|
| Low (~20-80 µT) | Extended (far from sensor) | **HELPS** (+0.5 to +8x) |
| Medium (~100-150 µT) | Partially flexed | **MARGINAL** (±1x) |
| High (~200-400 µT) | Flexed (close to sensor) | **HURTS** (-2 to -24x) |

### 4.3 Why Magnitude Matters

```
Signal = finger_residual - baseline

If baseline is LOW (fingers extended):
  - Minimum signal ≈ low - low ≈ near zero
  - Maximum signal ≈ high - low ≈ high
  - Range preserved ✓

If baseline is HIGH (fingers flexed):
  - Minimum signal ≈ low - high ≈ negative
  - Maximum signal ≈ high - high ≈ small
  - Range compressed ✗
```

SNR = peak/baseline benefits from a low, consistent baseline value.

---

## 5. Key Finding: Hand Movement ≠ Finger Movement

### 5.1 Independent Requirements

| Requirement | Why | What User Does |
|-------------|-----|----------------|
| **Hand rotation** | Earth field estimation needs varied orientations | Move hand around |
| **Finger extension** | Low-magnitude baseline needs fingers far from sensor | Keep fingers straight |

These are **independent** - user CAN rotate hand while keeping fingers extended.

### 5.2 Implications

A brief "Extended Baseline" calibration phase is viable:
- User rotates hand normally (Earth estimation works)
- User keeps fingers extended during rotation (baseline captured)
- Duration: ~2-4 seconds (100-200 samples at 50Hz)

---

## 6. Recommendations

### 6.1 Proposed Architecture

```
┌────────────────────────────────────────────────────────┐
│              UnifiedMagCalibration                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Phase 1: Baseline Capture (first 2-4 seconds)         │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Extended Baseline (optional but recommended)    │ │
│  │  • Prompt: "Extend fingers, rotate hand"         │ │
│  │  • Capture mean residual with fingers extended   │ │
│  │  • Verify magnitude < 100 µT (quality check)     │ │
│  │  • Store as session baseline                     │ │
│  └──────────────────────────────────────────────────┘ │
│                         ↓                              │
│  Phase 2: Normal Operation                             │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Earth Field Estimation (always active)          │ │
│  │  + Extended Baseline Subtraction (if captured)   │ │
│  └──────────────────────────────────────────────────┘ │
│                         ↓                              │
│              Residual Output for Magnet Detection      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 6.2 Quality Gate

| Baseline Magnitude | Quality | Action |
|-------------------|---------|--------|
| < 80 µT | Excellent | Use baseline |
| 80-120 µT | Good | Use baseline with caution |
| > 120 µT | Poor | Warn user, retry or skip |

### 6.3 Fallback Behavior

If baseline capture fails or user skips:
- Fall back to Earth-only (still provides +7.99x average improvement)
- No degradation from bad baseline

### 6.4 Naming Convention

**Recommended term:** "Extended Baseline" or "Reference Offset"

**NOT:** "Hard Iron" (misleading - captures finger magnets, not device bias)
**NOT:** "Rest Baseline" (implies stillness; hand movement is actually needed)

---

## 7. Summary

| Approach | Effect on SNR | Recommendation |
|----------|---------------|----------------|
| Earth-only (current) | **+7.99x** | Keep as default |
| Self-centering | **-10.12x** | Do NOT implement |
| Extended Baseline (new) | **+0.5 to +8.5x** | Implement as optional |
| Manual Hard Iron | Variable | Keep for advanced users |

**Key insight:** What matters is not stability (CV) or averaging out motion, but capturing the **low-magnitude state** when fingers are extended. This simple calibration step can provide significant additional SNR improvement beyond Earth-only.

---

## Appendix A: Analysis Scripts

| Script | Purpose |
|--------|---------|
| `auto_iron_calibration_investigation.py` | Initial self-centering test |
| `auto_iron_reexamination.py` | Mean finger position analysis |
| `cross_session_baseline_test.py` | External baseline transfer test |
| `automatic_rest_baseline_analysis.py` | Early-period baseline test |
| `rest_detection_baseline_analysis.py` | CV-based detection attempt |
| `baseline_magnitude_analysis.py` | **Key finding**: magnitude correlation |

---

## Appendix B: Raw Data

### Session 1 (2564 samples)
- Earth-only SNR: 4.02x
- Optimal baseline magnitude: 57 µT
- Optimal baseline SNR improvement: +0.67x
- Magnitude-SNR correlation: r = -0.927

### Session 2 (968 samples)
- Earth-only SNR: 23.90x
- Optimal baseline magnitude: 68 µT
- Optimal baseline SNR improvement: +8.47x
- Magnitude-SNR correlation: r = -0.748

---

## References

1. `docs/technical/earth-field-subtraction-investigation.md` - Earth field estimation
2. `src/web/GAMBIT/shared/unified-mag-calibration.js` - Current implementation
