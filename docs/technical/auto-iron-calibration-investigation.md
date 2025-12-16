# Automatic Iron Calibration Investigation

**Technical Investigation Report**
**Date:** 2025-12-16
**Author:** Claude (AI Assistant)
**Based on:** Earth Field Subtraction Investigation (2025-12-15)

---

## Abstract

This investigation examines whether hard iron calibration can be performed automatically from streaming magnetometer data when finger magnets are present. Building on the successful real-time Earth field estimation approach, we tested whether sensor-frame residual averaging could extract the hard iron offset.

**Key Finding:** Automatic hard iron calibration from streaming data with finger magnets present **DOES NOT WORK** and actually **DEGRADES** signal quality. The current architecture—automatic Earth field estimation + optional manual iron calibration—is the correct approach.

---

## 1. Background

### 1.1 Current Calibration Architecture

The `UnifiedMagCalibration` class implements:
1. **Hard/Soft Iron** - Manual wizard-based calibration (optional)
2. **Earth Field** - Automatic real-time estimation (always active)

The Earth field estimation works by:
- Transforming raw magnetometer readings to world frame
- Averaging in world frame (Earth is constant, biases rotate out)
- Subtracting orientation-rotated Earth estimate from raw readings

This investigation asked: **Can we apply the same principle to extract hard iron?**

### 1.2 Theoretical Basis

After Earth subtraction, the residual in sensor frame is:
```
residual = raw_mag - Earth_rotated_to_sensor
residual ≈ hard_iron + finger_magnets + noise
```

**Hypothesis:** If finger magnet positions vary over time, their contribution to the sensor-frame residual should average toward zero, leaving the hard iron offset.

---

## 2. Methodology

### 2.1 Approaches Tested

**Approach 1: Simultaneous Estimation**
- Estimate Earth and hard iron concurrently
- Earth from world-frame averaging (200 samples)
- Iron from sensor-frame residual averaging (500 samples)

**Approach 2: Two-Phase Estimation**
- Phase 1: Estimate Earth field until stable
- Phase 2: After Earth stabilizes, estimate iron from residuals

### 2.2 Test Data

Two sessions from 2025-12-15 with finger magnets attached:
- Session 1: 2564 samples, excellent rotation (91% active)
- Session 2: 968 samples, good rotation (68% active)

### 2.3 Evaluation Metric

Signal-to-Noise Ratio (SNR): Peak (95th percentile) / Baseline (25th percentile)

---

## 3. Results

### 3.1 Earth-Only Calibration (Already Implemented)

| Session | RAW SNR | Earth-Only SNR | Improvement |
|---------|---------|----------------|-------------|
| 1 (2564 samples) | 2.96x | 4.02x | **+1.06x** |
| 2 (968 samples) | 8.98x | 23.90x | **+14.93x** |
| **Average** | 5.97x | 13.96x | **+7.99x** |

✅ Earth field estimation provides **consistent, significant SNR improvement**.

### 3.2 Earth + Auto Iron Calibration

| Session | Earth-Only SNR | Earth+Iron SNR | Iron Effect |
|---------|----------------|----------------|-------------|
| 1 (2564 samples) | 4.02x | 3.66x | **-0.36x** |
| 2 (968 samples) | 23.90x | 6.62x | **-17.29x** |
| **Average** | 13.96x | 5.14x | **-8.82x** |

❌ Adding auto iron calibration **DEGRADES** SNR significantly.

### 3.3 Iron Estimate Behavior

Despite convergence indicators looking positive:
- Iron estimate variance decreased over time (✓ appears stable)
- Residual stability improved in Session 2

**The problem:** The iron estimate converges to **hard_iron + average_magnet_field**, not hard_iron alone.

---

## 4. Analysis

### 4.1 Why Auto Iron Calibration Fails

The fundamental assumption was flawed:

| Component | Frame | Behavior | Averages To |
|-----------|-------|----------|-------------|
| Earth field | World (constant) | Rotates in sensor frame | **Zero** in sensor frame |
| Hard iron | Sensor (constant) | Constant in sensor frame | **Itself** |
| Finger magnets | Sensor (varies slowly) | Quasi-constant in sensor frame | **Non-zero mean** |

**Key insight:** Finger magnets, while varying with finger position, remain roughly constant over short time scales in the sensor frame. Their field strength (~100-500 µT) is much larger than typical hard iron (~10-50 µT), so the residual average is dominated by magnet signals, not hard iron.

### 4.2 Mathematical Explanation

```
sensor_residual = raw - Earth_sensor
                = hard_iron + finger_magnets

E[sensor_residual] = hard_iron + E[finger_magnets]
                   ≈ hard_iron + (large magnet contribution)
                   ≠ hard_iron
```

For the averaging to work, we would need:
- `E[finger_magnets] ≈ 0`
- This requires finger positions to be uniformly distributed (unlikely during normal use)

### 4.3 Why Earth Estimation Works But Iron Doesn't

Earth estimation succeeds because:
1. Earth field is **truly constant** in world frame
2. **All** sensor-frame components (hard iron, magnets) rotate with the device
3. Rotation causes these to average toward zero in world frame

Iron estimation fails because:
1. In sensor frame, both hard iron AND magnets are quasi-constant
2. No rotation can separate them via simple averaging
3. Would need magnets to vary independently of device rotation

---

## 5. Conclusions

### 5.1 Negative Result

**Automatic hard iron calibration from streaming data with finger magnets present is not feasible using simple averaging techniques.**

The sensor-frame residual contains both hard iron and finger magnet fields, which cannot be separated through temporal averaging because both are approximately constant in the sensor frame.

### 5.2 Correct Architecture (Confirmed)

The current `UnifiedMagCalibration` design is correct:

```
┌────────────────────────────────────────────────────────┐
│              UnifiedMagCalibration                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Hard/Soft Iron Calibration                      │ │
│  │  • Manual wizard (without magnets)               │ │
│  │  • Persisted to localStorage                     │ │
│  │  • OPTIONAL but provides best results            │ │
│  └──────────────────────────────────────────────────┘ │
│                         ↓                              │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Earth Field Estimation                          │ │
│  │  • Real-time, automatic                          │ │
│  │  • World-frame averaging (200 samples)           │ │
│  │  • ALWAYS ACTIVE                                 │ │
│  └──────────────────────────────────────────────────┘ │
│                         ↓                              │
│              Residual Output for Magnet Detection      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 5.3 Recommendations

1. **Keep current architecture**: Earth-only automatic calibration is highly effective
2. **Do NOT implement auto iron**: It degrades signal quality
3. **Manual iron calibration**: Keep as optional enhancement for users wanting maximum accuracy
4. **Document limitation**: Users should remove finger magnets during iron calibration wizard

---

## 6. Future Work (If Needed)

Alternative approaches that MIGHT work for auto iron calibration:

1. **Gyroscope-gated estimation**: Only accumulate iron estimate when device is stationary (no finger movement assumed)
2. **Machine learning**: Train a model to separate hard iron from magnet signals based on signal characteristics
3. **Multi-session analysis**: Compare multiple sessions to find common sensor-frame bias
4. **User-initiated calibration period**: Ask user to keep fingers still while estimating iron

However, given the strong performance of Earth-only calibration (+7.99x average SNR improvement), the marginal benefit of these complex approaches is likely not worth the implementation cost.

---

## Appendix A: Scripts Created

| Script | Purpose |
|--------|---------|
| `auto_iron_calibration_investigation.py` | Initial hypothesis testing |
| `auto_iron_detailed_analysis.py` | Detailed dynamics analysis |

---

## References

1. `docs/technical/earth-field-subtraction-investigation.md` - Earth field estimation validation
2. `src/web/GAMBIT/shared/unified-mag-calibration.js` - Current calibration implementation
