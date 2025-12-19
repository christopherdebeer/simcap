# Magnetometer Y-Axis Sign Inversion Fix

**Date:** 2025-12-19
**Status:** Implemented and ready for testing

## Problem

The H/V ratio was consistently inverted (1.17 vs expected 0.33), causing:
- Incorrect Earth field direction estimation
- Poor finger magnet detection accuracy
- Yaw drift in orientation estimation

## Root Cause Analysis

### Correlation Analysis

Examining the correlation between accelerometer and magnetometer axes revealed:

| Correlation | Value | Expected |
|-------------|-------|----------|
| ax vs mx | +0.955 | Positive ✓ |
| ay vs my | **-0.919** | Positive ✗ |
| az vs mz | +0.645 | Positive ✓ |

The **negative correlation** between `ay` and `my` indicates the magnetometer Y-axis is inverted relative to the accelerometer Y-axis.

### Orientation-Specific Verification

When the device Y-axis points UP (against gravity):
- Accelerometer: ay = +1.0 g
- Magnetometer: my = **-43.2 µT** (should be +48 µT)

This confirms the Y-axis sign inversion.

## Solution

In `telemetry-processor.ts`, the magnetometer axis alignment was updated:

```typescript
// Before (incorrect):
const mx_ut = my_ut_raw;   // Mag Y -> aligned X
const my_ut = mx_ut_raw;   // Mag X -> aligned Y
const mz_ut = mz_ut_raw;   // Z unchanged

// After (correct):
const mx_ut = my_ut_raw;   // Mag Y -> aligned X
const my_ut = -mx_ut_raw;  // Mag X -> aligned Y, NEGATED
const mz_ut = mz_ut_raw;   // Z unchanged
```

## Results

### Live Diagnostics (2025-12-19T15:38)

```
[MagDiag] H/V components: H=15.1 µT (exp 16.0), V=39.6 µT (exp 47.8)
[MagDiag] H/V ratio: 0.38 (expected 0.33) ✓
```

### Offline Analysis Comparison

| Metric | Pre-fix Session | Post-fix Session | Expected |
|--------|-----------------|------------------|----------|
| ay vs my correlation | **-0.850** ✗ | **+0.744** ✓ | Positive |
| ax vs mx correlation | +0.471 | +0.823 | Positive |
| az vs mz correlation | +0.697 | +0.801 | Positive |

### Calibration Quality (Post-fix)

| Metric | Value | Target |
|--------|-------|--------|
| Earth residual (still) | 22.3 µT | <30 µT ✓ |
| Earth residual (moving) | 30.5 µT | - |
| Sphericity | 0.59 | >0.7 |
| H/V ratio (live) | 0.38 | 0.335 ✓ |

The Y-axis fix corrected the fundamental axis alignment issue. The remaining H/V error in offline analysis (0.63 vs 0.335) is due to soft iron distortion which the min-max calibration partially addresses.

## Impact

This fix affects:
1. **Magnetometer calibration** - Hard iron offset will now be computed correctly
2. **Earth field estimation** - H/V components will be properly separated
3. **Finger magnet detection** - Residual calculation will be more accurate
4. **Yaw estimation** - Heading will be more stable

## Testing

After this fix, the calibration diagnostics should show:
- H/V ratio close to 0.33 (within ±0.2)
- V component ~48 µT when level
- H component ~16 µT when level
- Positive correlation between ay and my

## Files Modified

- `apps/gambit/shared/telemetry-processor.ts` - Added Y-axis negation in magnetometer axis alignment

---

## Final Verification (17:15 session)

After combining the Y-axis fix with orientation-aware calibration, the final results show:

| Metric | Before Y-Axis Fix | After Y-Axis Fix + Orientation-Aware Cal |
|--------|-------------------|------------------------------------------|
| H/V ratio | 0.95 (inverted) | **0.36** ✓ |
| Expected H/V | 0.33 | 0.33 |
| Earth residual | 70-105 µT | **12-25 µT** |

The Y-axis fix was essential for enabling correct orientation-aware calibration. Without it, the calibration optimization would converge to incorrect parameters.

## Related Documentation

- [GAMBIT Telemetry Data Flow](./gambit-telemetry-data-flow.md) - Main documentation with full calibration details
- [Earth Residual Analysis](./earth-residual-analysis-2025-12-19.md) - Investigation leading to orientation-aware calibration
- [Gyro Bias Calibration Fix](./gyro-bias-calibration-fix-2025-12-19.md) - Related fix for yaw drift
