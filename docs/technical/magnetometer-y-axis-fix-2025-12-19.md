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

With the Y-axis negation applied:

| Metric | Before | After | Expected |
|--------|--------|-------|----------|
| H/V ratio | 1.17 | 0.54 | 0.335 |
| V component | 6.5 µT | 43.4 µT | 47.8 µT |
| H component | 51.9 µT | 23.4 µT | 16.0 µT |
| ay vs my correlation | -0.919 | +0.919 | Positive |

The H/V ratio improved from **1.17 to 0.54** (expected 0.335), a significant improvement. The remaining error is due to soft iron distortion which the min-max calibration addresses.

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
