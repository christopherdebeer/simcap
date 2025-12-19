# Earth Field Residual Analysis - 2025-12-19

## Summary

Investigation into why the Earth field residual is ~50 µT instead of near-zero when no finger magnets are present.

## Key Findings

### 1. Iron Calibration is Distorting Field Direction

The min-max hard/soft iron calibration produces:
- **Total magnitude**: 47.0 µT (close to expected 50.4 µT) ✓
- **Horizontal component**: 31.2 µT (expected ~16 µT) - **2x too high!** ✗
- **Vertical component**: 24.4 µT (expected ~48 µT) - **2x too low!** ✗

The H/V ratio is inverted. This means the soft iron correction is scaling the axes incorrectly, causing the field direction to be wrong even though the magnitude is approximately correct.

### 2. AHRS Orientation is Unreliable

- Dot product between corrected mag and expected Earth: **mean = 0.328** (should be ~1.0)
- Only 19% of samples have good alignment (dot > 0.9)
- Yaw difference between AHRS and magnetometer-derived: std = 95° (essentially random)

### 3. Circular Dependency Problem

The system has a circular dependency:
1. AHRS uses magnetometer to compute yaw
2. Yaw affects orientation quaternion
3. Orientation is used to rotate Earth field to device frame
4. Residual = measured_mag - rotated_earth
5. If yaw is wrong, residual is wrong
6. High residual → calibration thinks something is wrong
7. Low confidence → AHRS doesn't trust magnetometer
8. AHRS relies more on gyro → yaw drifts
9. Go to step 2 (vicious cycle)

### 4. Min-Max Calibration Limitations

The min-max method assumes:
- Hard iron offset is a simple 3D vector
- Soft iron distortion is axis-aligned (diagonal matrix)

Reality:
- Soft iron distortion includes cross-axis coupling (full 3x3 matrix)
- The device may have significant off-diagonal soft iron terms

## Root Cause

The **soft iron calibration is inadequate**. The simple axis-scaling approach cannot correct for:
1. Cross-axis coupling in the soft iron matrix
2. Non-orthogonal sensor axes
3. Mounting misalignment between magnetometer and accelerometer/gyroscope

## Evidence

Sample analysis showing H/V component errors:

```
Sample 0:
  Corrected mag: [-4.3, 2.7, 35.2] |35.6|
  Horizontal mag: |4.8| (expected ~16 µT)
  Vertical mag: 35.5 (expected ~48 µT)

Sample 100:
  Corrected mag: [-42.5, -19.2, -15.0] |49.0|
  Horizontal mag: |42.0| (expected ~16 µT)  ← 2.6x too high!
  Vertical mag: 17.0 (expected ~48 µT)      ← 2.8x too low!
```

## Solutions Tested

All three approaches were implemented and tested in Python:

### Option 1: Full Ellipsoid Fitting

Replace min-max calibration with ellipsoid fitting:
- Fit a 3x3 soft iron matrix (9 parameters) + 3D hard iron offset
- Use least squares to minimize |corrected_mag| - expected_mag

**Results:**
- Magnitude: 50.4 ± 1.3 µT ✓ (perfect)
- Horizontal: 34.7 µT (expected 16.0) ✗
- H/V ratio: 1.31 (expected 0.33) ✗
- **Problem**: Only constrains magnitude, not direction

### Option 2: Geomagnetic H/V Constraint

Constrain calibration to produce correct H/V ratio:
- Use Edinburgh reference: H=16.0 µT, V=47.8 µT
- Optimize to match both H and V components

**Results:**
- Magnitude: 46.4 ± 10.5 µT
- Horizontal: 15.8 ± 7.6 µT ✓ (good)
- Vertical: 36.4 ± 19.0 µT (expected 47.8) - improved but high variance
- H/V ratio: 0.43 (expected 0.33) - better

### Option 3: Orientation-Aware Calibration ✓ BEST

Use accelerometer orientation during calibration:
- Compute expected Earth field direction from accel (roll/pitch)
- Compute yaw from tilt-compensated magnetometer
- Optimize calibration to match full 3D Earth field vector in device frame

**Results:**
- Magnitude: 50.1 ± 1.5 µT ✓
- Horizontal: 15.0 ± 4.9 µT ✓
- Vertical: 33.4 ± 18.7 µT (still some variance)
- H/V ratio: 0.45 (improved)
- **Earth residual: 4.4 ± 3.3 µT** ✓ (90% reduction!)

## Comparison Summary

| Metric | Current (Min-Max) | Option 3 (Best) | Improvement |
|--------|-------------------|-----------------|-------------|
| Earth Residual | 43.9 µT | 4.4 µT | **90% reduction** |
| Magnitude | 47.0 ± 11.4 µT | 50.1 ± 1.5 µT | 87% std reduction |
| Horizontal | 31.2 µT | 15.0 µT | Closer to 16.0 |
| H/V Ratio | 1.28 | 0.45 | Closer to 0.33 |

## Best Calibration Parameters

From Option 3 (Orientation-Aware):

```
Hard iron offset: [30.72, -35.80, -36.07] µT

Soft iron matrix:
  [1.4590, -0.3133, -0.1090]
  [0.2151, -1.1729,  0.0683]
  [-0.3309, -0.5456, 0.9464]
```

Note the significant off-diagonal terms - this is why diagonal-only calibration fails.

## Metrics to Track

1. **Earth Residual**: Should be <10 µT (achieved 4.4 µT)
2. **H/V Ratio**: Should be ~0.33 (16/48) for Edinburgh
3. **Magnitude Error**: Should be <5% of expected
4. **Magnitude Std**: Should be <5 µT (achieved 1.5 µT)

## Files

- `ml/analyze_earth_residual.py` - Initial investigation script
- `ml/calibration_comparison.py` - Comprehensive comparison of all three approaches
- Session analyzed: `2025-12-19T13_16_59.786Z.json`

## Next Steps

1. **Implement Option 3 in TypeScript** - Port orientation-aware calibration to `unified-mag-calibration.ts`
2. **Add full 3x3 soft iron matrix support** - Replace diagonal scaling with matrix multiplication
3. **Use accelerometer for calibration** - Require accel data during calibration phase
4. **Add residual monitoring** - Track Earth residual as calibration quality metric
