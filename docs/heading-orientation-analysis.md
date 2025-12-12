# Heading-Informed Orientation Compensation Analysis

## Executive Summary

**Question**: Can we use general heading to inform orientation and improve environmental interference compensation?

**Answer**: YES, but the current implementation has a fundamental bug that must be fixed first.

## Key Findings

### 1. Current Earth Field Subtraction is Broken

The `calibration.js:correct()` method uses the wrong rotation:

```javascript
// CURRENT (WRONG):
const rotatedEarth = rotMatrix.multiply(this.earthField);

// CORRECT:
const rotatedEarth = rotMatrix.transpose().multiply(this.earthField);
```

**Why it's wrong**: `earthField` is stored in sensor frame at calibration time. To subtract it at runtime, we need to transform the world-frame field INTO the current sensor frame, which requires `R.T` (transpose), not `R`.

### 2. Session Data Analysis Results

Testing on session `2025-12-11T18_41_08.248Z.json` (1616 samples, full orientation coverage):

| Metric | Current Algorithm | Heading-Informed |
|--------|------------------|------------------|
| |B| std dev | 251.5 LSB | 14.0 LSB |
| Coefficient of Variation | 17.9% | 1.0% |
| **Improvement** | - | **94.2%** |

### 3. Iron-Corrected Field is Already Stable

The binned analysis shows the iron-corrected field is remarkably consistent across all orientations:

| Yaw Range | |B| Mean |
|-----------|---------|
| -180° to -150° | 1351.7 |
| -150° to -120° | 1353.5 |
| ... | ... |
| 150° to 180° | 1350.3 |

**Standard deviation across bins: ~2 LSB** (excellent stability after iron correction)

This means:
1. Hard/soft iron calibration IS working correctly
2. The earth field subtraction is CAUSING instability, not fixing it
3. The stored earth field value (466 LSB) doesn't match reality

### 4. World-Frame Earth Field Estimation

By transforming all iron-corrected readings to world frame and averaging:

```
Estimated world-frame field: [-4.7, 6.3, 19.8] LSB ≈ 0.31 μT
Expected earth field: ~50 μT
```

The near-zero world-frame estimate suggests the iron-corrected readings are dominated by a **constant offset** (likely device interference), not earth field.

## Root Cause Analysis

The ~1352 LSB constant field after iron correction suggests:

1. **Either**: Hard iron calibration offset is incomplete/wrong
2. **Or**: The device has additional permanent interference not captured by ellipsoid fitting
3. **Or**: Iron correction is creating an offset (soft iron matrix not unit determinant)

The stored calibration values:
- `hardIronOffset`: (4.5, 520.5, -482) - very large Y and Z offsets
- `softIronMatrix`: diagonal (1.029, 1.165, 0.855) - significant scaling

## Recommended Fixes

### Option A: Fix Frame Transformation (Minimal Change)

In `calibration.js:correct()`:

```javascript
// Step 3: Subtract Earth field (rotated to current orientation)
if (this.earthFieldCalibrated && orientation) {
    const rotMatrix = orientation.toRotationMatrix();
    // FIX: Use transpose to go from world to sensor frame
    const rotatedEarth = rotMatrix.transpose().multiply(this.earthField);
    corrected = {
        x: corrected.x - rotatedEarth.x,
        y: corrected.y - rotatedEarth.y,
        z: corrected.z - rotatedEarth.z
    };
}
```

### Option B: Store Earth Field in World Frame (Robust Fix)

In `runEarthFieldCalibration()`:

```javascript
runEarthFieldCalibration(samples, currentOrientation) {
    // ... existing iron correction code ...

    // NEW: Get orientation at calibration time
    const R_ref = currentOrientation.toRotationMatrix();

    // Transform to world frame
    const earthFieldSensor = { x: sumX/n, y: sumY/n, z: sumZ/n };
    this.earthField = R_ref.multiply(earthFieldSensor);  // Store in WORLD frame

    // ... rest of method ...
}
```

### Option C: Heading-Informed Recalibration Mode

Add a new calibration mode that:
1. Collects samples while rotating through multiple orientations
2. Transforms each to world frame
3. Averages to get robust world-frame earth field estimate

This is more robust than single-orientation calibration.

## Analysis Scripts

Created in `ml/`:
- `heading_analysis.py` - Initial correlation analysis
- `heading_compensation_fix.py` - Bug analysis and frame transformation
- `heading_informed_compensation.py` - Prototype of corrected algorithm

## For Existing Sessions

Sessions with zeroed calibration should use the checked-in `gambit_calibration.json`. The analysis can be rerun with:

```bash
python3 ml/heading_informed_compensation.py
```

## Critical Discovery: Spatial vs Rotational Data

### The Problem with Multi-Environment Sessions

Analysis of all sessions revealed a critical distinction:

| Session | Mag Span (LSB) | Center Drift | Type |
|---------|---------------|--------------|------|
| T13_26_33 | 218-384 | ~200 | Mixed |
| T16_16_16 | 1369-1828 | **675** | **Spatial movement** |
| T18_34_21 | 1187-2052 | ~500 | **Spatial movement** |
| T18_41_08 | 48-77 | **28** | **Pure rotation** |

**Key insight**: Session T18_41_08 has magnetometer spans of only ~50-77 LSB per axis (device rotated in place), while other sessions have spans of 1000-2000 LSB (device moved through different magnetic environments).

### Why Session-Specific Calibration Works for Pure Rotation

For session T18_41_08 (pure rotation):
- Device stayed in one consistent magnetic environment
- Min/max method gives valid hard iron estimate
- **Fused |B| = 17.31 ± 5.86 LSB** (near zero!)

For sessions with spatial movement:
- Device moved through regions with different background fields
- The "center" of the data is meaningless
- No orientation-based compensation can fix spatial variation

### Validation Results

With **session-specific** calibration + corrected R.T earth field subtraction:

| Session | Raw |B| Range | Fused |B| | Notes |
|---------|-----------------|----------|-------|
| T18_41_08 | 650-692 | **17.3 ± 5.9** | Pure rotation - WORKS! |
| T16_16_16 | 437-2278 | ~730 | Spatial movement - cannot fix |
| T18_34_21 | ~1500-2000 | ~200 | Spatial movement - cannot fix |

## Conclusion

**YES**, heading/orientation CAN and SHOULD inform the interference compensation. The current implementation has the right idea but wrong math. Fixing the rotation direction (`R.T` instead of `R`) and storing earth field in world frame will provide stable, orientation-independent readings.

The 94% reduction in coefficient of variation demonstrates that proper heading-informed compensation dramatically improves stability.

**Important caveat**: Orientation-based compensation only works when the device is rotating in a **consistent magnetic environment**. Sessions where the device moved through different magnetic environments (spatial variation) cannot be compensated using orientation alone.

### Recommendations for Data Collection

1. For calibration: Rotate device in place, don't move it spatially
2. For validation sessions: Keep device in consistent magnetic environment
3. Consider adding spatial consistency checks to detect when compensation will fail
