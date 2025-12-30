# Magnetometer Calibration Analysis - 2025-12-30

**Session:** `2025-12-30T22_46_28.771Z`
**Device:** GAMBIT (Firmware 0.4.2)
**Magnet Config:** None (no finger magnets present)
**Expected Result:** Near-zero residual (~<10µT)
**Actual Result:** 67µT residual with min-max calibration

## Executive Summary

This analysis investigates why the magnetometer calibration auto-completes at 100% but produces a high Earth field residual (~67µT) when no finger magnets are present. Expected residual with no magnets should be near zero.

### Key Findings

1. **Z-axis interference spikes**: 10 samples (0.2%) have extreme Z values up to 2569µT, corrupting min-max calibration
2. **Excessive raw data ranges**: Even clean data shows 207µT average range (expected ~100µT)
3. **Diagonal soft iron is inadequate**: H/V ratio with min-max = 10.36 (expected 0.33) - completely wrong
4. **Orientation-aware calibration works**: Reduces residual from 38.8µT to 12.1µT (69% improvement)
5. **Full 3x3 soft iron matrix required**: Significant off-diagonal terms detected

## Detailed Analysis

### 1. Z-Axis Anomaly Investigation

The session contained brief magnetic interference events:

| Period | Samples | Duration | Max |Z| |
|--------|---------|----------|--------|
| 1 | 553-554 | 0.04s | 2569µT |
| 2 | 4188-4193 | 0.1s | 245µT |
| 3 | 4282-4283 | 0.04s | 281µT |

**Impact:** These 10 outlier samples inflate the Z-axis range from 233µT to 2814µT, causing the soft iron Z scale to be compressed to 0.036 (destroying Z information).

**Root cause:** Likely external magnetic interference (phone, cable, nearby magnet) during brief moments.

### 2. Clean Data Analysis

After filtering samples with |mz| > 150µT:

| Metric | Clean Data | Expected |
|--------|------------|----------|
| X range | 159µT | ~100µT |
| Y range | 229µT | ~100µT |
| Z range | 233µT | ~100µT |
| Raw magnitude | 114 ± 36 µT | ~50µT |

The raw magnitude being 2x expected suggests:
- Strong hard iron offset in the environment
- Or sensor gain calibration issue

### 3. Calibration Method Comparison

| Method | Magnitude | Residual | H/V Ratio |
|--------|-----------|----------|-----------|
| Min-Max (diagonal) | 49.0 ± 17.1 µT | 38.8 µT | 10.36 |
| Orientation-Aware (3x3) | 46.4 ± 8.9 µT | **12.1 µT** | 1.10 |
| Target | 50.4 µT | <10 µT | 0.33 |

### 4. Why Calibration Auto-Completes Too Fast

The min-max calibration declares 100% progress when each axis has covered sufficient range. With the session's large ranges:

| Time | X Range | Y Range | Z Range | Progress |
|------|---------|---------|---------|----------|
| 0.5s | 10µT | 9µT | 35µT | 18% |
| 1.0s | 13µT | 21µT | 53µT | 25% |
| **2.5s** | **60µT** | **52µT** | **93µT** | **100%** |
| Final | 199µT | 229µT | 2814µT | 100% |

Calibration completes in 2.5s because the ranges quickly exceed the 50µT threshold, but:
- The ranges continue growing (indicating problem data)
- The Z-axis spikes weren't filtered out
- No quality gate checks H/V ratio or sphericity

### 5. Orientation-Aware Calibration Results

The full 3x3 soft iron matrix from optimization:

```
Hard iron: [35.97, -50.67, 58.49] µT

Soft iron matrix:
  [1.4425, 0.4451, -0.1405]
  [0.3589, 0.9821, 0.2789]
  [-0.3868, 0.4126, 0.9915]
```

Note the significant off-diagonal terms - the diagonal-only assumption is fundamentally wrong for this sensor.

## Root Causes

1. **Outlier sensitivity**: Min-max calibration is destroyed by a single outlier
2. **Diagonal assumption**: The sensor has significant cross-axis coupling requiring full 3x3 matrix
3. **No quality gates**: Calibration declares "complete" without validating results
4. **H/V ratio check missing**: Would catch inverted calibrations

## Recommendations

### Immediate Fixes

1. **Outlier filtering**: Use robust statistics (median, IQR) instead of min-max
   ```typescript
   // Instead of min/max
   const offset = (percentile(data, 95) + percentile(data, 5)) / 2;
   const range = percentile(data, 95) - percentile(data, 5);
   ```

2. **Quality gate**: Don't trust calibration if H/V ratio > 0.8 (inverted)
   ```typescript
   if (hvRatio > 0.8) {
     console.warn('[MagCal] H/V ratio inverted - calibration failed');
     return false;
   }
   ```

3. **Sphericity check**: Require sphericity > 0.5 for valid calibration
   ```typescript
   const sphericity = Math.min(...ranges) / Math.max(...ranges);
   if (sphericity < 0.5) {
     console.warn('[MagCal] Poor sphericity - retry calibration');
   }
   ```

### Longer-term Improvements

1. **Full 3x3 soft iron matrix**: Already implemented in `UnifiedMagCalibration.ts` but needs more samples and better convergence

2. **Progressive quality feedback**: Show user calibration quality during collection:
   - "Rotate more around X axis"
   - "Avoid magnetic interference"

3. **Environmental baseline**: Record baseline magnetic environment before session

## Files Generated

- `ml/analyze_mag_calibration_session.py` - Main analysis script
- `ml/analyze_z_axis_anomaly.py` - Z-axis spike investigation
- `ml/analyze_clean_residual.py` - Clean data calibration comparison
- `ml/mag_calibration_diagnostic.png` - Diagnostic visualization
- `ml/z_axis_anomaly_analysis.png` - Z-axis anomaly plot
- `ml/clean_data_calibration_comparison.png` - Calibration method comparison

## Session Metadata

```json
{
  "sample_rate": 26,
  "device": "GAMBIT",
  "firmware_version": "0.4.2",
  "magnet_config": "none",
  "location": "Edinburgh",
  "geomagnetic_field": {
    "horizontal_intensity": 16.0,
    "vertical_intensity": 47.8,
    "total_intensity": 50.5
  }
}
```

## Conclusion

The high residual is caused by:
1. Brief magnetic interference spikes corrupting min-max calibration
2. Fundamental inadequacy of diagonal soft iron correction
3. Lack of quality gates to detect poor calibration

**The orientation-aware calibration reduces residual by 69%** (38.8µT → 12.1µT) by using a full 3x3 matrix. Further improvement requires:
- Better outlier rejection
- More rotation coverage during calibration
- Possible sensor alignment calibration
