# Magnetometer Calibration Investigation - Updated Findings

**Session:** `2025-12-30T22_46_28.771Z`
**Updated:** Based on codebase analysis

## Summary of Implemented Features

### What's Already Working ✅

| Feature | Status | Location |
|---------|--------|----------|
| Outlier filtering | ✅ Working (150µT per axis) | `unified-mag-calibration.ts:1241-1262` |
| Full 3x3 soft iron matrix | ✅ Implemented | `unified-mag-calibration.ts:274-276` |
| Orientation-aware calibration | ✅ Implemented but buggy | `unified-mag-calibration.ts:972-1216` |
| Progressive trust scaling | ✅ Working | Lines 456-497 |
| Bootstrap from previous session | ✅ Working | Lines 317-327 |

### Key Issues Found ❌

## Issue 1: Orientation-Aware Calibration Epsilon Bug

**Location:** `unified-mag-calibration.ts:1071`

```typescript
const epsilon = 0.5;  // For numerical gradient - SAME FOR ALL PARAMETERS!
```

**Problem:**
- For hard iron offset (in µT), epsilon=0.5 is reasonable
- For soft iron matrix elements (dimensionless, ~1.0), epsilon=0.5 is a **50% perturbation**
- This produces extremely noisy gradients that don't converge

**Evidence from session logs:**
```
Soft iron matrix:
  [0.5000, -0.4386, -0.5000]  ← Hit clamp limits
  [-0.1315, 0.7445, -0.2248]
  [-0.5000, 0.5000, 0.5000]   ← Hit clamp limits
Corrected magnitude: 16.5 µT (expected 50.4 µT, error: 67.2%)
```

**Fix:**
```typescript
const epsilonOffset = 0.5;   // For hard iron (µT)
const epsilonMatrix = 0.01;  // For soft iron matrix (dimensionless)
```

## Issue 2: H/V Ratio Not Used as Quality Gate

**Current behavior:**
- H/V ratio is computed and logged
- But calibration is NOT blocked when H/V is inverted

**Session data:**
- H/V ratio = 10.81 (expected 0.33)
- Calibration still marked as "complete"

**Recommended fix:**
```typescript
if (hRatio > 0.8) {
  this._log('[UnifiedMagCal] ⚠️ Calibration failed: H/V ratio inverted');
  this._orientationAwareCalReady = false;
  return;  // Don't use bad calibration
}
```

## Issue 3: Sphericity Computed But Not Gated

**Location:** `unified-mag-calibration.ts:1310-1313`

```typescript
const sphericity = minRange / maxRange;
// Sphericity is logged but never checked!
```

**Session data:**
- Sphericity = 0.68 (borderline acceptable)

**Recommended fix:**
```typescript
if (sphericity < 0.5) {
  this._log('[UnifiedMagCal] ⚠️ Poor sphericity - calibration unreliable');
  // Could still use but with reduced trust
}
```

## Root Cause Analysis

### Why Magnitude is Correct but Direction is Wrong

1. **Raw magnitude:** 114 µT (2.26x expected)
2. **After diagonal soft iron:** 49 µT (correct!)
3. **But H/V ratio:** 10.81 (should be 0.33)

The diagonal soft iron correction:
- Scales each axis independently
- Normalizes overall magnitude correctly
- **Cannot correct cross-axis coupling**

The sensor has significant cross-axis coupling that requires a full 3x3 matrix with off-diagonal terms.

### Why Orientation-Aware Calibration Failed

1. Bad epsilon (0.5) for matrix elements
2. Only 50 iterations
3. Optimization diverges → hits clamp limits
4. Best found result is still bad (67% magnitude error)

## Verification: Outlier Filtering Works

```
Total samples: 4839
Rejected by firmware: 10 (0.2%)
- Sample 553-554: Z=2557-2569µT (rejected)
- Sample 4188-4193: Z=-245 to -229µT (rejected)

Firmware ranges (metadata): [159, 242, 246]µT
Python filtered ranges:     [159, 229, 233]µT
→ Match confirms filtering is active
```

## Recommended Code Changes

### 1. Fix epsilon for matrix gradients (HIGH PRIORITY)

```typescript
// unified-mag-calibration.ts line ~1071
const epsilonOffset = 0.5;   // µT for hard iron
const epsilonMatrix = 0.01;  // dimensionless for soft iron

// Line ~1086
gradOffset[axis] = (computeResidual(offsetPlus, S) - computeResidual(offsetMinus, S)) / (2 * epsilonOffset);

// Line ~1097
gradS[i][j] = (computeResidual(offset, Splus) - computeResidual(offset, Sminus)) / (2 * epsilonMatrix);
```

### 2. Add H/V ratio quality gate (HIGH PRIORITY)

```typescript
// After orientation-aware calibration completes
const hRatio = hComponent / Math.abs(vComponent);
const expectedRatio = this._geomagneticRef.horizontal / this._geomagneticRef.vertical;

if (hRatio > 0.8 || hRatio < 0.1) {
  this._log(`[UnifiedMagCal] ⚠️ H/V ratio ${hRatio.toFixed(2)} invalid (expected ${expectedRatio.toFixed(2)})`);
  this._orientationAwareCalReady = false;
  // Fall back to diagonal soft iron
}
```

### 3. Increase optimization iterations

```typescript
const maxIterations = 200;  // Was 50
```

### 4. Replace Gradient Descent with Levenberg-Marquardt ✅ IMPLEMENTED

**Validation across 22 sessions showed:**
- Epsilon fix alone: Mean residual 19.2 → 19.3 µT (no improvement)
- Scipy least_squares: Mean residual **3.5 µT** (5-6x better!)

**Implementation (2025-12-31):**
- Full LM algorithm implemented in `unified-mag-calibration.ts:1020-1262`
- Uses numerical Jacobian with separate epsilon for offset (0.5µT) vs matrix (0.01)
- Adaptive damping: λ starts at 0.01, increases 10x on failure, decreases 0.1x on success
- Gauss-Jordan elimination with partial pivoting for 12x12 linear system
- Soft iron matrix constraints: diagonal [0.5, 2.0], off-diagonal [-0.5, 0.5]
- Max 100 iterations with convergence threshold 1e-4

## Validation Results

Tested proposed fixes on 22 sessions:

| Metric | Gradient Descent | Epsilon Fix | Scipy LM | **TypeScript LM** |
|--------|------------------|-------------|----------|-------------------|
| Mean Residual | 19.2 µT | 19.3 µT | 3.5 µT | **~3-5 µT** |
| Convergence | Often stuck | Often stuck | Reliable | Reliable |

**Status:** LM algorithm now implemented in client-side TypeScript code.

## Files to Modify

1. `apps/gambit/shared/unified-mag-calibration.ts`
   - Line 1071: Separate epsilon for offset vs matrix
   - Line 1086, 1097: Use appropriate epsilon
   - After line 1153: Add H/V ratio validation
   - Line 1072: Increase maxIterations

## Test Plan

1. Run with fixed epsilon and verify:
   - Soft iron matrix doesn't hit clamp limits
   - Corrected magnitude ~50µT (not 16.5)
   - H/V ratio improves toward 0.33

2. Test H/V ratio gate:
   - Verify bad calibrations are rejected
   - Verify system falls back to diagonal soft iron

## Bootstrap Calibration Analysis

**Analysis date:** 2025-12-31
**Sessions analyzed:** 22

### Current vs Recommended Bootstrap Values

| Parameter | Current (2025-12-29) | Recommended (Median) | Difference |
|-----------|---------------------|---------------------|------------|
| Hard Iron X | -33.0 µT | 29.3 µT | 62.3 µT |
| Hard Iron Y | -69.1 µT | -9.9 µT | 59.2 µT |
| Hard Iron Z | -50.8 µT | -20.1 µT | 30.7 µT |
| **Total Offset Error** | - | - | **91.3 µT** |

### Offset Statistics Across Sessions

| Axis | Mean | Std Dev | Median |
|------|------|---------|--------|
| X | 15.6 µT | 24.7 µT | 29.3 µT |
| Y | -8.9 µT | 27.2 µT | -9.9 µT |
| Z | -5.9 µT | 48.3 µT | -20.1 µT |

### Range Statistics

| Axis | Mean Range | Std Dev | Median Range |
|------|------------|---------|--------------|
| X | 92.4 µT | 23.3 µT | 84.5 µT |
| Y | 110.8 µT | 35.8 µT | 99.0 µT |
| Z | 152.1 µT | 42.2 µT | 143.9 µT |

### Validation Results

| Metric | Optimal Bootstrap | No Bootstrap |
|--------|-------------------|--------------|
| Mean Residual | 3.4 µT | 3.5 µT |
| Mean Offset Error | 50.8 µT | - |

**Finding:** Bootstrap values have minimal impact on final calibration quality when using scipy's Levenberg-Marquardt optimizer. However, updated bootstrap values would:
1. Reduce offset error from 91.3 µT to ~50.8 µT
2. Provide better starting point for real-time gradient descent
3. Enable faster convergence in firmware

### Recommended TypeScript Code Update

```typescript
// Updated bootstrap values from offline analysis of 22 sessions
// Location: unified-mag-calibration.ts lines 317-327
if (this._autoHardIronEnabled) {
    this._autoHardIronEstimate = { x: 29.3, y: -9.9, z: -20.1 };
    this._autoHardIronMin = { x: 29.3 - 42.3, y: -9.9 - 49.5, z: -20.1 - 72.0 };
    this._autoHardIronMax = { x: 29.3 + 42.3, y: -9.9 + 49.5, z: -20.1 + 72.0 };
}
```

### Analysis Files

- `ml/analyze_bootstrap_impact.py` - Bootstrap comparison script
- `ml/bootstrap_analysis_results.json` - Full results data
- `ml/bootstrap_analysis_plot.png` - Visualization

## Soft Iron Bootstrap Analysis

**Analysis date:** 2025-12-31
**Sessions analyzed:** 22

### How Soft Iron Scales Are Computed

The firmware computes soft iron scale factors as:
```
scale = expectedRange / actualRange
expectedRange = 2 * expectedMagnitude = 2 * 50.4 = 100.8 µT
```

### Session Range Statistics

| Axis | Mean Range | Std Dev | Median Range |
|------|------------|---------|--------------|
| X | 92.4 µT | 23.3 µT | 84.5 µT |
| Y | 110.8 µT | 35.8 µT | 99.0 µT |
| Z | 152.1 µT | 42.2 µT | 143.9 µT |

### Optimal Soft Iron Scale Factors

| Axis | Current | Optimal | Calculation |
|------|---------|---------|-------------|
| X | 1.0 | 1.193 | 100.8 / 84.5 |
| Y | 1.0 | 1.018 | 100.8 / 99.0 |
| Z | 1.0 | 0.700 | 100.8 / 143.9 |

**Key insight:** Z-axis has significantly larger range (144 µT vs expected 100 µT), requiring 0.7x scale to normalize.

### Applied TypeScript Code

```typescript
// Location: unified-mag-calibration.ts line 271-273
private _autoSoftIronScale: Vector3 = { x: 1.193, y: 1.018, z: 0.700 };
```

### Analysis Files

- `ml/analyze_soft_iron_bootstrap.py` - Soft iron analysis script
- `ml/soft_iron_analysis_results.json` - Full results data
- `ml/soft_iron_analysis_plot.png` - Visualization

## Issue 4: Bootstrap Bounds Cause Instant 100% Progress

**Analysis date:** 2025-12-31
**Session analyzed:** `2025-12-31T12_34_54.116Z`

### Problem Description

The UI showed 100% calibration progress immediately on session start, before any device rotation. This was reported in session with finger magnet proximity testing.

### Root Cause Analysis

**Bootstrap bounds exceed the target range threshold:**

| Axis | Bootstrap Min | Bootstrap Max | Implied Range | Target Range | Progress |
|------|---------------|---------------|---------------|--------------|----------|
| X | -13.0 µT | 71.6 µT | 84.6 µT | 80.0 µT | 100% |
| Y | -59.4 µT | 39.6 µT | 99.0 µT | 80.0 µT | 100% |
| Z | -92.1 µT | 51.9 µT | 144.0 µT | 80.0 µT | 100% |

The calibration progress algorithm computes:
```typescript
axisProgress = currentRange / targetRange  // Where targetRange = 80µT
overallProgress = min(axisProgress)
```

When bootstrap bounds are applied, the "virtual" range from historical min/max already exceeds 80µT on all axes. The first sample immediately triggers 100% progress.

### Evidence from Session Logs

```
12:24:25 - Cal progress: 100% (sample 0!)
           ranges: [109, 99, 174] µT (bootstrap + first samples)
           offset: [41.3, -9.9, -5.3] µT (close to bootstrap)

12:24:28 - Calibration COMPLETE
           error: 23.7%, H/V ratio: 0.67 (wrong)

12:24:28 - MINIMAL TRUST: residual 125µT > 60µT threshold
           Quality gate correctly caught the problem

12:24:33 - LM calibration triggered
           RMS residual: 0.32 µT (fixed!)
           error: 0.2%, H/V ratio: 0.45 ✓
```

### Simulation Results

| Scenario | First 100% at | Notes |
|----------|---------------|-------|
| With bootstrap | Sample 0 | Instant completion |
| Without bootstrap | Sample 202 | After actual rotation |

### Quality Gates Working Correctly

Despite the premature 100% progress, the quality gates correctly detected the poor calibration:
- `MINIMAL TRUST: residual 125µT > 60µT` triggered
- LM optimization scheduled and achieved 0.32µT residual
- Final calibration quality was good

### Recommended Fixes

**Option A: Separate tracking for progress vs bootstrap (Recommended)**
```typescript
// Track actual measured min/max separately from bootstrap hints
private _measuredMin: Vector3;  // From actual samples only
private _measuredMax: Vector3;  // From actual samples only

// Progress uses measured range, not bootstrap range
const measuredRange = subtract(this._measuredMax, this._measuredMin);
const progress = Math.min(...Object.values(measuredRange).map(r => r / 80));
```

**Option B: Require minimum samples**
```typescript
// Don't mark complete until sufficient samples collected
if (this._sampleCount < 200) {
  return Math.min(progress, 0.99);  // Cap at 99% until enough samples
}
```

**Option C: Increase target range when using bootstrap**
```typescript
// Bootstrap provides starting point, but require actual expansion
const targetRange = this._useBootstrap ? 120 : 80;  // Higher threshold with bootstrap
```

### Analysis Files

- `ml/analyze_calibration_stages.py` - Calibration stages analysis
- `ml/analyze_calibration_timeline.py` - Timeline correlation analysis
- `ml/calibration_stages_analysis.png` - Stages visualization
- `ml/calibration_timeline_analysis.png` - Timeline visualization
- `ml/calibration_stages_results.json` - Numerical results
