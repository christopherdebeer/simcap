# Finger Magnet Detection Analysis

**Date:** 2025-12-15  
**Session:** `2025-12-15T16_31_04.482Z.json`  
**Status:** ✅ Magnets Successfully Detected

## Executive Summary

This document details the analysis of the first GAMBIT session recorded with finger magnets attached. The analysis confirms that finger magnets are **clearly detectable** using the existing magnetometer sensor, with residual magnetic fields approximately **13x higher** than baseline sessions without magnets.

## Session Details

| Property | Value |
|----------|-------|
| Filename | `2025-12-15T16_31_04.482Z.json` |
| Location | `data/GAMBIT/` |
| Samples | 804 |
| Duration | 16.1 seconds |
| Sample Rate | 50 Hz |
| Data Version | 2.1 |

## Analysis Scripts Executed

### 1. `ml/diagnose_live_calibration.py`

**Purpose:** Analyzes magnetometer readings and calibration confidence across sessions.

**Command:**
```bash
python ml/diagnose_live_calibration.py data/GAMBIT/2025-12-15T16_31_04.482Z.json
```

**Key Finding:**
- Magnetometer magnitude: **114.0 µT** (vs typical 5-17 µT in previous sessions)
- Status: ✓ GOOD (253.7% of expected Earth field)

### 2. `ml/analyze_raw_magnetic.py`

**Purpose:** Computes hard iron offsets and residual magnetic fields after Earth field subtraction.

**Command:**
```bash
python ml/analyze_raw_magnetic.py data/GAMBIT/2025-12-15T16_31_04.482Z.json
```

**Key Findings:**

| Metric | Value | Baseline (no magnets) |
|--------|-------|----------------------|
| Hard Iron Offset | [-28.86, -30.22, 67.09] µT | ~5-15 µT |
| Earth Field Estimate | 19.83 µT | 25-65 µT |
| **Residual Mean** | **66.99 µT** | **< 5 µT** |
| Residual Max | 149.45 µT | < 10 µT |
| Status | ✗ HIGH | ✓ GOOD |

**Orientation Coverage:**
- Roll: -179.8° to 179.9° (359.7° range) ✓
- Pitch: -69.1° to 85.7° (154.8° range) ✓
- Yaw: -178.1° to 179.8° (357.8° range) ✓

## Detailed Magnetometer Statistics

### Raw Magnetometer (µT)

| Axis | Mean | Std Dev | Min | Max |
|------|------|---------|-----|-----|
| X | -28.9 | 41.4 | -111.8 | 54.1 |
| Y | -23.8 | 35.4 | -104.3 | 43.8 |
| Z | 84.3 | 45.9 | -42.6 | 176.8 |
| **Magnitude** | **114.0** | **24.2** | **22.7** | **179.3** |

### Comparison Across Sessions

The `analyze_raw_magnetic.py` script analyzed 31 sessions total:

| Status | Count | Description |
|--------|-------|-------------|
| ✓ GOOD (< 5 µT) | 26 | Normal sessions without magnets |
| ⚠ MARGINAL (5-15 µT) | 4 | Slight interference or calibration issues |
| ✗ HIGH (> 15 µT) | **1** | **This session - magnets detected** |

## Detection Criteria

Based on empirical analysis, the following thresholds can be used for magnet detection:

### Primary Indicator: Residual Magnitude

```javascript
const MAGNET_DETECTION_THRESHOLDS = {
    NONE: 5,        // < 5 µT: No magnets detected
    POSSIBLE: 15,   // 5-15 µT: Possible magnet presence
    CONFIRMED: 30,  // 15-30 µT: Magnets likely present
    STRONG: 50      // > 50 µT: Strong magnet signal confirmed
};
```

### Secondary Indicators

1. **Hard Iron Offset Magnitude**
   - Normal: < 20 µT total offset
   - With magnets: > 50 µT total offset (this session: 79.4 µT)

2. **Field Variation Range**
   - Normal: < 30 µT range during motion
   - With magnets: > 100 µT range (this session: 156.6 µT)

3. **Z-axis Dominance**
   - Magnets on fingers tend to create strong Z-axis offsets
   - This session: Z offset of 67.09 µT (vs X: -28.86, Y: -30.22)

## Implementation Recommendations

### 1. Real-time Magnet Detection

Add to `telemetry-processor.js` or create new `magnet-detector.js`:

```javascript
class MagnetDetector {
    constructor() {
        this.residualHistory = [];
        this.windowSize = 50; // 1 second at 50Hz
        this.thresholds = {
            none: 5,
            possible: 15,
            confirmed: 30,
            strong: 50
        };
    }

    /**
     * Detect magnet presence from residual magnitude
     * @param {number} residualMagnitude - Current residual in µT
     * @returns {Object} Detection result
     */
    detect(residualMagnitude) {
        this.residualHistory.push(residualMagnitude);
        if (this.residualHistory.length > this.windowSize) {
            this.residualHistory.shift();
        }

        const avgResidual = this.residualHistory.reduce((a, b) => a + b, 0) 
                           / this.residualHistory.length;

        let status, confidence;
        if (avgResidual < this.thresholds.none) {
            status = 'none';
            confidence = 0;
        } else if (avgResidual < this.thresholds.possible) {
            status = 'possible';
            confidence = (avgResidual - this.thresholds.none) / 
                        (this.thresholds.possible - this.thresholds.none);
        } else if (avgResidual < this.thresholds.confirmed) {
            status = 'likely';
            confidence = 0.5 + 0.3 * (avgResidual - this.thresholds.possible) / 
                        (this.thresholds.confirmed - this.thresholds.possible);
        } else {
            status = 'confirmed';
            confidence = Math.min(1.0, 0.8 + 0.2 * (avgResidual - this.thresholds.confirmed) / 
                        (this.thresholds.strong - this.thresholds.confirmed));
        }

        return {
            status,
            confidence,
            avgResidual,
            currentResidual: residualMagnitude
        };
    }
}
```

### 2. UI Indicator

Add to GAMBIT index.html status panel:

```html
<div id="magnet-status" class="status-indicator">
    <span class="label">Magnets:</span>
    <span id="magnet-status-value" class="value">--</span>
    <span id="magnet-confidence" class="confidence"></span>
</div>
```

```css
.magnet-status-none { color: #888; }
.magnet-status-possible { color: #f0ad4e; }
.magnet-status-likely { color: #5bc0de; }
.magnet-status-confirmed { color: #5cb85c; font-weight: bold; }
```

### 3. Session Metadata

When saving sessions, include magnet detection status:

```javascript
const sessionMetadata = {
    // ... existing fields
    magnetDetection: {
        detected: true,
        confidence: 0.95,
        avgResidual: 66.99,
        maxResidual: 149.45,
        detectionMethod: 'residual_magnitude_v1'
    }
};
```

## Limitations & Future Work

### Current Limitations

1. **Position Inference Not Yet Possible**
   - We can detect magnet presence but not individual finger positions
   - Would require magnetic dipole modeling and multi-point sensing

2. **Calibration Interference**
   - Magnets interfere with standard magnetometer calibration
   - May need separate calibration mode for magnet sessions

3. **Environmental Sensitivity**
   - Strong external magnetic fields could cause false positives
   - Consider adding environmental baseline detection

### Future Enhancements

1. **Finger Position Estimation**
   - Train ML model on labeled finger position data
   - Use magnetic field vector direction, not just magnitude

2. **Multi-Magnet Discrimination**
   - Different magnets on different fingers
   - Polarity patterns (N/S alternating)

3. **Dynamic Calibration**
   - Adapt calibration to account for known magnet positions
   - Separate "magnet mode" calibration routine

## Appendix: Raw Analysis Output

### diagnose_live_calibration.py Output (excerpt)

```
2025-12-15T16_31_04.482Z  |     1167.2 |  114.0 |         253.7% | ✓ GOOD
```

### analyze_raw_magnetic.py Output

```
======================================================================
Session: 2025-12-15T16_31_04.482Z.json
Samples: 804
======================================================================

1. HARD IRON ESTIMATE (from session data):
   Offset: [-28.86, -30.22, 67.09] µT

2. EARTH FIELD ESTIMATE (world frame):
   Vector: [-4.76, -8.58, 17.23] µT
   Magnitude: 19.83 µT

3. RESIDUAL ANALYSIS (after Earth field subtraction):
   Mean:   66.99 µT
   Std:    22.97 µT
   Median: 64.93 µT
   Min:    18.04 µT
   Max:    149.45 µT

4. ORIENTATION COVERAGE:
   Roll:  [-179.8° to 179.9°], range=359.7°
   Pitch: [-69.1° to 85.7°], range=154.8°
   Yaw:   [-178.1° to 179.8°], range=357.8°

5. STATUS: ✗ HIGH
   Expected (no magnets): < 5 µT
   Actual mean: 66.99 µT
```

## References

- Session data: `data/GAMBIT/2025-12-15T16_31_04.482Z.json`
- Analysis scripts: `ml/diagnose_live_calibration.py`, `ml/analyze_raw_magnetic.py`
- Related docs: `docs/magnetometer-calibration-investigation.md`
