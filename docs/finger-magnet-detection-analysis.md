# Finger Magnet Detection Analysis

## Overview

This document describes the finger magnet detection system implemented in GAMBIT. The system detects the presence of finger magnets by analyzing magnetometer residual magnitude and comparing it against an established baseline.

## Background

Finger magnets are small neodymium magnets implanted in or attached to fingertips. When present near the magnetometer sensor, they create additional magnetic field that can be detected as an increase in the residual magnitude after Earth field subtraction.

### Key Insight

The detection uses **baseline comparison** rather than absolute thresholds because:
1. The incremental calibration residuals vary significantly based on calibration quality
2. Even without magnets, residuals can be 50-100 µT during calibration build-up
3. Magnets cause a **relative increase** of 30-100 µT above whatever the baseline is

## Detection Algorithm

### Phase 1: Calibration (Samples 0-200)

The incremental calibration system builds:
- **Hard iron offset**: Center of min/max magnetometer bounds
- **Earth field estimate**: Average magnetic field in world frame

```
[IncrementalCal] Hard iron computed at sample 200:
  Offset: [-39.1, -10.7, 81.3] µT
  Ranges: X=134.2, Y=111.1, Z=191.5 µT
  Sphericity: 58% (moderate)
  Octant coverage: 8/8

[IncrementalCal] Earth field computed at sample 200:
  Earth field (world): [16.0, -45.5, 61.6] µT
  Earth field magnitude: 78.2 µT
```

### Phase 2: Baseline Establishment (Samples 200-300)

Once Earth field is computed, the MagnetDetector starts receiving residual magnitudes and establishes a baseline from the first 100 samples:

```
[MagnetDetector] Baseline established: 86.3 µT (range: 21.2-142.8 µT)
```

The baseline represents the "normal" residual magnitude without magnets present.

### Phase 3: Detection (Samples 300+)

The detector compares current residual magnitude against the baseline:

```
Deviation = Average Residual - Baseline
```

Detection thresholds (deviation from baseline):

| Deviation | Status | Confidence | Icon |
|-----------|--------|------------|------|
| < 10 µT | none | 0% | ○ |
| 10-20 µT | possible | 0-30% | ◐ |
| 20-35 µT | likely | 30-70% | ◑ |
| > 50 µT | confirmed | 90%+ | 🧲 |

## Implementation

### MagnetDetector Class

Located in `src/web/GAMBIT/shared/magnet-detector.js`:

```javascript
import { MagnetDetector, createMagnetDetector, MagnetStatus } from './magnet-detector.js';

const detector = createMagnetDetector({
    windowSize: 50,        // Sliding window for averaging
    baselineSamples: 100,  // Samples to establish baseline
    onStatusChange: (newStatus, oldStatus, state) => {
        console.log(`Magnet status changed: ${oldStatus} → ${newStatus}`);
    }
});

// Feed residual magnitudes
const state = detector.update(residualMagnitude);
console.log(state.status);           // 'none', 'possible', 'likely', 'confirmed'
console.log(state.confidence);       // 0.0 - 1.0
console.log(state.baselineResidual); // Established baseline in µT
console.log(state.deviationFromBaseline); // Current deviation in µT
```

### Integration with TelemetryProcessor

The TelemetryProcessor automatically feeds the MagnetDetector with incremental calibration residuals:

```javascript
// In telemetry-processor.js
if (earthMag > 0) {
    const incResidual = this.incrementalCalibration.computeResidual(mag, orientation);
    if (incResidual) {
        const magnetState = this.magnetDetector.update(incResidual.magnitude);
        decorated.magnet_status = magnetState.status;
        decorated.magnet_confidence = magnetState.confidence;
        decorated.magnet_detected = magnetState.detected;
    }
}
```

### Telemetry Fields

The following fields are added to decorated telemetry:

| Field | Type | Description |
|-------|------|-------------|
| `magnet_status` | string | 'none', 'possible', 'likely', 'confirmed' |
| `magnet_confidence` | number | 0.0 - 1.0 confidence score |
| `magnet_detected` | boolean | true if status != 'none' |
| `magnet_baseline_established` | boolean | true after baseline phase |
| `magnet_baseline_residual` | number | Baseline residual in µT |
| `magnet_deviation` | number | Current deviation from baseline in µT |

## Test Results

### Session: 2025-12-15T16:31:04.482Z (with magnets)

```
[MagnetDetector] Baseline established: 86.3 µT (range: 21.2-142.8 µT)
... (magnets added around sample 500) ...
[TelemetryProcessor] 🧲 Finger magnets detected (incremental cal)!
  Status: likely
  Confidence: 36%
  Avg Residual: 108.7 µT
  Baseline: 86.3 µT
  Deviation: 22.4 µT
```

### Timeline

| Sample | Event |
|--------|-------|
| 0-200 | Incremental calibration builds hard iron + Earth field |
| 200 | Earth field computed, MagnetDetector starts receiving data |
| 300 | Baseline established (86.3 µT) |
| ~500 | Magnets physically added |
| ~710 | Detection triggered (22.4 µT deviation) |

## UI Integration

The GAMBIT index.html includes a magnet status indicator:

```html
<div id="magnet-status" class="status-indicator">
    <span class="icon">○</span>
    <span class="label">Magnets</span>
    <span class="value">--</span>
</div>
```

Updated via JavaScript:

```javascript
const magnetState = telemetryProcessor.getMagnetState();
const icon = magnetDetector.getStatusIcon();  // ○, ◐, ◑, or 🧲
const label = magnetDetector.getStatusLabel(); // 'No Magnets', 'Possible', etc.
const color = magnetDetector.getStatusColor(); // CSS color
```

## Limitations

1. **Detection Delay**: ~200 samples after magnets are added due to:
   - Sliding window (50 samples) needs to fill with affected data
   - Hysteresis (10 samples) prevents rapid status changes

2. **Baseline Sensitivity**: If magnets are present during baseline establishment, they become part of the baseline and won't be detected as anomalous.

3. **Calibration Dependency**: Detection only works after incremental calibration computes Earth field (~200 samples).

4. **Position Sensitivity**: Magnet detection is presence-based, not position-based. Future work could use residual vector direction to infer magnet positions.

## Future Enhancements

1. **Position Inference**: Use residual vector direction to estimate which finger has the magnet
2. **Multi-Magnet Detection**: Detect multiple magnets and their relative positions
3. **Gesture Integration**: Use magnet presence to enhance gesture recognition
4. **Adaptive Baseline**: Slowly adapt baseline over time to handle environmental changes

## References

- `src/web/GAMBIT/shared/magnet-detector.js` - MagnetDetector implementation
- `src/web/GAMBIT/shared/telemetry-processor.js` - Integration with telemetry pipeline
- `src/web/GAMBIT/shared/incremental-calibration.js` - Residual computation
- `ml/diagnose_live_calibration.py` - Python analysis script for validation
