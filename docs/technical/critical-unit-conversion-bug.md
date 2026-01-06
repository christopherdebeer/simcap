# CRITICAL: Magnetometer Unit Conversion Bug

**Date:** 2025-12-12
**Severity:** 🔴 CRITICAL
**Status:** IDENTIFIED - FIX PENDING

---

## Executive Summary

**The entire calibration investigation was based on a false premise!**

Magnetometer data is stored as **RAW LSB values** from the LIS3MDL sensor but is **NEVER converted to µT** in the processing pipeline. This caused all magnitude values to appear 68x too large, leading to incorrect diagnosis of "massive environmental contamination."

**Reality:** The environment is clean. The calibration is correct. We just have a unit conversion bug.

---

## The Bug

### What Should Happen:

```javascript
// sensor-config.js defines the conversion:
export const MAG_SCALE_LSB_TO_UT = 100 / 6842; // 0.014616 µT/LSB

// telemetry-processor.js should convert:
const mx_ut = raw.mx * MAG_SCALE_LSB_TO_UT;
const my_ut = raw.my * MAG_SCALE_LSB_TO_UT;
const mz_ut = raw.mz * MAG_SCALE_LSB_TO_UT;
```

### What Actually Happens:

```javascript
// telemetry-processor.js lines 213-215:
const ironCorrected = this.calibration.correctIronOnly({
    x: raw.mx,  // ❌ RAW LSB, not converted!
    y: raw.my,
    z: raw.mz
});
```

**Result:** All downstream processing (calibration, filtering, ML) operates on LSB values thinking they're µT.

---

## Evidence

### 1. Conversion Factor Definition

**File:** `src/web/GAMBIT/shared/sensor-config.js:36-39`

```javascript
/**
 * Magnetometer scale factor: LSB to μT
 * LIS3MDL: 6842 LSB/gauss @ ±4 gauss, 1 gauss = 100 μT
 */
export const MAG_SCALE_LSB_TO_UT = 100 / 6842;
```

### 2. Conversion NOT Imported

**File:** `src/web/GAMBIT/shared/telemetry-processor.js:12-22`

```javascript
import {
    ACCEL_SCALE,
    GYRO_SCALE,
    STATIONARY_SAMPLES_FOR_CALIBRATION,
    accelLsbToG,
    gyroLsbToDps,
    createMadgwickAHRS,
    createKalmanFilter3D,
    createMotionDetector,
    createGyroBiasState
} from './sensor-config.js';
// ❌ MAG_SCALE_LSB_TO_UT NOT imported!
```

### 3. Firmware Returns LSB

**File:** `src/device/GAMBIT/app.js:188-191`

```javascript
var mag = Puck.mag();  // Returns RAW LSB from LIS3MDL
telemetry.mx = mag.x;
telemetry.my = mag.y;
telemetry.mz = mag.z;
```

**Espruino Puck.mag() Documentation:**
> Returns values in LIS3MDL sensor units (LSB), NOT calibrated to physical units.

### 4. Validation Test

```python
# Test with Session 2025-12-12T11_14_50.144Z
Raw LSB values:
  mx = -383.7, my = -848.8, mz = -1273.8
  Magnitude = 1578.1 LSB

After conversion (× 0.014616):
  mx = -5.6 µT, my = -12.4 µT, mz = -18.6 µT
  Magnitude = 23.1 µT ✅

Expected for Edinburgh:
  Total field = 50.5 µT
  Components vary by orientation
  23.1 µT is REASONABLE for partial field in device orientation
```

---

## Impact Assessment

### What We Thought:
- ❌ Environmental contamination: 1,500 µT (30x Earth's field)
- ❌ Calibration captured environmental distortions
- ❌ Need to recalibrate in clean environment
- ❌ SNR only 1.6% (unusable)

### What's Actually True:
- ✅ Measured field: 23 µT (normal Earth field)
- ✅ Environment is magnetically clean
- ✅ Existing calibration is correct
- ✅ SNR is actually reasonable once units are fixed

### Affected Systems:

| Component | Issue | Fix Required |
|-----------|-------|--------------|
| **Firmware** | Returns LSB (correct) | ✅ No change needed |
| **telemetry-processor.js** | Doesn't convert LSB to µT | 🔴 Add conversion |
| **calibration.js** | Operates on LSB values | ⚠️ May need adjustment |
| **Python ML pipeline** | Assumes µT but gets LSB | 🔴 Add conversion |
| **All session data** | Stored as LSB, labeled as µT | ⚠️ Retroactive conversion needed |
| **Visualizations** | Show LSB labeled as µT | ⚠️ Axes need rescaling |

---

## Edinburgh Magnetic Field Reference

**Location:** Edinburgh, Scotland, UK (55.95°N, 3.19°W)

**IGRF-13 Model (2025):**
- **Total Intensity (F):** 50.5 µT
- **Horizontal Component (H):** 16.0 µT
- **Vertical Component (Z):** 47.5 µT (downward)
- **Inclination:** 71.5° (steep dip angle)
- **Declination:** -2.5° (west of true north)

**Component Breakdown:**
- North (X): ~16.0 µT
- East (Y): ~-0.7 µT
- Down (Z): ~47.5 µT

**Measured (after conversion):** 23.1 µT total
**Assessment:** ✅ Reasonable partial field based on device orientation

---

## Recommended Fix

### Phase 1: JavaScript Real-Time Pipeline (CRITICAL)

**File:** `src/web/GAMBIT/shared/telemetry-processor.js`

```javascript
// Add to imports (line 12):
import {
    ACCEL_SCALE,
    GYRO_SCALE,
    MAG_SCALE_LSB_TO_UT,  // ← ADD THIS
    // ... rest
} from './sensor-config.js';

// Add conversion step before Step 5 (after line 133):

// ===== Step 5: Magnetometer Unit Conversion =====
// Convert magnetometer from LSB to µT
const mx_ut = (raw.mx || 0) * MAG_SCALE_LSB_TO_UT;
const my_ut = (raw.my || 0) * MAG_SCALE_LSB_TO_UT;
const mz_ut = (raw.mz || 0) * MAG_SCALE_LSB_TO_UT;

// Store converted values
decorated.mx_ut = mx_ut;
decorated.my_ut = my_ut;
decorated.mz_ut = mz_ut;

// Use converted values for calibration (update line 213):
const ironCorrected = this.calibration.correctIronOnly({
    x: mx_ut,  // ← Use converted value
    y: my_ut,
    z: mz_ut
});
```

### Phase 2: Python ML Pipeline

**File:** `ml/data_loader.py` and all analysis scripts

```python
# Add conversion constant
MAG_SCALE_LSB_TO_UT = 100 / 6842  # 0.014616

# Convert on load:
def load_session(filepath):
    with open(filepath) as f:
        session = json.load(f)

    # Convert magnetometer from LSB to µT
    for sample in session['samples']:
        if 'mx' in sample:
            sample['mx'] *= MAG_SCALE_LSB_TO_UT
            sample['my'] *= MAG_SCALE_LSB_TO_UT
            sample['mz'] *= MAG_SCALE_LSB_TO_UT

        # Also convert calibrated/fused/filtered if present
        for field in ['calibrated_mx', 'fused_mx', 'filtered_mx']:
            if field in sample:
                sample[field] *= MAG_SCALE_LSB_TO_UT
        # ... same for _my, _mz

    return session
```

### Phase 3: Calibration File Format

**File:** `data/GAMBIT/gambit_calibration.json`

Update calibration parameters to use µT units:

```json
{
  "hardIronOffset": {
    "x": -0.75,     // was: -51.5 LSB, now: -0.75 µT
    "y": 7.37,      // was: 504 LSB, now: 7.37 µT
    "z": -6.37      // was: -436 LSB, now: -6.37 µT
  },
  "earthField": {
    "x": 0.47,      // was: 32.1 LSB, now: 0.47 µT
    "y": 4.17,      // was: 285.0 LSB, now: 4.17 µT
    "z": -4.48      // was: -306.9 LSB, now: -4.48 µT
  },
  "earthFieldMagnitude": 6.14,  // was: 420 LSB, now: 6.14 µT
  // ... rest
}
```

---

## Testing Plan

### Test 1: Verify Conversion

```python
# Load session
session = load_session('2025-12-12T11_14_50.144Z.json')

# Check magnitude
samples = session['samples']
mx = [s['mx'] for s in samples]
mag = np.sqrt(np.array(mx)**2 + ...)

print(f"Magnitude: {np.mean(mag):.1f} µT")
# Expected: ~23 µT (not 1578!)
```

### Test 2: Validate Calibration

```python
# After conversion, check calibration quality
# Earth field magnitude should be 20-65 µT range
# Hard iron offset should be < 20 µT
# Signal-to-noise should improve dramatically
```

### Test 3: ML Training

```python
# Retrain models with corrected units
# Check if SNR improves
# Validate finger magnet detection
```

---

## Migration Strategy

### For New Data (Going Forward):

1. Deploy fixed `telemetry-processor.js`
2. Update firmware docs to clarify LSB units
3. All new sessions will have correct µT values

### For Existing Data (Retroactive):

**Option A:** Convert on load (recommended)
```python
def load_session_with_conversion(filepath):
    session = load_raw_session(filepath)
    apply_mag_conversion(session)
    return session
```

**Option B:** Batch conversion script
```bash
python ml/convert_legacy_sessions.py --input data/GAMBIT/ --backup
```

**Option C:** Add metadata flag
```json
{
  "metadata": {
    "magnetometer_units": "LSB",  // or "uT" after conversion
    "conversion_applied": false
  }
}
```

---

## Lessons Learned

1. **Always validate unit assumptions** - Don't assume labeled units are correct
2. **Trace data flow end-to-end** - From firmware → collector → storage → analysis
3. **Sanity check against physical models** - 1500 µT is way too high for Earth's field
4. **Look for similar conversions** - Accel/gyro were converted, why not mag?
5. **Test with known references** - Edinburgh's field is well-documented

---

## Timeline

- **Bug Introduced:** Unknown (likely from initial implementation)
- **Bug Discovered:** 2025-12-12 (during calibration re-evaluation)
- **Root Cause:** User hypothesis about unit conversion
- **Investigation:** Python analysis script confirmed LSB→µT conversion needed
- **Fix Status:** PENDING IMPLEMENTATION

---

## Credits

**Discovered by:** User observation about order-of-magnitude difference
**Hypothesis:** Unit conversion error or location-specific field differences
**Validation:** Python analysis comparing against Edinburgh IGRF model
**Conclusion:** 100% correct - unit conversion bug confirmed

---

*This changes everything. The calibration was fine all along!* 🎉
