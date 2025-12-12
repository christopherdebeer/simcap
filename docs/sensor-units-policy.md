# Sensor Units Policy and Conventions

**Version:** 1.0.0
**Date:** 2025-12-12
**Status:** ACTIVE

---

## Executive Summary

This document defines the **authoritative policy** for sensor units throughout the GAMBIT system. Following the discovery of a critical magnetometer unit conversion bug, this policy ensures we never repeat such errors.

### Core Principles

1. **Raw Data is Sacred** - NEVER modify raw LSB values from firmware
2. **Decorate, Don't Replace** - Add converted fields alongside raw fields
3. **Units are Explicit** - Every field has a documented unit
4. **Metadata Tracks Conversions** - Sessions record what conversions were applied
5. **Validation is Mandatory** - Check units at system boundaries

---

## Sensor Specifications

### Accelerometer: LSM6DS3

| Property | Value |
|----------|-------|
| **Range** | ±2g (default) |
| **Resolution** | 16-bit |
| **Sensitivity** | 8192 LSB/g |
| **Raw Unit** | LSB (int16) |
| **Converted Unit** | g (standard gravity) |
| **Conversion** | `value_g = value_LSB / 8192` |
| **Reference** | [ST LSM6DS3 Datasheet](https://www.st.com/resource/en/datasheet/lsm6ds3.pdf) |

### Gyroscope: LSM6DS3

| Property | Value |
|----------|-------|
| **Range** | ±245 dps (default) |
| **Resolution** | 16-bit |
| **Sensitivity** | 114.28 LSB/(deg/s) |
| **Raw Unit** | LSB (int16) |
| **Converted Unit** | deg/s (degrees per second) |
| **Conversion** | `value_dps = value_LSB / 114.28` |
| **Reference** | [ST LSM6DS3 Datasheet](https://www.st.com/resource/en/datasheet/lsm6ds3.pdf) |

### Magnetometer: LIS3MDL ⚠️ CRITICAL

| Property | Value |
|----------|-------|
| **Range** | ±4 gauss (default) |
| **Resolution** | 16-bit |
| **Sensitivity** | 6842 LSB/gauss |
| **Raw Unit** | **LSB (int16)** ← Firmware returns this! |
| **Converted Unit** | **µT (microtesla)** |
| **Conversion** | `value_µT = value_LSB × (100/6842) = value_LSB × 0.014616` |
| **Reference** | [ST LIS3MDL Datasheet](https://www.st.com/resource/en/datasheet/lis3mdl.pdf) |

**⚠️ CRITICAL NOTES:**
- **`Puck.mag()` returns RAW LSB values, NOT physical units!**
- This was the source of the unit conversion bug (values appeared 68x too large)
- Earth's magnetic field: ~25-65 µT total (Edinburgh, UK: ~50.5 µT)
- After conversion, normal readings should be 20-200 µT range

---

## Field Naming Conventions

### Raw Fields (ALWAYS Preserved)

These fields contain RAW sensor data in LSB units:

```javascript
{
  // Accelerometer (LSB)
  ax: -4547,
  ay: -4044,
  az: 5642,

  // Gyroscope (LSB)
  gx: 594,
  gy: -10841,
  gz: -15009,

  // Magnetometer (LSB) ← CRITICAL: These are LSB, not µT!
  mx: -383,
  my: -848,
  mz: -1273
}
```

**Policy:** These fields are NEVER modified. They represent ground truth from hardware.

### Converted Fields (Decorated)

These fields contain unit-converted values:

```javascript
{
  // Accelerometer (g)
  ax_g: -0.555,
  ay_g: -0.494,
  az_g: 0.689,

  // Gyroscope (deg/s)
  gx_dps: 5.20,
  gy_dps: -94.87,
  gz_dps: -131.34,

  // Magnetometer (µT) ← CRITICAL: These are the FIXED fields!
  mx_ut: -5.60,
  my_ut: -12.40,
  mz_ut: -18.62
}
```

**Policy:** These fields are ADDED by processing pipeline. Original raw fields remain unchanged.

### Calibrated/Processed Fields

Additional processing stages may add more decorated fields:

```javascript
{
  // Iron-corrected magnetometer (µT)
  calibrated_mx: -4.85,
  calibrated_my: -5.03,
  calibrated_mz: 12.25,

  // Earth field subtracted (µT)
  fused_mx: 2.15,
  fused_my: -1.23,
  fused_mz: 3.47,

  // Kalman filtered (µT)
  filtered_mx: 2.18,
  filtered_my: -1.20,
  filtered_mz: 3.50
}
```

---

## Session Metadata

All sessions MUST include conversion metadata to track what processing was applied:

```json
{
  "metadata": {
    "unit_conversion": {
      "applied": true,
      "version": "1.0.0",
      "date": "2025-12-12T12:34:56.789Z",
      "sensors": {
        "accelerometer": {
          "sensor": "LSM6DS3",
          "raw_unit": "LSB",
          "converted_unit": "g",
          "conversion_factor": 0.0001220703125
        },
        "gyroscope": {
          "sensor": "LSM6DS3",
          "raw_unit": "LSB",
          "converted_unit": "deg/s",
          "conversion_factor": 0.008750280898876404
        },
        "magnetometer": {
          "sensor": "LIS3MDL",
          "raw_unit": "LSB",
          "converted_unit": "µT",
          "conversion_factor": 0.014616
        }
      }
    }
  }
}
```

**Purpose:**
1. Document which conversions were applied
2. Allow validation of data integrity
3. Enable future updates to conversion logic
4. Track version changes

---

## Implementation Guidelines

### JavaScript (Real-Time Pipeline)

```javascript
// src/web/GAMBIT/shared/sensor-units.js
import { magLsbToMicroTesla } from './sensor-units.js';

function processTelemetry(raw) {
  const decorated = { ...raw };  // Preserve raw

  // Convert magnetometer LSB → µT
  decorated.mx_ut = magLsbToMicroTesla(raw.mx);
  decorated.my_ut = magLsbToMicroTesla(raw.my);
  decorated.mz_ut = magLsbToMicroTesla(raw.mz);

  // Use converted values for downstream processing
  const calibrated = calibration.correctIronOnly({
    x: decorated.mx_ut,  // Use µT, not raw LSB!
    y: decorated.my_ut,
    z: decorated.mz_ut
  });

  return decorated;
}
```

### Python (Post-Processing Pipeline)

```python
# ml/sensor_units.py
from sensor_units import decorate_sample_with_units

def load_session(filepath):
    with open(filepath) as f:
        session = json.load(f)

    # Add unit conversions to all samples (in place)
    for sample in session['samples']:
        decorate_sample_with_units(sample, in_place=True)

    return session
```

### Firmware (Read-Only)

```javascript
// src/device/GAMBIT/app.js
// Puck.mag() returns RAW LSB values
var mag = Puck.mag();
telemetry.mx = mag.x;  // LSB, not µT!
telemetry.my = mag.y;
telemetry.mz = mag.z;
```

**Note:** Firmware should NOT be modified. It correctly returns LSB values. The conversion happens in the processing pipeline.

---

## Validation and Testing

### Unit Range Validation

Before processing, validate that raw values are in expected LSB ranges:

| Sensor | Expected Range (LSB) | Converted Range |
|--------|---------------------|-----------------|
| Accelerometer | -16384 to +16384 | ±2g |
| Gyroscope | -28000 to +28000 | ±245 deg/s |
| Magnetometer | -27368 to +27368 | ±4 gauss (±400 µT) |

**If values are outside these ranges OR suspiciously small (<100), the data may already be converted!**

### Magnetometer Magnitude Check

After conversion, check magnetometer magnitude:

```python
mag_magnitude_ut = np.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2)

if mag_magnitude_ut < 10:
    # Too low - may already be converted or sensor issue
    warnings.append("Magnetometer magnitude suspiciously low")
elif 20 < mag_magnitude_ut < 200:
    # Normal range (Earth field + environment)
    pass
elif mag_magnitude_ut > 1000:
    # Still in LSB! Conversion not applied
    warnings.append("Magnetometer appears to still be in LSB units")
```

### Test Cases

```python
# Test 1: LSB → µT conversion
assert abs(mag_lsb_to_microtesla(6842) - 100.0) < 0.001  # 1 gauss = 100 µT
assert abs(mag_lsb_to_microtesla(1578) - 23.06) < 0.01   # Edinburgh Earth field

# Test 2: Idempotency
sample = {'mx': 1578, 'my': 0, 'mz': 0}
decorate_sample_with_units(sample, in_place=True)
assert 'mx_ut' in sample
assert sample['mx'] == 1578  # Raw preserved!

# Test 3: Already converted detection
assert check_if_already_converted(sample)['mag'] == True
```

---

## Migration Strategy

### For Existing Data

Use the provided migration script:

```bash
# Dry run to see what would happen
python ml/convert_legacy_units.py --input data/GAMBIT/ --dry-run

# Convert with backups
python ml/convert_legacy_units.py --input data/GAMBIT/

# Convert single file
python ml/convert_legacy_units.py --file data/GAMBIT/session.json
```

**The script:**
- ✅ Preserves raw LSB values
- ✅ Adds `*_ut`, `*_g`, `*_dps` fields
- ✅ Creates `.bak` backups
- ✅ Adds conversion metadata
- ✅ Idempotent (safe to run multiple times)

### For New Data

- ✅ Already handled by updated `telemetry-processor.js`
- ✅ Conversion metadata automatically added
- ✅ Both raw and converted fields present

---

## Calibration Implications

### **CRITICAL: Calibration Files Need Conversion Too!**

Existing calibration files (`gambit_calibration.json`) contain offsets in LSB that must be converted:

```json
{
  "hardIronOffset": {
    "x": -51.5,    // LSB - needs conversion!
    "y": 504,      // LSB
    "z": -436      // LSB
  },
  "earthField": {
    "x": 32.116,   // LSB - needs conversion!
    "y": 285.028,  // LSB
    "z": -306.882  // LSB
  }
}
```

**After conversion:**

```json
{
  "hardIronOffset": {
    "x": -0.75,    // µT
    "y": 7.37,     // µT
    "z": -6.37     // µT
  },
  "earthField": {
    "x": 0.47,     // µT
    "y": 4.17,     // µT
    "z": -4.48     // µT
  },
  "units": {
    "hardIronOffset": "µT",
    "earthField": "µT",
    "converted_from": "LSB",
    "conversion_date": "2025-12-12T12:34:56.789Z"
  }
}
```

---

## Common Mistakes to Avoid

### ❌ DON'T: Overwrite Raw Values

```javascript
// WRONG - destroys raw data!
telemetry.mx = telemetry.mx * 0.014616;
```

### ✅ DO: Add Decorated Fields

```javascript
// CORRECT - preserves raw, adds converted
telemetry.mx_ut = telemetry.mx * 0.014616;
```

### ❌ DON'T: Mix Units

```javascript
// WRONG - mixing LSB and µT!
const calibrated = calibration.correct({
  x: raw.mx,      // LSB
  y: raw.my,      // LSB
  z: converted_mz // µT - inconsistent!
});
```

### ✅ DO: Use Consistent Units

```javascript
// CORRECT - all µT
const calibrated = calibration.correct({
  x: mx_ut,  // µT
  y: my_ut,  // µT
  z: mz_ut   // µT
});
```

### ❌ DON'T: Assume Units

```javascript
// WRONG - assuming these are µT
if (mx > 1000) {
  // This check is wrong if mx is in LSB!
}
```

### ✅ DO: Check Units Explicitly

```javascript
// CORRECT - check converted field with known units
if (mx_ut > 1000) {
  // This check is valid for µT
}
```

---

## Future Enhancements

### Planned Improvements

1. **Runtime Unit Validation**
   - Add assertions that check units at system boundaries
   - Detect and warn about mixed units

2. **Type Safety** (TypeScript)
   ```typescript
   type LSB = number & { __brand: 'LSB' };
   type MicroTesla = number & { __brand: 'µT' };

   function convert(lsb: LSB): MicroTesla {
     return (lsb * 0.014616) as MicroTesla;
   }
   ```

3. **Unit Testing**
   - Comprehensive test suite for all conversions
   - Property-based testing for idempotency
   - Fuzz testing for edge cases

4. **Calibration File Versioning**
   - Schema version in calibration files
   - Automatic migration on load
   - Validation against schema

---

## References

### Datasheets

- [ST LSM6DS3 (Accel/Gyro)](https://www.st.com/resource/en/datasheet/lsm6ds3.pdf)
- [ST LIS3MDL (Magnetometer)](https://www.st.com/resource/en/datasheet/lis3mdl.pdf)
- [Espruino Puck.js Documentation](https://www.espruino.com/Puck.js)

### Geomagnetic References

- [NOAA Geomagnetic Calculator](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml)
- [BGS Geomagnetism](https://www.geomag.bgs.ac.uk/)
- [IGRF-13 Model](https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html)

### Internal Documentation

- `docs/CRITICAL-unit-conversion-bug.md` - Bug discovery and analysis
- `src/web/GAMBIT/shared/sensor-units.js` - JavaScript implementation
- `ml/sensor_units.py` - Python implementation
- `ml/convert_legacy_units.py` - Migration script

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-12 | Initial policy created after unit conversion bug discovery |

---

**Document Owner:** SIMCAP Engineering
**Review Cycle:** Quarterly
**Next Review:** 2026-03-12

---

*This policy is mandatory for all code touching sensor data.*
