# Magnet Attachment Guide for Magnetic Finger Tracking

## Overview

This guide provides step-by-step instructions for attaching finger magnets to enable magnetic finger tracking with the GAMBIT system. Proper magnet configuration is **critical** for accurate tracking.

---

## Required Materials

### Magnets

**Recommended (Best Performance)**:
- **Size**: 6mm diameter × 3mm thickness
- **Grade**: N48 or N52 neodymium
- **Shape**: Disc (axial magnetization)
- **Quantity**: 5 (one per finger)
- **Expected SNR**: 30-40:1 at extended position (80mm)

**Minimum Viable (Budget Option)**:
- **Size**: 5mm diameter × 2mm thickness
- **Grade**: N42 neodymium
- **Expected SNR**: 14-20:1 at extended position (80mm)

**Purchase Links**:
- Amazon: Search "6mm x 3mm neodymium magnets N48"
- K&J Magnetics: Part #D62-N52
- SuperMagnetMan: 6mm disc magnets

### Attachment Hardware

**Option 1: Ring Mounting (Recommended)**:
- 5× adjustable finger rings (silicone or metal)
- Cyanoacrylate (super glue) or epoxy
- Small magnet holders (3D printed or purchased)

**Option 2: Adhesive Mounting (Prototyping)**:
- Medical-grade double-sided tape
- Skin-safe adhesive (e.g., Pros-Aide)
- Alcohol wipes for skin preparation

### Tools

- Small compass (for polarity testing)
- Permanent marker (for polarity marking)
- Ruler or caliper (for distance measurement)

---

## Safety Warnings

⚠️ **IMPORTANT SAFETY INFORMATION**:

1. **Pinch Hazard**: Neodymium magnets are extremely strong. Keep fingers clear when magnets attract.
2. **Swallowing Hazard**: If magnets are swallowed, they can cause serious internal injury. Keep away from children.
3. **Interference**: Keep magnets away from:
   - Pacemakers and medical devices (>30cm)
   - Credit cards and magnetic strips
   - Hard drives and electronic storage
   - Watches and compasses
4. **Skin Irritation**: Remove immediately if skin irritation occurs.

---

## Polarity Configuration

### Why Polarity Matters

Magnetic fields from multiple magnets **superpose linearly**. With all magnets having the same polarity, individual finger movements become difficult to distinguish. **Alternating polarity** creates unique vector signatures.

### Recommended Configuration

| Finger | Polarity | Orientation | Rationale |
|--------|----------|-------------|-----------|
| **Thumb** | **North → Palm** | ➕ Positive | Primary finger, strong signal |
| **Index** | **North → Palm** | ➕ Positive | Correlates with thumb |
| **Middle** | **North ← Away** | ➖ Negative | **Breaks symmetry** |
| **Ring** | **North → Palm** | ➕ Positive | Correlates with pinky |
| **Pinky** | **North ← Away** | ➖ Negative | **Breaks symmetry** |

**Pattern**: +, +, -, +, - (alternating with bias toward palm)

### Testing Polarity

**Method 1: Compass**
1. Hold magnet near compass
2. If compass North pole is attracted → South pole of magnet
3. If compass North pole is repelled → North pole of magnet
4. Mark North pole with permanent marker

**Method 2: Sensor Reading**
1. Open `collector.html`
2. Connect device
3. Hold magnet 5cm from sensor with marked side facing palm
4. Observe `mz` value:
   - If positive → North pole toward sensor (correct for + fingers)
   - If negative → South pole toward sensor (flip or use for - fingers)

---

## Attachment Procedure

### Preparation (5 minutes)

1. **Verify Calibration**
   - Open `collector.html`
   - Run calibration wizard (see main guide)
   - Ensure calibration quality is "Good" or "Excellent"

2. **Test Polarity**
   - Use compass or sensor to identify North pole of each magnet
   - Mark with permanent marker or nail polish
   - Label magnets: T+, I+, M-, R+, P-

3. **Clean Mounting Surface**
   - If using rings: clean with alcohol
   - If using skin: wipe with alcohol pad and dry completely

### Ring Mounting (Permanent - Recommended)

**Per Finger** (repeat 5 times):

1. **Position Ring**
   - Place ring on finger
   - Adjust so magnet will sit on **dorsal side** (back of hand)
   - Magnet should be **centered over middle phalanx** (middle bone segment)
   - Distance from palm sensor: 50-100mm when flexed/extended

2. **Attach Magnet**
   - Apply thin layer of epoxy or super glue to ring surface
   - Place magnet with **correct polarity orientation**:
     - Thumb, Index, Ring: North → palm (marked side away from skin)
     - Middle, Pinky: North ← away (marked side toward skin)
   - Hold for 30 seconds
   - Let cure per adhesive instructions (typically 5-10 minutes)

3. **Verify Attachment**
   - Gently tug magnet to ensure secure bond
   - Flex/extend finger to check for interference with movement
   - Measure distance from palm sensor: should be 50-100mm range

4. **Test Signal**
   - With ring on finger, open `collector.html`
   - Hold hand in reference pose (palm down, fingers extended)
   - Check magnetometer readings:
     - Should see significant change (>20 μT) when finger flexes
     - Polarity should match expected orientation

### Adhesive Mounting (Temporary - Prototyping)

**Per Finger**:

1. **Prepare Skin**
   - Wipe back of middle phalanx with alcohol pad
   - Dry completely (30 seconds)

2. **Apply Adhesive**
   - Cut small piece of medical tape (1cm × 1cm)
   - Apply thin layer of skin-safe adhesive if using
   - Place magnet with correct polarity on tape

3. **Attach to Finger**
   - Press firmly onto dorsal side of finger
   - Hold for 10 seconds
   - Smooth edges to prevent peeling

4. **Verify**
   - Flex/extend finger to check adhesion
   - Should stay secure through full range of motion

---

## Validation Procedure

### Phase 0: SNR Test (Critical)

Before collecting full dataset, **validate signal quality**:

1. **Collect SNR Test Data**
   ```
   In collector.html:
   - Add custom label: "snr_test"
   - For each finger with magnet:
     1. Set finger state to "extended", motion to "static"
     2. Hold for 5 seconds
     3. Set finger state to "flexed", motion to "static"
     4. Hold for 5 seconds
     5. Repeat 10 times
   - Export session
   ```

2. **Run SNR Analysis**
   ```bash
   python -m ml.analyze_snr --data-dir data/GAMBIT --finger index --plot
   ```

3. **Interpret Results**
   - **SNR (extended) > 10:1**: ✅ Adequate → Proceed
   - **SNR (extended) < 10:1**: ❌ Poor → Troubleshoot

### Troubleshooting Low SNR

| Symptom | Cause | Solution |
|---------|-------|----------|
| SNR < 10:1 at extended | Magnet too small | Use 6mm × 3mm instead of 5mm × 2mm |
| High noise floor (>2 μT) | Poor calibration | Re-run calibration wizard |
| No signal change | Wrong polarity | Flip magnet orientation |
| Inconsistent readings | Loose attachment | Re-attach with stronger adhesive |
| High signal but low SNR | Environmental interference | Move away from metal objects |

---

## Maintenance

### Daily Use

- **Before Each Session**:
  - Run calibration wizard
  - Check magnet attachment security
  - Test range of motion

- **After Each Session**:
  - Clean magnets with dry cloth
  - Inspect for loosening
  - Store rings in non-magnetic case

### Long-Term Care

- **Weekly**:
  - Re-test SNR to check for degradation
  - Inspect adhesive for wear
  - Clean sensor housing

- **Monthly**:
  - Replace adhesive if using temporary mounting
  - Check magnet polarity (can degrade over time)
  - Recalibrate if SNR drops below threshold

### Replacement

**Replace magnets if**:
- SNR drops by >30% from baseline
- Visible corrosion or damage
- Polarity test shows weakened field
- Physical damage (chipping, cracking)

**Typical Lifespan**:
- Ring-mounted: 6-12 months
- Adhesive-mounted: 1-2 weeks per application

---

## Advanced: Multi-Environment Calibration

For best results across different locations:

1. **Home/Office Environment**
   - Run calibration wizard
   - Save as `gambit_cal_home.json`
   - Load before home sessions

2. **Lab Environment**
   - Run calibration wizard
   - Save as `gambit_cal_lab.json`
   - Load before lab sessions

3. **Outdoor Environment**
   - Run calibration wizard
   - Save as `gambit_cal_outdoor.json`
   - Minimal interference, best SNR

**Switching Environments**:
```javascript
// In collector.html
$('loadCalibration').click() → Select appropriate calibration file
```

---

## Appendix: Physics Reference

### Magnetic Dipole Field Strength

At distance `r` from a magnetic dipole with moment `m`:

```
B = (μ₀ / 4π) × (m / r³)
```

**For 6mm × 3mm N48 magnet**:
- Magnetic moment: ~0.01 A·m²
- Field at 50mm: **141 μT** (flexed)
- Field at 80mm: **34 μT** (extended)
- Field at 100mm: **18 μT** (full extension)

**Signal falloff**: Follows **1/r³** law → distance is critical!

### Earth's Magnetic Field

- **Total field**: 25-65 μT (location-dependent)
- **Horizontal component**: 15-30 μT
- **Vertical component**: 20-50 μT
- **Variation from orientation**: ±10 μT

**Key Insight**: Without calibration, Earth field (±10 μT) dominates finger magnet signal at extended positions (18-34 μT) → **Calibration is essential, not optional**.

---

## Quick Reference Checklist

- [ ] Purchase 5× magnets (6mm × 3mm N48 recommended)
- [ ] Test polarity with compass
- [ ] Mark North pole on each magnet
- [ ] Assign polarities: T+, I+, M-, R+, P-
- [ ] Attach to rings or skin (dorsal side, middle phalanx)
- [ ] Run calibration wizard in collector.html
- [ ] Collect SNR test data (10 reps × extended/flexed per finger)
- [ ] Run `python -m ml.analyze_snr --plot`
- [ ] Verify SNR > 10:1 at extended position
- [ ] If SNR adequate → proceed to Phase 1 data collection
- [ ] If SNR poor → troubleshoot (see table above)

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-09  
**Related Documents**:
- `docs/design/magnetic-tracking-pipeline-analysis.md` - Technical deep dive
- `docs/design/magnetic-finger-tracking-analysis.md` - Physics foundation  
**Support**: File issues at github.com/simcap/simcap

---

<link rel="stylesheet" href="../../src/simcap.css">
