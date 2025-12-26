# GAMBIT Capacitive Sensor Wiring Guide

**Purpose:** Physical wiring instructions for grip detection on the Puck.js device

## Overview

The Puck.js v2 has a capacitive sensing capability on pin **D11**. This guide covers how to wire an external electrode for grip detection when the device is mounted in its silicone housing.

```
┌─────────────────────────────────────┐
│          GRIP DETECTION             │
│                                     │
│   ┌───────────────────────────┐     │
│   │     Silicone Case         │     │
│   │  ┌─────────────────────┐  │     │
│   │  │    Puck.js PCB      │  │     │
│   │  │                     │  │     │
│   │  │   D11 ●────────────────────► Electrode (copper tape/fabric)
│   │  │                     │  │     │
│   │  └─────────────────────┘  │     │
│   │                           │     │
│   └───────────────────────────┘     │
│                                     │
└─────────────────────────────────────┘
```

## Required Materials

| Material | Specification | Source |
|----------|--------------|--------|
| Thin wire | 30 AWG stranded, silicone insulated | Generic |
| Copper tape | 6-25mm with conductive adhesive | [Adafruit #1128](https://www.adafruit.com/product/1128) |
| **OR** Conductive fabric | <10Ω resistance | [EM shielding fabric](https://www.sciencedirect.com/topics/engineering/conductive-textile) |
| Solder | Standard rosin core | Generic |
| Heat shrink | 1.5mm diameter | Generic |

## Puck.js Pin Layout

The D11 capacitive sensing pin is located on the edge of the PCB:

```
     ┌───────────────────┐
     │      BUTTON       │
     │                   │
     │   ┌───────────┐   │
     │   │  nRF52840 │   │
     │   │           │   │
     │   └───────────┘   │
     │                   │
     │ ● ● ● ● ● ● ● ● ● │  ← 0.1" pitch pads
     │ D28   D31  FET D11│
     └───────────────────┘
           ↓
         D11 = Capacitive sense
```

Reference: [Espruino Puck.js pinout](https://www.espruino.com/Puck.js)

## Wiring Options

### Option 1: Through-Silicone Wire Exit (Recommended)

The silicone cover is flexible and can accommodate a thin wire exiting at the edge.

**Steps:**
1. Remove silicone cover from Puck.js
2. Solder 30 AWG wire to D11 pad (use minimal solder)
3. Route wire along PCB edge toward the gap between silicone and ABS plastic
4. Replace silicone cover - wire exits at seam
5. Connect wire to external electrode

```
Side view:
    ┌─ Silicone cover
    │
    ├──────────────────┤
    │   PCB            │
    │ D11 ●────────────┼──► Wire exits at seam
    │                  │
    ├──────────────────┤
    │   ABS plastic    │
    └──────────────────┘
```

**Pros:** Clean, weather-resistant, no case modification
**Cons:** Wire can shift if not secured

### Option 2: Lanyard Mount Exit

The ABS plastic rear has a small lanyard mount hole that can accommodate wire.

**Steps:**
1. Remove silicone cover
2. Solder wire to D11
3. Route wire around PCB edge to lanyard hole
4. Thread wire through lanyard hole
5. Replace silicone cover

**Pros:** Secure wire routing, strain relief
**Cons:** Blocks lanyard use

### Option 3: Internal Electrode (Experimental)

Use conductive material inside the silicone case that capacitively couples through the silicone.

**Steps:**
1. Apply thin copper tape or conductive fabric to inside of silicone cover
2. Connect to D11 with short wire
3. Reassemble - silicone acts as dielectric

**Note:** Silicone thickness (~2mm) may reduce sensitivity. Requires calibration.

## Electrode Design for Grip Detection

### Palm-Mount Configuration

When the Puck.js is mounted in the palm (e.g., on a glove), the electrode should:
- Cover the bottom/rear of the device (contact with palm)
- Have sufficient surface area (≥15mm × 15mm recommended)
- Be durable for repeated grip/release cycles

```
Top view (device on palm):

        ┌───────────────┐
        │   Button      │  ← Facing up (accessible)
        │               │
        │   Silicone    │
        │     Case      │
        │               │
        └───────────────┘
              ↓
    ┌─────────────────────┐
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  ← Copper tape electrode (palm side)
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │     Connected to D11
    └─────────────────────┘
              ↓
           PALM
```

### Recommended Electrode Configurations

#### A. Wrap-Around Copper Tape

```
Side view:
         ┌─────────┐
     ▓▓▓▓│ Puck.js │▓▓▓▓
     ▓   └─────────┘   ▓
     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Continuous copper tape wrap
           (bottom)
```

- Wrap 25mm copper tape around bottom third of device
- Covers sides for finger contact + bottom for palm contact
- ~100mm² surface area

#### B. Conductive Fabric Sleeve

```
         ┌─────────┐
    ╔════╡ Puck.js ╞════╗
    ║    └─────────┘    ║
    ║  Conductive Fabric║
    ╚═══════════════════╝
```

- Sew/glue conductive fabric sleeve on bottom half
- More comfortable than copper tape
- Better for glove integration
- Use fabric with <10Ω resistance per [research](https://pmc.ncbi.nlm.nih.gov/articles/PMC7219339/)

#### C. Strip Electrode (Minimal)

```
Bottom view:
    ┌─────────────────┐
    │                 │
    │   ═══════════   │  ← Single 25mm × 6mm copper strip
    │                 │
    └─────────────────┘
```

- Minimal modification
- 6mm copper tape across bottom
- Good for testing, may need threshold adjustment

## Calibration

### Software Calibration

The firmware includes capacitive baseline calibration:

```javascript
// Long-press button when NOT gripping device
// to establish baseline reading

// Or call via BLE:
calibrateContext()
```

### Expected Values

| Condition | capSense() Reading | Notes |
|-----------|-------------------|-------|
| No contact | 0-200 (baseline) | Varies by electrode size |
| Proximity | +200-500 | Hand near but not touching |
| Light grip | +500-1000 | Fingertips touching |
| Firm grip | +1000-3000 | Palm contact |

These are approximate - calibration establishes actual baseline.

### Firmware Constants

```javascript
// In app.js - adjust based on electrode configuration
var CAP_GRIP_THRESHOLD = 500;  // Delta from baseline for grip detection
var CAP_PROXIMITY_THRESHOLD = 200;  // Delta for proximity detection
```

## Integration with Glove/Mount

### Glove Integration

For palm-mounted use in a glove:

1. **External electrode approach:**
   - Attach copper tape to outside of glove palm
   - Route wire inside glove to Puck.js pocket
   - Connect wire to D11

2. **Internal electrode approach:**
   - Use conductive thread sewn into glove palm
   - Connect thread ends to Puck.js D11 wire

```
Glove cross-section:

    ─────────────────────── Outer fabric
    ═══════════════════════ Copper tape electrode
    ─────────────────────── Glove lining
    ~~~~~~~~~~~~~~~~~~~~~~~ Hand/palm
```

### Mounting Housing

For rigid mounting in a custom housing:

1. Include copper electrode in housing design
2. Spring contacts or pogo pins for D11 connection
3. Position electrode on grip surfaces

## Testing

### Basic Continuity Test

```javascript
// Run on Puck.js via Espruino IDE
setInterval(function() {
    console.log("Cap:", Puck.capSense());
}, 500);

// Touch/release electrode and observe values
```

### Grip Detection Test

```javascript
// After wiring, test grip detection
calibrateContext();  // Establish baseline (don't grip)

// Wait a few seconds, then grip device
setInterval(function() {
    var val = Puck.capSense();
    var grip = val > (CAP_BASELINE + CAP_GRIP_THRESHOLD);
    LED2.write(grip ? 1 : 0);  // Green LED when gripping
    console.log("Cap:", val, "Grip:", grip);
}, 200);
```

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No reading change | Bad solder joint | Reflow D11 connection |
| Very low readings | Electrode too small | Increase surface area |
| Erratic readings | Poor grounding | Ensure electrode is connected only to D11 |
| Always high | Electrode touching GND | Check for shorts |
| Low sensitivity | Silicone blocking | Use external electrode |

## Safety Notes

- Copper tape edges can be sharp - fold edges or cover with tape
- Avoid skin irritation with prolonged contact - use fabric for wearables
- Keep connections dry - moisture affects capacitive readings
- Don't exceed 3.3V on D11 - it's a GPIO pin

## References

- [Espruino Puck.js Documentation](https://www.espruino.com/Puck.js)
- [Capacitive Touch Sensing Electrodes - NXP AN3863](https://www.nxp.com/docs/en/application-note/AN3863.pdf)
- [E-Textile 3D Gesture Sensors - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7219339/)
- [Adafruit Copper Tape](https://www.adafruit.com/product/1128)
- [Wyss Institute Fabric Sensors](https://wyss.harvard.edu/news/soft-and-stretchy-fabric-based-sensors-for-wearable-robots/)

---

*Part of the GAMBIT firmware documentation*
