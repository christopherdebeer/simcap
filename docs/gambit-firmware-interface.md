# GAMBIT Firmware Interface Reference

This document describes the LED feedback system, button interactions, sampling modes, and context-aware behavior of the GAMBIT firmware.

## Overview

The GAMBIT firmware runs on a Puck.js device and provides:
- 9-axis IMU (accelerometer, gyroscope, magnetometer)
- Environmental sensing (ambient light, capacitive touch)
- Button-based gesture recognition
- Context-aware automatic mode switching
- Binary telemetry streaming over BLE

**Firmware Version:** 0.4.2
**Hardware:** Puck.js v2 with MMC5603NJ magnetometer

---

## LED Feedback Reference

The Puck.js has three LEDs: **Red** (LED1), **Green** (LED2), and **Blue** (LED3).

### Startup Sequence

| LED | Pattern | Meaning |
|-----|---------|---------|
| Green | Single 200ms flash | Firmware initialized and ready |

### Mode Indicators

When the mode changes (via double-tap or auto-switching):

| Mode | LED | Pattern | Description |
|------|-----|---------|-------------|
| LOW_POWER | Green | Quick 50ms flash | Power-saving mode |
| NORMAL | Blue | 100ms flash | Standard tracking (default) |
| HIGH_RES | Blue | Double flash (50-50-50ms) | High resolution sampling |
| BURST | Purple | Red + Blue 50ms | Maximum sample rate |

### Battery Level Indicators

Shown on long press (1s hold):

| Level | LED | Pattern | Description |
|-------|-----|---------|-------------|
| >60% | Green | Triple flash | Good battery |
| 30-60% | Yellow | Double flash (red + green) | Medium battery |
| 10-30% | Red | Double flash | Low battery |
| <10% | Red | Rapid 6-flash burst | Critical - charge soon |

### Streaming Indicators

| State | LED | Pattern | Description |
|-------|-----|---------|-------------|
| Streaming active | Blue | Brief 20ms pulse every 2s | Data being transmitted |
| Stream started | Green | 500ms flash | BLE streaming begun |
| Collection complete | Green | 200ms flash | Sample collection finished |

### Button Gesture Feedback

Visual confirmation when a gesture is recognized:

| Gesture | LED | Pattern |
|---------|-----|---------|
| Single tap | Blue | 100ms flash |
| Double tap | Blue | Double flash |
| Triple tap | Green | 100ms flash |
| Long press (1s) | Green | Quad flash |
| Very long press (3s) | Red | 500ms flash |

### Other Indicators

| Event | LED | Pattern | Description |
|-------|-----|---------|-------------|
| Error | Red | Rapid 5-flash | Operation failed |
| Success | Green | Double 200ms flash | Operation succeeded |
| Context calibrated | Green | Double flash | Sensors calibrated |
| Wake-on-touch triggered | Blue | 100ms flash | Touch detected, starting stream |

---

## Button Interactions

The Puck.js button supports gesture recognition with the following timing:

| Parameter | Value |
|-----------|-------|
| Tap window | 300ms (time between taps for multi-tap) |
| Long press threshold | 1000ms |
| Very long press threshold | 3000ms |
| Debounce | 30ms |

### Gesture Actions

| Gesture | Detection | Action |
|---------|-----------|--------|
| **Single Tap** | 1 quick press | Toggle streaming on/off |
| **Double Tap** | 2 quick presses | Cycle through sampling modes |
| **Triple Tap** | 3 quick presses | Mark event (annotation timestamp) |
| **Long Press** | Hold 1-3 seconds | Show battery level (or calibrate context if uncalibrated) |
| **Very Long Press** | Hold >3 seconds | Enter deep sleep / low power mode |

### Gesture Detection Flow

```
Button Press → Hold Time Check
                ├── >3s: VERY_LONG_PRESS → Deep sleep
                ├── 1-3s: LONG_PRESS → Battery/Calibrate
                └── <1s: Count as tap
                         ├── Wait 300ms for more taps
                         └── Evaluate: 1=SINGLE, 2=DOUBLE, 3+=TRIPLE
```

---

## Sampling Modes

Four modes balance power consumption versus data quality:

| Mode | Accel/Gyro Rate | Magnetometer | Light | Battery | Use Case |
|------|-----------------|--------------|-------|---------|----------|
| **LOW_POWER** | 26 Hz | Every 5th (5 Hz) | Every 20th | Every 200th | Background/sleep |
| **NORMAL** | 26 Hz | Every 2nd (13 Hz) | Every 10th | Every 100th | Standard tracking |
| **HIGH_RES** | 52 Hz | Every sample (52 Hz) | Every 5th | Every 100th | Calibration, precision |
| **BURST** | 104 Hz | Every sample (104 Hz) | Every 10th | Every 200th | Fast gestures, debugging |

### Mode Selection

- **Default:** NORMAL mode at startup
- **Manual:** Double-tap cycles: LOW_POWER → NORMAL → HIGH_RES → BURST → LOW_POWER
- **Automatic:** Context-aware switching (when enabled)

### Recommended Modes by Task

| Task | Recommended Mode | Reason |
|------|------------------|--------|
| Magnetometer calibration | HIGH_RES | 52 Hz mag, 1:1 sampling |
| Gesture recognition | NORMAL | Balanced power/precision |
| Battery conservation | LOW_POWER | Minimal sensor activity |
| Motion analysis | BURST | 104 Hz captures fast movements |

---

## Context Awareness

> **Status:** Context detection is implemented but **not fully integrated** into the web application. The sensors work but automatic mode switching may not behave as expected in all scenarios.

### Context States

The firmware attempts to detect usage context using ambient sensors:

| Context | Detection Criteria | Description |
|---------|-------------------|-------------|
| **STORED** | Dark + No grip | In pocket, bag, or case |
| **HELD** | Grip detected | Being held, stationary |
| **ACTIVE** | Grip + Motion (>1g) | Active use, gesturing |
| **TABLE** | Light + No grip | Resting on surface |
| **UNKNOWN** | Ambiguous readings | Transitional state |

### Sensor Thresholds

| Sensor | Threshold | Meaning |
|--------|-----------|---------|
| Light (dark) | < 0.02 | Considered dark/occluded |
| Capacitive grip | > baseline + 500 | Firm grip detected |
| Motion | > 9500 (raw units) | Significant acceleration |

### Context → Mode Mapping (Auto Mode)

When auto-mode is enabled, context changes trigger mode switches:

| Context | Target Mode | Rationale |
|---------|-------------|-----------|
| STORED | LOW_POWER | Minimize battery drain |
| HELD | HIGH_RES | Anticipate precise input |
| ACTIVE | NORMAL | Efficient for gestures |
| TABLE | LOW_POWER | Idle, ready for pickup |

### Hysteresis

Context changes require **majority agreement** over a 5-sample window to prevent rapid switching from sensor noise.

---

## Calibration

### Context Sensor Calibration

**When:** Long press when uncalibrated (first use or after reset)

**Process:**
1. Device should be placed on table (not held)
2. Takes 10 samples of light and capacitive sensors
3. Establishes baseline for grip detection
4. Green double-flash confirms completion

**Required for:**
- Grip detection (context awareness)
- Wake-on-touch feature

### Magnetometer Degauss

**When:** Automatically at firmware startup

**Process:**
1. SET pulse (375ns) to establish reference
2. 2ms delay
3. RESET pulse (375ns) to clear residual magnetization

**Purpose:** Clears null field offset in the MMC5603NJ magnetometer's AMR sensors after power-up or exposure to strong fields.

---

## Appendix A: Context Determination Logic

```
detectContext(light, cap, accelMag):
    isDark = light < 0.02
    isGripped = (cap - capBaseline) > 500
    isMoving = accelMag > 9500

    if isDark AND NOT isGripped:
        return STORED
    else if isGripped AND isMoving:
        return ACTIVE
    else if isGripped:
        return HELD
    else if NOT isDark AND NOT isGripped:
        return TABLE
    else:
        return UNKNOWN
```

### Impact on Modes

```
Context Change Detected
    │
    ├── Auto-mode enabled?
    │   ├── Yes → Apply mode mapping
    │   └── No → Ignore (manual control)
    │
    └── Streaming active?
        ├── Yes → Mode change takes effect immediately
        └── No → Mode saved for next stream
```

---

## Appendix B: Integration Status

### Fully Integrated

| Feature | Status | Notes |
|---------|--------|-------|
| Button gestures | Active | All gestures work |
| Mode switching | Active | Manual and API control |
| LED feedback | Active | All patterns functional |
| Magnetometer degauss | Active | Runs at startup |
| Binary streaming | Active | Primary telemetry format |

### Partially Integrated

| Feature | Status | Notes |
|---------|--------|-------|
| Context detection | Firmware ready | Web UI doesn't display context |
| Auto-mode switching | Firmware ready | Disabled by default, untested in web |
| Grip detection | Firmware ready | `grip` field in telemetry, unused |
| Wake-on-touch | Firmware ready | No web UI to enable |

### Known Limitations

1. **Sample Rate:** Default 26 Hz (NORMAL) may be too slow for magnetometer calibration. Consider HIGH_RES (52 Hz) or BURST (104 Hz) for calibration tasks.

2. **Context Calibration:** Must be performed manually via long-press. No web UI calibration flow exists yet.

3. **Light Sensor:** Readings included in telemetry but not displayed or used in web application.

4. **Capacitive Sensor:** Baseline calibration required before grip detection works. Currently no reminder/prompt in web UI.

---

## Appendix C: Frame Protocol Reference

Gesture and context events are sent as JSON frames:

```javascript
// Button gesture
{ "BTN": { "gesture": "DOUBLE_TAP", "time": 12345, "pressCount": 7 } }

// Context change
{ "CTX": { "context": "active", "from": "held" } }

// Mode change
{ "MODE": { "mode": "HIGH_RES", "config": { "accelHz": 52, ... } } }

// Context calibration
{ "CAL": { "type": "context", "light": 0.043, "cap": 12500 } }

// Event mark (triple-tap)
{ "MARK": { "time": 54321, "sampleCount": 1234 } }
```

---

## Appendix D: Quick Reference Card

### Button Gestures
| Taps/Hold | Action |
|-----------|--------|
| 1 tap | Toggle streaming |
| 2 taps | Cycle mode |
| 3 taps | Mark event |
| 1s hold | Battery level |
| 3s hold | Deep sleep |

### Mode Cycle Order
LOW_POWER → NORMAL → HIGH_RES → BURST → (repeat)

### LED Colors
| Color | Meaning |
|-------|---------|
| Green | Success, good, ready |
| Blue | Normal operation, mode |
| Red | Warning, error, critical |
| Yellow (R+G) | Moderate/medium |
| Purple (R+B) | BURST mode |
