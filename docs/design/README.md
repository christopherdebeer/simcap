# SIMCAP Design Documents

This directory contains speculative designs, proposals, research directions, and conceptual analysis documents.

These documents describe **potential future directions** rather than current implementations. For documentation on what's actually implemented, see the README files in the respective `src/` directories.

## Documents

### [GAMBIT E2E Process Analysis](gambit-e2e-process-analysis.md)

Detailed analysis of the GAMBIT calibration and data collection system, with an ideal early-phase end-to-end process for gathering data and training an initial model to assess accuracy.

**Contents:**
- Current state assessment (strengths and gaps)
- Ideal early-phase end-to-end process (6 phases)
- Non-generative modeling considerations
- Filtering and calibration best practices
- Recommended immediate actions (2-week sprint)
- Quick reference commands

### [Revisiting SIMCAP](revisiting-simcap.md)

Comprehensive conceptual analysis contrasting the theoretical vision with current implementation.

**Contents:**
- Complete theoretical foundation (magnets + IMU + ML)
- Current implementation analysis (GAMBIT firmware + web UI)
- Gap analysis between vision and reality
- Three-tier roadmap (static poses → gestures → hand tracking)
- Design considerations and known challenges
- Concrete ML architecture proposals

---

## Concepts

### JOYPAD - Dual-Hand Game Controller

See [src/web/JOYPAD/](../../src/web/JOYPAD/) for the concept document.

A proposed dual-device BLE coordination system for emulating game controller HID from sensor data using two SIMCAP devices.

### FFO$$ - Fist Full Of Dollars

See [src/web/FFO$$/](../../src/web/FFO$$/) for the research direction document.

Research direction exploring $1/$P/$N family gesture recognition algorithms applied to IMU sensor data.

---

[← Back to Documentation](../)

---

<link rel="stylesheet" href="../../src/simcap.css">
