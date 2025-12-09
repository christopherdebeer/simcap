# Magnetic Finger Tracking: Physics & Information Theoretic Analysis

## Executive Summary

This document provides a first-principles analysis of using passive magnets on fingertips/rings to enable finger position tracking via a palm-mounted magnetometer. We examine the fundamental physics constraints, expected signal magnitudes, noise characteristics, and information theoretic limits of this approach.

**Key Findings:**
- Fingertip magnets can produce measurable signals at palm distance, but with significant constraints
- Signal-to-noise ratio is the critical challenge, not absolute field strength
- Orientation matters profoundly—the problem is fundamentally 3D
- Alternating polarity is essential for multi-finger differentiation
- Magnetic superposition is linear (additive), creating both opportunities and challenges
- The approach is feasible but requires careful engineering of magnet selection, placement, and signal processing

---

## 1. Physical Setup & Geometry

### 1.1 Hand Anatomy Relevant to Magnetic Tracking

| Parameter | Typical Value | Range |
|-----------|---------------|-------|
| Palm center to index fingertip | 80-100 mm | 70-120 mm |
| Palm center to thumb tip | 60-80 mm | 50-100 mm |
| Palm center to pinky tip | 70-90 mm | 60-100 mm |
| Inter-finger spacing (adjacent fingertips) | 15-25 mm | 10-35 mm |
| Finger flexion range (tip to palm) | 20-100 mm | varies by finger |

### 1.2 Sensor Location

The Puck.js v2 magnetometer (assumed palm-mounted) sits at the coordinate origin. Finger magnets are distributed in an arc around this center at varying distances depending on finger extension/flexion.

```
           INDEX   MIDDLE   RING   PINKY
              ●       ●       ●       ●     ← Fingertip magnets
               \      |      /      /
                \     |     /      /
                 \    |    /      /
                  \   |   /      /
    THUMB ●--------[PALM]--------          ← Magnetometer
                   SENSOR
```

---

## 2. Physics of Magnetic Dipoles

### 2.1 Magnetic Dipole Field Equations

A permanent magnet (neodymium, etc.) at macroscopic distances behaves as a magnetic dipole. The magnetic field **B** at position **r** from a dipole with moment **m** is:

```
B(r) = (μ₀/4π) × [ 3(m·r̂)r̂ - m ] / r³
```

Where:
- `μ₀ = 4π × 10⁻⁷ T·m/A` (permeability of free space)
- `m` = magnetic dipole moment (A·m²)
- `r` = distance from dipole (m)
- `r̂` = unit vector from dipole to observation point

**Critical insight: Field strength falls off as 1/r³** — this is the dominant physical constraint.

### 2.2 On-Axis Field (Simplified Case)

For a dipole aligned with the measurement axis, the field at distance `r` along the axis is:

```
B_axial = (μ₀/2π) × m / r³
```

### 2.3 Magnetic Moment of Small Magnets

The dipole moment of a cylindrical magnet is:

```
m = Br × V / μ₀ = Br × (π × radius² × height) / μ₀
```

Where `Br` is the remanent magnetization (typically 1.0-1.4 T for N35-N52 neodymium).

---

## 3. Expected Signal Magnitudes

### 3.1 Candidate Magnet Specifications

| Magnet Type | Diameter | Height | Volume | Br | Dipole Moment |
|-------------|----------|--------|--------|-----|---------------|
| Tiny disc (ring-safe) | 3 mm | 1 mm | 7.1 mm³ | 1.2 T | 6.8 × 10⁻³ A·m² |
| Small disc | 5 mm | 2 mm | 39 mm³ | 1.2 T | 3.7 × 10⁻² A·m² |
| Medium disc | 6 mm | 3 mm | 85 mm³ | 1.3 T | 8.8 × 10⁻² A·m² |
| Ring magnet | 8 mm OD, 4 mm ID | 3 mm | 113 mm³ | 1.2 T | 1.1 × 10⁻¹ A·m² |

### 3.2 Field Strength at Palm Distance

Using `B = (μ₀/2π) × m / r³` for on-axis field:

| Magnet | m (A·m²) | At 50 mm | At 80 mm | At 100 mm |
|--------|----------|----------|----------|-----------|
| Tiny (3×1 mm) | 6.8 × 10⁻³ | 10.9 μT | 2.7 μT | 1.4 μT |
| Small (5×2 mm) | 3.7 × 10⁻² | 59.2 μT | 14.5 μT | 7.4 μT |
| Medium (6×3 mm) | 8.8 × 10⁻² | 140.8 μT | 34.4 μT | 17.6 μT |
| Ring (8×3 mm) | 1.1 × 10⁻¹ | 176.0 μT | 43.0 μT | 22.0 μT |

### 3.3 Comparison to Ambient Field

| Field Source | Magnitude |
|--------------|-----------|
| Earth's magnetic field | 25-65 μT (location dependent) |
| Urban magnetic noise (buildings, wiring) | 0.1-10 μT |
| Electronic devices nearby | 1-100 μT (highly variable) |
| **Puck.js LIS3MDL noise floor** | **~5-10 mgauss RMS (0.5-1.0 μT)** |
| LIS3MDL resolution (±4 gauss range) | 0.146 mgauss/LSB (0.0146 μT) |

**Key observation:** At typical finger-to-palm distances (50-100 mm), even small magnets produce fields comparable to or larger than Earth's field variation, but the r³ falloff means positioning within that range changes signal dramatically.

**Important sensor limitation:** The LIS3MDL has been observed to exhibit ~2× higher noise than datasheet specifications in practice (see ST Community discussions). Real-world RMS noise of 5-10 mgauss (0.5-1.0 μT) should be assumed for SNR calculations.

---

## 4. The Signal-to-Noise Challenge

### 4.1 Noise Sources

1. **Sensor Noise (Intrinsic)**
   - Puck.js uses **STMicroelectronics LIS3MDL** magnetometer
   - Datasheet spec: ~5 mgauss RMS at ±12 gauss full-scale
   - **Real-world measured: ~5-10 mgauss RMS (0.5-1.0 μT)**
   - Higher noise than competing sensors (ST recommends LIS2MDL for lower noise applications)
   - Sensitivity: 6,842 LSB/gauss at ±4 gauss (0.146 mgauss/LSB resolution)

2. **Earth's Field (Static but Orientation-Dependent)**
   - Magnitude: 25-65 μT depending on location
   - As palm orientation changes, projection onto sensor axes changes
   - This is the dominant "signal" the sensor sees
   - Must be compensated via IMU-based orientation tracking

3. **Environmental Magnetic Interference**
   - Steel structures, rebar, furniture: 1-50 μT gradients
   - Electrical wiring (60 Hz): 0.1-5 μT oscillating
   - Electronic devices: highly variable
   - Moving ferromagnetic objects: sporadic transients

4. **Hand Motion Artifacts**
   - As hand moves through spatially varying ambient field
   - Appears as low-frequency drift
   - Correlated with accelerometer/gyro data

### 4.2 Signal-to-Noise Ratio Estimates

For a **5×2 mm magnet** at various distances (using LIS3MDL realistic noise of ~1.0 μT RMS):

| Distance | Signal | Sensor Noise | Earth Field Variation | Effective SNR |
|----------|--------|--------------|----------------------|---------------|
| 50 mm (flexed) | 59 μT | 1.0 μT | ~10 μT (orientation) | ~5.4:1 |
| 80 mm (extended) | 14 μT | 1.0 μT | ~10 μT | ~1.3:1 |
| 100 mm (full reach) | 7 μT | 1.0 μT | ~10 μT | ~0.6:1 |

For a **6×3 mm magnet** (recommended minimum):

| Distance | Signal | Sensor Noise | Earth Field Variation | Effective SNR |
|----------|--------|--------------|----------------------|---------------|
| 50 mm (flexed) | 141 μT | 1.0 μT | ~10 μT | ~12.8:1 |
| 80 mm (extended) | 34 μT | 1.0 μT | ~10 μT | ~3.1:1 |
| 100 mm (full reach) | 18 μT | 1.0 μT | ~10 μT | ~1.6:1 |

**Critical insight:** The LIS3MDL's higher-than-expected noise floor (~1 μT vs. ~0.5 μT for better sensors) makes the Earth field variation (~10 μT from orientation changes) the dominant noise source. **Earth field compensation via IMU fusion is essential, not optional.**

**Implication for magnet sizing:** With realistic LIS3MDL noise, the 5×2 mm magnet is marginal. A 6×3 mm or larger magnet provides necessary SNR margin for extended finger positions.

---

## 5. Orientation Effects

### 5.1 Why Orientation Matters Profoundly

The dipole field is **not spherically symmetric**. For a dipole moment **m** pointing in direction **ẑ**:

```
B_z = (μ₀/4π) × m × (3cos²θ - 1) / r³
B_r = (μ₀/4π) × m × (3cosθ sinθ) / r³
```

Where θ is the angle from the dipole axis.

| Angle from dipole axis | Relative field strength | Field direction |
|------------------------|------------------------|-----------------|
| 0° (along axis) | 1.0 | Parallel to dipole |
| 45° | 0.35 | Complex angle |
| 90° (perpendicular) | 0.5 | Opposite to dipole |
| ~54.7° | 0 (null) | Zero field! |

**Implication:** A magnet oriented perpendicular to the palm produces different field patterns than one parallel. Finger flexion changes both distance AND orientation of the magnet relative to the sensor.

### 5.2 Information in Orientation

This is actually **useful information**—if we can measure the full 3D field vector:

1. **Magnet axial orientation** affects the field pattern
2. **Fingertip pose** (not just position) becomes observable
3. **Multiple magnets** with different orientations create distinguishable signatures

### 5.3 Ring Mounting Considerations

If magnets are embedded in finger rings:
- **Axial mounting** (N-S along finger): Field points toward/away from palm
- **Radial mounting** (N-S across finger): Field perpendicular to finger axis
- **Tangential mounting** (N-S around finger): Complex field pattern

Recommendation: **Axial mounting with alternating polarity** provides cleanest signal separation.

---

## 6. Multi-Finger Superposition

### 6.1 The Principle of Superposition

Magnetic fields **add linearly** (vector superposition):

```
B_total = B_earth + B_finger1 + B_finger2 + B_finger3 + B_finger4 + B_finger5 + B_noise
```

This is both a blessing and a challenge:
- **Blessing:** Each finger contributes independently
- **Challenge:** Signals can cancel or reinforce unpredictably

### 6.2 Same-Polarity Problems

If all fingers have magnets oriented the same way (all N toward palm):

```
All fingers extended:
  B_total = B_earth + (5 × small positive contribution)
           ≈ B_earth + 5 × 7 μT = B_earth + 35 μT

All fingers flexed (closer):
  B_total = B_earth + (5 × larger positive contribution)
           ≈ B_earth + 5 × 50 μT = B_earth + 250 μT
```

Problems:
- Only observe **aggregate** signal, not per-finger
- One finger flexing is masked by four fingers extended
- Cannot distinguish which fingers are flexed

### 6.3 Alternating Polarity Solution

Assign alternating orientations:
- Index: N toward palm (+)
- Middle: N away from palm (-)
- Ring: N toward palm (+)
- Pinky: N away from palm (-)
- Thumb: N toward palm (+)

Now individual finger movements create **distinguishable vector changes**:

```
Index flexes (closer, + contribution increases):
  ΔB ≈ +40 μT in +z direction

Middle flexes (closer, - contribution magnitude increases):
  ΔB ≈ +40 μT in -z direction (opposite!)
```

### 6.4 Optimal Polarity Assignment

For maximum distinguishability with 5 fingers:

| Finger | Polarity | Rationale |
|--------|----------|-----------|
| Thumb | + | Spatially isolated, any polarity works |
| Index | + | Reference finger |
| Middle | - | Adjacent to index, opposite polarity |
| Ring | + | Adjacent to middle, opposite polarity |
| Pinky | - | Adjacent to ring, opposite polarity |

This creates a spatial + polarity pattern that maximizes the uniqueness of each finger's contribution to the field vector.

### 6.5 3D Vector Decomposition

With a 3-axis magnetometer and alternating polarities:
- Each finger contributes a unique vector based on:
  - Position (x, y, z relative to sensor)
  - Polarity (+/-)
  - Distance (affects magnitude)

The problem becomes: **solve for 5 unknown positions given 3 measurements + temporal correlation**

This is underdetermined instantaneously but tractable with:
- Temporal filtering (fingers don't teleport)
- Kinematic constraints (anatomically possible poses)
- Machine learning on the composite signal

---

## 7. Information Theoretic Limits

### 7.1 Degrees of Freedom

**Input (what we're trying to sense):**
- 5 fingers × 2 DoF each (flexion, abduction) = 10 DoF minimum
- Plus wrist/palm orientation: 3 DoF
- Total: ~13 DoF hand pose

**Output (what we measure):**
- Magnetometer: 3 values (Bx, By, Bz)
- Accelerometer: 3 values (ax, ay, az)
- Gyroscope: 3 values (ωx, ωy, ωz)
- Total: 9 values per timestep

### 7.2 Instantaneous Under-Determination

At any single instant:
- 13 unknowns (hand pose)
- 9 measurements
- **Under-determined** by 4 degrees of freedom

However, with temporal integration:
- Fingers move continuously
- Prior frames constrain current state
- Bayesian filtering can recover missing DoF

### 7.3 Effective Channel Capacity

From magnetometer only (3 measurements):
- Dynamic range: ~16 bits (65536 levels, 0.15 μT resolution typical)
- Sample rate: 50-100 Hz
- Raw capacity: 3 × 16 × 50 = 2400 bits/second

But considering SNR limitations:
- Effective resolution: ~8-10 bits (noise limits precision)
- Useful capacity: ~1500 bits/second from magnetometer

For discrete pose classification (32 poses = 5 bits):
- Theoretically: 300 pose classifications/second possible
- Practically: 10-20 Hz reliable classification more realistic

### 7.4 Pose Classification Feasibility

| Pose Vocabulary Size | Bits Required | Feasibility |
|---------------------|---------------|-------------|
| 8 poses | 3 bits | Highly feasible |
| 16 poses | 4 bits | Feasible |
| 32 poses | 5 bits | Feasible with care |
| 64 poses | 6 bits | Challenging |
| 128+ poses | 7+ bits | Requires very clean signals |

---

## 8. Environmental Calibration Requirements

### 8.1 Earth Field Compensation

The Earth's field (25-65 μT) is the elephant in the room:
- Dominates the raw magnetometer signal
- Rotates through sensor frame as palm orientation changes
- Must be tracked and subtracted

**Solution:** Use gyroscope + accelerometer to track palm orientation, then project known Earth field into sensor frame and subtract.

```
B_fingers = B_measured - R(orientation) × B_earth_local
```

Where R is the rotation matrix from world to sensor frame.

### 8.2 Hard/Soft Iron Calibration

Ferromagnetic materials on the hand create:
- **Hard iron:** Constant offset (rings, watches, etc.)
- **Soft iron:** Orientation-dependent distortion

Standard magnetometer calibration:
1. Rotate hand through all orientations
2. Fit ellipsoid to data
3. Transform to sphere + offset

### 8.3 Per-User Calibration

Hand geometry varies significantly:
- Finger lengths differ by ±15%
- Palm sizes differ by ±20%
- Ring positions vary

Calibration sequence needed:
1. Finger-by-finger isolation (flex each finger alone)
2. Full extension pose
3. Full flexion pose
4. Neutral rest pose

---

## 9. Practical Magnet Recommendations

### 9.1 Minimum Viable Magnet

For detectable signal at 80 mm (extended finger):
- Need >5 μT to exceed noise floor with margin
- **Minimum: 5 mm diameter × 2 mm height N42 disc**
- Produces ~14 μT at 80 mm

### 9.2 Recommended Magnet

For robust operation with SNR margin:
- **Recommended: 6 mm diameter × 3 mm height N48 disc**
- Produces ~35 μT at 80 mm
- Provides 5:1 SNR over noise

### 9.3 Ring Integration Options

| Option | Pros | Cons |
|--------|------|------|
| Magnet embedded in ring | Comfortable, stable | Limited size, orientation fixed |
| Magnet on ring surface | Larger magnet possible | May snag, aesthetics |
| Magnetic ring (whole ring is magnet) | Strong signal | Heavy, expensive |
| Silicone band with embedded magnet | Comfortable, adjustable | May shift, weaker hold |

### 9.4 Safety Considerations

- Neodymium magnets can pinch skin if two collide
- Keep away from magnetic media, credit cards
- Medical: check for pacemaker contraindications
- Small magnets: choking hazard if loose

---

## 10. Comparison to Alternative Approaches

### 10.1 Active RF Tracking

| Aspect | Magnetic (Passive) | RF (Active) |
|--------|-------------------|-------------|
| Finger hardware | Magnet only | Battery + transmitter |
| Power | None on finger | Requires charging |
| Accuracy | ~5-10 mm (estimated) | Sub-mm possible |
| Cost | <$1/finger | $10+/finger |
| Interference | Ferromagnetic objects | RF congestion |

### 10.2 Vision-Based Tracking

| Aspect | Magnetic | Camera-based |
|--------|----------|--------------|
| Occlusion | No occlusion issues | Finger blocking is common |
| Lighting | Works in any light | Needs illumination |
| Privacy | No images captured | Camera always on |
| Setup | Self-contained | Requires external camera |
| Accuracy | Lower | Higher (with ML) |

### 10.3 IMU Per Finger

| Aspect | Central Magnet Sensing | IMU Per Finger |
|--------|----------------------|----------------|
| Finger hardware | Passive magnet | IMU + battery |
| Finger pose | Position only | Full orientation |
| Complexity | 1 smart + 5 dumb | 6 smart devices |
| Latency | <10 ms | <5 ms |
| Cost | Low | High |

---

## 11. Conclusions & Recommendations

### 11.1 Is Magnetic Finger Tracking Feasible?

**Yes, with caveats:**

✓ Physics permits detection of fingertip magnets at palm distance
✓ Signal magnitudes are workable with appropriate magnet sizing
✓ Information theory supports discrete pose classification
✓ Alternating polarity enables multi-finger differentiation

**Critical success factors:**
- Proper magnet sizing (≥5×2 mm N42 minimum)
- Alternating polarity across adjacent fingers
- Robust Earth field compensation via IMU fusion
- Machine learning to handle the under-determined inverse problem

### 11.2 Expected Performance Envelope

| Metric | Conservative Estimate | Optimistic Estimate |
|--------|----------------------|---------------------|
| Pose vocabulary | 16-32 poses | 64+ poses |
| Classification rate | 10 Hz | 30+ Hz |
| Position accuracy | ~10 mm | ~5 mm |
| Orientation sensing | Limited | Per-finger (with optimization) |

### 11.3 Recommended Development Path

1. **Phase 1: Proof of Concept**
   - Single finger with 6×3 mm magnet
   - Verify detection at extension/flexion extremes
   - Characterize SNR empirically

2. **Phase 2: Multi-Finger Static**
   - All 5 fingers with alternating polarity
   - Collect data for 16-32 static poses
   - Train classifier, evaluate accuracy

3. **Phase 3: Dynamic Gestures**
   - Temporal models for gesture sequences
   - Sliding window classification
   - Real-time demo

4. **Phase 4: Continuous Tracking**
   - Regression to continuous pose vector
   - Kalman/particle filter for temporal smoothing
   - Hand model visualization

### 11.4 Key Open Questions for Empirical Testing

1. What is the actual SNR with real hardware in real environments?
2. How much does environmental calibration help?
3. Can ML models generalize across users with different hand sizes?
4. What is the minimum calibration sequence for new users?
5. How do dynamic gestures perform vs. static poses?

---

## Appendix A: Derivation of Dipole Field Equations

The magnetic field of a dipole at the origin with moment **m** = m**ẑ** is:

In spherical coordinates:
```
B_r = (μ₀/4π) × (2m cosθ) / r³
B_θ = (μ₀/4π) × (m sinθ) / r³
B_φ = 0
```

Converting to Cartesian with dipole along z:
```
B_x = (μ₀/4π) × m × (3xz) / r⁵
B_y = (μ₀/4π) × m × (3yz) / r⁵
B_z = (μ₀/4π) × m × (3z² - r²) / r⁵
```

Or in vector form:
```
B = (μ₀/4π) × [ 3(m·r̂)r̂ - m ] / r³
```

## Appendix B: Puck.js LIS3MDL Magnetometer Specifications

*Source: [ST LIS3MDL Datasheet](https://www.st.com/resource/en/datasheet/lis3mdl.pdf), [ST Community Discussions](https://community.st.com/t5/mems-sensors/the-rms-noise-of-lis3mdl-and-lsm303d/td-p/401561)*

| Parameter | Value |
|-----------|-------|
| Manufacturer | STMicroelectronics |
| Type | MEMS Magnetometer |
| Full-Scale Range | ±4 / ±8 / ±12 / ±16 gauss (user selectable) |
| Resolution | 16-bit |
| **Sensitivity (±4 gauss)** | **6,842 LSB/gauss (0.146 mgauss/LSB)** |
| Sensitivity (±8 gauss) | 3,421 LSB/gauss (0.29 mgauss/LSB) |
| Sensitivity (±12 gauss) | 2,281 LSB/gauss (0.43 mgauss/LSB) |
| Sensitivity (±16 gauss) | 1,711 LSB/gauss (0.58 mgauss/LSB) |
| **Noise (datasheet)** | ~5 mgauss RMS at ±12 gauss |
| **Noise (measured)** | **~5-10 mgauss RMS (0.5-1.0 μT)** |
| Output Data Rate | Up to 155 Hz (high precision) or 1000 Hz (lower precision) |
| Temperature Sensor | Yes (8 LSB/°C, 0 = 25°C) |
| Operating Temperature | -40°C to +85°C |
| Interface | I²C / SPI |
| Sample rate (Puck.js firmware) | 10 Hz (every 5th sample at 50 Hz loop) |

### LIS3MDL vs. Alternative Sensors

| Sensor | Noise (RMS) | Notes |
|--------|-------------|-------|
| LIS3MDL | ~5-10 mgauss | Current Puck.js sensor |
| LSM303D | ~5 mgauss | Lower noise in practice |
| **LIS2MDL** | **~3 mgauss** | **ST-recommended upgrade for low-noise applications** |
| IIS2MDC | ~5-10 mgauss | Similar to LIS3MDL |
| MMC5603 | ~0.4 μT/√Hz | MEMSIC alternative, potentially lower noise |

**Practical implication:** For future hardware revisions, consider LIS2MDL or MMC5603 for improved SNR.

### Full-Scale Range Selection for Finger Tracking

The LIS3MDL offers four full-scale ranges. For magnetic finger tracking:

| Range | Max Field | Resolution | Best For |
|-------|-----------|------------|----------|
| ±4 gauss (±400 μT) | 400 μT | 0.146 mgauss | **Recommended** - best resolution for finger magnets |
| ±8 gauss (±800 μT) | 800 μT | 0.29 mgauss | High-interference environments |
| ±12 gauss (±1200 μT) | 1200 μT | 0.43 mgauss | Very strong magnets or very close range |
| ±16 gauss (±1600 μT) | 1600 μT | 0.58 mgauss | Rarely needed |

**Analysis:** Expected finger magnet signals (7-140 μT) plus Earth's field (~50 μT) fit comfortably within ±4 gauss (±400 μT). Using the ±4 gauss range provides maximum sensitivity (6,842 LSB/gauss) and best discrimination of small field changes.

**Warning:** At ±4 gauss, the sensor will saturate if a strong magnet gets too close. A 6×3mm N48 magnet at 20mm produces ~800 μT—exceeding the ±4 gauss range. Consider ±8 gauss if fingers can come very close to the sensor.

## Appendix C: Neodymium Magnet Grade Reference

| Grade | Br (Remanence) | BHmax | Typical Use |
|-------|----------------|-------|-------------|
| N35 | 1.17-1.22 T | 263-287 kJ/m³ | General purpose |
| N42 | 1.28-1.32 T | 318-342 kJ/m³ | Good balance |
| N48 | 1.37-1.42 T | 366-390 kJ/m³ | High performance |
| N52 | 1.43-1.48 T | 398-422 kJ/m³ | Maximum strength |

Higher grades provide stronger fields but cost more and are more brittle.

---

*Document created: December 2024*
*Updated: December 2024 - Corrected sensor specifications to LIS3MDL per Puck.js datasheet*
*SIMCAP Project - Magnetic Finger Tracking Feasibility Analysis*

## References

1. [ST LIS3MDL Datasheet](https://www.st.com/resource/en/datasheet/lis3mdl.pdf)
2. [ST Application Note AN4602 - LIS3MDL Configuration](https://www.st.com/resource/en/application_note/an4602-lis3mdl-threeaxis-digital-output-magnetometer-stmicroelectronics.pdf)
3. [ST Community - LIS3MDL Noise Discussion](https://community.st.com/t5/mems-sensors/the-rms-noise-of-lis3mdl-and-lsm303d/td-p/401561)
4. [ST Community - LIS3MDL Sensitivity](https://community.st.com/t5/mems-sensors/sensitivity-of-lis3mdl-lsb-gauss/td-p/401626)
5. [Espruino Puck.js Documentation](https://www.espruino.com/Puck.js)
6. [Espruino LIS3MDL Datasheet Mirror](https://www.espruino.com/datasheets/LIS3MDL.pdf)

---

<link rel="stylesheet" href="../../src/simcap.css">
