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
| Magnetometer noise floor (typical MEMS) | 0.1-0.5 μT RMS |
| Puck.js MMC5603 noise density | ~0.4 μT at 100 Hz BW |

**Key observation:** At typical finger-to-palm distances (50-100 mm), even small magnets produce fields comparable to or larger than Earth's field variation, but the r³ falloff means positioning within that range changes signal dramatically.

---

## 4. The Signal-to-Noise Challenge

### 4.1 Noise Sources

1. **Sensor Noise (Intrinsic)**
   - Puck.js uses MMC5603NJ magnetometer
   - Typical noise: 0.4-2 μT RMS depending on bandwidth
   - At 50 Hz sample rate: ~0.6 μT noise floor

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

For a **5×2 mm magnet** at various distances:

| Distance | Signal | Sensor Noise | Earth Field Variation | Effective SNR |
|----------|--------|--------------|----------------------|---------------|
| 50 mm | 59 μT | 0.6 μT | ~10 μT (orientation) | ~6:1 |
| 80 mm | 14 μT | 0.6 μT | ~10 μT | ~1.4:1 |
| 100 mm | 7 μT | 0.6 μT | ~10 μT | ~0.7:1 |

**Critical insight:** At extended finger positions (80-100 mm), the magnetic signal approaches or falls below the noise floor from orientation-dependent Earth field variation. **Robust tracking requires either stronger magnets, closer distances, or sophisticated compensation.**

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

## Appendix B: Puck.js MMC5603 Magnetometer Specifications

| Parameter | Value |
|-----------|-------|
| Manufacturer | MEMSIC |
| Type | AMR (Anisotropic Magnetoresistance) |
| Range | ±30 Gauss (±3000 μT) |
| Resolution | 0.0625 μT (16-bit) |
| Noise density | 0.4 μT/√Hz |
| Bandwidth | Up to 1000 Hz |
| Sample rate (Puck.js) | Typically 10-100 Hz |

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
*SIMCAP Project - Magnetic Finger Tracking Feasibility Analysis*
