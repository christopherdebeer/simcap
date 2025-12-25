# Magnetic Field Simulation for Training Data Bootstrap

## Executive Summary

This document explores using physics-based magnetic field simulation to generate synthetic training data for finger magnet tracking. By modeling magnetic dipole physics, we can generate large volumes of labeled training data without manual collection, potentially accelerating model development by 10-100x.

**Key Opportunity**: Real data collection is slow (~100 samples/session), but simulation can generate millions of labeled samples per hour with perfect ground truth labels.

---

## 1. Current Data Collection Bottleneck

### 1.1 Real Data Collection Constraints

| Metric | Current Reality | Desired |
|--------|-----------------|---------|
| Samples per session | ~2,500 (100s @ 26Hz) | - |
| Sessions collected | ~15 sessions | 1,000+ |
| Total samples | ~37,500 | 1,000,000+ |
| Labels per sample | Manual/semi-auto | Automatic |
| Collection time | 5 min setup + 2 min recording | - |
| Effort per 1K samples | ~3 minutes human time | < 1 second |

### 1.2 The Physics Advantage

Unlike many ML domains where synthetic data lacks realism, **magnetic fields follow exact physics equations**:

```
B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³
```

Where:
- `B` = magnetic field vector at position `r`
- `m` = magnetic dipole moment (fixed for a given magnet)
- `r` = distance from magnet to sensor
- `μ₀` = permeability of free space

**Key insight**: Given magnet position and orientation, the magnetic field is **deterministic**. This means we can simulate ground truth sensor readings from any finger pose.

---

## 2. Proposed Simulation Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAGNETIC FIELD SIMULATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Hand Pose    │    │ Magnetic     │    │ Sensor       │               │
│  │ Generator    │───▶│ Field        │───▶│ Model        │               │
│  │              │    │ Calculator   │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│        │                    │                   │                        │
│        │                    │                   │                        │
│        ▼                    ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Finger       │    │ Sum of       │    │ Noisy        │               │
│  │ Positions    │    │ Dipoles      │    │ Measurement  │               │
│  │ + Magnets    │    │              │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                             │                   │                        │
│                             │                   ▼                        │
│                             │         ┌──────────────────────┐          │
│                             └────────▶│ Synthetic Training   │          │
│                                       │ Data (JSON)          │          │
│                                       │ - mx, my, mz         │          │
│                                       │ - finger_states      │          │
│                                       │ - ground_truth_poses │          │
│                                       └──────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### A. Hand Pose Generator

Generates realistic 5-finger poses with:
- **Kinematic constraints**: Fingers can't bend backwards, joints have limits
- **Pose distributions**: Common poses weighted higher (fist, open, pointing)
- **Temporal dynamics**: Smooth transitions between poses

```python
class HandPoseGenerator:
    """Generate kinematically valid hand poses."""

    def __init__(self, sensor_position: np.ndarray):
        """
        Args:
            sensor_position: 3D position of magnetometer on wrist (mm)
        """
        self.sensor_pos = sensor_position

        # Finger base positions relative to wrist sensor
        self.finger_bases = {
            'thumb':  np.array([20, 30, 0]),   # mm from sensor
            'index':  np.array([45, 60, 0]),
            'middle': np.array([50, 65, 0]),
            'ring':   np.array([45, 60, 0]),
            'pinky':  np.array([40, 55, 0])
        }

        # Finger lengths (base to tip)
        self.finger_lengths = {
            'thumb':  55,  # mm
            'index':  75,
            'middle': 80,
            'ring':   75,
            'pinky':  60
        }

    def generate_pose(self, finger_states: dict) -> dict:
        """
        Generate fingertip positions from finger states.

        Args:
            finger_states: {'thumb': 'extended', 'index': 'flexed', ...}

        Returns:
            {'thumb': np.array([x,y,z]), ...} - fingertip positions in mm
        """
        positions = {}
        for finger, state in finger_states.items():
            base = self.finger_bases[finger]
            length = self.finger_lengths[finger]

            if state == 'extended':
                # Tip is length away from base, roughly parallel to palm
                tip = base + np.array([0, length, 0])
            elif state == 'flexed':
                # Tip curls toward palm, much closer to sensor
                tip = base + np.array([0, 20, -30])  # Closer in Y and Z
            else:  # partial
                tip = base + np.array([0, length * 0.6, -15])

            positions[finger] = tip

        return positions
```

#### B. Magnetic Field Calculator

Uses dipole equation to compute field from each finger magnet:

```python
import numpy as np

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

def magnetic_dipole_field(
    observation_point: np.ndarray,  # Position of sensor (m)
    dipole_position: np.ndarray,    # Position of magnet (m)
    dipole_moment: np.ndarray       # Magnetic moment vector (A·m²)
) -> np.ndarray:
    """
    Calculate magnetic field from a dipole at a given point.

    Based on the exact dipole field equation:
    B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³

    Returns:
        Magnetic field vector in Tesla (convert to μT for sensor units)
    """
    # Vector from dipole to observation point
    r_vec = observation_point - dipole_position
    r_mag = np.linalg.norm(r_vec)

    if r_mag < 1e-6:  # Avoid division by zero
        return np.zeros(3)

    r_hat = r_vec / r_mag

    # Dipole field equation
    m_dot_r = np.dot(dipole_moment, r_hat)
    B = (MU_0 / (4 * np.pi)) * (3 * m_dot_r * r_hat - dipole_moment) / (r_mag ** 3)

    return B


def compute_total_field(
    sensor_pos: np.ndarray,
    finger_positions: dict,
    magnet_config: dict,
    earth_field: np.ndarray = np.array([16.0, 0.0, 47.8])  # Edinburgh, μT
) -> np.ndarray:
    """
    Compute total magnetic field at sensor from all finger magnets + Earth.

    Args:
        sensor_pos: Sensor position in mm
        finger_positions: {'thumb': np.array([x,y,z]), ...} in mm
        magnet_config: {'thumb': {'moment': [mx, my, mz]}, ...} in A·m²
        earth_field: Earth's magnetic field in μT (default: Edinburgh)

    Returns:
        Total magnetic field at sensor in μT
    """
    # Start with Earth's field
    total_field = earth_field.copy()

    for finger, pos in finger_positions.items():
        if finger not in magnet_config:
            continue

        moment = np.array(magnet_config[finger]['moment'])

        # Convert mm to meters for physics calculation
        sensor_m = sensor_pos / 1000.0
        pos_m = pos / 1000.0

        # Add dipole contribution (convert Tesla to μT)
        B_dipole = magnetic_dipole_field(sensor_m, pos_m, moment)
        total_field += B_dipole * 1e6  # Tesla to μT

    return total_field
```

#### C. Sensor Noise Model

Add realistic sensor noise to match real hardware:

```python
class MMC5603Simulator:
    """Simulate MMC5603NJ magnetometer characteristics."""

    def __init__(
        self,
        noise_density: float = 1.0,      # μT RMS
        bias_drift: float = 0.5,          # μT/hour
        quantization: int = 16,           # bits
        range_gauss: float = 30.0,        # ±30 gauss
        sample_rate: float = 26.0         # Hz
    ):
        self.noise_density = noise_density
        self.bias_drift = bias_drift
        self.bits = quantization
        self.range_ut = range_gauss * 100  # Convert gauss to μT
        self.sample_rate = sample_rate

        # Initialize random bias (hard iron simulation)
        self.bias = np.random.uniform(-20, 20, size=3)  # μT

        # Soft iron distortion (slight ellipsoid)
        self.soft_iron = np.eye(3) + np.random.uniform(-0.05, 0.05, size=(3,3))

    def measure(self, true_field: np.ndarray) -> dict:
        """
        Simulate a magnetometer measurement with realistic noise.

        Args:
            true_field: True magnetic field in μT

        Returns:
            dict with raw (LSB), unit-converted, and decorated fields
        """
        # Apply soft iron distortion
        distorted = self.soft_iron @ true_field

        # Apply hard iron bias
        biased = distorted + self.bias

        # Add Gaussian noise
        noisy = biased + np.random.normal(0, self.noise_density, size=3)

        # Quantize to sensor resolution
        lsb_per_ut = 1024 / 100  # 1024 LSB/gauss = 10.24 LSB/μT
        raw_lsb = np.round(noisy * lsb_per_ut).astype(int)

        # Clamp to sensor range
        max_lsb = int(self.range_ut * lsb_per_ut)
        raw_lsb = np.clip(raw_lsb, -max_lsb, max_lsb)

        # Convert back to μT for unit-converted fields
        ut_values = raw_lsb / lsb_per_ut

        return {
            # Raw sensor output (LSB)
            'mx': int(raw_lsb[0]),
            'my': int(raw_lsb[1]),
            'mz': int(raw_lsb[2]),
            # Unit-converted (μT)
            'mx_ut': float(ut_values[0]),
            'my_ut': float(ut_values[1]),
            'mz_ut': float(ut_values[2]),
            # Ground truth (for validation)
            '_true_mx': float(true_field[0]),
            '_true_my': float(true_field[1]),
            '_true_mz': float(true_field[2])
        }
```

### 2.3 Recommended Library: Magpylib

For production implementation, use **Magpylib** - a mature Python library for magnetic field calculation:

```python
import magpylib as magpy
import numpy as np

# Create sensor
sensor = magpy.Sensor(position=(0, 0, 0))  # Wrist position

# Create finger magnets with alternating polarity
magnets = []
finger_configs = [
    ('thumb',  (20, 30, 0),  (0, 0, 0.01)),    # N toward palm (+Z)
    ('index',  (45, 60, 0),  (0, 0, -0.01)),   # N away from palm (-Z)
    ('middle', (50, 65, 0),  (0, 0, 0.01)),    # N toward palm
    ('ring',   (45, 60, 0),  (0, 0, -0.01)),   # N away from palm
    ('pinky',  (40, 55, 0),  (0, 0, 0.01))     # N toward palm
]

for name, pos, moment in finger_configs:
    mag = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, 1200),  # kA/m for N48 neodymium
        dimension=(1.5, 0, 3, 0, 360),  # 3mm dia, 2mm height
        position=pos
    )
    magnets.append((name, mag))

# Calculate field at sensor from all magnets
collection = magpy.Collection(*[m for _, m in magnets])
B_total = sensor.getB(collection)  # Returns field in mT

print(f"Total field at sensor: {B_total * 1000} μT")
```

**Magpylib Advantages**:
- Vectorized NumPy operations (fast)
- Supports arbitrary magnet shapes (not just dipoles)
- Built-in visualization
- Well-documented API

---

## 3. Synthetic Data Generation Strategy

### 3.1 Pose Distribution Design

Generate synthetic poses following realistic distributions:

```python
POSE_WEIGHTS = {
    # Static poses (50% of data)
    'all_extended':    0.10,  # Open palm
    'all_flexed':      0.10,  # Fist
    'index_extended':  0.08,  # Pointing
    'thumb_extended':  0.05,  # Thumbs up
    'peace_sign':      0.07,  # Index + middle
    'pinky_extended':  0.05,  # Pinky out
    'three_fingers':   0.05,  # Index + middle + ring

    # Transitions (30% of data)
    'opening':         0.10,  # Fist → open
    'closing':         0.10,  # Open → fist
    'pointing_start':  0.05,  # Open → pointing
    'wave_motion':     0.05,  # Animated wave

    # Random variations (20% of data)
    'random_static':   0.10,  # Random finger combinations
    'random_dynamic':  0.10,  # Random smooth transitions
}

def generate_training_batch(
    num_samples: int = 10000,
    sample_rate: float = 26.0,
    window_size: int = 50
) -> list:
    """
    Generate a batch of synthetic training samples.

    Returns:
        List of sessions, each containing 'samples' and 'labels'
    """
    sessions = []

    for pose_type, weight in POSE_WEIGHTS.items():
        num_pose_samples = int(num_samples * weight)

        if 'static' in pose_type or pose_type in ['all_extended', 'all_flexed']:
            # Generate static pose samples
            samples = generate_static_pose_samples(
                pose_type,
                num_samples=num_pose_samples,
                sample_rate=sample_rate
            )
        else:
            # Generate dynamic transition samples
            samples = generate_transition_samples(
                pose_type,
                num_samples=num_pose_samples,
                sample_rate=sample_rate
            )

        sessions.extend(samples)

    return sessions
```

### 3.2 Domain Randomization

To ensure synthetic data generalizes to real sensors, randomize:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| **Magnet strength** | 0.8-1.2× nominal | Manufacturing variance |
| **Magnet position** | ±5mm | Attachment variance |
| **Sensor noise** | 0.5-2.0 μT | Environmental conditions |
| **Hard iron bias** | ±50 μT | Location-dependent |
| **Soft iron distortion** | ±10% | Device variance |
| **Earth field** | 40-60 μT | Geographic location |
| **Sensor orientation** | ±15° | Wrist angle variance |
| **Finger lengths** | ±10% | Hand size variance |

```python
class DomainRandomizer:
    """Apply random variations to simulation parameters."""

    def randomize_magnet_config(self, base_config: dict) -> dict:
        """Vary magnet properties within realistic ranges."""
        config = base_config.copy()

        for finger in config:
            # Vary moment strength (±20%)
            config[finger]['moment'] = [
                m * np.random.uniform(0.8, 1.2)
                for m in config[finger]['moment']
            ]

            # Vary attachment position (±5mm)
            config[finger]['offset'] = np.random.uniform(-5, 5, size=3)

        return config

    def randomize_hand_geometry(self, base_geometry: dict) -> dict:
        """Vary hand dimensions (±10%)."""
        geometry = base_geometry.copy()

        for key in geometry:
            geometry[key] *= np.random.uniform(0.9, 1.1)

        return geometry

    def randomize_sensor(self, base_params: dict) -> dict:
        """Vary sensor characteristics."""
        params = base_params.copy()
        params['noise'] = np.random.uniform(0.5, 2.0)
        params['bias'] = np.random.uniform(-50, 50, size=3)
        params['soft_iron'] = np.eye(3) + np.random.uniform(-0.1, 0.1, size=(3,3))
        return params
```

### 3.3 Data Format Compatibility

Generate synthetic data in the exact same format as real sessions:

```python
def generate_synthetic_session(
    num_samples: int = 2500,
    finger_states: list = None,  # List of per-sample finger states
    sample_rate: float = 26.0
) -> dict:
    """
    Generate a synthetic session in SIMCAP v2.1 format.

    Returns:
        Session dict compatible with data_loader.py
    """
    pose_gen = HandPoseGenerator(sensor_position=np.zeros(3))
    sensor = MMC5603Simulator()

    samples = []
    labels = []

    for i, states in enumerate(finger_states):
        # Generate pose and field
        fingertip_positions = pose_gen.generate_pose(states)
        true_field = compute_total_field(
            sensor_pos=np.zeros(3),
            finger_positions=fingertip_positions,
            magnet_config=MAGNET_CONFIG
        )

        # Simulate sensor reading
        measurement = sensor.measure(true_field)

        # Add IMU data (can be from real recordings or also simulated)
        sample = {
            # Accelerometer (at rest, Z = 1g)
            'ax': 0, 'ay': 0, 'az': 8192,
            'ax_g': 0.0, 'ay_g': 0.0, 'az_g': 1.0,
            # Gyroscope (stationary)
            'gx': 0, 'gy': 0, 'gz': 0,
            'gx_dps': 0.0, 'gy_dps': 0.0, 'gz_dps': 0.0,
            # Magnetometer (simulated)
            **measurement,
            # Timing
            'dt': 1.0 / sample_rate,
            't': i * (1000 / sample_rate),  # ms
            # Additional fields
            'isMoving': False,
            'filtered_mx': measurement['mx_ut'],
            'filtered_my': measurement['my_ut'],
            'filtered_mz': measurement['mz_ut']
        }

        samples.append(sample)

        # Generate label
        if i % 50 == 0:  # Start of each window
            labels.append({
                'start_sample': i,
                'end_sample': min(i + 50, num_samples),
                'labels': {
                    'pose': states_to_pose_name(states),
                    'motion': 'static',
                    'calibration': 'none',
                    'fingers': states
                }
            })

    return {
        'version': '2.1',
        'timestamp': f'synthetic_{datetime.now().isoformat()}',
        'samples': samples,
        'labels': labels,
        'metadata': {
            'synthetic': True,
            'generator_version': '1.0',
            'magnet_config': MAGNET_CONFIG,
            'domain_randomization': True
        }
    }
```

---

## 4. Training Pipeline Integration

### 4.1 Mixed Training Strategy

Combine synthetic and real data for best results:

```python
class MixedDataset:
    """Load both real and synthetic data for training."""

    def __init__(
        self,
        real_data_dir: str,
        synthetic_data_dir: str,
        real_ratio: float = 0.3  # 30% real, 70% synthetic
    ):
        self.real_dataset = GambitDataset(real_data_dir)
        self.synthetic_dataset = GambitDataset(synthetic_data_dir)
        self.real_ratio = real_ratio

    def load_mixed(self) -> tuple:
        """Load mixed training data."""
        X_real, y_real = self.real_dataset.load_finger_tracking_sessions()
        X_synth, y_synth = self.synthetic_dataset.load_finger_tracking_sessions()

        # Undersample synthetic to achieve target ratio
        n_real = len(X_real)
        n_synth_target = int(n_real * (1 - self.real_ratio) / self.real_ratio)

        if len(X_synth) > n_synth_target:
            indices = np.random.choice(len(X_synth), n_synth_target, replace=False)
            X_synth = X_synth[indices]
            y_synth = y_synth[indices]

        # Combine and shuffle
        X = np.concatenate([X_real, X_synth])
        y = np.concatenate([y_real, y_synth])

        shuffle_idx = np.random.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx]
```

### 4.2 Curriculum Learning

Start with synthetic data, gradually add real data:

```python
def curriculum_training(
    model,
    synthetic_data: tuple,
    real_data: tuple,
    epochs_per_stage: int = 10
):
    """
    Train with curriculum: synthetic → mixed → real.

    Stage 1: 100% synthetic (learn physics)
    Stage 2: 50% synthetic, 50% real (domain adaptation)
    Stage 3: 100% real (fine-tune)
    """
    X_synth, y_synth = synthetic_data
    X_real, y_real = real_data

    # Stage 1: Synthetic only
    print("Stage 1: Training on synthetic data...")
    model.fit(X_synth, y_synth, epochs=epochs_per_stage, validation_split=0.2)

    # Stage 2: Mixed
    print("Stage 2: Training on mixed data...")
    X_mixed = np.concatenate([X_synth[:len(X_real)], X_real])
    y_mixed = np.concatenate([y_synth[:len(y_real)], y_real])
    model.fit(X_mixed, y_mixed, epochs=epochs_per_stage, validation_split=0.2)

    # Stage 3: Real only (fine-tune)
    print("Stage 3: Fine-tuning on real data...")
    model.fit(X_real, y_real, epochs=epochs_per_stage, validation_split=0.2)

    return model
```

### 4.3 Sim-to-Real Validation

Validate that synthetic training transfers to real data:

```python
def validate_sim_to_real_transfer(
    model,
    synthetic_test: tuple,
    real_test: tuple
) -> dict:
    """
    Measure accuracy on both synthetic and real test sets.

    Good transfer: real accuracy within 10% of synthetic accuracy.
    Poor transfer: real accuracy much lower → need more domain randomization.
    """
    X_synth, y_synth = synthetic_test
    X_real, y_real = real_test

    synth_acc = model.evaluate(X_synth, y_synth)[1]  # Accuracy
    real_acc = model.evaluate(X_real, y_real)[1]

    transfer_gap = synth_acc - real_acc

    return {
        'synthetic_accuracy': synth_acc,
        'real_accuracy': real_acc,
        'transfer_gap': transfer_gap,
        'transfer_quality': 'good' if transfer_gap < 0.1 else 'needs_improvement'
    }
```

---

## 5. Implementation Roadmap

### Phase 1: Core Simulation (1-2 weeks)

**Goal**: Build basic magnetic field simulator

1. **Create `ml/simulation/` module**
   - `dipole.py`: Magnetic dipole field equations
   - `hand_model.py`: Hand geometry and pose generation
   - `sensor_model.py`: MMC5603 noise simulation
   - `generator.py`: Synthetic session generation

2. **Validate against real data**
   - Compare simulated vs real magnetic field magnitudes
   - Verify 1/r³ falloff matches theory
   - Calibrate noise parameters

**Deliverable**: Generate 10,000 synthetic samples, validate format compatibility

### Phase 2: Domain Randomization (1 week)

**Goal**: Make synthetic data realistic enough for training

1. **Implement randomization**
   - Magnet strength/position variance
   - Hand geometry variance
   - Sensor characteristic variance

2. **Measure transfer gap**
   - Train on synthetic, test on real
   - Iterate on randomization parameters

**Deliverable**: <15% accuracy gap between synthetic and real test sets

### Phase 3: Large-Scale Generation (1 week)

**Goal**: Generate production training dataset

1. **Parallelized generation**
   - GPU-accelerated field calculation (optional)
   - Multiprocessing for batch generation

2. **Dataset creation**
   - Generate 1M+ samples
   - Multiple pose categories
   - Temporal dynamics (transitions)

**Deliverable**: Production synthetic dataset in `data/GAMBIT/synthetic/`

### Phase 4: Training Integration (ongoing)

**Goal**: Integrate synthetic data into training workflow

1. **Mixed training pipeline**
   - Curriculum learning
   - Continual learning as real data grows

2. **Model improvement**
   - Compare synthetic-only vs mixed vs real-only
   - Find optimal mixture ratio

**Deliverable**: Models trained with synthetic data, validated on real data

---

## 6. Expected Benefits

| Metric | Current (Real Only) | With Simulation |
|--------|---------------------|-----------------|
| Training samples | ~10,000 | 1,000,000+ |
| Data collection time | 10+ hours | 0 (automated) |
| Pose coverage | Limited | Exhaustive |
| Label quality | Manual/noisy | Perfect ground truth |
| Model accuracy (projected) | 60-70% | 80-90% |
| Iteration speed | Days | Hours |

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Sim-to-real gap | Domain randomization, mixed training |
| Unrealistic poses | Kinematic constraints, real pose distributions |
| Physics model errors | Validate against real measurements |
| Sensor model mismatch | Calibrate from real sensor data |

---

## 7. Conclusion

Magnetic field simulation presents a **unique opportunity** for SIMCAP because:

1. **Physics is deterministic**: Unlike vision or audio, magnetic fields follow exact equations
2. **Ground truth is free**: Pose labels come directly from simulation parameters
3. **Infinite data**: Can generate millions of samples with arbitrary pose distributions
4. **Fast iteration**: Test model architectures without waiting for data collection

**Recommended Next Step**: Implement Phase 1 (core simulation) and validate against existing real session data. If simulated field magnitudes match real measurements within 20%, proceed with full pipeline.

---

## Appendix A: Magnet Specifications

### Recommended Magnets for Finger Tracking

| Size | Grade | Moment (A·m²) | Field at 50mm | Field at 80mm |
|------|-------|---------------|---------------|---------------|
| 6mm × 3mm | N48 | 0.0135 | ~140 μT | ~35 μT |
| 5mm × 2mm | N42 | 0.0065 | ~65 μT | ~16 μT |
| 3mm × 1mm | N35 | 0.0020 | ~20 μT | ~5 μT |

### Magnetic Dipole Moment Calculation

```python
def calculate_dipole_moment(
    diameter_mm: float,
    height_mm: float,
    Br_mT: float = 1430  # N48 residual flux density
) -> float:
    """
    Calculate magnetic dipole moment for a cylindrical magnet.

    Returns:
        Dipole moment in A·m²
    """
    volume_m3 = np.pi * (diameter_mm/2000)**2 * (height_mm/1000)
    Br_T = Br_mT / 1000
    M = Br_T / (4 * np.pi * 1e-7)  # Magnetization in A/m
    moment = M * volume_m3  # A·m²
    return moment
```

---

## Appendix B: Code Locations

| File | Purpose |
|------|---------|
| `ml/simulation/__init__.py` | Module initialization |
| `ml/simulation/dipole.py` | Magnetic field physics |
| `ml/simulation/hand_model.py` | Hand pose generation |
| `ml/simulation/sensor_model.py` | Magnetometer simulation |
| `ml/simulation/generator.py` | Synthetic data generation |
| `ml/simulation/randomization.py` | Domain randomization |
| `ml/train_with_synthetic.py` | Training pipeline integration |

---

**Document Version**: 1.0
**Created**: 2025-12-25
**Author**: Magnetic Field Simulation Exploration
**Related Documents**:
- `magnetic-tracking-pipeline-analysis.md` - Current pipeline status
- `magnetic-finger-tracking-analysis.md` - Physics foundation
- `calibration-filtering-guide.md` - Sensor calibration details
