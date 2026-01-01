#!/usr/bin/env python3
"""
Parameterized Synthetic Data Generator for Finger State Inference

This module extends the existing simulation infrastructure to support:
1. Different magnet sizes and grades
2. Empirically calibrated field strengths matching real observations
3. Easy configuration for testing reduced magnet strength scenarios

The key insight from validation: real-world effective magnetic fields are ~16-20%
of theoretical dipole calculations due to geometry, multi-finger interference,
and device effects. This module includes calibration factors based on observed data.

Usage:
    from ml.simulation.parameterized_generator import ParameterizedGenerator

    # Current setup (6x3mm N48)
    gen = ParameterizedGenerator.current_setup()

    # Reduced magnet scenario (5x2mm N42)
    gen = ParameterizedGenerator.reduced_magnet(diameter=5, height=2, grade='N42')

    # Generate training data
    session = gen.generate_session(poses=['open_palm', 'fist', 'pointing'])

Author: Physics Simulation Data Generation Task
Date: 2025-01-01
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

# Import existing simulation components
from .dipole import (
    MU_0, MU_0_OVER_4PI, EARTH_FIELD_EDINBURGH,
    magnetic_dipole_field, compute_total_field
)
from .hand_model import (
    HandPoseGenerator, HandPose, FingerState,
    POSE_TEMPLATES, pose_template_to_states
)
from .sensor_model import MMC5603Simulator, IMUSimulator, SensorCharacteristics


# ============================================================================
# MAGNET CONFIGURATION
# ============================================================================

# Empirical calibration factor based on real data validation
# Normal sessions show P50 ≈ 65-75 μT (compared to Earth field ~50 μT)
# Deviation from Earth is ~15-25 μT, requiring moment scaling to match
#
# Calibration analysis (Dec 19 sessions):
# - P50 magnitude: 65-75 μT (deviation from Earth: ~15-25 μT)
# - Theoretical single-dipole at 50mm: ~154 μT
# - Multi-finger with alternating polarity causes partial cancellation
# - Effective moment scale: ~25% gives P50 ≈ 68 μT (matching observations)
EMPIRICAL_MOMENT_SCALE = 0.25  # Calibrated to match Dec 19 session data

MAGNET_GRADES = {
    'N35': 1170,  # mT
    'N38': 1250,
    'N42': 1320,
    'N45': 1370,
    'N48': 1430,
    'N50': 1450,
    'N52': 1480,
}


@dataclass
class MagnetConfig:
    """Configuration for a finger magnet."""
    diameter_mm: float
    height_mm: float
    grade: str = 'N48'
    polarity: int = 1  # +1 = north toward palm, -1 = north away

    @property
    def Br_mT(self) -> float:
        return MAGNET_GRADES.get(self.grade, 1430)

    @property
    def volume_m3(self) -> float:
        radius_m = (self.diameter_mm / 2) / 1000.0
        height_m = self.height_mm / 1000.0
        return np.pi * radius_m**2 * height_m

    @property
    def theoretical_moment(self) -> float:
        """Theoretical dipole moment (A·m²) from remanence formula."""
        Br_T = self.Br_mT / 1000.0
        M = Br_T / MU_0
        return M * self.volume_m3

    @property
    def effective_moment(self) -> float:
        """Empirically calibrated effective moment (A·m²)."""
        return self.theoretical_moment * EMPIRICAL_MOMENT_SCALE

    def moment_vector(self, scale: float = 1.0) -> np.ndarray:
        """
        Get moment vector (z-oriented with polarity).

        Args:
            scale: Optional additional scaling factor

        Returns:
            3D moment vector in A·m²
        """
        m = self.effective_moment * self.polarity * scale
        return np.array([0.0, 0.0, m])


@dataclass
class HandMagnetSetup:
    """Complete magnet configuration for a hand (5 fingers)."""
    thumb: MagnetConfig
    index: MagnetConfig
    middle: MagnetConfig
    ring: MagnetConfig
    pinky: MagnetConfig

    # Magnet attachment offset from fingertip (mm)
    # Positive X = toward thumb, positive Y = toward fingertip, positive Z = toward palm
    attachment_offsets: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'thumb': np.array([0.0, -15.0, 0.0]),   # Middle phalanx
        'index': np.array([0.0, -20.0, 0.0]),
        'middle': np.array([0.0, -22.0, 0.0]),
        'ring': np.array([0.0, -20.0, 0.0]),
        'pinky': np.array([0.0, -18.0, 0.0]),
    })

    @classmethod
    def uniform(
        cls,
        diameter_mm: float = 6.0,
        height_mm: float = 3.0,
        grade: str = 'N48',
        alternating_polarity: bool = True
    ) -> 'HandMagnetSetup':
        """
        Create setup with identical magnets on all fingers.

        Args:
            diameter_mm: Magnet diameter
            height_mm: Magnet height/thickness
            grade: Magnet grade (N35-N52)
            alternating_polarity: Use alternating polarity pattern (recommended)

        Returns:
            HandMagnetSetup instance
        """
        polarities = [1, -1, 1, -1, 1] if alternating_polarity else [1, 1, 1, 1, 1]

        return cls(
            thumb=MagnetConfig(diameter_mm, height_mm, grade, polarities[0]),
            index=MagnetConfig(diameter_mm, height_mm, grade, polarities[1]),
            middle=MagnetConfig(diameter_mm, height_mm, grade, polarities[2]),
            ring=MagnetConfig(diameter_mm, height_mm, grade, polarities[3]),
            pinky=MagnetConfig(diameter_mm, height_mm, grade, polarities[4]),
        )

    @classmethod
    def current_setup(cls) -> 'HandMagnetSetup':
        """Current production setup: 6x3mm N48 with alternating polarity."""
        return cls.uniform(6.0, 3.0, 'N48', alternating_polarity=True)

    @classmethod
    def reduced_v1(cls) -> 'HandMagnetSetup':
        """Reduced size v1: 5x2mm N42."""
        return cls.uniform(5.0, 2.0, 'N42', alternating_polarity=True)

    @classmethod
    def reduced_v2(cls) -> 'HandMagnetSetup':
        """Reduced size v2: 4x2mm N38."""
        return cls.uniform(4.0, 2.0, 'N38', alternating_polarity=True)

    @classmethod
    def minimal(cls) -> 'HandMagnetSetup':
        """Minimal viable: 3x1mm N35."""
        return cls.uniform(3.0, 1.0, 'N35', alternating_polarity=True)

    def get_config(self, finger: str) -> MagnetConfig:
        return getattr(self, finger)

    def get_offset(self, finger: str) -> np.ndarray:
        return self.attachment_offsets.get(finger, np.zeros(3))

    def to_legacy_format(self) -> Dict[str, Dict]:
        """Convert to format expected by existing compute_total_field()."""
        return {
            finger: {
                'moment': self.get_config(finger).moment_vector().tolist(),
                'offset': self.get_offset(finger).tolist()
            }
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']
        }

    def summary(self) -> str:
        cfg = self.thumb  # Assume uniform
        total_moment = sum(
            abs(self.get_config(f).effective_moment)
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        )
        return (
            f"{cfg.diameter_mm}×{cfg.height_mm}mm {cfg.grade}, "
            f"m_eff={cfg.effective_moment:.4f} A·m² per finger, "
            f"total={total_moment:.4f} A·m²"
        )


# ============================================================================
# PARAMETERIZED GENERATOR
# ============================================================================

class ParameterizedGenerator:
    """
    Generate synthetic training data with configurable magnet parameters.

    This extends MagneticFieldSimulator with:
    - Empirically calibrated field strengths
    - Easy configuration for different magnet setups
    - Comparison metrics between setups
    """

    def __init__(
        self,
        magnet_setup: HandMagnetSetup,
        earth_field: np.ndarray = None,
        sample_rate: float = 26.0,
        randomize_geometry: bool = True,
        randomize_sensor: bool = True,
        sensor_noise_ut: float = None,
    ):
        """
        Initialize parameterized generator.

        Args:
            magnet_setup: Hand magnet configuration
            earth_field: Earth field vector (μT). Default: Edinburgh
            sample_rate: Sample rate in Hz
            randomize_geometry: Apply random hand geometry variation
            randomize_sensor: Apply random sensor characteristics
            sensor_noise_ut: Override sensor noise level (μT RMS)
        """
        self.magnet_setup = magnet_setup
        self.earth_field = earth_field if earth_field is not None else EARTH_FIELD_EDINBURGH
        self.sample_rate = sample_rate

        # Initialize hand pose generator
        self.hand_generator = HandPoseGenerator(randomize_geometry=randomize_geometry)

        # Initialize sensor simulator
        self.mag_sensor = MMC5603Simulator()
        if randomize_sensor:
            self.mag_sensor.randomize_parameters(realistic_mode=True)
        if sensor_noise_ut is not None:
            self.mag_sensor._noise_ut = sensor_noise_ut

        self.imu_sensor = IMUSimulator()

        # Legacy format for existing dipole functions
        self._legacy_config = magnet_setup.to_legacy_format()

    @classmethod
    def current_setup(cls, **kwargs) -> 'ParameterizedGenerator':
        """Create generator with current production magnet setup."""
        return cls(HandMagnetSetup.current_setup(), **kwargs)

    @classmethod
    def reduced_magnet(
        cls,
        diameter: float = 5.0,
        height: float = 2.0,
        grade: str = 'N42',
        **kwargs
    ) -> 'ParameterizedGenerator':
        """Create generator with custom reduced magnet configuration."""
        setup = HandMagnetSetup.uniform(diameter, height, grade)
        return cls(setup, **kwargs)

    def compute_field_for_pose(
        self,
        pose: HandPose,
        include_earth: bool = True,
        device_orientation: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute magnetic field at sensor for a given hand pose.

        Args:
            pose: HandPose with fingertip positions
            include_earth: Include Earth's magnetic field
            device_orientation: Optional rotation matrix for device orientation

        Returns:
            Magnetic field vector at sensor in μT
        """
        # Apply device orientation to Earth field if provided
        if device_orientation is not None and include_earth:
            rotated_earth = device_orientation @ self.earth_field
        else:
            rotated_earth = self.earth_field

        # Use existing compute_total_field with legacy config
        return compute_total_field(
            sensor_position=np.zeros(3),
            finger_positions=pose.fingertip_positions,
            magnet_config=self._legacy_config,
            earth_field=rotated_earth,
            include_earth=include_earth
        )

    def random_rotation_matrix(self, max_angle_deg: float = 30.0) -> np.ndarray:
        """Generate random rotation matrix for orientation variation."""
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def generate_sample(
        self,
        pose: HandPose,
        sample_index: int = 0,
        add_orientation_variation: bool = True
    ) -> Dict:
        """Generate a single telemetry sample."""
        orientation = self.random_rotation_matrix() if add_orientation_variation else None
        true_field = self.compute_field_for_pose(pose, device_orientation=orientation)

        # Simulate sensor reading
        mag_reading = self.mag_sensor.measure(true_field)
        imu_reading = self.imu_sensor.measure_static()

        dt = 1.0 / self.sample_rate
        sample = {
            **imu_reading,
            **{k: v for k, v in mag_reading.items() if not k.startswith('_')},
            'dt': dt,
            't': sample_index * dt * 1000,
            'filtered_mx': mag_reading['mx_ut'],
            'filtered_my': mag_reading['my_ut'],
            'filtered_mz': mag_reading['mz_ut'],
            '_ground_truth': {
                'finger_states': {k: v.value for k, v in pose.finger_states.items()},
                'true_field': true_field.tolist()
            }
        }
        return sample

    def generate_static_samples(
        self,
        pose_name: str,
        num_samples: int,
        position_noise_mm: float = 1.0,
        start_index: int = 0
    ) -> Tuple[List[Dict], Dict]:
        """Generate samples for a static pose."""
        samples = []
        for i in range(num_samples):
            pose = self.hand_generator.generate_static_pose(pose_name, position_noise_mm)
            sample = self.generate_sample(pose, start_index + i)
            samples.append(sample)

        pose_template = POSE_TEMPLATES.get(pose_name, {})
        finger_states = pose_template_to_states(pose_template) if pose_template else {}

        label = {
            'start_sample': start_index,
            'end_sample': start_index + num_samples,
            'labels': {
                'pose': pose_name,
                'motion': 'static',
                'calibration': 'none',
                'fingers': {k: v.value for k, v in finger_states.items()}
            }
        }
        return samples, label

    def generate_session(
        self,
        poses: List[str] = None,
        samples_per_pose: int = 500,
        include_transitions: bool = True,
        position_noise_mm: float = 1.0
    ) -> Dict:
        """
        Generate a complete synthetic session.

        Args:
            poses: List of pose names. Default: common poses
            samples_per_pose: Samples per static pose
            include_transitions: Include transition samples between poses
            position_noise_mm: Fingertip position noise (mm)

        Returns:
            Session dict in SIMCAP v2.1 format
        """
        if poses is None:
            poses = ['open_palm', 'fist', 'pointing', 'peace', 'thumbs_up', 'rest']

        all_samples = []
        all_labels = []
        current_index = 0

        for pose in poses:
            samples, label = self.generate_static_samples(
                pose, samples_per_pose, position_noise_mm, current_index
            )
            all_samples.extend(samples)
            all_labels.append(label)
            current_index += samples_per_pose

        session = {
            'version': '2.1',
            'timestamp': f'synthetic_{datetime.now().isoformat()}',
            'samples': all_samples,
            'labels': all_labels,
            'metadata': {
                'synthetic': True,
                'generator_version': '2.0-parameterized',
                'magnet_setup': self.magnet_setup.summary(),
                'sample_rate': self.sample_rate,
                'earth_field': self.earth_field.tolist(),
                'poses_included': poses,
                'samples_per_pose': samples_per_pose,
                'sensor_calibration': self.mag_sensor.get_calibration_info()
            }
        }
        return session

    def expected_field_range(self) -> Dict[str, float]:
        """
        Calculate expected field magnitude range for this magnet setup.

        Returns:
            Dict with min/max/typical field magnitudes in μT
        """
        cfg = self.magnet_setup.thumb
        moment = cfg.effective_moment

        # Field at various distances (single magnet, on-axis)
        def field_at_dist(d_mm):
            r_m = d_mm / 1000.0
            return MU_0_OVER_4PI * 2 * moment / (r_m**3) * 1e6

        earth_mag = np.linalg.norm(self.earth_field)

        # Extended fingers: 70-90mm, Flexed: 40-60mm
        B_extended = field_at_dist(80)
        B_flexed = field_at_dist(50)

        return {
            'single_magnet_extended_ut': B_extended,
            'single_magnet_flexed_ut': B_flexed,
            'signal_delta_ut': B_flexed - B_extended,
            'earth_field_ut': earth_mag,
            'expected_min_ut': earth_mag * 0.8,  # With cancellation
            'expected_max_ut': earth_mag + B_flexed * 3,  # With reinforcement
            'snr_delta': (B_flexed - B_extended) / 1.0,  # Assuming 1 μT noise
        }


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_magnet_setups(
    setups: List[Tuple[str, HandMagnetSetup]],
    num_samples: int = 500
) -> Dict:
    """
    Compare field distributions from different magnet setups.

    Args:
        setups: List of (name, HandMagnetSetup) tuples
        num_samples: Samples to generate per setup

    Returns:
        Comparison results
    """
    results = []

    for name, setup in setups:
        gen = ParameterizedGenerator(setup, randomize_geometry=True, randomize_sensor=False)

        # Generate samples across poses
        magnitudes = []
        for pose_name in ['open_palm', 'fist', 'pointing']:
            samples, _ = gen.generate_static_samples(pose_name, num_samples // 3)
            for s in samples:
                mag = np.sqrt(s['mx_ut']**2 + s['my_ut']**2 + s['mz_ut']**2)
                magnitudes.append(mag)

        mags = np.array(magnitudes)
        expected = gen.expected_field_range()

        results.append({
            'name': name,
            'setup': setup.summary(),
            'n_samples': len(mags),
            'magnitude_mean': float(np.mean(mags)),
            'magnitude_std': float(np.std(mags)),
            'magnitude_p5': float(np.percentile(mags, 5)),
            'magnitude_p50': float(np.percentile(mags, 50)),
            'magnitude_p95': float(np.percentile(mags, 95)),
            'expected_snr_delta': expected['snr_delta'],
        })

    return {'comparison': results}


def generate_training_dataset(
    output_dir: str,
    magnet_setup: HandMagnetSetup = None,
    num_sessions: int = 100,
    poses_per_session: int = 6,
    samples_per_pose: int = 400
) -> List[str]:
    """
    Generate a complete training dataset.

    Args:
        output_dir: Output directory
        magnet_setup: Magnet configuration (default: current setup)
        num_sessions: Number of sessions to generate
        poses_per_session: Poses per session
        samples_per_pose: Samples per pose

    Returns:
        List of generated file paths
    """
    if magnet_setup is None:
        magnet_setup = HandMagnetSetup.current_setup()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_poses = list(POSE_TEMPLATES.keys())
    generated = []

    for i in range(num_sessions):
        # Random pose selection
        poses = list(np.random.choice(all_poses, size=poses_per_session, replace=False))

        # New generator with randomization for each session
        gen = ParameterizedGenerator(
            magnet_setup,
            randomize_geometry=True,
            randomize_sensor=True
        )

        session = gen.generate_session(
            poses=poses,
            samples_per_pose=samples_per_pose
        )

        filename = f"synthetic_{magnet_setup.thumb.grade}_{i:04d}.json"
        filepath = output_path / filename

        with open(filepath, 'w') as f:
            json.dump(session, f)

        generated.append(str(filepath))

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_sessions} sessions")

    print(f"Generated {len(generated)} sessions in {output_dir}")
    print(f"Magnet setup: {magnet_setup.summary()}")
    return generated


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Parameterized Synthetic Data Generator'
    )
    parser.add_argument('--compare', action='store_true',
                        help='Compare different magnet setups')
    parser.add_argument('--generate', action='store_true',
                        help='Generate training dataset')
    parser.add_argument('--output-dir', '-o', type=str, default='synthetic_data',
                        help='Output directory for generated data')
    parser.add_argument('--magnet-size', type=str, default='6x3',
                        help='Magnet size as DxH (e.g., 5x2)')
    parser.add_argument('--grade', type=str, default='N48',
                        help='Magnet grade (N35-N52)')
    parser.add_argument('--num-sessions', '-n', type=int, default=100,
                        help='Number of sessions to generate')

    args = parser.parse_args()

    if args.compare:
        print("=" * 70)
        print("MAGNET SETUP COMPARISON")
        print("=" * 70)

        setups = [
            ('Current (6x3 N48)', HandMagnetSetup.current_setup()),
            ('Reduced v1 (5x2 N42)', HandMagnetSetup.reduced_v1()),
            ('Reduced v2 (4x2 N38)', HandMagnetSetup.reduced_v2()),
            ('Minimal (3x1 N35)', HandMagnetSetup.minimal()),
        ]

        print("\nGenerating samples for each configuration...")
        results = compare_magnet_setups(setups, num_samples=600)

        print("\nResults:")
        print(f"{'Name':20s} {'P50 (μT)':10s} {'P95 (μT)':10s} {'SNR':8s}")
        print("-" * 50)
        for r in results['comparison']:
            print(f"{r['name']:20s} {r['magnitude_p50']:10.1f} {r['magnitude_p95']:10.1f} {r['expected_snr_delta']:8.1f}")

    if args.generate:
        size_parts = args.magnet_size.split('x')
        diameter = float(size_parts[0])
        height = float(size_parts[1])

        setup = HandMagnetSetup.uniform(diameter, height, args.grade)

        print(f"\nGenerating dataset with {setup.summary()}")
        generate_training_dataset(
            args.output_dir,
            magnet_setup=setup,
            num_sessions=args.num_sessions
        )


if __name__ == '__main__':
    main()
