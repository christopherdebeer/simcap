"""
Synthetic Training Data Generator

Combines hand pose generation, magnetic field physics, and sensor simulation
to generate complete synthetic training sessions in SIMCAP v2.1 format.

Usage:
    from ml.simulation import MagneticFieldSimulator, DEFAULT_MAGNET_CONFIG

    # Create simulator
    sim = MagneticFieldSimulator(magnet_config=DEFAULT_MAGNET_CONFIG)

    # Generate a session
    session = sim.generate_session(
        num_samples=2500,
        poses=['open_palm', 'fist', 'pointing'],
        samples_per_pose=833
    )

    # Save to file
    import json
    with open('synthetic_session.json', 'w') as f:
        json.dump(session, f)
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .dipole import compute_total_field, EARTH_FIELD_EDINBURGH
from .hand_model import (
    HandPoseGenerator, HandPose, FingerState,
    POSE_TEMPLATES, pose_template_to_states
)
from .sensor_model import MMC5603Simulator, IMUSimulator

# Try to import Magpylib for high-fidelity simulation
try:
    from .magpylib_sim import MagpylibSimulator, DEFAULT_MAGNET_SPECS
    HAS_MAGPYLIB = True
except ImportError:
    HAS_MAGPYLIB = False
    MagpylibSimulator = None


class MagneticFieldSimulator:
    """
    Complete magnetic field simulation pipeline for training data generation.

    This class orchestrates:
    1. Hand pose generation (geometry + kinematics)
    2. Magnetic field calculation (dipole physics)
    3. Sensor simulation (noise, bias, quantization)
    4. Session formatting (SIMCAP v2.1 compatible JSON)
    """

    def __init__(
        self,
        magnet_config: Dict,
        earth_field: Optional[np.ndarray] = None,
        sample_rate: float = 26.0,
        randomize_geometry: bool = False,
        randomize_sensor: bool = False,
        use_magpylib: bool = True
    ):
        """
        Initialize the magnetic field simulator.

        Args:
            magnet_config: Dict mapping finger names to magnet properties
                Each entry should have 'moment' (A·m²) and optional 'offset' (mm)
            earth_field: Earth's magnetic field vector (μT). Default: Edinburgh
            sample_rate: Sample rate in Hz
            randomize_geometry: Apply random variation to hand geometry
            randomize_sensor: Apply random variation to sensor characteristics
            use_magpylib: Use Magpylib for accurate cylinder magnet simulation
                         (falls back to dipole approximation if not available)
        """
        self.magnet_config = magnet_config
        self.earth_field = earth_field if earth_field is not None else EARTH_FIELD_EDINBURGH
        self.sample_rate = sample_rate
        self.use_magpylib = use_magpylib and HAS_MAGPYLIB

        # Initialize components
        self.hand_generator = HandPoseGenerator(randomize_geometry=randomize_geometry)
        self.mag_sensor = MMC5603Simulator()
        self.imu_sensor = IMUSimulator()

        # Initialize Magpylib simulator if available and requested
        self.magpylib_sim = None
        if self.use_magpylib:
            self.magpylib_sim = MagpylibSimulator()

        if randomize_sensor:
            self.mag_sensor.randomize_parameters()

    def compute_field_for_pose(
        self,
        pose: HandPose,
        include_earth: bool = True,
        device_orientation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute magnetic field at sensor for a given hand pose.

        Args:
            pose: HandPose with fingertip positions
            include_earth: Include Earth's magnetic field
            device_orientation: Optional 3x3 rotation matrix for device orientation

        Returns:
            Magnetic field vector at sensor in μT
        """
        # Get Earth field (optionally rotated by device orientation)
        if device_orientation is not None and include_earth:
            rotated_earth = device_orientation @ self.earth_field
        else:
            rotated_earth = self.earth_field

        # Use Magpylib if available for more accurate field calculation
        if self.use_magpylib and self.magpylib_sim is not None:
            field = self.magpylib_sim.compute_field(
                finger_positions_mm=pose.fingertip_positions,
                include_earth=include_earth,
                earth_field_ut=rotated_earth
            )
            return field

        # Fall back to dipole approximation
        return compute_total_field(
            sensor_position=np.zeros(3),
            finger_positions=pose.fingertip_positions,
            magnet_config=self.magnet_config,
            earth_field=rotated_earth,
            include_earth=include_earth
        )

    def random_rotation_matrix(self, max_angle_deg: float = 30.0) -> np.ndarray:
        """Generate a random rotation matrix for device orientation variation."""
        # Random rotation axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Random rotation angle
        angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0

        # Rodrigues rotation formula
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
        """
        Generate a single telemetry sample for a given pose.

        Returns a dict in SIMCAP v2.1 sample format.
        """
        # Random device orientation (simulates wrist movement)
        orientation = self.random_rotation_matrix(max_angle_deg=25.0) if add_orientation_variation else None

        # Compute magnetic field with orientation effects
        true_field = self.compute_field_for_pose(pose, device_orientation=orientation)

        # Simulate magnetometer reading
        mag_reading = self.mag_sensor.measure(true_field)

        # Simulate IMU reading (static)
        imu_reading = self.imu_sensor.measure_static()

        # Combine into sample
        dt = 1.0 / self.sample_rate
        sample = {
            # IMU data
            **imu_reading,

            # Magnetometer data
            **{k: v for k, v in mag_reading.items() if not k.startswith('_')},

            # Timing
            'dt': dt,
            't': sample_index * dt * 1000,  # milliseconds

            # Additional fields matching GAMBIT format
            'gyroBiasCalibrated': False,
            'mag_cal_ready': False,
            'mag_cal_confidence': 0.0,
            'mag_cal_mean_residual': None,
            'mag_cal_earth_magnitude': float(np.linalg.norm(self.earth_field)),
            'mag_cal_hard_iron': False,
            'mag_cal_soft_iron': False,

            # Filtered values (in simulation, same as raw)
            'filtered_mx': mag_reading['mx_ut'],
            'filtered_my': mag_reading['my_ut'],
            'filtered_mz': mag_reading['mz_ut'],

            # Ground truth (for validation, can be removed in production)
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
        """
        Generate samples for a static pose.

        Args:
            pose_name: Name of the pose (e.g., 'open_palm', 'fist')
            num_samples: Number of samples to generate
            position_noise_mm: Fingertip position noise (mm)
            start_index: Starting sample index for timing

        Returns:
            Tuple of (samples list, label dict)
        """
        samples = []

        for i in range(num_samples):
            pose = self.hand_generator.generate_static_pose(pose_name, position_noise_mm)
            sample = self.generate_sample(pose, start_index + i)
            samples.append(sample)

        # Create label for this segment
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

    def generate_transition_samples(
        self,
        start_pose: str,
        end_pose: str,
        num_samples: int,
        position_noise_mm: float = 1.0,
        start_index: int = 0
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate samples for a transition between poses.

        Returns:
            Tuple of (samples list, labels list)
        """
        start_states = pose_template_to_states(POSE_TEMPLATES[start_pose])
        end_states = pose_template_to_states(POSE_TEMPLATES[end_pose])

        # Generate pose sequence
        poses = self.hand_generator.generate_transition(
            start_states, end_states, num_samples, position_noise_mm
        )

        samples = []
        for i, pose in enumerate(poses):
            sample = self.generate_sample(pose, start_index + i)
            samples.append(sample)

        # Labels: start, transition, end
        labels = [
            {
                'start_sample': start_index,
                'end_sample': start_index + num_samples // 3,
                'labels': {
                    'pose': start_pose,
                    'motion': 'transition',
                    'calibration': 'none',
                    'fingers': {k: v.value for k, v in start_states.items()}
                }
            },
            {
                'start_sample': start_index + num_samples // 3,
                'end_sample': start_index + 2 * num_samples // 3,
                'labels': {
                    'pose': 'transition',
                    'motion': 'moving',
                    'calibration': 'none',
                    'fingers': {}  # Undefined during transition
                }
            },
            {
                'start_sample': start_index + 2 * num_samples // 3,
                'end_sample': start_index + num_samples,
                'labels': {
                    'pose': end_pose,
                    'motion': 'transition',
                    'calibration': 'none',
                    'fingers': {k: v.value for k, v in end_states.items()}
                }
            }
        ]

        return samples, labels

    def generate_session(
        self,
        poses: List[str],
        samples_per_pose: int = 500,
        include_transitions: bool = True,
        transition_samples: int = 50,
        position_noise_mm: float = 1.0
    ) -> Dict:
        """
        Generate a complete synthetic session with multiple poses.

        Args:
            poses: List of pose names to include
            samples_per_pose: Number of samples per static pose
            include_transitions: Generate transition samples between poses
            transition_samples: Number of samples per transition
            position_noise_mm: Fingertip position noise (mm)

        Returns:
            Complete session dict in SIMCAP v2.1 format
        """
        all_samples = []
        all_labels = []
        current_index = 0

        for i, pose in enumerate(poses):
            # Generate static samples for this pose
            samples, label = self.generate_static_samples(
                pose, samples_per_pose, position_noise_mm, current_index
            )
            all_samples.extend(samples)
            all_labels.append(label)
            current_index += samples_per_pose

            # Generate transition to next pose
            if include_transitions and i < len(poses) - 1:
                next_pose = poses[i + 1]
                trans_samples, trans_labels = self.generate_transition_samples(
                    pose, next_pose, transition_samples, position_noise_mm, current_index
                )
                all_samples.extend(trans_samples)
                all_labels.extend(trans_labels)
                current_index += transition_samples

        # Create session metadata
        session = {
            'version': '2.1',
            'timestamp': f'synthetic_{datetime.now().isoformat()}',
            'samples': all_samples,
            'labels': all_labels,
            'metadata': {
                'synthetic': True,
                'generator_version': '1.1',
                'physics_engine': 'magpylib' if self.use_magpylib else 'dipole_approximation',
                'sample_rate': self.sample_rate,
                'magnet_config': {
                    k: {
                        'moment': list(v['moment']),
                        'offset': list(v.get('offset', [0, 0, 0]))
                    }
                    for k, v in self.magnet_config.items()
                },
                'earth_field': self.earth_field.tolist(),
                'poses_included': poses,
                'samples_per_pose': samples_per_pose,
                'include_transitions': include_transitions,
                'position_noise_mm': position_noise_mm,
                'sensor_calibration': self.mag_sensor.get_calibration_info()
            }
        }

        return session


def generate_synthetic_session(
    output_path: Optional[str] = None,
    poses: Optional[List[str]] = None,
    samples_per_pose: int = 500,
    magnet_config: Optional[Dict] = None,
    randomize: bool = True
) -> Dict:
    """
    Convenience function to generate a synthetic session.

    Args:
        output_path: Path to save JSON file (optional)
        poses: List of poses to include. Default: common poses
        samples_per_pose: Samples per pose
        magnet_config: Magnet configuration. Default: alternating polarity
        randomize: Apply domain randomization

    Returns:
        Session dict
    """
    from . import DEFAULT_MAGNET_CONFIG

    if poses is None:
        poses = ['open_palm', 'fist', 'pointing', 'peace', 'thumbs_up']

    if magnet_config is None:
        magnet_config = DEFAULT_MAGNET_CONFIG

    # Create simulator with optional randomization
    sim = MagneticFieldSimulator(
        magnet_config=magnet_config,
        randomize_geometry=randomize,
        randomize_sensor=randomize
    )

    # Generate session
    session = sim.generate_session(
        poses=poses,
        samples_per_pose=samples_per_pose,
        include_transitions=True
    )

    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(session, f, indent=2)
        print(f"Saved synthetic session to {output_path}")

    return session


def generate_dataset(
    output_dir: str,
    num_sessions: int = 100,
    poses_per_session: int = 5,
    samples_per_pose: int = 500
) -> List[str]:
    """
    Generate multiple synthetic sessions for training.

    Args:
        output_dir: Directory to save sessions
        num_sessions: Number of sessions to generate
        poses_per_session: Poses per session
        samples_per_pose: Samples per pose

    Returns:
        List of generated file paths
    """
    from . import DEFAULT_MAGNET_CONFIG

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_poses = list(POSE_TEMPLATES.keys())
    generated = []

    for i in range(num_sessions):
        # Randomly select poses for this session
        poses = list(np.random.choice(all_poses, size=poses_per_session, replace=False))

        # Generate with randomization
        sim = MagneticFieldSimulator(
            magnet_config=DEFAULT_MAGNET_CONFIG,
            randomize_geometry=True,
            randomize_sensor=True
        )

        session = sim.generate_session(
            poses=poses,
            samples_per_pose=samples_per_pose,
            include_transitions=True
        )

        # Save
        filename = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}.json"
        filepath = output_path / filename

        with open(filepath, 'w') as f:
            json.dump(session, f)

        generated.append(str(filepath))

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_sessions} sessions")

    print(f"Generated {len(generated)} sessions in {output_dir}")
    return generated


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output', '-o', type=str, default='synthetic_session.json',
                        help='Output file path')
    parser.add_argument('--poses', '-p', nargs='+', default=['open_palm', 'fist', 'pointing'],
                        help='Poses to include')
    parser.add_argument('--samples', '-n', type=int, default=500,
                        help='Samples per pose')
    parser.add_argument('--no-randomize', action='store_true',
                        help='Disable domain randomization')

    args = parser.parse_args()

    print(f"Generating synthetic session with poses: {args.poses}")
    session = generate_synthetic_session(
        output_path=args.output,
        poses=args.poses,
        samples_per_pose=args.samples,
        randomize=not args.no_randomize
    )

    print(f"Generated {len(session['samples'])} samples")
    print(f"Labels: {len(session['labels'])}")
