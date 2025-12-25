"""
Magnetic Field Simulation Module

Physics-based simulation of finger magnet fields for synthetic training data generation.
Uses magnetic dipole equations to compute sensor readings from arbitrary hand poses.

Usage:
    from ml.simulation import MagneticFieldSimulator, HandPoseGenerator

    # Create simulator with magnet configuration
    simulator = MagneticFieldSimulator(magnet_config=DEFAULT_MAGNET_CONFIG)

    # Generate synthetic session
    session = simulator.generate_session(
        num_samples=2500,
        pose_sequence=['open_palm', 'fist', 'pointing'],
        sample_rate=26.0
    )
"""

from .dipole import magnetic_dipole_field, compute_total_field
from .hand_model import HandPoseGenerator, FingerState, HandPose
from .sensor_model import MMC5603Simulator
from .generator import MagneticFieldSimulator, generate_synthetic_session

# Default magnet configuration with alternating polarity
DEFAULT_MAGNET_CONFIG = {
    'thumb': {
        'moment': [0, 0, 0.0135],     # N48 6x3mm, N toward palm (+Z)
        'offset': [0, 0, 0]           # Attachment offset from fingertip
    },
    'index': {
        'moment': [0, 0, -0.0135],    # N away from palm (-Z)
        'offset': [0, 0, 0]
    },
    'middle': {
        'moment': [0, 0, 0.0135],     # N toward palm
        'offset': [0, 0, 0]
    },
    'ring': {
        'moment': [0, 0, -0.0135],    # N away from palm
        'offset': [0, 0, 0]
    },
    'pinky': {
        'moment': [0, 0, 0.0135],     # N toward palm
        'offset': [0, 0, 0]
    }
}

__all__ = [
    'magnetic_dipole_field',
    'compute_total_field',
    'HandPoseGenerator',
    'FingerState',
    'HandPose',
    'MMC5603Simulator',
    'MagneticFieldSimulator',
    'generate_synthetic_session',
    'DEFAULT_MAGNET_CONFIG'
]
