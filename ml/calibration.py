"""
Magnetometer Calibration for Magnetic Finger Tracking

Provides environmental calibration utilities to compensate for:
1. Earth's magnetic field (constant background field)
2. Hard iron distortion (constant offset from nearby ferromagnetic materials)
3. Soft iron distortion (field distortion from conductive materials)

Python equivalent of calibration.js for ML pipeline consistency.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class EnvironmentalCalibration:
    """
    Environmental calibration for magnetometer data.

    Handles three types of calibration:
    - Earth field: Background magnetic field
    - Hard iron: Constant offset correction
    - Soft iron: Ellipsoid-to-sphere transformation
    """

    def __init__(self):
        self.calibrations = {}
        self.earth_field = np.zeros(3)
        self.hard_iron_offset = np.zeros(3)
        self.soft_iron_matrix = np.eye(3)

    def run_earth_field_calibration(self, samples: List[Dict[str, float]]) -> Dict:
        """
        Calibrate for Earth's magnetic field.

        Requires: 50+ samples in reference orientation (hand still, away from magnets)

        Args:
            samples: List of {x, y, z} magnetometer readings

        Returns:
            dict with 'field' and 'quality' keys
        """
        if len(samples) < 50:
            raise ValueError(f"Need at least 50 samples for earth field calibration, got {len(samples)}")

        # Convert to numpy array
        data = np.array([[s['x'], s['y'], s['z']] for s in samples])

        # Earth field is the mean of all samples
        self.earth_field = np.mean(data, axis=0)

        # Quality metric: lower standard deviation = better (more stable)
        std = np.std(data, axis=0)
        quality = 1.0 / (1.0 + np.mean(std))

        self.calibrations['earth_field'] = True

        return {
            'field': {'x': float(self.earth_field[0]),
                     'y': float(self.earth_field[1]),
                     'z': float(self.earth_field[2])},
            'quality': float(quality)
        }

    def run_hard_iron_calibration(self, samples: List[Dict[str, float]]) -> Dict:
        """
        Calibrate for hard iron distortion.

        Requires: 100+ samples collected while rotating sensor in all directions

        Args:
            samples: List of {x, y, z} magnetometer readings

        Returns:
            dict with 'offset', 'quality', and metrics
        """
        if len(samples) < 100:
            raise ValueError(f"Need at least 100 samples for hard iron calibration, got {len(samples)}")

        # Convert to numpy array
        data = np.array([[s['x'], s['y'], s['z']] for s in samples])

        # Hard iron offset is the center of the ellipsoid
        # Simple approach: use min/max midpoint for each axis
        offset = (np.max(data, axis=0) + np.min(data, axis=0)) / 2.0
        self.hard_iron_offset = offset

        # Quality metrics
        # 1. Sphericity: how close to a sphere (vs ellipsoid)
        centered = data - offset
        radii = np.linalg.norm(centered, axis=1)
        mean_radius = np.mean(radii)
        std_radius = np.std(radii)
        sphericity = 1.0 - (std_radius / (mean_radius + 1e-6))

        # 2. Coverage: angular coverage of samples
        # Check how well we covered different directions
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        phi = np.arccos(centered[:, 2] / (radii + 1e-6))

        # Divide sphere into bins and check coverage
        theta_bins = np.linspace(-np.pi, np.pi, 12)
        phi_bins = np.linspace(0, np.pi, 6)
        coverage = 0
        total_bins = len(theta_bins) * len(phi_bins)

        for i in range(len(theta_bins) - 1):
            for j in range(len(phi_bins) - 1):
                in_bin = ((theta >= theta_bins[i]) & (theta < theta_bins[i+1]) &
                         (phi >= phi_bins[j]) & (phi < phi_bins[j+1]))
                if np.any(in_bin):
                    coverage += 1

        coverage_ratio = coverage / total_bins
        quality = (sphericity + coverage_ratio) / 2.0

        self.calibrations['hard_iron'] = True

        return {
            'offset': {'x': float(offset[0]), 'y': float(offset[1]), 'z': float(offset[2])},
            'quality': float(quality),
            'sphericity': float(sphericity),
            'coverage': float(coverage_ratio)
        }

    def run_soft_iron_calibration(self, samples: List[Dict[str, float]]) -> Dict:
        """
        Calibrate for soft iron distortion.

        Requires: 200+ samples collected while rotating sensor

        Args:
            samples: List of {x, y, z} magnetometer readings

        Returns:
            dict with 'matrix' and 'quality' keys
        """
        if len(samples) < 200:
            raise ValueError(f"Need at least 200 samples for soft iron calibration, got {len(samples)}")

        # Convert to numpy array
        data = np.array([[s['x'], s['y'], s['z']] for s in samples])

        # Remove hard iron offset first
        centered = data - self.hard_iron_offset

        # Soft iron correction: transform ellipsoid to sphere
        # Use covariance-based approach
        cov = np.cov(centered.T)

        # Decompose: cov = U * S * U.T
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Correction matrix: scale each axis by 1/sqrt(eigenvalue)
        # This makes the ellipsoid spherical
        scale = np.diag(1.0 / np.sqrt(eigenvalues + 1e-6))
        self.soft_iron_matrix = eigenvectors @ scale @ eigenvectors.T

        # Quality: how ellipsoidal was the data
        # Higher eigenvalue variance = more ellipsoidal = worse quality before correction
        eigenvalue_ratio = np.min(eigenvalues) / (np.max(eigenvalues) + 1e-6)
        quality = eigenvalue_ratio  # Closer to 1.0 = more spherical = better

        self.calibrations['soft_iron'] = True

        return {
            'matrix': self.soft_iron_matrix.tolist(),
            'quality': float(quality)
        }

    def correct(self, measurement: Dict[str, float]) -> Dict[str, float]:
        """
        Apply all calibrations to a magnetometer reading.

        Args:
            measurement: {x, y, z} magnetometer reading

        Returns:
            Corrected {x, y, z} reading
        """
        # Convert to numpy vector
        m = np.array([measurement['x'], measurement['y'], measurement['z']])

        # 1. Remove hard iron offset
        m = m - self.hard_iron_offset

        # 2. Apply soft iron correction
        m = self.soft_iron_matrix @ m

        # 3. Remove earth field
        m = m - self.earth_field

        return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}

    def has_calibration(self, cal_type: str) -> bool:
        """Check if a specific calibration has been performed."""
        return self.calibrations.get(cal_type, False)

    def save(self, filepath: str):
        """Save calibration to JSON file."""
        data = {
            'calibrations': self.calibrations,
            'earth_field': self.earth_field.tolist(),
            'hard_iron_offset': self.hard_iron_offset.tolist(),
            'soft_iron_matrix': self.soft_iron_matrix.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load calibration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.calibrations = data['calibrations']
        self.earth_field = np.array(data['earth_field'])
        self.hard_iron_offset = np.array(data['hard_iron_offset'])
        self.soft_iron_matrix = np.array(data['soft_iron_matrix'])


def decorate_telemetry_with_calibration(telemetry_data: List[Dict],
                                       calibration: EnvironmentalCalibration) -> List[Dict]:
    """
    Decorate telemetry data with calibrated magnetometer fields.

    IMPORTANT: Preserves raw data, only adds calibrated_ fields.

    Args:
        telemetry_data: List of telemetry dictionaries with mx, my, mz fields
        calibration: EnvironmentalCalibration instance

    Returns:
        List of telemetry dictionaries with added calibrated_mx, calibrated_my, calibrated_mz fields
    """
    decorated = []

    for sample in telemetry_data:
        # Create decorated copy
        decorated_sample = sample.copy()

        # Apply calibration if available
        if (calibration.has_calibration('earth_field') and
            calibration.has_calibration('hard_iron') and
            calibration.has_calibration('soft_iron')):

            try:
                corrected = calibration.correct({
                    'x': sample['mx'],
                    'y': sample['my'],
                    'z': sample['mz']
                })
                decorated_sample['calibrated_mx'] = corrected['x']
                decorated_sample['calibrated_my'] = corrected['y']
                decorated_sample['calibrated_mz'] = corrected['z']
            except Exception as e:
                # Calibration failed, skip decoration
                pass

        decorated.append(decorated_sample)

    return decorated
