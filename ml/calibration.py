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
from typing import Dict, List, Tuple, Optional, Union


def quaternion_to_rotation_matrix(q: Union[Dict, np.ndarray]) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion as dict with keys 'w', 'x', 'y', 'z' or numpy array [w, x, y, z]

    Returns:
        3x3 rotation matrix as numpy array

    Reference:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    if isinstance(q, dict):
        w, x, y, z = q['w'], q['x'], q['y'], q['z']
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R


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

    def correct(self, measurement: Dict[str, float], orientation: Optional[Union[Dict, np.ndarray]] = None) -> Dict[str, float]:
        """
        Apply all calibrations to a magnetometer reading.

        Args:
            measurement: {x, y, z} magnetometer reading
            orientation: Optional orientation quaternion (dict with w, x, y, z keys or array [w, x, y, z])
                        If provided, Earth field will be rotated to sensor frame before subtraction.
                        This is CRITICAL for accurate Earth field removal during device movement.

        Returns:
            Corrected {x, y, z} reading with iron correction and Earth field subtraction

        Note:
            Without orientation, this uses static Earth field subtraction which is only
            valid if device orientation hasn't changed since calibration. For dynamic
            tracking, always provide orientation.
        """
        # Convert to numpy vector
        m = np.array([measurement['x'], measurement['y'], measurement['z']])

        # 1. Remove hard iron offset
        m = m - self.hard_iron_offset

        # 2. Apply soft iron correction
        m = self.soft_iron_matrix @ m

        # 3. Remove Earth field (with orientation compensation if available)
        if orientation is not None:
            # Rotate Earth field from world frame to sensor frame
            # The quaternion represents sensor orientation in world frame
            # To transform a vector from world to sensor, use R^T (transpose)
            R = quaternion_to_rotation_matrix(orientation)
            earth_rotated = R.T @ self.earth_field
            m = m - earth_rotated
        else:
            # Fall back to static subtraction (only valid if orientation unchanged)
            m = m - self.earth_field

        return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}

    def correct_iron_only(self, measurement: Dict[str, float]) -> Dict[str, float]:
        """
        Apply only hard and soft iron corrections, no Earth field subtraction.

        Use this when:
        - Orientation is not available
        - You want to see iron-corrected signal before Earth compensation
        - Generating calibrated_ fields (iron only)

        Args:
            measurement: {x, y, z} magnetometer reading

        Returns:
            Iron-corrected {x, y, z} reading (Earth field still present)
        """
        # Convert to numpy vector
        m = np.array([measurement['x'], measurement['y'], measurement['z']])

        # 1. Remove hard iron offset
        m = m - self.hard_iron_offset

        # 2. Apply soft iron correction
        m = self.soft_iron_matrix @ m

        return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2])}

    def has_calibration(self, cal_type: str) -> bool:
        """Check if a specific calibration has been performed."""
        return self.calibrations.get(cal_type, False)

    def save(self, filepath: str):
        """
        Save calibration to JSON file.

        Uses a format compatible with both JS (web) and Python (ML) pipelines.
        The format uses camelCase keys for JS compatibility, but arrays for values.
        """
        # Flatten 3x3 matrix to 9-element array for JS Matrix3.fromArray()
        matrix_flat = self.soft_iron_matrix.flatten().tolist()

        data = {
            # camelCase for JS compatibility
            'hardIronOffset': {'x': float(self.hard_iron_offset[0]),
                              'y': float(self.hard_iron_offset[1]),
                              'z': float(self.hard_iron_offset[2])},
            'softIronMatrix': matrix_flat,
            'earthField': {'x': float(self.earth_field[0]),
                          'y': float(self.earth_field[1]),
                          'z': float(self.earth_field[2])},
            'earthFieldMagnitude': float(np.linalg.norm(self.earth_field)),
            'hardIronCalibrated': self.calibrations.get('hard_iron', False),
            'softIronCalibrated': self.calibrations.get('soft_iron', False),
            'earthFieldCalibrated': self.calibrations.get('earth_field', False),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load calibration from JSON file.

        Supports both formats:
        - JS format (camelCase): {hardIronOffset: {x,y,z}, softIronMatrix: [...], ...}
        - Python format (snake_case): {hard_iron_offset: [...], soft_iron_matrix: [[...]], ...}
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Detect format by checking for camelCase or snake_case keys
        if 'hardIronOffset' in data:
            # JS format (camelCase)
            offset = data['hardIronOffset']
            self.hard_iron_offset = np.array([offset['x'], offset['y'], offset['z']])

            field = data.get('earthField', {'x': 0, 'y': 0, 'z': 0})
            self.earth_field = np.array([field['x'], field['y'], field['z']])

            matrix_data = data.get('softIronMatrix', [1,0,0,0,1,0,0,0,1])
            if len(matrix_data) == 9:
                self.soft_iron_matrix = np.array(matrix_data).reshape(3, 3)
            else:
                self.soft_iron_matrix = np.eye(3)

            # Load calibration flags
            self.calibrations = {
                'earth_field': data.get('earthFieldCalibrated', False),
                'hard_iron': data.get('hardIronCalibrated', False),
                'soft_iron': data.get('softIronCalibrated', False)
            }
        else:
            # Python format (snake_case) - legacy support
            self.calibrations = data.get('calibrations', {})
            self.earth_field = np.array(data.get('earth_field', [0, 0, 0]))
            self.hard_iron_offset = np.array(data.get('hard_iron_offset', [0, 0, 0]))
            self.soft_iron_matrix = np.array(data.get('soft_iron_matrix', np.eye(3).tolist()))


def decorate_telemetry_with_calibration(telemetry_data: List[Dict],
                                       calibration: EnvironmentalCalibration,
                                       use_orientation: bool = True) -> List[Dict]:
    """
    Decorate telemetry data with calibrated and fused magnetometer fields.

    IMPORTANT: Preserves raw data, only adds decorated fields.

    This function reproduces the same calibration stages as the real-time JavaScript
    pipeline, allowing validation and post-processing of data even if real-time
    decoration wasn't applied.

    Args:
        telemetry_data: List of telemetry dictionaries with mx, my, mz fields
        calibration: EnvironmentalCalibration instance
        use_orientation: If True and orientation fields present, apply orientation-based
                        Earth subtraction. If False, use static subtraction (legacy).

    Returns:
        List of telemetry dictionaries with added fields:
        - calibrated_mx/my/mz: Iron corrected only (hard + soft iron)
        - fused_mx/my/mz: Iron + orientation-compensated Earth field subtraction
                          (only if Earth calibration and orientation available)

    Note:
        This allows the Python backend to reproduce calibration stages from raw data
        + calibration file, enabling validation even if JavaScript real-time processing
        didn't persist the decorated fields.
    """
    decorated = []

    for sample in telemetry_data:
        # Create decorated copy
        decorated_sample = sample.copy()

        # Check if we have iron calibration (minimum requirement)
        has_iron_cal = (calibration.has_calibration('hard_iron') and
                       calibration.has_calibration('soft_iron'))
        has_earth_cal = calibration.has_calibration('earth_field')

        if has_iron_cal:
            try:
                # Stage 1: Iron correction only (calibrated_ fields)
                iron_corrected = calibration.correct_iron_only({
                    'x': sample['mx'],
                    'y': sample['my'],
                    'z': sample['mz']
                })
                decorated_sample['calibrated_mx'] = iron_corrected['x']
                decorated_sample['calibrated_my'] = iron_corrected['y']
                decorated_sample['calibrated_mz'] = iron_corrected['z']

                # Stage 2: Fused (iron + Earth subtraction with orientation if available)
                if has_earth_cal:
                    orientation = None
                    if use_orientation and 'orientation_w' in sample:
                        # Extract orientation quaternion from sample
                        orientation = {
                            'w': sample['orientation_w'],
                            'x': sample['orientation_x'],
                            'y': sample['orientation_y'],
                            'z': sample['orientation_z']
                        }

                    # Apply full correction (iron + Earth field)
                    fused = calibration.correct({
                        'x': sample['mx'],
                        'y': sample['my'],
                        'z': sample['mz']
                    }, orientation=orientation)

                    decorated_sample['fused_mx'] = fused['x']
                    decorated_sample['fused_my'] = fused['y']
                    decorated_sample['fused_mz'] = fused['z']

            except Exception as e:
                # Calibration failed, skip decoration for this sample
                pass

        decorated.append(decorated_sample)

    return decorated
