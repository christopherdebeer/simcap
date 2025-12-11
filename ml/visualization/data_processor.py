"""
SIMCAP Data Processor

Handles loading and processing of sensor data from JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np


class SensorDataProcessor:
    """Processes and prepares sensor data for visualization."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sessions = []
        self.load_sessions()

    def load_sessions(self):
        """Load all JSON data files from data directory."""
        json_files = sorted(self.data_dir.glob("*.json"))
        # Exclude metadata files
        json_files = [f for f in json_files if not f.name.endswith('.meta.json')]

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if not data or not isinstance(data, list):
                    continue

                # Load metadata if available
                meta_file = json_file.with_suffix('.meta.json')
                metadata = None
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                session = {
                    'filename': json_file.name,
                    'timestamp': json_file.stem,
                    'data': data,
                    'metadata': metadata,
                    'duration': len(data) / 50.0  # 50Hz sampling
                }

                self.sessions.append(session)
                print(f"Loaded {json_file.name}: {len(data)} samples ({session['duration']:.1f}s)")

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"\nTotal sessions loaded: {len(self.sessions)}")

    def extract_sensor_arrays(self, data: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract sensor data into numpy arrays, including calibrated/fused/filtered fields."""
        # Initialize arrays
        n_samples = len(data)
        sensors = {
            # Raw IMU data
            'ax': np.zeros(n_samples),
            'ay': np.zeros(n_samples),
            'az': np.zeros(n_samples),
            'gx': np.zeros(n_samples),
            'gy': np.zeros(n_samples),
            'gz': np.zeros(n_samples),
            'mx': np.zeros(n_samples),
            'my': np.zeros(n_samples),
            'mz': np.zeros(n_samples),
            # Calibrated magnetometer (iron correction only)
            'calibrated_mx': np.zeros(n_samples),
            'calibrated_my': np.zeros(n_samples),
            'calibrated_mz': np.zeros(n_samples),
            # Fused magnetometer (iron + Earth field subtraction)
            'fused_mx': np.zeros(n_samples),
            'fused_my': np.zeros(n_samples),
            'fused_mz': np.zeros(n_samples),
            # Filtered magnetometer (Kalman smoothed)
            'filtered_mx': np.zeros(n_samples),
            'filtered_my': np.zeros(n_samples),
            'filtered_mz': np.zeros(n_samples),
            # Device orientation quaternion
            'orientation_w': np.zeros(n_samples),
            'orientation_x': np.zeros(n_samples),
            'orientation_y': np.zeros(n_samples),
            'orientation_z': np.zeros(n_samples),
            # Device orientation euler angles (degrees)
            'euler_roll': np.zeros(n_samples),
            'euler_pitch': np.zeros(n_samples),
            'euler_yaw': np.zeros(n_samples),
            # Auxiliary sensors
            'light': np.zeros(n_samples),
            'temp': np.zeros(n_samples),
            'capacitive': np.zeros(n_samples),
        }

        # Track which decorated fields are present
        has_calibrated = False
        has_fused = False
        has_filtered = False
        has_orientation = False

        for i, sample in enumerate(data):
            # Raw IMU
            sensors['ax'][i] = sample.get('ax', 0)
            sensors['ay'][i] = sample.get('ay', 0)
            sensors['az'][i] = sample.get('az', 0)
            sensors['gx'][i] = sample.get('gx', 0)
            sensors['gy'][i] = sample.get('gy', 0)
            sensors['gz'][i] = sample.get('gz', 0)
            sensors['mx'][i] = sample.get('mx', 0)
            sensors['my'][i] = sample.get('my', 0)
            sensors['mz'][i] = sample.get('mz', 0)

            # Calibrated (iron corrected)
            if 'calibrated_mx' in sample:
                has_calibrated = True
                sensors['calibrated_mx'][i] = sample.get('calibrated_mx', 0)
                sensors['calibrated_my'][i] = sample.get('calibrated_my', 0)
                sensors['calibrated_mz'][i] = sample.get('calibrated_mz', 0)

            # Fused (Earth field subtracted)
            if 'fused_mx' in sample:
                has_fused = True
                sensors['fused_mx'][i] = sample.get('fused_mx', 0)
                sensors['fused_my'][i] = sample.get('fused_my', 0)
                sensors['fused_mz'][i] = sample.get('fused_mz', 0)

            # Filtered (Kalman smoothed)
            if 'filtered_mx' in sample:
                has_filtered = True
                sensors['filtered_mx'][i] = sample.get('filtered_mx', 0)
                sensors['filtered_my'][i] = sample.get('filtered_my', 0)
                sensors['filtered_mz'][i] = sample.get('filtered_mz', 0)

            # Orientation quaternion
            if 'orientation_w' in sample:
                has_orientation = True
                sensors['orientation_w'][i] = sample.get('orientation_w', 1)
                sensors['orientation_x'][i] = sample.get('orientation_x', 0)
                sensors['orientation_y'][i] = sample.get('orientation_y', 0)
                sensors['orientation_z'][i] = sample.get('orientation_z', 0)

            # Euler angles
            if 'euler_roll' in sample:
                sensors['euler_roll'][i] = sample.get('euler_roll', 0)
                sensors['euler_pitch'][i] = sample.get('euler_pitch', 0)
                sensors['euler_yaw'][i] = sample.get('euler_yaw', 0)

            # Auxiliary
            sensors['light'][i] = sample.get('l', 0)
            sensors['temp'][i] = sample.get('t', 0)
            sensors['capacitive'][i] = sample.get('c', 0)

        # Compute magnitudes for raw magnetometer
        sensors['accel_mag'] = np.sqrt(sensors['ax']**2 + sensors['ay']**2 + sensors['az']**2)
        sensors['gyro_mag'] = np.sqrt(sensors['gx']**2 + sensors['gy']**2 + sensors['gz']**2)
        sensors['mag_mag'] = np.sqrt(sensors['mx']**2 + sensors['my']**2 + sensors['mz']**2)

        # Compute magnitudes for decorated magnetometer fields
        if has_calibrated:
            sensors['calibrated_mag'] = np.sqrt(
                sensors['calibrated_mx']**2 + sensors['calibrated_my']**2 + sensors['calibrated_mz']**2
            )
        if has_fused:
            sensors['fused_mag'] = np.sqrt(
                sensors['fused_mx']**2 + sensors['fused_my']**2 + sensors['fused_mz']**2
            )
        if has_filtered:
            sensors['filtered_mag'] = np.sqrt(
                sensors['filtered_mx']**2 + sensors['filtered_my']**2 + sensors['filtered_mz']**2
            )

        # Time axis
        sensors['time'] = np.arange(n_samples) / 50.0  # 50Hz sampling

        # Store calibration status flags
        sensors['_has_calibrated'] = has_calibrated
        sensors['_has_fused'] = has_fused
        sensors['_has_filtered'] = has_filtered
        sensors['_has_orientation'] = has_orientation

        return sensors
