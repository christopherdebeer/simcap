#!/usr/bin/env python3
"""
SIMCAP Data Visualization Pipeline

Generates comprehensive visualizations from GAMBIT sensor data:
- Composite session images
- Per-second window images
- Raw axis/orientation plots
- Interactive HTML explorer

Usage:
    python visualize.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime
import colorsys

# Import schema if available
try:
    from schema import Gesture
    GESTURE_NAMES = {g.value: g.name for g in Gesture}
except ImportError:
    GESTURE_NAMES = {
        0: "REST", 1: "FIST", 2: "OPEN_PALM", 3: "INDEX_UP", 4: "PEACE",
        5: "THUMBS_UP", 6: "OK_SIGN", 7: "PINCH", 8: "GRAB", 9: "WAVE"
    }


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

                # Handle v1, v2.0, and v2.1 formats
                version = '1.0'
                labels = []
                firmware_version = None
                session_type = 'recording'
                custom_labels = []
                calibration_types = []
                metadata = {}

                if isinstance(data, dict) and 'samples' in data:
                    # New format with wrapper (v2.0 or v2.1)
                    samples = data['samples']
                    metadata = data.get('metadata', {})
                    version = data.get('version', '2.0')
                    labels = data.get('labels', [])

                    # Extract v2.1 fields from embedded metadata
                    if metadata:
                        firmware_version = metadata.get('firmware_version')
                        session_type = metadata.get('session_type', 'recording')

                    # Extract custom labels and calibration types from labels
                    for label in labels:
                        if 'labels' in label:
                            lab = label['labels']
                            if lab.get('custom'):
                                custom_labels.extend(lab['custom'])
                            if lab.get('calibration') and lab['calibration'] != 'none':
                                calibration_types.append(lab['calibration'])

                elif isinstance(data, list):
                    # Old format: direct array
                    samples = data
                    # Load metadata from separate file if available
                    meta_file = json_file.with_suffix('.meta.json')
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                else:
                    print(f"Skipping {json_file.name}: unexpected format")
                    continue

                if not samples or not isinstance(samples, list):
                    print(f"Skipping {json_file.name}: no valid samples")
                    continue

                # Determine actual sample rate from metadata or data
                sample_rate = metadata.get('sample_rate', 50) if metadata else 50
                # Use dt from first sample if available for more accurate rate
                if samples and 'dt' in samples[0]:
                    dt = samples[0].get('dt', 0.02)
                    if dt > 0:
                        sample_rate = 1.0 / dt

                session = {
                    'filename': json_file.name,
                    'timestamp': json_file.stem,
                    'data': samples,
                    'metadata': metadata,
                    'duration': len(samples) / sample_rate,
                    'sample_rate': sample_rate,
                    # V2.1 extended fields
                    'version': version,
                    'firmware_version': firmware_version,
                    'session_type': session_type,
                    'labels': labels,
                    'custom_labels': list(set(custom_labels)),
                    'calibration_types': list(set(calibration_types)),
                    # Extended metadata fields for VIZ display
                    'device': metadata.get('device', 'unknown') if metadata else 'unknown',
                    'subject_id': metadata.get('subject_id', '') if metadata else '',
                    'environment': metadata.get('environment', '') if metadata else '',
                    'hand': metadata.get('hand', '') if metadata else '',
                    'magnet_config': metadata.get('magnet_config', '') if metadata else '',
                    'magnet_type': metadata.get('magnet_type', '') if metadata else '',
                    'notes': metadata.get('notes', '') if metadata else '',
                    'location': metadata.get('location', {}) if metadata else {},
                    'calibration_state': metadata.get('calibration', {}) if metadata else {},
                }

                self.sessions.append(session)
                fw_info = f" | fw:{firmware_version}" if firmware_version else ""
                print(f"Loaded {json_file.name}: {len(samples)} samples ({session['duration']:.1f}s @ {sample_rate:.0f}Hz){fw_info}")

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"\nTotal sessions loaded: {len(self.sessions)}")

    def extract_sensor_arrays(self, data: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract sensor data into numpy arrays, including calibrated/fused/filtered fields.

        Physical Units (per unified-mag-calibration.js and sensor-config.js):
        - Accelerometer: g (1g = 9.81 m/s²)
        - Gyroscope: °/s (degrees per second)
        - Magnetometer: µT (microtesla)

        Magnetometer Calibration Stages (per magnetometer-calibration-complete-analysis.md):
        1. Raw (mx, my, mz): Unit-converted readings in µT
        2. Iron Corrected (calibrated_*): Hard/soft iron correction applied
        3. Residual/Earth-Subtracted (residual_* or fused_*): Earth field removed via orientation-compensated averaging
        4. Filtered (filtered_*): Kalman smoothed for noise reduction

        Note: 'fused_*' is legacy naming for 'residual_*' (Earth-subtracted) fields.
        Both field names are supported for backward compatibility.
        """
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
            # Fused/Residual magnetometer (Earth field subtracted)
            # Note: Internally uses 'fused_*' but loads both new (residual_*) and legacy (fused_*) field names
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
        has_fused = False  # Supports both new (residual_*) and legacy (fused_*) field names
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
            # Magnetometer: prefer converted µT values, fall back to raw LSB
            # IMPORTANT: mx_ut, my_ut, mz_ut are the proper unit-converted fields
            sensors['mx'][i] = sample.get('mx_ut', sample.get('mx', 0))
            sensors['my'][i] = sample.get('my_ut', sample.get('my', 0))
            sensors['mz'][i] = sample.get('mz_ut', sample.get('mz', 0))

            # Calibrated (iron corrected)
            if 'calibrated_mx' in sample:
                has_calibrated = True
                sensors['calibrated_mx'][i] = sample.get('calibrated_mx', 0)
                sensors['calibrated_my'][i] = sample.get('calibrated_my', 0)
                sensors['calibrated_mz'][i] = sample.get('calibrated_mz', 0)

            # Fused/Residual (Earth field subtracted) - supports new (residual_*) and legacy (fused_*) field names
            if 'residual_mx' in sample or 'fused_mx' in sample:
                has_fused = True
                sensors['fused_mx'][i] = sample.get('residual_mx', sample.get('fused_mx', 0))
                sensors['fused_my'][i] = sample.get('residual_my', sample.get('fused_my', 0))
                sensors['fused_mz'][i] = sample.get('residual_mz', sample.get('fused_mz', 0))

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


class VisualDistinctionEngine:
    """Creates visually distinct representations based on sensor patterns."""

    @staticmethod
    def sensor_to_color(sensors: Dict[str, np.ndarray], window_start: int, window_end: int) -> Tuple[float, float, float]:
        """Generate a unique color based on sensor data in a window."""
        # Use accelerometer for hue (0-360 degrees)
        ax_mean = np.mean(sensors['ax'][window_start:window_end])
        ay_mean = np.mean(sensors['ay'][window_start:window_end])
        hue = (np.arctan2(ay_mean, ax_mean) + np.pi) / (2 * np.pi)  # Normalize to 0-1

        # Use gyroscope magnitude for saturation
        gyro_std = np.std(sensors['gyro_mag'][window_start:window_end])
        saturation = min(gyro_std / 30000.0, 1.0)  # Normalize

        # Use accelerometer magnitude for value/brightness
        accel_mean = np.mean(sensors['accel_mag'][window_start:window_end])
        value = min(accel_mean / 20000.0, 1.0)  # Normalize
        value = max(value, 0.3)  # Ensure minimum brightness

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (r, g, b)

    @staticmethod
    def create_signature_pattern(sensors: Dict[str, np.ndarray], window_start: int, window_end: int, size: int = 64) -> np.ndarray:
        """Create a unique visual pattern/fingerprint for a data window."""
        pattern = np.zeros((size, size, 3))

        window_data = {
            'ax': sensors['ax'][window_start:window_end],
            'ay': sensors['ay'][window_start:window_end],
            'az': sensors['az'][window_start:window_end],
            'gx': sensors['gx'][window_start:window_end],
            'gy': sensors['gy'][window_start:window_end],
            'gz': sensors['gz'][window_start:window_end],
        }

        # Normalize data to 0-1 range for visualization
        for key in window_data:
            data = window_data[key]
            if np.max(np.abs(data)) > 0:
                window_data[key] = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Create radial pattern from center
        center = size // 2
        y, x = np.ogrid[:size, :size]
        distances = np.sqrt((x - center)**2 + (y - center)**2)
        angles = np.arctan2(y - center, x - center)

        # Map accelerometer to radius (distance from center)
        accel_pattern = np.interp(distances, [0, center], [0, len(window_data['ax']) - 1]).astype(int)
        accel_pattern = np.clip(accel_pattern, 0, len(window_data['ax']) - 1)

        # Map gyroscope to angular patterns
        angle_pattern = ((angles + np.pi) / (2 * np.pi) * len(window_data['gx'])).astype(int)
        angle_pattern = np.clip(angle_pattern, 0, len(window_data['gx']) - 1)

        # Assign RGB channels
        pattern[:, :, 0] = window_data['ax'][accel_pattern]  # Accel X -> Red (radial)
        pattern[:, :, 1] = window_data['gy'][angle_pattern]  # Gyro Y -> Green (angular)
        pattern[:, :, 2] = window_data['az'][accel_pattern]  # Accel Z -> Blue (radial)

        return pattern


class SessionVisualizer:
    """Generates comprehensive visualizations for sensor data sessions."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distinction_engine = VisualDistinctionEngine()

    def create_composite_session_image(self, session: Dict, processor: SensorDataProcessor) -> Path:
        """Create a comprehensive composite visualization for an entire session."""
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Title with session info
        timestamp_str = session['timestamp']
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            title_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            title_time = timestamp_str

        fig.suptitle(f'Session: {session["filename"]}\n{title_time} | Duration: {session["duration"]:.1f}s | Samples: {len(data)}',
                     fontsize=16, fontweight='bold')

        # 1. Accelerometer 3-axis time series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(sensors['time'], sensors['ax'], label='X', alpha=0.7, linewidth=1)
        ax1.plot(sensors['time'], sensors['ay'], label='Y', alpha=0.7, linewidth=1)
        ax1.plot(sensors['time'], sensors['az'], label='Z', alpha=0.7, linewidth=1)
        ax1.set_title('Accelerometer (3-axis)', fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (g)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. Gyroscope 3-axis time series
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(sensors['time'], sensors['gx'], label='X', alpha=0.7, linewidth=1)
        ax2.plot(sensors['time'], sensors['gy'], label='Y', alpha=0.7, linewidth=1)
        ax2.plot(sensors['time'], sensors['gz'], label='Z', alpha=0.7, linewidth=1)
        ax2.set_title('Gyroscope (3-axis)', fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular velocity (°/s)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Magnetometer 3-axis time series with ALL calibration stages
        ax3 = fig.add_subplot(gs[2, :])

        # Determine calibration status for title
        has_calibrated = sensors.get('_has_calibrated', False)
        has_fused = sensors.get('_has_fused', False)
        has_filtered = sensors.get('_has_filtered', False)

        if has_filtered:
            calib_status = 'Full (Iron + Earth + Filtered)'
            status_color = '#4daf4a'  # Green
        elif has_fused:
            calib_status = 'Residual (Iron + Earth Subtracted)'
            status_color = '#377eb8'  # Blue
        elif has_calibrated:
            calib_status = 'Partial (Iron Only)'
            status_color = '#ff7f00'  # Orange
        else:
            calib_status = 'None (Raw)'
            status_color = '#e41a1c'  # Red

        # Plot ALL calibration stages with distinct colors per documentation:
        # Raw (gray, dashed), Calibrated (blue), Fused (green), Filtered (red, bold)

        # Stage 1: Raw magnetometer (gray, dashed for reference)
        ax3.plot(sensors['time'], sensors['mx'], color='gray', alpha=0.4, linewidth=1, linestyle='--', label='Raw X')
        ax3.plot(sensors['time'], sensors['my'], color='gray', alpha=0.4, linewidth=1, linestyle='-.', label='Raw Y')
        ax3.plot(sensors['time'], sensors['mz'], color='gray', alpha=0.4, linewidth=1, linestyle=':', label='Raw Z')

        # Stage 2: Iron corrected (blue tones) - if available
        if has_calibrated:
            ax3.plot(sensors['time'], sensors['calibrated_mx'], color='#1f77b4', alpha=0.6, linewidth=1, label='Iron X')
            ax3.plot(sensors['time'], sensors['calibrated_my'], color='#17becf', alpha=0.6, linewidth=1, label='Iron Y')
            ax3.plot(sensors['time'], sensors['calibrated_mz'], color='#9467bd', alpha=0.6, linewidth=1, label='Iron Z')

        # Stage 3: Residual / Earth field subtracted (green tones) - if available
        if has_fused:
            ax3.plot(sensors['time'], sensors['fused_mx'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Residual X')
            ax3.plot(sensors['time'], sensors['fused_my'], color='#98df8a', alpha=0.7, linewidth=1.2, label='Residual Y')
            ax3.plot(sensors['time'], sensors['fused_mz'], color='#006400', alpha=0.7, linewidth=1.2, label='Residual Z')

        # Stage 4: Filtered / Kalman smoothed (red tones, bold) - if available
        if has_filtered:
            ax3.plot(sensors['time'], sensors['filtered_mx'], color='#d62728', alpha=0.9, linewidth=1.8, label='Filtered X')
            ax3.plot(sensors['time'], sensors['filtered_my'], color='#ff9896', alpha=0.9, linewidth=1.8, label='Filtered Y')
            ax3.plot(sensors['time'], sensors['filtered_mz'], color='#8b0000', alpha=0.9, linewidth=1.8, label='Filtered Z')

        ax3.set_title(f'Magnetometer (3-axis) | Calibration: {calib_status}', fontweight='bold', color=status_color)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Value (μT)')
        ax3.legend(loc='upper right', fontsize=7, ncol=4)
        ax3.grid(True, alpha=0.3)

        # 4. Magnitudes comparison (with ALL calibration stages)
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(sensors['time'], sensors['accel_mag'], label='Accel', alpha=0.8)
        ax4.plot(sensors['time'], sensors['gyro_mag'], label='Gyro', alpha=0.8)

        # Plot ALL magnetometer magnitude stages
        ax4.plot(sensors['time'], sensors['mag_mag'], color='gray', alpha=0.4, linestyle='--', label='Mag (Raw)')
        if has_calibrated and 'calibrated_mag' in sensors:
            ax4.plot(sensors['time'], sensors['calibrated_mag'], color='#1f77b4', alpha=0.6, linewidth=1, label='Mag (Iron)')
        if has_fused and 'fused_mag' in sensors:
            ax4.plot(sensors['time'], sensors['fused_mag'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Mag (Residual)')
        if has_filtered and 'filtered_mag' in sensors:
            ax4.plot(sensors['time'], sensors['filtered_mag'], color='#d62728', alpha=0.9, linewidth=1.5, label='Mag (Filtered)')

        ax4.set_title('Magnitude Comparison', fontweight='bold')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnitude')
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(True, alpha=0.3)

        # 5. Auxiliary sensors OR Orientation (if available)
        ax5 = fig.add_subplot(gs[3, 1])
        has_orientation = sensors.get('_has_orientation', False)

        if has_orientation and np.any(sensors['euler_roll'] != 0):
            # Show device orientation (Euler angles)
            ax5.plot(sensors['time'], sensors['euler_roll'], 'r-', label='Roll', alpha=0.8, linewidth=1)
            ax5.plot(sensors['time'], sensors['euler_pitch'], 'g-', label='Pitch', alpha=0.8, linewidth=1)
            ax5.plot(sensors['time'], sensors['euler_yaw'], 'b-', label='Yaw', alpha=0.8, linewidth=1)
            ax5.set_title('Device Orientation (Euler)', fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Angle (degrees)')
            ax5.legend(loc='upper right', fontsize=8)
            ax5.grid(True, alpha=0.3)
        else:
            # Show auxiliary sensors
            ax5_twin = ax5.twinx()
            line1 = ax5.plot(sensors['time'], sensors['light'], 'g-', label='Light', alpha=0.7)
            line2 = ax5_twin.plot(sensors['time'], sensors['capacitive'], 'orange', label='Capacitive', alpha=0.7)
            ax5.set_title('Auxiliary Sensors', fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Light Level', color='g')
            ax5_twin.set_ylabel('Capacitive Value', color='orange')
            ax5.tick_params(axis='y', labelcolor='g')
            ax5_twin.tick_params(axis='y', labelcolor='orange')
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='upper right')
            ax5.grid(True, alpha=0.3)

        # 6. Spectral signature (visual fingerprint) with calibration status badge
        ax6 = fig.add_subplot(gs[3, 2])
        signature = self.distinction_engine.create_signature_pattern(sensors, 0, len(data))
        ax6.imshow(signature)
        ax6.set_title('Spectral Signature', fontweight='bold')
        ax6.axis('off')

        # Add calibration status badge
        badge_colors = {'Full (Iron + Earth Field)': '#4daf4a', 'Partial (Iron Only)': '#ff7f00', 'None (Raw)': '#e41a1c'}
        ax6.text(0.05, 0.95, f'Calibration: {calib_status}',
                transform=ax6.transAxes, fontsize=8, fontweight='bold',
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=status_color, alpha=0.8))

        # Save composite image
        output_file = self.output_dir / f"composite_{session['timestamp']}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Created composite: {output_file.name}")
        return output_file

    def create_window_images(self, session: Dict, processor: SensorDataProcessor) -> List[Dict]:
        """Create individual images for each 1-second window in the session.
        
        Generates both:
        1. Composite window image (backward compatible)
        2. Individual figure images (new: timeseries, trajectories, signature, stats, trajectory_comparison)
        """
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)

        window_size = 50  # 1 second at 50Hz
        n_samples = len(data)
        n_windows = n_samples // window_size

        window_info = []

        # Create output directory for this session's windows
        session_dir = self.output_dir / f"windows_{session['timestamp']}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Check calibration status
        has_calibrated = sensors.get('_has_calibrated', False)
        has_fused = sensors.get('_has_fused', False)
        has_filtered = sensors.get('_has_filtered', False)

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size

            if end_idx > n_samples:
                break

            window_time_start = start_idx / 50.0
            window_time_end = end_idx / 50.0
            time_window = sensors['time'][start_idx:end_idx]

            # Create per-window subdirectory for individual images
            window_subdir = session_dir / f"window_{i+1:03d}"
            window_subdir.mkdir(parents=True, exist_ok=True)

            # Dictionary to store individual image paths
            individual_images = {}

            # ============================================================
            # Generate Individual Images
            # ============================================================

            # 1. Timeseries - Accelerometer
            fig_ts_accel, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_window, sensors['ax'][start_idx:end_idx], label='X', linewidth=2)
            ax.plot(time_window, sensors['ay'][start_idx:end_idx], label='Y', linewidth=2)
            ax.plot(time_window, sensors['az'][start_idx:end_idx], label='Z', linewidth=2)
            ax.set_title(f'Accelerometer | Window {i+1} ({window_time_start:.1f}s - {window_time_end:.1f}s)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration (g)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ts_accel_path = window_subdir / 'timeseries_accel.png'
            plt.savefig(ts_accel_path, dpi=100, bbox_inches='tight')
            plt.close(fig_ts_accel)
            individual_images['timeseries_accel'] = str(ts_accel_path.relative_to(self.output_dir))

            # 2. Timeseries - Gyroscope
            fig_ts_gyro, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_window, sensors['gx'][start_idx:end_idx], label='X', linewidth=2)
            ax.plot(time_window, sensors['gy'][start_idx:end_idx], label='Y', linewidth=2)
            ax.plot(time_window, sensors['gz'][start_idx:end_idx], label='Z', linewidth=2)
            ax.set_title(f'Gyroscope | Window {i+1} ({window_time_start:.1f}s - {window_time_end:.1f}s)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angular velocity (°/s)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ts_gyro_path = window_subdir / 'timeseries_gyro.png'
            plt.savefig(ts_gyro_path, dpi=100, bbox_inches='tight')
            plt.close(fig_ts_gyro)
            individual_images['timeseries_gyro'] = str(ts_gyro_path.relative_to(self.output_dir))

            # 3. Timeseries - Magnetometer (all calibration stages)
            fig_ts_mag, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_window, sensors['mx'][start_idx:end_idx], color='gray', alpha=0.4, linestyle='--', label='Raw X')
            ax.plot(time_window, sensors['my'][start_idx:end_idx], color='gray', alpha=0.4, linestyle='-.', label='Raw Y')
            ax.plot(time_window, sensors['mz'][start_idx:end_idx], color='gray', alpha=0.4, linestyle=':', label='Raw Z')
            if has_filtered:
                ax.plot(time_window, sensors['filtered_mx'][start_idx:end_idx], color='#d62728', linewidth=2, label='Filtered X')
                ax.plot(time_window, sensors['filtered_my'][start_idx:end_idx], color='#ff9896', linewidth=2, label='Filtered Y')
                ax.plot(time_window, sensors['filtered_mz'][start_idx:end_idx], color='#8b0000', linewidth=2, label='Filtered Z')
            elif has_fused:
                ax.plot(time_window, sensors['fused_mx'][start_idx:end_idx], color='#2ca02c', linewidth=2, label='Residual X')
                ax.plot(time_window, sensors['fused_my'][start_idx:end_idx], color='#98df8a', linewidth=2, label='Residual Y')
                ax.plot(time_window, sensors['fused_mz'][start_idx:end_idx], color='#006400', linewidth=2, label='Residual Z')
            ax.set_title(f'Magnetometer | Window {i+1} ({window_time_start:.1f}s - {window_time_end:.1f}s)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value (μT)')
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ts_mag_path = window_subdir / 'timeseries_mag.png'
            plt.savefig(ts_mag_path, dpi=100, bbox_inches='tight')
            plt.close(fig_ts_mag)
            individual_images['timeseries_mag'] = str(ts_mag_path.relative_to(self.output_dir))

            # 4. 3D Trajectory - Accelerometer
            fig_traj_accel = plt.figure(figsize=(6, 6))
            ax = fig_traj_accel.add_subplot(111, projection='3d')
            ax.plot(sensors['ax'][start_idx:end_idx], sensors['ay'][start_idx:end_idx], sensors['az'][start_idx:end_idx],
                   linewidth=2, alpha=0.7, color='tab:blue')
            ax.scatter(sensors['ax'][start_idx], sensors['ay'][start_idx], sensors['az'][start_idx], c='green', s=50, label='Start')
            ax.scatter(sensors['ax'][end_idx-1], sensors['ay'][end_idx-1], sensors['az'][end_idx-1], c='red', s=50, label='End')
            ax.set_title(f'Accel 3D | Window {i+1}', fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend(fontsize=7)
            traj_accel_path = window_subdir / 'trajectory_accel_3d.png'
            plt.savefig(traj_accel_path, dpi=100, bbox_inches='tight')
            plt.close(fig_traj_accel)
            individual_images['trajectory_accel_3d'] = str(traj_accel_path.relative_to(self.output_dir))

            # 5. 3D Trajectory - Gyroscope
            fig_traj_gyro = plt.figure(figsize=(6, 6))
            ax = fig_traj_gyro.add_subplot(111, projection='3d')
            ax.plot(sensors['gx'][start_idx:end_idx], sensors['gy'][start_idx:end_idx], sensors['gz'][start_idx:end_idx],
                   linewidth=2, alpha=0.7, color='tab:orange')
            ax.scatter(sensors['gx'][start_idx], sensors['gy'][start_idx], sensors['gz'][start_idx], c='green', s=50, label='Start')
            ax.scatter(sensors['gx'][end_idx-1], sensors['gy'][end_idx-1], sensors['gz'][end_idx-1], c='red', s=50, label='End')
            ax.set_title(f'Gyro 3D | Window {i+1}', fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend(fontsize=7)
            traj_gyro_path = window_subdir / 'trajectory_gyro_3d.png'
            plt.savefig(traj_gyro_path, dpi=100, bbox_inches='tight')
            plt.close(fig_traj_gyro)
            individual_images['trajectory_gyro_3d'] = str(traj_gyro_path.relative_to(self.output_dir))

            # 6. 3D Trajectory - Magnetometer
            fig_traj_mag = plt.figure(figsize=(6, 6))
            ax = fig_traj_mag.add_subplot(111, projection='3d')
            ax.plot(sensors['mx'][start_idx:end_idx], sensors['my'][start_idx:end_idx], sensors['mz'][start_idx:end_idx],
                   linewidth=2, alpha=0.7, color='tab:green')
            ax.scatter(sensors['mx'][start_idx], sensors['my'][start_idx], sensors['mz'][start_idx], c='green', s=50, label='Start')
            ax.scatter(sensors['mx'][end_idx-1], sensors['my'][end_idx-1], sensors['mz'][end_idx-1], c='red', s=50, label='End')
            ax.set_title(f'Mag 3D | Window {i+1}', fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend(fontsize=7)
            traj_mag_path = window_subdir / 'trajectory_mag_3d.png'
            plt.savefig(traj_mag_path, dpi=100, bbox_inches='tight')
            plt.close(fig_traj_mag)
            individual_images['trajectory_mag_3d'] = str(traj_mag_path.relative_to(self.output_dir))

            # 7. Visual Signature
            fig_sig, ax = plt.subplots(figsize=(4, 4))
            signature = self.distinction_engine.create_signature_pattern(sensors, start_idx, end_idx, size=128)
            ax.imshow(signature)
            ax.set_title(f'Signature | Window {i+1}', fontweight='bold')
            ax.axis('off')
            sig_path = window_subdir / 'signature.png'
            plt.savefig(sig_path, dpi=100, bbox_inches='tight')
            plt.close(fig_sig)
            individual_images['signature'] = str(sig_path.relative_to(self.output_dir))

            # 8. Statistics Panel
            fig_stats, ax = plt.subplots(figsize=(6, 6))
            ax.axis('off')
            stats_text = f"""
Window {i+1} Statistics
Time: {window_time_start:.2f}s - {window_time_end:.2f}s

Accelerometer:
  Mean: [{sensors['ax'][start_idx:end_idx].mean():.0f}, {sensors['ay'][start_idx:end_idx].mean():.0f}, {sensors['az'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['ax'][start_idx:end_idx].std():.0f}, {sensors['ay'][start_idx:end_idx].std():.0f}, {sensors['az'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['accel_mag'][start_idx:end_idx].mean():.0f} ± {sensors['accel_mag'][start_idx:end_idx].std():.0f}

Gyroscope:
  Mean: [{sensors['gx'][start_idx:end_idx].mean():.0f}, {sensors['gy'][start_idx:end_idx].mean():.0f}, {sensors['gz'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['gx'][start_idx:end_idx].std():.0f}, {sensors['gy'][start_idx:end_idx].std():.0f}, {sensors['gz'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['gyro_mag'][start_idx:end_idx].mean():.0f} ± {sensors['gyro_mag'][start_idx:end_idx].std():.0f}

Magnetometer:
  Mean: [{sensors['mx'][start_idx:end_idx].mean():.0f}, {sensors['my'][start_idx:end_idx].mean():.0f}, {sensors['mz'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['mx'][start_idx:end_idx].std():.0f}, {sensors['my'][start_idx:end_idx].std():.0f}, {sensors['mz'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['mag_mag'][start_idx:end_idx].mean():.0f} ± {sensors['mag_mag'][start_idx:end_idx].std():.0f}
            """.strip()
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            stats_path = window_subdir / 'stats.png'
            plt.savefig(stats_path, dpi=100, bbox_inches='tight')
            plt.close(fig_stats)
            individual_images['stats'] = str(stats_path.relative_to(self.output_dir))

            # 9. Per-Window Trajectory Comparison (if calibration data available)
            trajectory_images_dict = None
            if has_calibrated or has_fused or has_filtered:
                trajectory_images_dict = self._create_window_trajectory_comparison(
                    sensors, start_idx, end_idx, i+1, n_windows, window_subdir,
                    has_calibrated, has_fused, has_filtered
                )
                if trajectory_images_dict:
                    # Keep composite for backward compatibility
                    individual_images['trajectory_comparison'] = trajectory_images_dict.get('combined', '')

            # 10. Combined 3D Trajectory
            fig_combined = plt.figure(figsize=(8, 8))
            ax = fig_combined.add_subplot(111, projection='3d')
            # Normalize each sensor
            ax_norm = (sensors['ax'][start_idx:end_idx] - sensors['ax'][start_idx:end_idx].mean()) / (sensors['ax'][start_idx:end_idx].std() + 1e-6)
            ay_norm = (sensors['ay'][start_idx:end_idx] - sensors['ay'][start_idx:end_idx].mean()) / (sensors['ay'][start_idx:end_idx].std() + 1e-6)
            az_norm = (sensors['az'][start_idx:end_idx] - sensors['az'][start_idx:end_idx].mean()) / (sensors['az'][start_idx:end_idx].std() + 1e-6)
            gx_norm = (sensors['gx'][start_idx:end_idx] - sensors['gx'][start_idx:end_idx].mean()) / (sensors['gx'][start_idx:end_idx].std() + 1e-6)
            gy_norm = (sensors['gy'][start_idx:end_idx] - sensors['gy'][start_idx:end_idx].mean()) / (sensors['gy'][start_idx:end_idx].std() + 1e-6)
            gz_norm = (sensors['gz'][start_idx:end_idx] - sensors['gz'][start_idx:end_idx].mean()) / (sensors['gz'][start_idx:end_idx].std() + 1e-6)
            mx_norm = (sensors['mx'][start_idx:end_idx] - sensors['mx'][start_idx:end_idx].mean()) / (sensors['mx'][start_idx:end_idx].std() + 1e-6)
            my_norm = (sensors['my'][start_idx:end_idx] - sensors['my'][start_idx:end_idx].mean()) / (sensors['my'][start_idx:end_idx].std() + 1e-6)
            mz_norm = (sensors['mz'][start_idx:end_idx] - sensors['mz'][start_idx:end_idx].mean()) / (sensors['mz'][start_idx:end_idx].std() + 1e-6)
            ax.plot(ax_norm, ay_norm, az_norm, color='blue', linewidth=1.5, alpha=0.8, label='Accel')
            ax.plot(gx_norm, gy_norm, gz_norm, color='orange', linewidth=1.5, alpha=0.8, label='Gyro')
            ax.plot(mx_norm, my_norm, mz_norm, color='green', linewidth=1.5, alpha=0.8, label='Mag')
            ax.set_title(f'Combined 3D | Window {i+1}', fontweight='bold')
            ax.set_xlabel('X (norm)'); ax.set_ylabel('Y (norm)'); ax.set_zlabel('Z (norm)')
            ax.legend(fontsize=8)
            combined_path = window_subdir / 'trajectory_combined_3d.png'
            plt.savefig(combined_path, dpi=100, bbox_inches='tight')
            plt.close(fig_combined)
            individual_images['trajectory_combined_3d'] = str(combined_path.relative_to(self.output_dir))

            # ============================================================
            # Generate Composite Image (backward compatible)
            # ============================================================
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

            fig.suptitle(f'Window {i+1}/{n_windows} | Time: {window_time_start:.1f}s - {window_time_end:.1f}s',
                        fontsize=14, fontweight='bold')

            # 1. Accelerometer time series
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(time_window, sensors['ax'][start_idx:end_idx], label='X', linewidth=2)
            ax1.plot(time_window, sensors['ay'][start_idx:end_idx], label='Y', linewidth=2)
            ax1.plot(time_window, sensors['az'][start_idx:end_idx], label='Z', linewidth=2)
            ax1.set_title('Accelerometer', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 2. Gyroscope time series
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(time_window, sensors['gx'][start_idx:end_idx], label='X', linewidth=2)
            ax2.plot(time_window, sensors['gy'][start_idx:end_idx], label='Y', linewidth=2)
            ax2.plot(time_window, sensors['gz'][start_idx:end_idx], label='Z', linewidth=2)
            ax2.set_title('Gyroscope', fontweight='bold')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Value')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 3. Magnetometer time series
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(time_window, sensors['mx'][start_idx:end_idx], label='X', linewidth=2)
            ax3.plot(time_window, sensors['my'][start_idx:end_idx], label='Y', linewidth=2)
            ax3.plot(time_window, sensors['mz'][start_idx:end_idx], label='Z', linewidth=2)
            ax3.set_title('Magnetometer', fontweight='bold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Value')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)

            # 4. 3D Trajectory - Accelerometer
            ax4 = fig.add_subplot(gs[0, 1], projection='3d')
            ax4.plot(sensors['ax'][start_idx:end_idx],
                    sensors['ay'][start_idx:end_idx],
                    sensors['az'][start_idx:end_idx],
                    linewidth=2, alpha=0.7, color='tab:blue')
            ax4.scatter(sensors['ax'][start_idx], sensors['ay'][start_idx], sensors['az'][start_idx],
                       c='green', s=50, label='Start', zorder=5)
            ax4.scatter(sensors['ax'][end_idx-1], sensors['ay'][end_idx-1], sensors['az'][end_idx-1],
                       c='red', s=50, label='End', zorder=5)
            ax4.set_title('Accel 3D Trajectory', fontweight='bold')
            ax4.set_xlabel('X', fontsize=8)
            ax4.set_ylabel('Y', fontsize=8)
            ax4.set_zlabel('Z', fontsize=8)
            ax4.legend(fontsize=7)

            # 5. 3D Trajectory - Gyroscope
            ax5 = fig.add_subplot(gs[1, 1], projection='3d')
            ax5.plot(sensors['gx'][start_idx:end_idx],
                    sensors['gy'][start_idx:end_idx],
                    sensors['gz'][start_idx:end_idx],
                    linewidth=2, alpha=0.7, color='tab:orange')
            ax5.scatter(sensors['gx'][start_idx], sensors['gy'][start_idx], sensors['gz'][start_idx],
                       c='green', s=50, label='Start', zorder=5)
            ax5.scatter(sensors['gx'][end_idx-1], sensors['gy'][end_idx-1], sensors['gz'][end_idx-1],
                       c='red', s=50, label='End', zorder=5)
            ax5.set_title('Gyro 3D Trajectory', fontweight='bold')
            ax5.set_xlabel('X', fontsize=8)
            ax5.set_ylabel('Y', fontsize=8)
            ax5.set_zlabel('Z', fontsize=8)
            ax5.legend(fontsize=7)

            # 6. 3D Trajectory - Magnetometer
            ax6 = fig.add_subplot(gs[2, 1], projection='3d')
            ax6.plot(sensors['mx'][start_idx:end_idx],
                    sensors['my'][start_idx:end_idx],
                    sensors['mz'][start_idx:end_idx],
                    linewidth=2, alpha=0.7, color='tab:green')
            ax6.scatter(sensors['mx'][start_idx], sensors['my'][start_idx], sensors['mz'][start_idx],
                       c='green', s=50, label='Start', zorder=5)
            ax6.scatter(sensors['mx'][end_idx-1], sensors['my'][end_idx-1], sensors['mz'][end_idx-1],
                       c='red', s=50, label='End', zorder=5)
            ax6.set_title('Mag 3D Trajectory', fontweight='bold')
            ax6.set_xlabel('X', fontsize=8)
            ax6.set_ylabel('Y', fontsize=8)
            ax6.set_zlabel('Z', fontsize=8)
            ax6.legend(fontsize=7)

            # 7. Spectral Signature
            ax7 = fig.add_subplot(gs[0, 2])
            signature = self.distinction_engine.create_signature_pattern(sensors, start_idx, end_idx, size=128)
            ax7.imshow(signature)
            ax7.set_title('Visual Signature', fontweight='bold')
            ax7.axis('off')

            # 8. Combined 3D Trajectory - All sensors intertwined over time
            ax8 = fig.add_subplot(gs[1, 2], projection='3d')
            
            # Normalize each sensor to similar scale for combined visualization
            ax_norm = (sensors['ax'][start_idx:end_idx] - sensors['ax'][start_idx:end_idx].mean()) / (sensors['ax'][start_idx:end_idx].std() + 1e-6)
            ay_norm = (sensors['ay'][start_idx:end_idx] - sensors['ay'][start_idx:end_idx].mean()) / (sensors['ay'][start_idx:end_idx].std() + 1e-6)
            az_norm = (sensors['az'][start_idx:end_idx] - sensors['az'][start_idx:end_idx].mean()) / (sensors['az'][start_idx:end_idx].std() + 1e-6)
            
            gx_norm = (sensors['gx'][start_idx:end_idx] - sensors['gx'][start_idx:end_idx].mean()) / (sensors['gx'][start_idx:end_idx].std() + 1e-6)
            gy_norm = (sensors['gy'][start_idx:end_idx] - sensors['gy'][start_idx:end_idx].mean()) / (sensors['gy'][start_idx:end_idx].std() + 1e-6)
            gz_norm = (sensors['gz'][start_idx:end_idx] - sensors['gz'][start_idx:end_idx].mean()) / (sensors['gz'][start_idx:end_idx].std() + 1e-6)
            
            mx_norm = (sensors['mx'][start_idx:end_idx] - sensors['mx'][start_idx:end_idx].mean()) / (sensors['mx'][start_idx:end_idx].std() + 1e-6)
            my_norm = (sensors['my'][start_idx:end_idx] - sensors['my'][start_idx:end_idx].mean()) / (sensors['my'][start_idx:end_idx].std() + 1e-6)
            mz_norm = (sensors['mz'][start_idx:end_idx] - sensors['mz'][start_idx:end_idx].mean()) / (sensors['mz'][start_idx:end_idx].std() + 1e-6)
            
            n_points = len(ax_norm)
            time_colors = np.linspace(0, 1, n_points)
            
            # Plot all three trajectories with time-based coloring
            for j in range(n_points - 1):
                # Accelerometer - solid line
                ax8.plot([ax_norm[j], ax_norm[j+1]], 
                        [ay_norm[j], ay_norm[j+1]], 
                        [az_norm[j], az_norm[j+1]], 
                        color=plt.cm.Blues(0.3 + time_colors[j] * 0.7), linewidth=1.5, alpha=0.8)
                # Gyroscope - solid line
                ax8.plot([gx_norm[j], gx_norm[j+1]], 
                        [gy_norm[j], gy_norm[j+1]], 
                        [gz_norm[j], gz_norm[j+1]], 
                        color=plt.cm.Oranges(0.3 + time_colors[j] * 0.7), linewidth=1.5, alpha=0.8)
                # Magnetometer - solid line
                ax8.plot([mx_norm[j], mx_norm[j+1]], 
                        [my_norm[j], my_norm[j+1]], 
                        [mz_norm[j], mz_norm[j+1]], 
                        color=plt.cm.Greens(0.3 + time_colors[j] * 0.7), linewidth=1.5, alpha=0.8)
            
            # Add start/end markers for each trajectory
            ax8.scatter([ax_norm[0]], [ay_norm[0]], [az_norm[0]], c='blue', s=30, marker='o', label='Accel', zorder=5)
            ax8.scatter([gx_norm[0]], [gy_norm[0]], [gz_norm[0]], c='orange', s=30, marker='s', label='Gyro', zorder=5)
            ax8.scatter([mx_norm[0]], [my_norm[0]], [mz_norm[0]], c='green', s=30, marker='^', label='Mag', zorder=5)
            
            ax8.set_title('Combined 3D Trajectory', fontweight='bold')
            ax8.set_xlabel('X (norm)', fontsize=7)
            ax8.set_ylabel('Y (norm)', fontsize=7)
            ax8.set_zlabel('Z (norm)', fontsize=7)
            ax8.legend(fontsize=6, loc='upper left')
            ax8.tick_params(axis='both', which='major', labelsize=6)

            # 9. Statistics
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.axis('off')

            stats_text = f"""
Window Statistics:

Accelerometer:
  Mean: [{sensors['ax'][start_idx:end_idx].mean():.0f}, {sensors['ay'][start_idx:end_idx].mean():.0f}, {sensors['az'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['ax'][start_idx:end_idx].std():.0f}, {sensors['ay'][start_idx:end_idx].std():.0f}, {sensors['az'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['accel_mag'][start_idx:end_idx].mean():.0f} ± {sensors['accel_mag'][start_idx:end_idx].std():.0f}

Gyroscope:
  Mean: [{sensors['gx'][start_idx:end_idx].mean():.0f}, {sensors['gy'][start_idx:end_idx].mean():.0f}, {sensors['gz'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['gx'][start_idx:end_idx].std():.0f}, {sensors['gy'][start_idx:end_idx].std():.0f}, {sensors['gz'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['gyro_mag'][start_idx:end_idx].mean():.0f} ± {sensors['gyro_mag'][start_idx:end_idx].std():.0f}

Magnetometer:
  Mean: [{sensors['mx'][start_idx:end_idx].mean():.0f}, {sensors['my'][start_idx:end_idx].mean():.0f}, {sensors['mz'][start_idx:end_idx].mean():.0f}]
  Std:  [{sensors['mx'][start_idx:end_idx].std():.0f}, {sensors['my'][start_idx:end_idx].std():.0f}, {sensors['mz'][start_idx:end_idx].std():.0f}]
  Mag:  {sensors['mag_mag'][start_idx:end_idx].mean():.0f} ± {sensors['mag_mag'][start_idx:end_idx].std():.0f}
            """.strip()

            ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # Save window image
            output_file = session_dir / f"window_{i+1:03d}.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Get color for this window
            color = self.distinction_engine.sensor_to_color(sensors, start_idx, end_idx)
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

            window_entry = {
                'window_num': i + 1,
                'time_start': window_time_start,
                'time_end': window_time_end,
                'filename': output_file.name,
                'filepath': str(output_file.relative_to(self.output_dir)),
                'color': color_hex,
                'accel_mag_mean': float(sensors['accel_mag'][start_idx:end_idx].mean()),
                'gyro_mag_mean': float(sensors['gyro_mag'][start_idx:end_idx].mean()),
                'images': individual_images,  # Individual image paths
            }
            
            # Add trajectory images dict if available
            if trajectory_images_dict:
                window_entry['trajectory_images'] = trajectory_images_dict
            
            window_info.append(window_entry)

        print(f"  Created {len(window_info)} window images in {session_dir.name}/ (with individual figures)")
        return window_info

    def _create_window_trajectory_comparison(self, sensors: Dict, start_idx: int, end_idx: int, 
                                              window_num: int, n_windows: int, window_subdir: Path,
                                              has_calibrated: bool, has_fused: bool, has_filtered: bool) -> Optional[Dict]:
        """Create individual trajectory images for a window showing all calibration stages.
        
        Generates separate images for each stage:
        - Raw, Iron, Fused, Filtered (individual 3D plots)
        - Combined overlay
        - Statistics panel
        
        Returns dict with paths to individual images.
        """
        # Color scheme
        colors = {
            'raw': 'gray',
            'iron': '#1f77b4',
            'fused': '#2ca02c',
            'filtered': '#d62728'
        }

        n_samples = end_idx - start_idx
        time_colors = np.linspace(0, 1, n_samples)

        # Helper function to plot 3D trajectory with time coloring
        def plot_trajectory(mx, my, mz, title, color, cmap_name, output_path):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory with time-based coloring
            for j in range(len(mx) - 1):
                ax.plot([mx[j], mx[j+1]], [my[j], my[j+1]], [mz[j], mz[j+1]],
                       color=matplotlib.colormaps[cmap_name](time_colors[j]), linewidth=1.5, alpha=0.8)

            # Start/end markers
            ax.scatter([mx[0]], [my[0]], [mz[0]], c='green', s=80, marker='o', label='Start', zorder=10)
            ax.scatter([mx[-1]], [my[-1]], [mz[-1]], c='red', s=80, marker='s', label='End', zorder=10)

            ax.set_title(f'{title}\nWindow {window_num} | {start_idx/50.0:.1f}s - {end_idx/50.0:.1f}s', 
                        fontweight='bold', fontsize=11, color=color)
            ax.set_xlabel('X (μT)', fontsize=9)
            ax.set_ylabel('Y (μT)', fontsize=9)
            ax.set_zlabel('Z (μT)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=7)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        # Helper function to compute trajectory statistics
        def compute_traj_stats(mx, my, mz):
            spread = np.sqrt(np.std(mx)**2 + np.std(my)**2 + np.std(mz)**2)
            dx, dy, dz = np.diff(mx), np.diff(my), np.diff(mz)
            path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
            center = (np.mean(mx), np.mean(my), np.mean(mz))
            return {'spread': spread, 'path_length': path_length, 'center': center}

        # Dictionary to store image paths
        trajectory_images = {}

        # 1. Raw trajectory
        raw_path = window_subdir / 'trajectory_raw.png'
        plot_trajectory(sensors['mx'][start_idx:end_idx], sensors['my'][start_idx:end_idx], 
                       sensors['mz'][start_idx:end_idx], 'Raw Magnetometer', colors['raw'], 'Greys', raw_path)
        trajectory_images['raw'] = str(raw_path.relative_to(self.output_dir))

        # 2. Iron corrected (if available)
        if has_calibrated:
            iron_path = window_subdir / 'trajectory_iron.png'
            plot_trajectory(sensors['calibrated_mx'][start_idx:end_idx], 
                           sensors['calibrated_my'][start_idx:end_idx], 
                           sensors['calibrated_mz'][start_idx:end_idx], 
                           'Iron Corrected', colors['iron'], 'Blues', iron_path)
            trajectory_images['iron'] = str(iron_path.relative_to(self.output_dir))

        # 3. Fused (if available)
        if has_fused:
            fused_path = window_subdir / 'trajectory_fused.png'
            plot_trajectory(sensors['fused_mx'][start_idx:end_idx],
                           sensors['fused_my'][start_idx:end_idx],
                           sensors['fused_mz'][start_idx:end_idx],
                           'Residual (Earth Subtracted)', colors['fused'], 'Greens', fused_path)
            trajectory_images['fused'] = str(fused_path.relative_to(self.output_dir))

        # 4. Filtered (if available)
        if has_filtered:
            filtered_path = window_subdir / 'trajectory_filtered.png'
            plot_trajectory(sensors['filtered_mx'][start_idx:end_idx], 
                           sensors['filtered_my'][start_idx:end_idx], 
                           sensors['filtered_mz'][start_idx:end_idx], 
                           'Filtered (Kalman)', colors['filtered'], 'Reds', filtered_path)
            trajectory_images['filtered'] = str(filtered_path.relative_to(self.output_dir))

        # 5. Combined overlay trajectory
        combined_path = window_subdir / 'trajectory_combined.png'
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(sensors['mx'][start_idx:end_idx], sensors['my'][start_idx:end_idx], 
                sensors['mz'][start_idx:end_idx], color=colors['raw'], alpha=0.3, 
                linewidth=0.8, linestyle='--', label='Raw')
        if has_calibrated:
            ax.plot(sensors['calibrated_mx'][start_idx:end_idx], 
                    sensors['calibrated_my'][start_idx:end_idx], 
                    sensors['calibrated_mz'][start_idx:end_idx],
                    color=colors['iron'], alpha=0.5, linewidth=1, label='Iron')
        if has_fused:
            ax.plot(sensors['fused_mx'][start_idx:end_idx], 
                    sensors['fused_my'][start_idx:end_idx], 
                    sensors['fused_mz'][start_idx:end_idx],
                    color=colors['fused'], alpha=0.7, linewidth=1.2, label='Residual')
        if has_filtered:
            ax.plot(sensors['filtered_mx'][start_idx:end_idx], 
                    sensors['filtered_my'][start_idx:end_idx], 
                    sensors['filtered_mz'][start_idx:end_idx],
                    color=colors['filtered'], alpha=0.9, linewidth=1.5, label='Filtered')

        ax.set_title(f'Combined Overlay\nWindow {window_num} | {start_idx/50.0:.1f}s - {end_idx/50.0:.1f}s',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('X (μT)', fontsize=9)
        ax.set_ylabel('Y (μT)', fontsize=9)
        ax.set_zlabel('Z (μT)', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=7)

        plt.tight_layout()
        plt.savefig(combined_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        trajectory_images['combined'] = str(combined_path.relative_to(self.output_dir))

        # 6. Statistics panel
        stats_path = window_subdir / 'trajectory_stats.png'
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')

        raw_stats = compute_traj_stats(sensors['mx'][start_idx:end_idx], 
                                       sensors['my'][start_idx:end_idx], 
                                       sensors['mz'][start_idx:end_idx])

        stats_text = f"""
WINDOW {window_num} TRAJECTORY STATS
{'='*40}

◆ RAW
  Spread: {raw_stats['spread']:.2f} μT
  Path: {raw_stats['path_length']:.2f} μT
  Center: ({raw_stats['center'][0]:.1f}, {raw_stats['center'][1]:.1f}, {raw_stats['center'][2]:.1f})
"""
        if has_calibrated:
            iron_stats = compute_traj_stats(sensors['calibrated_mx'][start_idx:end_idx],
                                           sensors['calibrated_my'][start_idx:end_idx],
                                           sensors['calibrated_mz'][start_idx:end_idx])
            stats_text += f"""
◆ IRON
  Spread: {iron_stats['spread']:.2f} μT ({(iron_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {iron_stats['path_length']:.2f} μT
"""
        if has_fused:
            fused_stats = compute_traj_stats(sensors['fused_mx'][start_idx:end_idx],
                                            sensors['fused_my'][start_idx:end_idx],
                                            sensors['fused_mz'][start_idx:end_idx])
            stats_text += f"""
◆ RESIDUAL (Earth Subtracted)
  Spread: {fused_stats['spread']:.2f} μT ({(fused_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {fused_stats['path_length']:.2f} μT
  Center: ({fused_stats['center'][0]:.1f}, {fused_stats['center'][1]:.1f}, {fused_stats['center'][2]:.1f})
"""
        if has_filtered:
            filtered_stats = compute_traj_stats(sensors['filtered_mx'][start_idx:end_idx],
                                               sensors['filtered_my'][start_idx:end_idx],
                                               sensors['filtered_mz'][start_idx:end_idx])
            stats_text += f"""
◆ FILTERED
  Spread: {filtered_stats['spread']:.2f} μT ({(filtered_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {filtered_stats['path_length']:.2f} μT
"""

        stats_text += f"""
{'='*40}
Time: {start_idx/50.0:.2f}s - {end_idx/50.0:.2f}s
"""

        ax.text(0.05, 0.95, stats_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.95, edgecolor='#dee2e6'))

        plt.tight_layout()
        plt.savefig(stats_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        trajectory_images['statistics'] = str(stats_path.relative_to(self.output_dir))

        # Also create composite for backward compatibility
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.2, height_ratios=[1.2, 1])

        fig.suptitle(f'Window {window_num}/{n_windows} Trajectory Comparison\n'
                     f'Time: {start_idx/50.0:.2f}s - {end_idx/50.0:.2f}s',
                     fontsize=14, fontweight='bold')

        # Plot individual trajectories in row 1
        col = 0
        ax_raw = fig.add_subplot(gs[0, col], projection='3d')
        for j in range(n_samples - 1):
            ax_raw.plot([sensors['mx'][start_idx+j], sensors['mx'][start_idx+j+1]], 
                       [sensors['my'][start_idx+j], sensors['my'][start_idx+j+1]], 
                       [sensors['mz'][start_idx+j], sensors['mz'][start_idx+j+1]],
                       color=matplotlib.colormaps['Greys'](time_colors[j]), linewidth=1.5, alpha=0.8)
        ax_raw.scatter([sensors['mx'][start_idx]], [sensors['my'][start_idx]], [sensors['mz'][start_idx]], 
                      c='green', s=60, marker='o', label='Start', zorder=10)
        ax_raw.scatter([sensors['mx'][end_idx-1]], [sensors['my'][end_idx-1]], [sensors['mz'][end_idx-1]], 
                      c='red', s=60, marker='s', label='End', zorder=10)
        ax_raw.set_title('Raw', fontweight='bold', fontsize=10, color=colors['raw'])
        ax_raw.set_xlabel('X', fontsize=8); ax_raw.set_ylabel('Y', fontsize=8); ax_raw.set_zlabel('Z', fontsize=8)
        ax_raw.legend(fontsize=6, loc='upper left')
        ax_raw.tick_params(axis='both', which='major', labelsize=6)
        col += 1

        if has_calibrated:
            ax_iron = fig.add_subplot(gs[0, col], projection='3d')
            for j in range(n_samples - 1):
                ax_iron.plot([sensors['calibrated_mx'][start_idx+j], sensors['calibrated_mx'][start_idx+j+1]], 
                           [sensors['calibrated_my'][start_idx+j], sensors['calibrated_my'][start_idx+j+1]], 
                           [sensors['calibrated_mz'][start_idx+j], sensors['calibrated_mz'][start_idx+j+1]],
                           color=matplotlib.colormaps['Blues'](time_colors[j]), linewidth=1.5, alpha=0.8)
            ax_iron.scatter([sensors['calibrated_mx'][start_idx]], [sensors['calibrated_my'][start_idx]], 
                          [sensors['calibrated_mz'][start_idx]], c='green', s=60, marker='o', label='Start', zorder=10)
            ax_iron.scatter([sensors['calibrated_mx'][end_idx-1]], [sensors['calibrated_my'][end_idx-1]], 
                          [sensors['calibrated_mz'][end_idx-1]], c='red', s=60, marker='s', label='End', zorder=10)
            ax_iron.set_title('Iron', fontweight='bold', fontsize=10, color=colors['iron'])
            ax_iron.set_xlabel('X', fontsize=8); ax_iron.set_ylabel('Y', fontsize=8); ax_iron.set_zlabel('Z', fontsize=8)
            ax_iron.legend(fontsize=6, loc='upper left')
            ax_iron.tick_params(axis='both', which='major', labelsize=6)
            col += 1

        if has_fused:
            ax_fused = fig.add_subplot(gs[0, col], projection='3d')
            for j in range(n_samples - 1):
                ax_fused.plot([sensors['fused_mx'][start_idx+j], sensors['fused_mx'][start_idx+j+1]], 
                            [sensors['fused_my'][start_idx+j], sensors['fused_my'][start_idx+j+1]], 
                            [sensors['fused_mz'][start_idx+j], sensors['fused_mz'][start_idx+j+1]],
                            color=matplotlib.colormaps['Greens'](time_colors[j]), linewidth=1.5, alpha=0.8)
            ax_fused.scatter([sensors['fused_mx'][start_idx]], [sensors['fused_my'][start_idx]], 
                           [sensors['fused_mz'][start_idx]], c='green', s=60, marker='o', label='Start', zorder=10)
            ax_fused.scatter([sensors['fused_mx'][end_idx-1]], [sensors['fused_my'][end_idx-1]], 
                           [sensors['fused_mz'][end_idx-1]], c='red', s=60, marker='s', label='End', zorder=10)
            ax_fused.set_title('Residual', fontweight='bold', fontsize=10, color=colors['fused'])
            ax_fused.set_xlabel('X', fontsize=8); ax_fused.set_ylabel('Y', fontsize=8); ax_fused.set_zlabel('Z', fontsize=8)
            ax_fused.legend(fontsize=6, loc='upper left')
            ax_fused.tick_params(axis='both', which='major', labelsize=6)
            col += 1

        if has_filtered:
            ax_filtered = fig.add_subplot(gs[0, col], projection='3d')
            for j in range(n_samples - 1):
                ax_filtered.plot([sensors['filtered_mx'][start_idx+j], sensors['filtered_mx'][start_idx+j+1]], 
                               [sensors['filtered_my'][start_idx+j], sensors['filtered_my'][start_idx+j+1]], 
                               [sensors['filtered_mz'][start_idx+j], sensors['filtered_mz'][start_idx+j+1]],
                               color=matplotlib.colormaps['Reds'](time_colors[j]), linewidth=1.5, alpha=0.8)
            ax_filtered.scatter([sensors['filtered_mx'][start_idx]], [sensors['filtered_my'][start_idx]], 
                              [sensors['filtered_mz'][start_idx]], c='green', s=60, marker='o', label='Start', zorder=10)
            ax_filtered.scatter([sensors['filtered_mx'][end_idx-1]], [sensors['filtered_my'][end_idx-1]], 
                              [sensors['filtered_mz'][end_idx-1]], c='red', s=60, marker='s', label='End', zorder=10)
            ax_filtered.set_title('Filtered', fontweight='bold', fontsize=10, color=colors['filtered'])
            ax_filtered.set_xlabel('X', fontsize=8); ax_filtered.set_ylabel('Y', fontsize=8); ax_filtered.set_zlabel('Z', fontsize=8)
            ax_filtered.legend(fontsize=6, loc='upper left')
            ax_filtered.tick_params(axis='both', which='major', labelsize=6)
            col += 1

        # Row 2: Combined overlay + stats
        ax_combined = fig.add_subplot(gs[1, :2], projection='3d')
        ax_combined.plot(sensors['mx'][start_idx:end_idx], sensors['my'][start_idx:end_idx], 
                        sensors['mz'][start_idx:end_idx], color=colors['raw'], alpha=0.3, 
                        linewidth=0.8, linestyle='--', label='Raw')
        if has_calibrated:
            ax_combined.plot(sensors['calibrated_mx'][start_idx:end_idx], 
                            sensors['calibrated_my'][start_idx:end_idx], 
                            sensors['calibrated_mz'][start_idx:end_idx],
                            color=colors['iron'], alpha=0.5, linewidth=1, label='Iron')
        if has_fused:
            ax_combined.plot(sensors['fused_mx'][start_idx:end_idx], 
                            sensors['fused_my'][start_idx:end_idx], 
                            sensors['fused_mz'][start_idx:end_idx],
                            color=colors['fused'], alpha=0.7, linewidth=1.2, label='Residual')
        if has_filtered:
            ax_combined.plot(sensors['filtered_mx'][start_idx:end_idx], 
                            sensors['filtered_my'][start_idx:end_idx], 
                            sensors['filtered_mz'][start_idx:end_idx],
                            color=colors['filtered'], alpha=0.9, linewidth=1.5, label='Filtered')
        ax_combined.set_title('Combined Overlay', fontweight='bold', fontsize=10)
        ax_combined.set_xlabel('X', fontsize=8); ax_combined.set_ylabel('Y', fontsize=8); ax_combined.set_zlabel('Z', fontsize=8)
        ax_combined.legend(fontsize=7, loc='upper left')

        # Row 2: Statistics panel
        ax_stats = fig.add_subplot(gs[1, 2:])
        ax_stats.axis('off')

        raw_stats = compute_traj_stats(sensors['mx'][start_idx:end_idx], 
                                       sensors['my'][start_idx:end_idx], 
                                       sensors['mz'][start_idx:end_idx])

        stats_text = f"""
WINDOW {window_num} TRAJECTORY STATS
{'='*35}

◆ RAW
  Spread: {raw_stats['spread']:.2f} μT
  Path: {raw_stats['path_length']:.2f} μT
  Center: ({raw_stats['center'][0]:.1f}, {raw_stats['center'][1]:.1f}, {raw_stats['center'][2]:.1f})
"""
        if has_calibrated:
            iron_stats = compute_traj_stats(sensors['calibrated_mx'][start_idx:end_idx],
                                           sensors['calibrated_my'][start_idx:end_idx],
                                           sensors['calibrated_mz'][start_idx:end_idx])
            stats_text += f"""
◆ IRON
  Spread: {iron_stats['spread']:.2f} μT ({(iron_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {iron_stats['path_length']:.2f} μT
"""
        if has_fused:
            fused_stats = compute_traj_stats(sensors['fused_mx'][start_idx:end_idx],
                                            sensors['fused_my'][start_idx:end_idx],
                                            sensors['fused_mz'][start_idx:end_idx])
            stats_text += f"""
◆ RESIDUAL (Earth Subtracted)
  Spread: {fused_stats['spread']:.2f} μT ({(fused_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {fused_stats['path_length']:.2f} μT
  Center: ({fused_stats['center'][0]:.1f}, {fused_stats['center'][1]:.1f}, {fused_stats['center'][2]:.1f})
"""
        if has_filtered:
            filtered_stats = compute_traj_stats(sensors['filtered_mx'][start_idx:end_idx],
                                               sensors['filtered_my'][start_idx:end_idx],
                                               sensors['filtered_mz'][start_idx:end_idx])
            stats_text += f"""
◆ FILTERED
  Spread: {filtered_stats['spread']:.2f} μT ({(filtered_stats['spread']/raw_stats['spread']*100):.0f}%)
  Path: {filtered_stats['path_length']:.2f} μT
"""

        ax_stats.text(0.02, 0.98, stats_text.strip(), transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))

        # Save composite
        composite_path = window_subdir / 'trajectory_comparison.png'
        plt.savefig(composite_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return trajectory_images

    def create_raw_axis_images(self, session: Dict, processor: SensorDataProcessor) -> List[Path]:
        """Create detailed raw axis/orientation visualization images."""
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)
        output_files = []

        # Calibration status
        has_calibrated = sensors.get('_has_calibrated', False)
        has_fused = sensors.get('_has_fused', False)
        has_orientation = sensors.get('_has_orientation', False)

        # 1. Comprehensive Axis View
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Raw Axis Data: {session["filename"]}', fontsize=16, fontweight='bold')

        # Accelerometer
        axes[0, 0].plot(sensors['time'], sensors['ax'], 'r-', linewidth=1)
        axes[0, 0].set_title('Accelerometer X', fontweight='bold')
        axes[0, 0].set_ylabel('Acceleration (g)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(sensors['time'], sensors['ay'], 'g-', linewidth=1)
        axes[0, 1].set_title('Accelerometer Y', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(sensors['time'], sensors['az'], 'b-', linewidth=1)
        axes[0, 2].set_title('Accelerometer Z', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)

        # Gyroscope
        axes[1, 0].plot(sensors['time'], sensors['gx'], 'r-', linewidth=1)
        axes[1, 0].set_title('Gyroscope X', fontweight='bold')
        axes[1, 0].set_ylabel('Angular velocity (°/s)')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(sensors['time'], sensors['gy'], 'g-', linewidth=1)
        axes[1, 1].set_title('Gyroscope Y', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(sensors['time'], sensors['gz'], 'b-', linewidth=1)
        axes[1, 2].set_title('Gyroscope Z', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        # Magnetometer (with ALL calibration stages overlay)
        # Colors: Raw=gray dashed, Iron=blue, Fused=green, Filtered=red bold
        has_filtered = sensors.get('_has_filtered', False)

        # Mag X
        axes[2, 0].plot(sensors['time'], sensors['mx'], color='gray', alpha=0.4, linewidth=1, linestyle='--', label='Raw')
        if has_calibrated:
            axes[2, 0].plot(sensors['time'], sensors['calibrated_mx'], color='#1f77b4', alpha=0.6, linewidth=1, label='Iron')
        if has_fused:
            axes[2, 0].plot(sensors['time'], sensors['fused_mx'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Residual')
        if has_filtered:
            axes[2, 0].plot(sensors['time'], sensors['filtered_mx'], color='#d62728', alpha=0.9, linewidth=1.5, label='Filtered')
        axes[2, 0].set_title('Magnetometer X', fontweight='bold')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Value (μT)')
        axes[2, 0].legend(fontsize=6)
        axes[2, 0].grid(True, alpha=0.3)

        # Mag Y
        axes[2, 1].plot(sensors['time'], sensors['my'], color='gray', alpha=0.4, linewidth=1, linestyle='--', label='Raw')
        if has_calibrated:
            axes[2, 1].plot(sensors['time'], sensors['calibrated_my'], color='#1f77b4', alpha=0.6, linewidth=1, label='Iron')
        if has_fused:
            axes[2, 1].plot(sensors['time'], sensors['fused_my'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Residual')
        if has_filtered:
            axes[2, 1].plot(sensors['time'], sensors['filtered_my'], color='#d62728', alpha=0.9, linewidth=1.5, label='Filtered')
        axes[2, 1].set_title('Magnetometer Y', fontweight='bold')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].legend(fontsize=6)
        axes[2, 1].grid(True, alpha=0.3)

        # Mag Z
        axes[2, 2].plot(sensors['time'], sensors['mz'], color='gray', alpha=0.4, linewidth=1, linestyle='--', label='Raw')
        if has_calibrated:
            axes[2, 2].plot(sensors['time'], sensors['calibrated_mz'], color='#1f77b4', alpha=0.6, linewidth=1, label='Iron')
        if has_fused:
            axes[2, 2].plot(sensors['time'], sensors['fused_mz'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Residual')
        if has_filtered:
            axes[2, 2].plot(sensors['time'], sensors['filtered_mz'], color='#d62728', alpha=0.9, linewidth=1.5, label='Filtered')
        axes[2, 2].set_title('Magnetometer Z', fontweight='bold')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].legend(fontsize=6)
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f"raw_axes_{session['timestamp']}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        output_files.append(output_file)

        # 2. 3D Orientation View
        fig = plt.figure(figsize=(18, 6))

        # Accelerometer 3D
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(sensors['ax'], sensors['ay'], sensors['az'],
                              c=sensors['time'], cmap='viridis', s=5, alpha=0.6)
        ax1.set_title('Accelerometer 3D Space', fontweight='bold', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1, label='Time (s)')

        # Gyroscope 3D
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(sensors['gx'], sensors['gy'], sensors['gz'],
                              c=sensors['time'], cmap='plasma', s=5, alpha=0.6)
        ax2.set_title('Gyroscope 3D Space', fontweight='bold', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2, label='Time (s)')

        # Magnetometer 3D
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(sensors['mx'], sensors['my'], sensors['mz'],
                              c=sensors['time'], cmap='coolwarm', s=5, alpha=0.6)
        ax3.set_title('Magnetometer 3D Space', fontweight='bold', fontsize=12)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3, label='Time (s)')

        fig.suptitle(f'3D Orientation Spaces: {session["filename"]}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f"orientation_3d_{session['timestamp']}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        output_files.append(output_file)

        # 3. Orientation Track (if orientation data is available)
        if has_orientation and np.any(sensors['euler_roll'] != 0):
            fig = plt.figure(figsize=(18, 8))
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # Euler angles time series
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(sensors['time'], sensors['euler_roll'], 'r-', label='Roll', linewidth=1)
            ax1.plot(sensors['time'], sensors['euler_pitch'], 'g-', label='Pitch', linewidth=1)
            ax1.plot(sensors['time'], sensors['euler_yaw'], 'b-', label='Yaw', linewidth=1)
            ax1.set_title('Device Orientation (Euler Angles)', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Angle (degrees)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Roll vs Pitch scatter
            ax2 = fig.add_subplot(gs[1, 0])
            scatter = ax2.scatter(sensors['euler_roll'], sensors['euler_pitch'],
                                 c=sensors['time'], cmap='viridis', s=5, alpha=0.6)
            ax2.set_title('Roll vs Pitch', fontweight='bold')
            ax2.set_xlabel('Roll (degrees)')
            ax2.set_ylabel('Pitch (degrees)')
            plt.colorbar(scatter, ax=ax2, label='Time (s)')
            ax2.grid(True, alpha=0.3)

            # Magnetometer ALL stages magnitude comparison
            ax3 = fig.add_subplot(gs[1, 1])
            has_filtered = sensors.get('_has_filtered', False)

            # Plot all available magnitude stages
            ax3.plot(sensors['time'], sensors['mag_mag'], color='gray', alpha=0.4, linewidth=1, linestyle='--', label='Raw')
            if has_calibrated and 'calibrated_mag' in sensors:
                ax3.plot(sensors['time'], sensors['calibrated_mag'], color='#1f77b4', alpha=0.6, linewidth=1, label='Iron')
            if has_fused and 'fused_mag' in sensors:
                ax3.plot(sensors['time'], sensors['fused_mag'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Residual')
            if has_filtered and 'filtered_mag' in sensors:
                ax3.plot(sensors['time'], sensors['filtered_mag'], color='#d62728', alpha=0.9, linewidth=1.5, label='Filtered')

            ax3.set_title('Magnetometer: All Calibration Stages', fontweight='bold')
            ax3.set_ylabel('Magnitude (μT)')
            ax3.set_xlabel('Time (s)')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)

            fig.suptitle(f'Orientation Analysis: {session["filename"]}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            output_file = self.output_dir / f"orientation_track_{session['timestamp']}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            output_files.append(output_file)

        print(f"  Created raw axis images: {len(output_files)} files")
        return output_files

    def create_trajectory_comparison_image(self, session: Dict, processor: SensorDataProcessor) -> Optional[Dict]:
        """Create individual 3D trajectory images for all calibration stages.

        Generates separate images for each stage:
        - Raw (gray)
        - Iron Corrected (blue)
        - Fused (green)
        - Filtered (red)
        - Combined overlay
        - Statistics panel

        Returns dict with paths to individual images for flexible HTML display.
        """
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)

        has_calibrated = sensors.get('_has_calibrated', False)
        has_fused = sensors.get('_has_fused', False)
        has_filtered = sensors.get('_has_filtered', False)

        # Only create if we have calibration data
        if not (has_calibrated or has_fused or has_filtered):
            return None

        # Create subdirectory for trajectory images
        traj_dir = self.output_dir / f"trajectory_comparison_{session['timestamp']}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        colors = {
            'raw': 'gray',
            'iron': '#1f77b4',
            'fused': '#2ca02c',
            'filtered': '#d62728'
        }

        # Time-based coloring for trajectories
        n_samples = len(data)
        time_colors = np.linspace(0, 1, n_samples)

        # Helper function to plot 3D trajectory with time coloring
        def plot_trajectory(mx, my, mz, title, color, cmap_name, output_path):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory with time-based coloring
            for i in range(len(mx) - 1):
                ax.plot([mx[i], mx[i+1]], [my[i], my[i+1]], [mz[i], mz[i+1]],
                       color=plt.cm.get_cmap(cmap_name)(time_colors[i]), linewidth=1.5, alpha=0.8)

            # Start/end markers
            ax.scatter([mx[0]], [my[0]], [mz[0]], c='green', s=100, marker='o', label='Start', zorder=10)
            ax.scatter([mx[-1]], [my[-1]], [mz[-1]], c='red', s=100, marker='s', label='End', zorder=10)

            ax.set_title(f'{title}\n{session["filename"]} | {session["duration"]:.1f}s', 
                        fontweight='bold', fontsize=12, color=color)
            ax.set_xlabel('X (μT)', fontsize=10)
            ax.set_ylabel('Y (μT)', fontsize=10)
            ax.set_zlabel('Z (μT)', fontsize=10)
            ax.legend(fontsize=9, loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

        # Helper function to compute trajectory statistics
        def compute_traj_stats(mx, my, mz):
            spread_x = np.std(mx)
            spread_y = np.std(my)
            spread_z = np.std(mz)
            total_spread = np.sqrt(spread_x**2 + spread_y**2 + spread_z**2)

            dx = np.diff(mx)
            dy = np.diff(my)
            dz = np.diff(mz)
            path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

            bbox_x = np.max(mx) - np.min(mx)
            bbox_y = np.max(my) - np.min(my)
            bbox_z = np.max(mz) - np.min(mz)

            com = (np.mean(mx), np.mean(my), np.mean(mz))

            return {
                'spread': total_spread,
                'spread_xyz': (spread_x, spread_y, spread_z),
                'path_length': path_length,
                'bbox': (bbox_x, bbox_y, bbox_z),
                'center': com
            }

        # Dictionary to store image paths
        trajectory_images = {}

        # 1. Raw trajectory
        raw_path = traj_dir / 'raw_3d.png'
        plot_trajectory(sensors['mx'], sensors['my'], sensors['mz'],
                       'Raw Magnetometer', colors['raw'], 'Greys', raw_path)
        trajectory_images['raw'] = str(raw_path.relative_to(self.output_dir))

        # 2. Iron corrected (if available)
        if has_calibrated:
            iron_path = traj_dir / 'iron_3d.png'
            plot_trajectory(sensors['calibrated_mx'], sensors['calibrated_my'], sensors['calibrated_mz'],
                           'Iron Corrected', colors['iron'], 'Blues', iron_path)
            trajectory_images['iron'] = str(iron_path.relative_to(self.output_dir))

        # 3. Fused (if available)
        if has_fused:
            fused_path = traj_dir / 'fused_3d.png'
            plot_trajectory(sensors['fused_mx'], sensors['fused_my'], sensors['fused_mz'],
                           'Fused (Earth Subtracted)', colors['fused'], 'Greens', fused_path)
            trajectory_images['fused'] = str(fused_path.relative_to(self.output_dir))

        # 4. Filtered (if available)
        if has_filtered:
            filtered_path = traj_dir / 'filtered_3d.png'
            plot_trajectory(sensors['filtered_mx'], sensors['filtered_my'], sensors['filtered_mz'],
                           'Filtered (Kalman)', colors['filtered'], 'Reds', filtered_path)
            trajectory_images['filtered'] = str(filtered_path.relative_to(self.output_dir))

        # 5. Combined overlay trajectory
        combined_path = traj_dir / 'combined_overlay.png'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(sensors['mx'], sensors['my'], sensors['mz'],
                color=colors['raw'], alpha=0.3, linewidth=1, linestyle='--', label='Raw')
        if has_calibrated:
            ax.plot(sensors['calibrated_mx'], sensors['calibrated_my'], sensors['calibrated_mz'],
                    color=colors['iron'], alpha=0.5, linewidth=1.2, label='Iron')
        if has_fused:
            ax.plot(sensors['fused_mx'], sensors['fused_my'], sensors['fused_mz'],
                    color=colors['fused'], alpha=0.7, linewidth=1.5, label='Residual')
        if has_filtered:
            ax.plot(sensors['filtered_mx'], sensors['filtered_my'], sensors['filtered_mz'],
                    color=colors['filtered'], alpha=0.9, linewidth=1.8, label='Filtered')

        ax.set_title(f'Combined Trajectory Overlay\n{session["filename"]} | {session["duration"]:.1f}s',
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('X (μT)', fontsize=10)
        ax.set_ylabel('Y (μT)', fontsize=10)
        ax.set_zlabel('Z (μT)', fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(combined_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        trajectory_images['combined'] = str(combined_path.relative_to(self.output_dir))

        # 6. Statistics panel
        stats_path = traj_dir / 'statistics.png'
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        raw_stats = compute_traj_stats(sensors['mx'], sensors['my'], sensors['mz'])

        stats_text = f"""
TRAJECTORY STATISTICS
{'='*50}

◆ RAW MAGNETOMETER
  Spread: {raw_stats['spread']:.1f} μT
    (X: {raw_stats['spread_xyz'][0]:.1f}, Y: {raw_stats['spread_xyz'][1]:.1f}, Z: {raw_stats['spread_xyz'][2]:.1f})
  Path Length: {raw_stats['path_length']:.1f} μT
  Bounding Box: {raw_stats['bbox'][0]:.1f} × {raw_stats['bbox'][1]:.1f} × {raw_stats['bbox'][2]:.1f} μT
  Center: ({raw_stats['center'][0]:.1f}, {raw_stats['center'][1]:.1f}, {raw_stats['center'][2]:.1f})
"""

        if has_calibrated:
            iron_stats = compute_traj_stats(sensors['calibrated_mx'], sensors['calibrated_my'], sensors['calibrated_mz'])
            stats_text += f"""
◆ IRON CORRECTED
  Spread: {iron_stats['spread']:.1f} μT ({(iron_stats['spread']/raw_stats['spread']*100):.0f}% of raw)
  Path Length: {iron_stats['path_length']:.1f} μT
  Center: ({iron_stats['center'][0]:.1f}, {iron_stats['center'][1]:.1f}, {iron_stats['center'][2]:.1f})
"""

        if has_fused:
            fused_stats = compute_traj_stats(sensors['fused_mx'], sensors['fused_my'], sensors['fused_mz'])
            stats_text += f"""
◆ FUSED (Earth Subtracted)
  Spread: {fused_stats['spread']:.1f} μT ({(fused_stats['spread']/raw_stats['spread']*100):.0f}% of raw)
  Path Length: {fused_stats['path_length']:.1f} μT
  Center: ({fused_stats['center'][0]:.1f}, {fused_stats['center'][1]:.1f}, {fused_stats['center'][2]:.1f})
  ↳ Center near origin indicates good Earth field removal
"""

        if has_filtered:
            filtered_stats = compute_traj_stats(sensors['filtered_mx'], sensors['filtered_my'], sensors['filtered_mz'])
            stats_text += f"""
◆ FILTERED (Kalman)
  Spread: {filtered_stats['spread']:.1f} μT ({(filtered_stats['spread']/raw_stats['spread']*100):.0f}% of raw)
  Path Length: {filtered_stats['path_length']:.1f} μT
  Center: ({filtered_stats['center'][0]:.1f}, {filtered_stats['center'][1]:.1f}, {filtered_stats['center'][2]:.1f})
  ↳ Shorter path = smoother trajectory (less noise)
"""

        stats_text += f"""
{'='*50}
INTERPRETATION GUIDE:
• Spread: Lower = more concentrated signal
• Path Length: Lower = smoother trajectory
• Center near (0,0,0): Good Earth field removal
• Fused trajectory shows finger magnet signal only

Session: {session['filename']}
Duration: {session['duration']:.1f}s | Samples: {len(data)}
"""

        ax.text(0.05, 0.95, stats_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.95, edgecolor='#dee2e6'))

        plt.tight_layout()
        plt.savefig(stats_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        trajectory_images['statistics'] = str(stats_path.relative_to(self.output_dir))

        print(f"  Created {len(trajectory_images)} trajectory comparison images in {traj_dir.name}/")
        return trajectory_images

    def create_calibration_stages_image(self, session: Dict, processor: SensorDataProcessor) -> Optional[Path]:
        """Create a dedicated multi-stage magnetometer calibration comparison visualization.

        Shows all 4 calibration stages (Raw, Iron Corrected, Fused, Filtered)
        in a clear comparison format per the documentation requirements.
        """
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)

        has_calibrated = sensors.get('_has_calibrated', False)
        has_fused = sensors.get('_has_fused', False)
        has_filtered = sensors.get('_has_filtered', False)

        # Only create this visualization if we have at least some calibration data
        if not (has_calibrated or has_fused or has_filtered):
            return None

        # Create figure with 4 rows (one per axis + magnitude) and stages as overlaid traces
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25, width_ratios=[3, 1])

        # Title with calibration status summary
        stages_available = []
        if has_calibrated:
            stages_available.append('Iron')
        if has_fused:
            stages_available.append('Fused')
        if has_filtered:
            stages_available.append('Filtered')

        fig.suptitle(f'Magnetometer Calibration Stages: {session["filename"]}\n'
                     f'Available: Raw + {", ".join(stages_available)}',
                     fontsize=16, fontweight='bold')

        # Color scheme per documentation:
        # Raw (gray, dashed), Iron Corrected (blue), Fused (green), Filtered (red, bold)
        colors = {
            'raw': {'color': 'gray', 'alpha': 0.5, 'linewidth': 1, 'linestyle': '--', 'label': 'Raw'},
            'iron': {'color': '#1f77b4', 'alpha': 0.7, 'linewidth': 1.2, 'linestyle': '-', 'label': 'Iron Corrected'},
            'fused': {'color': '#2ca02c', 'alpha': 0.8, 'linewidth': 1.5, 'linestyle': '-', 'label': 'Fused (Earth Sub)'},
            'filtered': {'color': '#d62728', 'alpha': 0.95, 'linewidth': 2, 'linestyle': '-', 'label': 'Filtered (Kalman)'}
        }

        # Row 1: X-axis comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(sensors['time'], sensors['mx'], **{k: v for k, v in colors['raw'].items() if k != 'label'}, label=colors['raw']['label'])
        if has_calibrated:
            ax1.plot(sensors['time'], sensors['calibrated_mx'], **{k: v for k, v in colors['iron'].items() if k != 'label'}, label=colors['iron']['label'])
        if has_fused:
            ax1.plot(sensors['time'], sensors['fused_mx'], **{k: v for k, v in colors['fused'].items() if k != 'label'}, label=colors['fused']['label'])
        if has_filtered:
            ax1.plot(sensors['time'], sensors['filtered_mx'], **{k: v for k, v in colors['filtered'].items() if k != 'label'}, label=colors['filtered']['label'])
        ax1.set_title('Magnetometer X-Axis', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Value (μT)')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Row 2: Y-axis comparison
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(sensors['time'], sensors['my'], **{k: v for k, v in colors['raw'].items() if k != 'label'}, label=colors['raw']['label'])
        if has_calibrated:
            ax2.plot(sensors['time'], sensors['calibrated_my'], **{k: v for k, v in colors['iron'].items() if k != 'label'}, label=colors['iron']['label'])
        if has_fused:
            ax2.plot(sensors['time'], sensors['fused_my'], **{k: v for k, v in colors['fused'].items() if k != 'label'}, label=colors['fused']['label'])
        if has_filtered:
            ax2.plot(sensors['time'], sensors['filtered_my'], **{k: v for k, v in colors['filtered'].items() if k != 'label'}, label=colors['filtered']['label'])
        ax2.set_title('Magnetometer Y-Axis', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Value (μT)')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Row 3: Z-axis comparison
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(sensors['time'], sensors['mz'], **{k: v for k, v in colors['raw'].items() if k != 'label'}, label=colors['raw']['label'])
        if has_calibrated:
            ax3.plot(sensors['time'], sensors['calibrated_mz'], **{k: v for k, v in colors['iron'].items() if k != 'label'}, label=colors['iron']['label'])
        if has_fused:
            ax3.plot(sensors['time'], sensors['fused_mz'], **{k: v for k, v in colors['fused'].items() if k != 'label'}, label=colors['fused']['label'])
        if has_filtered:
            ax3.plot(sensors['time'], sensors['filtered_mz'], **{k: v for k, v in colors['filtered'].items() if k != 'label'}, label=colors['filtered']['label'])
        ax3.set_title('Magnetometer Z-Axis', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Value (μT)')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Row 4: Magnitude comparison
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(sensors['time'], sensors['mag_mag'], **{k: v for k, v in colors['raw'].items() if k != 'label'}, label=colors['raw']['label'])
        if has_calibrated and 'calibrated_mag' in sensors:
            ax4.plot(sensors['time'], sensors['calibrated_mag'], **{k: v for k, v in colors['iron'].items() if k != 'label'}, label=colors['iron']['label'])
        if has_fused and 'fused_mag' in sensors:
            ax4.plot(sensors['time'], sensors['fused_mag'], **{k: v for k, v in colors['fused'].items() if k != 'label'}, label=colors['fused']['label'])
        if has_filtered and 'filtered_mag' in sensors:
            ax4.plot(sensors['time'], sensors['filtered_mag'], **{k: v for k, v in colors['filtered'].items() if k != 'label'}, label=colors['filtered']['label'])
        ax4.set_title('Magnetometer Magnitude', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnitude (μT)')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Right column: Statistics per stage
        ax_stats = fig.add_subplot(gs[:, 1])
        ax_stats.axis('off')

        # Compute statistics for each stage including SNR metrics
        def compute_stats(mx, my, mz):
            mag = np.sqrt(mx**2 + my**2 + mz**2)
            # Basic stats
            mean_mag = np.mean(mag)
            std_mag = np.std(mag)
            # SNR = mean / std (signal-to-noise ratio)
            snr = mean_mag / std_mag if std_mag > 0 else float('inf')
            snr_db = 20 * np.log10(snr) if snr > 0 and snr != float('inf') else 0
            # Drift = max cumulative deviation from mean
            drift = np.max(np.abs(np.cumsum(mag - mean_mag))) / len(mag) if len(mag) > 0 else 0
            # Noise floor = std during the session (proxy for rest noise)
            noise_floor = std_mag
            return {
                'mean_x': np.mean(mx),
                'mean_y': np.mean(my),
                'mean_z': np.mean(mz),
                'std_x': np.std(mx),
                'std_y': np.std(my),
                'std_z': np.std(mz),
                'mean_mag': mean_mag,
                'std_mag': std_mag,
                'snr': snr,
                'snr_db': snr_db,
                'drift': drift,
                'noise_floor': noise_floor,
            }

        def snr_quality(snr_db):
            """Return quality indicator for SNR in dB"""
            if snr_db >= 20:
                return '✓ EXCELLENT'
            elif snr_db >= 10:
                return '✓ GOOD'
            elif snr_db >= 6:
                return '⚠ MARGINAL'
            else:
                return '✗ POOR'

        def noise_quality(noise):
            """Return quality indicator for noise floor in μT"""
            if noise < 1.0:
                return '✓ EXCELLENT'
            elif noise < 3.0:
                return '✓ GOOD'
            elif noise < 5.0:
                return '⚠ MARGINAL'
            else:
                return '✗ HIGH'

        raw_stats = compute_stats(sensors['mx'], sensors['my'], sensors['mz'])

        stats_text = f"""
CALIBRATION STAGE STATISTICS
{'='*40}

◆ RAW (Original Sensor Data)
  Mean: [{raw_stats['mean_x']:.1f}, {raw_stats['mean_y']:.1f}, {raw_stats['mean_z']:.1f}] μT
  Std:  [{raw_stats['std_x']:.1f}, {raw_stats['std_y']:.1f}, {raw_stats['std_z']:.1f}] μT
  Magnitude: {raw_stats['mean_mag']:.1f} ± {raw_stats['std_mag']:.1f} μT
"""

        if has_calibrated:
            iron_stats = compute_stats(sensors['calibrated_mx'], sensors['calibrated_my'], sensors['calibrated_mz'])
            stats_text += f"""
◆ IRON CORRECTED (Hard/Soft Iron)
  Mean: [{iron_stats['mean_x']:.1f}, {iron_stats['mean_y']:.1f}, {iron_stats['mean_z']:.1f}] μT
  Std:  [{iron_stats['std_x']:.1f}, {iron_stats['std_y']:.1f}, {iron_stats['std_z']:.1f}] μT
  Magnitude: {iron_stats['mean_mag']:.1f} ± {iron_stats['std_mag']:.1f} μT
"""

        if has_fused:
            fused_stats = compute_stats(sensors['fused_mx'], sensors['fused_my'], sensors['fused_mz'])
            stats_text += f"""
◆ FUSED (Earth Field Subtracted)
  Mean: [{fused_stats['mean_x']:.1f}, {fused_stats['mean_y']:.1f}, {fused_stats['mean_z']:.1f}] μT
  Std:  [{fused_stats['std_x']:.1f}, {fused_stats['std_y']:.1f}, {fused_stats['std_z']:.1f}] μT
  Magnitude: {fused_stats['mean_mag']:.1f} ± {fused_stats['std_mag']:.1f} μT
"""

        if has_filtered:
            filtered_stats = compute_stats(sensors['filtered_mx'], sensors['filtered_my'], sensors['filtered_mz'])
            stats_text += f"""
◆ FILTERED (Kalman Smoothed)
  Mean: [{filtered_stats['mean_x']:.1f}, {filtered_stats['mean_y']:.1f}, {filtered_stats['mean_z']:.1f}] μT
  Std:  [{filtered_stats['std_x']:.1f}, {filtered_stats['std_y']:.1f}, {filtered_stats['std_z']:.1f}] μT
  Magnitude: {filtered_stats['mean_mag']:.1f} ± {filtered_stats['std_mag']:.1f} μT
"""

        # Add Signal Quality Metrics section
        stats_text += f"""
{'='*40}
SIGNAL QUALITY METRICS
{'='*40}
"""
        # Determine best available stage for quality metrics
        if has_filtered:
            best_stats = filtered_stats
            best_name = 'Filtered'
        elif has_fused:
            best_stats = fused_stats
            best_name = 'Fused'
        elif has_calibrated:
            best_stats = iron_stats
            best_name = 'Iron'
        else:
            best_stats = raw_stats
            best_name = 'Raw'

        stats_text += f"""
Using: {best_name} data

  SNR:         {best_stats['snr']:.1f}:1 ({best_stats['snr_db']:.1f} dB)
               {snr_quality(best_stats['snr_db'])}

  Noise Floor: {best_stats['noise_floor']:.2f} μT
               {noise_quality(best_stats['noise_floor'])}

  Drift:       {best_stats['drift']:.2f} μT/sample
"""
        # Compare stages if multiple available
        if has_filtered and has_calibrated:
            improvement = ((raw_stats['std_mag'] - filtered_stats['std_mag']) / raw_stats['std_mag'] * 100) if raw_stats['std_mag'] > 0 else 0
            stats_text += f"""
  Noise Reduction: {improvement:.1f}% (raw→filtered)
"""
        stats_text += f"""
{'='*40}
COLOR LEGEND:
  ▬▬▬ Gray (dashed): Raw
  ▬▬▬ Blue: Iron Corrected
  ▬▬▬ Green: Fused
  ▬▬▬ Red (bold): Filtered
"""

        ax_stats.text(0.05, 0.95, stats_text.strip(), transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))

        # Save
        output_file = self.output_dir / f"calibration_stages_{session['timestamp']}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Created calibration stages image: {output_file.name}")
        return output_file


class HTMLDataGenerator:
    """Generates interactive HTML viewer for exploring visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_explorer(self, sessions_data: List[Dict]):
        """Generate interactive HTML Data file to explore all visualizations."""
        js_content = """const sessionsData = """ + json.dumps(sessions_data, indent=2) + """;"""

        output_file = self.output_dir / 'session-data.js'
        with open(output_file, 'w') as f:
            f.write(js_content)

        print(f"\n✅ Generated interactive HTML explorer: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='SIMCAP Data Visualization Pipeline')
    parser.add_argument('--data-dir', type=str, default='../data/GAMBIT',
                       help='Directory containing JSON data files')
    parser.add_argument('--output-dir', type=str, default='../images',
                       help='Output directory for generated visualizations (images branch worktree)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of sessions to process (for testing)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 70)
    print("SIMCAP DATA VISUALIZATION PIPELINE")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()

    # Initialize processor
    processor = SensorDataProcessor(data_dir)

    if not processor.sessions:
        print("❌ No sessions found. Exiting.")
        return

    # Limit sessions if requested
    sessions_to_process = processor.sessions
    if args.limit:
        sessions_to_process = sessions_to_process[:args.limit]
        print(f"⚠️  Processing limited to {args.limit} sessions\n")

    # Initialize visualizer
    visualizer = SessionVisualizer(output_dir)

    # Process each session
    sessions_data = []

    for i, session in enumerate(sessions_to_process, 1):
        print(f"\n[{i}/{len(sessions_to_process)}] Processing: {session['filename']}")

        try:
            # Generate composite image
            composite_path = visualizer.create_composite_session_image(session, processor)

            # Generate window images
            windows_info = visualizer.create_window_images(session, processor)

            # Generate raw axis images
            raw_images = visualizer.create_raw_axis_images(session, processor)

            # Generate calibration stages comparison image (if calibration data available)
            calibration_stages_path = visualizer.create_calibration_stages_image(session, processor)

            # Generate 3D trajectory comparison images (if calibration data available)
            trajectory_comparison_images = visualizer.create_trajectory_comparison_image(session, processor)

            # Store session data for HTML generation
            session_entry = {
                'filename': session['filename'],
                'timestamp': session['timestamp'],
                'duration': session['duration'],
                'sample_rate': session.get('sample_rate', 50),
                'composite_image': str(composite_path.relative_to(output_dir)),
                'windows': windows_info,
                'raw_images': [str(img.relative_to(output_dir)) for img in raw_images],
                # V2.1 extended fields
                'firmware_version': session.get('firmware_version'),
                'session_type': session.get('session_type', 'recording'),
                'labels': session.get('labels', []),
                'custom_labels': session.get('custom_labels', []),
                'calibration_types': session.get('calibration_types', []),
                # Extended metadata for VIZ display
                'device': session.get('device', 'unknown'),
                'subject_id': session.get('subject_id', ''),
                'environment': session.get('environment', ''),
                'hand': session.get('hand', ''),
                'magnet_config': session.get('magnet_config', ''),
                'magnet_type': session.get('magnet_type', ''),
                'notes': session.get('notes', ''),
                'location': session.get('location', {}),
                'calibration_state': session.get('calibration_state', {}),
            }

            # Include calibration stages image if it was created
            if calibration_stages_path:
                session_entry['calibration_stages_image'] = str(calibration_stages_path.relative_to(output_dir))

            # Include trajectory comparison images dict if it was created
            if trajectory_comparison_images:
                session_entry['trajectory_comparison_images'] = trajectory_comparison_images

            sessions_data.append(session_entry)

        except Exception as e:
            print(f"  ❌ Error processing session: {e}")
            continue

    # Generate HTML explorer
    print("\n" + "=" * 70)
    print("Generating HTML Explorer...")
    print("=" * 70)

    html_data_generator = HTMLDataGenerator(output_dir)
    html_file = html_data_generator.generate_explorer(sessions_data)

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(sessions_data)} session visualizations")
    print(f"Total windows: {sum(len(s['windows']) for s in sessions_data)}")
    print(f"\n🌐 Open in browser: {html_file}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
