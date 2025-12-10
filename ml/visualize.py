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
        ax1.set_ylabel('Value (raw ADC)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. Gyroscope 3-axis time series
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(sensors['time'], sensors['gx'], label='X', alpha=0.7, linewidth=1)
        ax2.plot(sensors['time'], sensors['gy'], label='Y', alpha=0.7, linewidth=1)
        ax2.plot(sensors['time'], sensors['gz'], label='Z', alpha=0.7, linewidth=1)
        ax2.set_title('Gyroscope (3-axis)', fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Value (raw ADC)')
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
            calib_status = 'Fused (Iron + Earth Field)'
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

        # Stage 3: Fused / Earth field subtracted (green tones) - if available
        if has_fused:
            ax3.plot(sensors['time'], sensors['fused_mx'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Fused X')
            ax3.plot(sensors['time'], sensors['fused_my'], color='#98df8a', alpha=0.7, linewidth=1.2, label='Fused Y')
            ax3.plot(sensors['time'], sensors['fused_mz'], color='#006400', alpha=0.7, linewidth=1.2, label='Fused Z')

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
            ax4.plot(sensors['time'], sensors['fused_mag'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Mag (Fused)')
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
        """Create individual images for each 1-second window in the session."""
        data = session['data']
        sensors = processor.extract_sensor_arrays(data)

        window_size = 50  # 1 second at 50Hz
        n_samples = len(data)
        n_windows = n_samples // window_size

        window_info = []

        # Create output directory for this session's windows
        session_dir = self.output_dir / f"windows_{session['timestamp']}"
        session_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size

            if end_idx > n_samples:
                break

            # Create figure with expanded layout for 3 3D plots
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

            window_time_start = start_idx / 50.0
            window_time_end = end_idx / 50.0

            fig.suptitle(f'Window {i+1}/{n_windows} | Time: {window_time_start:.1f}s - {window_time_end:.1f}s',
                        fontsize=14, fontweight='bold')

            # Extract window data
            time_window = sensors['time'][start_idx:end_idx]

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

            window_info.append({
                'window_num': i + 1,
                'time_start': window_time_start,
                'time_end': window_time_end,
                'filename': output_file.name,
                'filepath': str(output_file.relative_to(self.output_dir)),
                'color': color_hex,
                'accel_mag_mean': float(sensors['accel_mag'][start_idx:end_idx].mean()),
                'gyro_mag_mean': float(sensors['gyro_mag'][start_idx:end_idx].mean()),
            })

        print(f"  Created {len(window_info)} window images in {session_dir.name}/")
        return window_info

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
        axes[0, 0].set_ylabel('Value (raw)')
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
        axes[1, 0].set_ylabel('Value (raw)')
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
            axes[2, 0].plot(sensors['time'], sensors['fused_mx'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Fused')
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
            axes[2, 1].plot(sensors['time'], sensors['fused_my'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Fused')
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
            axes[2, 2].plot(sensors['time'], sensors['fused_mz'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Fused')
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
                ax3.plot(sensors['time'], sensors['fused_mag'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Fused')
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


class HTMLGenerator:
    """Generates interactive HTML viewer for exploring visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_explorer(self, sessions_data: List[Dict]):
        """Generate interactive HTML file to explore all visualizations."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIMCAP Data Visualization Explorer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }

        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: bold;
        }

        .stat-box .number {
            font-size: 2em;
            display: block;
        }

        .stat-box .label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .session-list {
            display: grid;
            gap: 20px;
        }

        .session-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .session-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .session-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .session-title {
            font-size: 1.3em;
            font-weight: bold;
        }

        .session-info {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .expand-icon {
            font-size: 1.5em;
            transition: transform 0.3s;
        }

        .session-card.expanded .expand-icon {
            transform: rotate(180deg);
        }

        .session-content {
            display: none;
            padding: 20px;
        }

        .session-card.expanded .session-content {
            display: block;
        }

        .image-section {
            margin-bottom: 30px;
        }

        .image-section.hidden {
            display: none;
        }

        .section-title {
            font-size: 1.4em;
            color: #667eea;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .composite-image {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }

        .windows-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .window-card {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.2s;
            cursor: pointer;
            border: 3px solid transparent;
        }

        .window-card:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .window-color-bar {
            height: 8px;
        }

        .window-preview {
            width: 100%;
            aspect-ratio: 4/3;
            object-fit: contain;
            background: #f0f0f0;
        }

        .window-info {
            padding: 10px;
        }

        .window-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }

        .window-time {
            font-size: 0.85em;
            color: #666;
        }

        .window-stats {
            font-size: 0.75em;
            color: #888;
            margin-top: 5px;
        }

        .raw-images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 15px;
        }

        .raw-image {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            padding: 20px;
        }

        .modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }

        .modal-close {
            position: absolute;
            top: 30px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }

        .modal-close:hover {
            color: #667eea;
        }

        .filter-bar {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }

        .filter-label {
            font-weight: bold;
            color: #667eea;
        }

        input[type="text"], select {
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .no-sessions {
            text-align: center;
            padding: 60px;
            background: white;
            border-radius: 15px;
            color: #999;
            font-size: 1.2em;
        }

        .filter-group {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .checkbox-group {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 0.95em;
            color: #555;
        }

        .checkbox-label input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: #667eea;
        }

        .toggle-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            color: #667eea;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }

        .toggle-btn:hover {
            background: #667eea;
            color: white;
        }

        .toggle-btn.active {
            background: #667eea;
            color: white;
        }

        .divider {
            width: 1px;
            height: 30px;
            background: #e0e0e0;
            margin: 0 5px;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 SIMCAP Data Visualization Explorer</h1>
            <p class="subtitle">Interactive viewer for 9-axis IMU sensor data and gesture recognition</p>
            <div class="stats">
                <div class="stat-box">
                    <span class="number" id="total-sessions">0</span>
                    <span class="label">Sessions</span>
                </div>
                <div class="stat-box">
                    <span class="number" id="total-windows">0</span>
                    <span class="label">Windows</span>
                </div>
                <div class="stat-box">
                    <span class="number" id="total-duration">0</span>
                    <span class="label">Total Duration (s)</span>
                </div>
                <div class="stat-box" style="cursor: pointer;" onclick="openModal('snr_analysis_report.png')">
                    <span class="number">📊</span>
                    <span class="label">SNR Report</span>
                </div>
            </div>
        </header>

        <div class="filter-bar">
            <div class="filter-group">
                <span class="filter-label">Search:</span>
                <input type="text" id="search-box" placeholder="Search sessions..." style="min-width: 200px;">
            </div>

            <div class="divider"></div>

            <div class="filter-group">
                <span class="filter-label">Sort:</span>
                <select id="sort-select">
                    <option value="timestamp-desc">Newest First</option>
                    <option value="timestamp-asc">Oldest First</option>
                    <option value="duration-desc">Longest First</option>
                    <option value="duration-asc">Shortest First</option>
                </select>
            </div>

            <div class="divider"></div>

            <div class="filter-group">
                <span class="filter-label">Show:</span>
                <div class="checkbox-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="show-composite" checked>
                        📊 Composite
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="show-calibration" checked>
                        🧲 Calibration
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="show-windows" checked>
                        🔍 Windows
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="show-raw" checked>
                        📐 Raw Axis
                    </label>
                </div>
            </div>

            <div class="divider"></div>

            <div class="filter-group">
                <span class="filter-label">Sessions:</span>
                <div class="action-buttons">
                    <button class="toggle-btn" id="expand-all-btn" onclick="expandAll()">Expand All</button>
                    <button class="toggle-btn" id="collapse-all-btn" onclick="collapseAll()">Collapse All</button>
                </div>
            </div>
        </div>

        <div class="session-list" id="session-list"></div>
    </div>

    <div class="modal" id="image-modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modal-image">
    </div>

    <script>
        const sessionsData = """ + json.dumps(sessions_data, indent=2) + """;

        let filteredSessions = [...sessionsData];
        let defaultExpanded = false;

        // View settings
        let viewSettings = {
            showComposite: true,
            showCalibration: true,
            showWindows: true,
            showRaw: true
        };

        function initializePage() {
            updateStats();
            renderSessions();
            setupEventListeners();
        }

        function updateStats() {
            document.getElementById('total-sessions').textContent = filteredSessions.length;

            const totalWindows = filteredSessions.reduce((sum, s) => sum + s.windows.length, 0);
            document.getElementById('total-windows').textContent = totalWindows;

            const totalDuration = filteredSessions.reduce((sum, s) => sum + s.duration, 0);
            document.getElementById('total-duration').textContent = totalDuration.toFixed(1);
        }

        function renderSessions() {
            const container = document.getElementById('session-list');

            if (filteredSessions.length === 0) {
                container.innerHTML = '<div class="no-sessions">No sessions found matching your criteria</div>';
                return;
            }

            container.innerHTML = filteredSessions.map((session, idx) => `
                <div class="session-card ${defaultExpanded ? 'expanded' : ''}" id="session-${idx}">
                    <div class="session-header" onclick="toggleSession(${idx})">
                        <div>
                            <div class="session-title">${session.filename}</div>
                            <div class="session-info">${formatTimestamp(session.timestamp)} | ${session.duration.toFixed(1)}s | ${session.windows.length} windows</div>
                        </div>
                        <div class="expand-icon">▼</div>
                    </div>
                    <div class="session-content">
                        <div class="image-section composite-section ${viewSettings.showComposite ? '' : 'hidden'}">
                            <h3 class="section-title">📊 Composite Session View</h3>
                            <img src="${session.composite_image}" class="composite-image" onclick="openModal(this.src)" alt="Composite view">
                        </div>

                        ${session.calibration_stages_image ? `
                        <div class="image-section calibration-section ${viewSettings.showCalibration ? '' : 'hidden'}">
                            <h3 class="section-title">🧲 Magnetometer Calibration Stages (Raw → Iron → Fused → Filtered)</h3>
                            <img src="${session.calibration_stages_image}" class="composite-image" onclick="openModal(this.src)" alt="Calibration stages comparison">
                        </div>
                        ` : ''}

                        <div class="image-section windows-section ${viewSettings.showWindows ? '' : 'hidden'}">
                            <h3 class="section-title">🔍 Per-Second Windows (${session.windows.length})</h3>
                            <div class="windows-grid">
                                ${session.windows.map(w => `
                                    <div class="window-card" onclick="openModal('${w.filepath}')">
                                        <div class="window-color-bar" style="background-color: ${w.color}"></div>
                                        <img src="${w.filepath}" class="window-preview" alt="Window ${w.window_num}">
                                        <div class="window-info">
                                            <div class="window-title">Window ${w.window_num}</div>
                                            <div class="window-time">${w.time_start.toFixed(2)}s - ${w.time_end.toFixed(2)}s</div>
                                            <div class="window-stats">
                                                Accel: ${w.accel_mag_mean.toFixed(0)} | Gyro: ${w.gyro_mag_mean.toFixed(0)}
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>

                        <div class="image-section raw-section ${viewSettings.showRaw ? '' : 'hidden'}">
                            <h3 class="section-title">📐 Raw Axis & Orientation Views</h3>
                            <div class="raw-images">
                                ${session.raw_images.map(img => `
                                    <img src="${img}" class="raw-image" onclick="openModal(this.src)" alt="Raw axis view">
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function toggleSession(idx) {
            const card = document.getElementById(`session-${idx}`);
            card.classList.toggle('expanded');
        }

        function expandAll() {
            document.querySelectorAll('.session-card').forEach(card => {
                card.classList.add('expanded');
            });
            defaultExpanded = true;
        }

        function collapseAll() {
            document.querySelectorAll('.session-card').forEach(card => {
                card.classList.remove('expanded');
            });
            defaultExpanded = false;
        }

        function updateViewSettings() {
            viewSettings.showComposite = document.getElementById('show-composite').checked;
            viewSettings.showCalibration = document.getElementById('show-calibration').checked;
            viewSettings.showWindows = document.getElementById('show-windows').checked;
            viewSettings.showRaw = document.getElementById('show-raw').checked;

            // Update visibility of sections
            document.querySelectorAll('.composite-section').forEach(el => {
                el.classList.toggle('hidden', !viewSettings.showComposite);
            });
            document.querySelectorAll('.calibration-section').forEach(el => {
                el.classList.toggle('hidden', !viewSettings.showCalibration);
            });
            document.querySelectorAll('.windows-section').forEach(el => {
                el.classList.toggle('hidden', !viewSettings.showWindows);
            });
            document.querySelectorAll('.raw-section').forEach(el => {
                el.classList.toggle('hidden', !viewSettings.showRaw);
            });
        }

        function openModal(imageSrc) {
            const modal = document.getElementById('image-modal');
            const modalImg = document.getElementById('modal-image');
            modal.classList.add('active');
            modalImg.src = imageSrc;
        }

        function closeModal() {
            document.getElementById('image-modal').classList.remove('active');
        }

        function formatTimestamp(timestamp) {
            try {
                const date = new Date(timestamp);
                return date.toLocaleString();
            } catch {
                return timestamp;
            }
        }

        function setupEventListeners() {
            // Search functionality
            document.getElementById('search-box').addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                filteredSessions = sessionsData.filter(s =>
                    s.filename.toLowerCase().includes(query) ||
                    s.timestamp.toLowerCase().includes(query)
                );
                updateStats();
                renderSessions();
            });

            // Sort functionality
            document.getElementById('sort-select').addEventListener('change', (e) => {
                const sortBy = e.target.value;

                filteredSessions.sort((a, b) => {
                    switch(sortBy) {
                        case 'timestamp-desc':
                            return b.timestamp.localeCompare(a.timestamp);
                        case 'timestamp-asc':
                            return a.timestamp.localeCompare(b.timestamp);
                        case 'duration-desc':
                            return b.duration - a.duration;
                        case 'duration-asc':
                            return a.duration - b.duration;
                        default:
                            return 0;
                    }
                });

                renderSessions();
            });

            // View filter checkboxes
            document.getElementById('show-composite').addEventListener('change', updateViewSettings);
            document.getElementById('show-calibration').addEventListener('change', updateViewSettings);
            document.getElementById('show-windows').addEventListener('change', updateViewSettings);
            document.getElementById('show-raw').addEventListener('change', updateViewSettings);

            // Close modal on click outside
            document.getElementById('image-modal').addEventListener('click', (e) => {
                if (e.target.id === 'image-modal') {
                    closeModal();
                }
            });

            // Close modal on Escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    closeModal();
                }
            });
        }

        // Initialize on page load
        initializePage();
    </script>
</body>
</html>"""

        output_file = self.output_dir / 'index.html'
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"\n✅ Generated interactive HTML explorer: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='SIMCAP Data Visualization Pipeline')
    parser.add_argument('--data-dir', type=str, default='../data/GAMBIT',
                       help='Directory containing JSON data files')
    parser.add_argument('--output-dir', type=str, default='../visualizations',
                       help='Output directory for generated visualizations')
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

            # Store session data for HTML generation
            session_entry = {
                'filename': session['filename'],
                'timestamp': session['timestamp'],
                'duration': session['duration'],
                'composite_image': str(composite_path.relative_to(output_dir)),
                'windows': windows_info,
                'raw_images': [str(img.relative_to(output_dir)) for img in raw_images],
            }

            # Include calibration stages image if it was created
            if calibration_stages_path:
                session_entry['calibration_stages_image'] = str(calibration_stages_path.relative_to(output_dir))

            sessions_data.append(session_entry)

        except Exception as e:
            print(f"  ❌ Error processing session: {e}")
            continue

    # Generate HTML explorer
    print("\n" + "=" * 70)
    print("Generating HTML Explorer...")
    print("=" * 70)

    html_generator = HTMLGenerator(output_dir)
    html_file = html_generator.generate_explorer(sessions_data)

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
