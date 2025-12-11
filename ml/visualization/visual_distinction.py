"""
SIMCAP Visual Distinction Engine

Creates visually distinct representations based on sensor patterns.
"""

from typing import Dict, Tuple
import numpy as np
import colorsys


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
