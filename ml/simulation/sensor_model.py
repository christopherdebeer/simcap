"""
Magnetometer Sensor Simulation

Models the characteristics of the MMC5603NJ magnetometer used in GAMBIT devices,
including noise, quantization, bias, and soft iron distortion.

The goal is to generate synthetic sensor readings that closely match real sensor
behavior, enabling training data that transfers well to real devices.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SensorCharacteristics:
    """MMC5603NJ magnetometer specifications and noise model."""

    # Sensor range and resolution
    range_gauss: float = 30.0          # ±30 Gauss full scale
    bits: int = 16                      # ADC resolution
    lsb_per_gauss: float = 1024.0      # Sensitivity (1024 LSB/Gauss)

    # Noise characteristics
    noise_density_ut: float = 1.0       # RMS noise floor (μT)

    # Bias and drift (hard iron simulation)
    bias_ut: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_drift_ut_per_hour: float = 0.5  # Temporal drift

    # Soft iron distortion (non-spherical response)
    soft_iron_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Timing
    sample_rate_hz: float = 26.0        # Default GAMBIT sample rate

    @property
    def range_ut(self) -> float:
        """Range in microTesla."""
        return self.range_gauss * 100  # 1 Gauss = 100 μT

    @property
    def lsb_per_ut(self) -> float:
        """Sensitivity in LSB per microTesla."""
        return self.lsb_per_gauss / 100  # Convert from Gauss to μT


class MMC5603Simulator:
    """
    Simulate MMC5603NJ magnetometer readings.

    This class adds realistic sensor effects to ideal magnetic field values:
    1. Hard iron bias (constant offset)
    2. Soft iron distortion (axis scaling/rotation)
    3. Gaussian noise
    4. Quantization
    5. Range limiting (saturation)

    The output matches the format expected by SIMCAP's telemetry pipeline.
    """

    def __init__(
        self,
        noise_ut: Optional[float] = None,
        bias_ut: Optional[np.ndarray] = None,
        soft_iron: Optional[np.ndarray] = None,
        characteristics: Optional[SensorCharacteristics] = None
    ):
        """
        Initialize the sensor simulator.

        Args:
            noise_ut: Override noise level (μT RMS). If None, uses characteristics.
            bias_ut: Override hard iron bias (μT). If None, uses characteristics.
            soft_iron: Override soft iron matrix. If None, uses characteristics.
            characteristics: Full sensor characteristics. If None, uses defaults.
        """
        self.chars = characteristics or SensorCharacteristics()

        # Apply overrides
        self._noise_ut = noise_ut if noise_ut is not None else self.chars.noise_density_ut
        self._bias_ut = bias_ut if bias_ut is not None else self.chars.bias_ut.copy()
        self._soft_iron = soft_iron if soft_iron is not None else self.chars.soft_iron_matrix.copy()

        # Time tracking for drift
        self._elapsed_time = 0.0

    def randomize_parameters(
        self,
        noise_range: Tuple[float, float] = (1.0, 5.0),
        bias_range: Tuple[float, float] = (-40, 40),
        soft_iron_deviation: float = 0.15,
        realistic_mode: bool = True
    ):
        """
        Randomize sensor parameters for domain randomization.

        Call this to simulate different devices or environmental conditions.
        When realistic_mode=True, uses parameters calibrated from real sensor data.

        Args:
            noise_range: Min/max noise level (μT)
            bias_range: Min/max bias per axis (μT)
            soft_iron_deviation: Max deviation in soft iron matrix elements
            realistic_mode: Use parameters calibrated from real data
        """
        if realistic_mode:
            # Parameters calibrated from real GAMBIT sensor data
            # Real data shows ~66 µT mean magnitude vs Earth field ~50 µT
            # Increased noise range based on sim-to-real analysis
            self._noise_ut = np.random.uniform(3.0, 12.0)  # Increased from 2-8

            # Larger bias to match real sensor offsets (20-50 µT per axis typical)
            self._bias_ut = np.array([
                np.random.uniform(20, 60) * np.random.choice([-1, 1]),  # Increased from 10-40
                np.random.uniform(10, 50) * np.random.choice([-1, 1]),  # Increased from 5-30
                np.random.uniform(-40, 40)  # Increased from -20, 20
            ])

            # More aggressive soft iron distortion
            scale_perturbation = np.diag(np.random.uniform(0.85, 1.15, size=3))  # Axis scaling
            rotation_perturbation = np.random.uniform(-0.15, 0.15, size=(3, 3))
            self._soft_iron = scale_perturbation @ (np.eye(3) + rotation_perturbation)
        else:
            self._noise_ut = np.random.uniform(*noise_range)
            self._bias_ut = np.random.uniform(bias_range[0], bias_range[1], size=3)

            perturbation = np.random.uniform(
                -soft_iron_deviation,
                soft_iron_deviation,
                size=(3, 3)
            )
            self._soft_iron = np.eye(3) + perturbation

    def measure(
        self,
        true_field_ut: np.ndarray,
        add_noise: bool = True,
        add_bias: bool = True,
        add_soft_iron: bool = True,
        quantize: bool = True
    ) -> Dict:
        """
        Simulate a magnetometer measurement from true magnetic field.

        Args:
            true_field_ut: True magnetic field vector in μT
            add_noise: Add Gaussian sensor noise
            add_bias: Add hard iron bias
            add_soft_iron: Apply soft iron distortion
            quantize: Quantize to sensor resolution

        Returns:
            Dict with sensor reading in multiple formats:
            - mx, my, mz: Raw LSB values (as sensor would output)
            - mx_ut, my_ut, mz_ut: Unit-converted values (μT)
            - _true_*: Ground truth values (for validation)
        """
        field = true_field_ut.copy()

        # Apply soft iron distortion (measured field is distorted)
        if add_soft_iron:
            field = self._soft_iron @ field

        # Add hard iron bias (constant offset from nearby metal)
        if add_bias:
            field = field + self._bias_ut

        # Add Gaussian noise
        if add_noise:
            noise = np.random.normal(0, self._noise_ut, size=3)
            field = field + noise

        # Convert to raw LSB values
        if quantize:
            raw_lsb = np.round(field * self.chars.lsb_per_ut).astype(int)

            # Clamp to sensor range
            max_lsb = int(self.chars.range_ut * self.chars.lsb_per_ut)
            raw_lsb = np.clip(raw_lsb, -max_lsb, max_lsb)

            # Convert back to μT with quantization effects
            field_quantized = raw_lsb / self.chars.lsb_per_ut
        else:
            # Use ideal values (floating point)
            raw_lsb = (field * self.chars.lsb_per_ut).astype(int)
            field_quantized = field

        return {
            # Raw sensor output (LSB) - matches GAMBIT data format
            'mx': int(raw_lsb[0]),
            'my': int(raw_lsb[1]),
            'mz': int(raw_lsb[2]),

            # Unit-converted (μT) - decorated fields
            'mx_ut': float(field_quantized[0]),
            'my_ut': float(field_quantized[1]),
            'mz_ut': float(field_quantized[2]),

            # Ground truth (hidden, for validation only)
            '_true_mx_ut': float(true_field_ut[0]),
            '_true_my_ut': float(true_field_ut[1]),
            '_true_mz_ut': float(true_field_ut[2]),

            # Measurement metadata
            '_noise_added': float(self._noise_ut) if add_noise else 0,
            '_bias_applied': add_bias,
            '_quantized': quantize
        }

    def measure_sequence(
        self,
        fields: np.ndarray,
        sample_rate: Optional[float] = None
    ) -> list:
        """
        Simulate multiple sequential measurements.

        Useful for generating a full session of data.

        Args:
            fields: Array of shape (N, 3) with true field vectors in μT
            sample_rate: Sample rate (Hz). Used for timing. Default from characteristics.

        Returns:
            List of measurement dicts
        """
        sample_rate = sample_rate or self.chars.sample_rate_hz
        dt = 1.0 / sample_rate

        measurements = []
        for i, field in enumerate(fields):
            measurement = self.measure(field)
            measurement['dt'] = dt
            measurement['t'] = i * dt * 1000  # milliseconds
            measurements.append(measurement)

        return measurements

    def get_calibration_info(self) -> Dict:
        """
        Return sensor calibration parameters.

        This can be used to generate a calibration file that matches
        the simulated sensor characteristics.
        """
        return {
            'hardIronOffset': {
                'x': float(self._bias_ut[0]),
                'y': float(self._bias_ut[1]),
                'z': float(self._bias_ut[2])
            },
            'softIronMatrix': self._soft_iron.flatten().tolist(),
            'noiseLevel': float(self._noise_ut),
            'synthetic': True
        }


class IMUSimulator:
    """
    Simulate accelerometer and gyroscope readings.

    Provides basic IMU simulation for generating complete session data.
    The magnetic field simulation focuses on magnetometer data, but sessions
    need accelerometer and gyroscope values too.
    """

    def __init__(
        self,
        accel_noise_g: float = 0.005,      # Accelerometer noise (g)
        gyro_noise_dps: float = 0.1,        # Gyroscope noise (deg/s)
        gravity_direction: np.ndarray = None  # Gravity vector
    ):
        """
        Initialize IMU simulator.

        Args:
            accel_noise_g: Accelerometer noise in g
            gyro_noise_dps: Gyroscope noise in degrees per second
            gravity_direction: Direction of gravity. Default: [0, 0, -1] (palm down)
        """
        self.accel_noise_g = accel_noise_g
        self.gyro_noise_dps = gyro_noise_dps
        self.gravity = gravity_direction if gravity_direction is not None else np.array([0, 0, -1])

        # LSM6DS3 conversion factors
        self.accel_lsb_per_g = 8192     # ±2g range
        self.gyro_lsb_per_dps = 114.28  # ±245 dps range

    def measure_static(self, orientation: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate IMU measurement for a static (not moving) sensor.

        Args:
            orientation: Optional rotation to apply to gravity

        Returns:
            Dict with accelerometer and gyroscope readings
        """
        # Gravity in sensor frame
        gravity_g = self.gravity.copy()
        if orientation is not None:
            # orientation could be a rotation matrix
            gravity_g = orientation @ gravity_g

        # Add noise
        accel_g = gravity_g + np.random.normal(0, self.accel_noise_g, size=3)
        gyro_dps = np.random.normal(0, self.gyro_noise_dps, size=3)

        # Convert to LSB
        accel_lsb = (accel_g * self.accel_lsb_per_g).astype(int)
        gyro_lsb = (gyro_dps * self.gyro_lsb_per_dps).astype(int)

        return {
            # Raw LSB values
            'ax': int(accel_lsb[0]),
            'ay': int(accel_lsb[1]),
            'az': int(accel_lsb[2]),
            'gx': int(gyro_lsb[0]),
            'gy': int(gyro_lsb[1]),
            'gz': int(gyro_lsb[2]),

            # Unit-converted values
            'ax_g': float(accel_g[0]),
            'ay_g': float(accel_g[1]),
            'az_g': float(accel_g[2]),
            'gx_dps': float(gyro_dps[0]),
            'gy_dps': float(gyro_dps[1]),
            'gz_dps': float(gyro_dps[2]),

            # Motion detection (static = not moving)
            'isMoving': False,
            'accelStd': 0,
            'gyroStd': 0
        }


if __name__ == '__main__':
    print("Sensor Simulation Test")
    print("=" * 50)

    # Create sensor simulator
    sensor = MMC5603Simulator()

    # Test with a known field
    true_field = np.array([20.0, -10.0, 45.0])  # μT
    print(f"\nTrue field: {true_field} μT")

    # Measure without any effects
    clean = sensor.measure(true_field, add_noise=False, add_bias=False,
                           add_soft_iron=False, quantize=False)
    print(f"Clean measurement: [{clean['mx_ut']:.2f}, {clean['my_ut']:.2f}, {clean['mz_ut']:.2f}] μT")

    # Measure with all effects
    noisy = sensor.measure(true_field)
    print(f"Noisy measurement: [{noisy['mx_ut']:.2f}, {noisy['my_ut']:.2f}, {noisy['mz_ut']:.2f}] μT")
    print(f"Raw LSB: [{noisy['mx']}, {noisy['my']}, {noisy['mz']}]")

    # Randomize and measure again
    print("\nWith domain randomization:")
    sensor.randomize_parameters(
        noise_range=(0.5, 3.0),
        bias_range=(-30, 30),
        soft_iron_deviation=0.15
    )
    randomized = sensor.measure(true_field)
    print(f"Randomized: [{randomized['mx_ut']:.2f}, {randomized['my_ut']:.2f}, {randomized['mz_ut']:.2f}] μT")
    print(f"Calibration info: {sensor.get_calibration_info()}")

    # Test IMU simulator
    print("\nIMU Simulation:")
    imu = IMUSimulator()
    imu_reading = imu.measure_static()
    print(f"Accelerometer: [{imu_reading['ax_g']:.4f}, {imu_reading['ay_g']:.4f}, {imu_reading['az_g']:.4f}] g")
    print(f"Gyroscope: [{imu_reading['gx_dps']:.2f}, {imu_reading['gy_dps']:.2f}, {imu_reading['gz_dps']:.2f}] dps")
