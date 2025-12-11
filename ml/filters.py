"""
Advanced Filtering for Magnetic Finger Tracking

Provides:
- KalmanFilter3D: Multi-dimensional Kalman filter for 3D position/velocity tracking
- ParticleFilter: Multi-hypothesis particle filter for finger pose estimation
- magneticLikelihood: Dipole-based likelihood function

Python equivalent of filters.js for ML pipeline consistency.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class KalmanFilter3D:
    """
    3D Kalman Filter for position and velocity tracking.

    State vector: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    def __init__(self,
                 process_noise: float = 1.0,
                 measurement_noise: float = 1.0,
                 initial_covariance: float = 100.0,
                 dt: float = 0.02):
        """
        Initialize 3D Kalman filter.

        Args:
            process_noise: Process noise covariance (Q) - increased from 0.1 to 1.0 for better responsiveness
            measurement_noise: Measurement noise covariance (R)
            initial_covariance: Initial state uncertainty (P0)
            dt: Time step in seconds (default 0.02 = 50Hz)
        """
        # State dimension: position (3) + velocity (3) = 6
        self.state_dim = 6
        self.meas_dim = 3

        # State vector [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # Covariance matrix (6x6)
        self.P = np.eye(6) * initial_covariance

        # Process noise covariance
        self.Q = np.eye(6) * process_noise

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        # Time step
        self.dt = dt

        self.initialized = False

    def _get_F(self, dt: float) -> np.ndarray:
        """
        State transition matrix F.
        Models constant velocity: x_new = x + v*dt
        """
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F

    def _get_H(self) -> np.ndarray:
        """
        Measurement matrix H.
        We only measure position, not velocity.
        """
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        return H

    def predict(self, dt: Optional[float] = None):
        """
        Prediction step.

        Args:
            dt: Time step (uses self.dt if not provided)
        """
        if dt is None:
            dt = self.dt

        F = self._get_F(dt)

        # State prediction: x = F * x
        self.state = F @ self.state

        # Covariance prediction: P = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: Dict[str, float]) -> Dict[str, float]:
        """
        Update step with measurement.

        Args:
            measurement: {x, y, z} position measurement

        Returns:
            Filtered {x, y, z} position estimate
        """
        # Convert measurement to vector
        z = np.array([measurement['x'], measurement['y'], measurement['z']])

        # Initialize state on first measurement
        if not self.initialized:
            self.state[:3] = z
            self.state[3:] = 0  # Zero initial velocity
            self.initialized = True
            return {'x': float(z[0]), 'y': float(z[1]), 'z': float(z[2])}

        # Predict
        self.predict()

        # Update
        H = self._get_H()

        # Innovation: y = z - H * x
        y = z - H @ self.state

        # Innovation covariance: S = H * P * H.T + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P * H.T * inv(S)
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.state = self.state + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        # Return position estimate
        return {
            'x': float(self.state[0]),
            'y': float(self.state[1]),
            'z': float(self.state[2])
        }

    def get_position(self) -> Dict[str, float]:
        """Get current position estimate."""
        return {
            'x': float(self.state[0]),
            'y': float(self.state[1]),
            'z': float(self.state[2])
        }

    def get_velocity(self) -> Dict[str, float]:
        """Get current velocity estimate."""
        return {
            'x': float(self.state[3]),
            'y': float(self.state[4]),
            'z': float(self.state[5])
        }

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 100
        self.initialized = False


class ParticleFilter:
    """
    Particle filter for multi-hypothesis finger pose tracking.

    Handles ambiguous/multimodal magnetic field distributions.
    """

    def __init__(self,
                 num_particles: int = 500,
                 position_noise: float = 5.0,
                 velocity_noise: float = 2.0):
        """
        Initialize particle filter.

        Args:
            num_particles: Number of particles
            position_noise: Position noise for motion model (mm)
            velocity_noise: Velocity noise for motion model (mm/s)
        """
        self.num_particles = num_particles
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise

        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        self.initialized = False

    def initialize(self, initial_pose: Dict[str, Dict[str, float]]):
        """
        Initialize particles around initial pose.

        Args:
            initial_pose: {thumb: {x, y, z}, index: {x, y, z}, ...}
        """
        self.particles = []

        for _ in range(self.num_particles):
            particle = {}
            for finger, pos in initial_pose.items():
                # Add noise around initial position
                particle[finger] = {
                    'x': pos['x'] + np.random.randn() * self.position_noise,
                    'y': pos['y'] + np.random.randn() * self.position_noise,
                    'z': pos['z'] + np.random.randn() * self.position_noise
                }
            self.particles.append(particle)

        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True

    def predict(self, dt: float):
        """
        Prediction step: apply motion model.

        Args:
            dt: Time step in seconds
        """
        if not self.initialized:
            return

        # Simple random walk motion model
        for particle in self.particles:
            for finger in particle:
                particle[finger]['x'] += np.random.randn() * self.position_noise * np.sqrt(dt)
                particle[finger]['y'] += np.random.randn() * self.position_noise * np.sqrt(dt)
                particle[finger]['z'] += np.random.randn() * self.position_noise * np.sqrt(dt)

    def update(self, measurement: Dict[str, float], likelihood_fn):
        """
        Update step: reweight particles based on measurement.

        Args:
            measurement: {x, y, z} magnetic field measurement
            likelihood_fn: Function(particle, measurement) -> likelihood
        """
        if not self.initialized:
            return

        # Compute likelihood for each particle
        for i, particle in enumerate(self.particles):
            self.weights[i] = likelihood_fn(particle, measurement)

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights zero - reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Resample if effective sample size too low
        eff_sample_size = 1.0 / np.sum(self.weights ** 2)
        if eff_sample_size < self.num_particles / 2:
            self._resample()

    def _resample(self):
        """Systematic resampling."""
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)

        # Systematic resampling
        new_particles = []
        step = 1.0 / self.num_particles
        u = np.random.uniform(0, step)

        for i in range(self.num_particles):
            threshold = u + i * step
            idx = np.searchsorted(cumsum, threshold)
            idx = min(idx, len(self.particles) - 1)
            new_particles.append(self.particles[idx].copy())

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self) -> Dict[str, Dict[str, float]]:
        """
        Get weighted mean estimate of hand pose.

        Returns:
            {thumb: {x, y, z}, index: {x, y, z}, ...}
        """
        if not self.initialized or len(self.particles) == 0:
            return {}

        # Get finger names from first particle
        fingers = list(self.particles[0].keys())

        estimate = {}
        for finger in fingers:
            x = sum(p[finger]['x'] * w for p, w in zip(self.particles, self.weights))
            y = sum(p[finger]['y'] * w for p, w in zip(self.particles, self.weights))
            z = sum(p[finger]['z'] * w for p, w in zip(self.particles, self.weights))
            estimate[finger] = {'x': x, 'y': y, 'z': z}

        return estimate

    def reset(self):
        """Reset filter."""
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = False


def magnetic_dipole_field(magnet_pos: Dict[str, float],
                          magnet_moment: Dict[str, float],
                          sensor_pos: Dict[str, float] = None) -> Dict[str, float]:
    """
    Compute magnetic field at sensor due to a magnetic dipole.

    Uses dipole equation: B = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³

    Args:
        magnet_pos: Magnet position {x, y, z} in mm
        magnet_moment: Magnetic moment vector {x, y, z} in A·m²
        sensor_pos: Sensor position {x, y, z} in mm (default {0,0,0})

    Returns:
        Magnetic field {x, y, z} in arbitrary units
    """
    if sensor_pos is None:
        sensor_pos = {'x': 0, 'y': 0, 'z': 0}

    # Position vector from magnet to sensor (mm to m)
    rx = (sensor_pos['x'] - magnet_pos['x']) * 0.001
    ry = (sensor_pos['y'] - magnet_pos['y']) * 0.001
    rz = (sensor_pos['z'] - magnet_pos['z']) * 0.001

    # Distance
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    # Avoid singularity at r=0
    if r < 0.001:  # 1mm threshold
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    # Unit vector r̂
    rx_hat = rx / r
    ry_hat = ry / r
    rz_hat = rz / r

    # Dot product m·r̂
    m_dot_r = (magnet_moment['x'] * rx_hat +
               magnet_moment['y'] * ry_hat +
               magnet_moment['z'] * rz_hat)

    # Dipole field: B = k * (3(m·r̂)r̂ - m) / r³
    k = 1.0  # Simplified constant (units absorbed into calibration)
    r3 = r ** 3

    Bx = k * (3 * m_dot_r * rx_hat - magnet_moment['x']) / r3
    By = k * (3 * m_dot_r * ry_hat - magnet_moment['y']) / r3
    Bz = k * (3 * m_dot_r * rz_hat - magnet_moment['z']) / r3

    return {'x': Bx, 'y': By, 'z': Bz}


def magnetic_likelihood(particle: Dict[str, Dict[str, float]],
                       measurement: Dict[str, float],
                       magnet_config: Optional[Dict] = None) -> float:
    """
    Compute likelihood of measurement given particle pose using dipole model.

    Args:
        particle: Finger positions {thumb: {x,y,z}, index: {x,y,z}, ...}
        measurement: Measured magnetic field {x, y, z}
        magnet_config: Magnet configuration (moments and orientations)

    Returns:
        Likelihood probability (0 to 1)
    """
    # Default magnet configuration
    if magnet_config is None:
        magnet_config = {
            'thumb': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},
            'index': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},
            'middle': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},
            'ring': {'moment': {'x': 0, 'y': 0, 'z': 0.01}},
            'pinky': {'moment': {'x': 0, 'y': 0, 'z': 0.01}}
        }

    sensor_pos = {'x': 0, 'y': 0, 'z': 0}

    # Calculate expected field as sum of all dipole contributions
    expected = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        if finger in particle and finger in magnet_config:
            field = magnetic_dipole_field(
                particle[finger],
                magnet_config[finger]['moment'],
                sensor_pos
            )
            expected['x'] += field['x']
            expected['y'] += field['y']
            expected['z'] += field['z']

    # Compute residual
    dx = measurement['x'] - expected['x']
    dy = measurement['y'] - expected['y']
    dz = measurement['z'] - expected['z']

    residual = np.sqrt(dx**2 + dy**2 + dz**2)

    # Gaussian likelihood
    sigma = 10.0  # Tunable parameter
    likelihood = np.exp(-(residual**2) / (2 * sigma**2))

    return likelihood


def decorate_telemetry_with_filtering(telemetry_data: List[Dict],
                                     filter_instance: KalmanFilter3D) -> List[Dict]:
    """
    Decorate telemetry data with Kalman-filtered magnetometer fields.

    IMPORTANT: Preserves raw data, only adds filtered_ fields.

    Args:
        telemetry_data: List of telemetry dictionaries
        filter_instance: KalmanFilter3D instance

    Returns:
        List with added filtered_mx, filtered_my, filtered_mz fields
    """
    decorated = []

    for sample in telemetry_data:
        decorated_sample = sample.copy()

        # Use calibrated fields if available, otherwise raw
        mx = sample.get('calibrated_mx', sample.get('mx', 0))
        my = sample.get('calibrated_my', sample.get('my', 0))
        mz = sample.get('calibrated_mz', sample.get('mz', 0))

        try:
            filtered = filter_instance.update({'x': mx, 'y': my, 'z': mz})
            decorated_sample['filtered_mx'] = filtered['x']
            decorated_sample['filtered_my'] = filtered['y']
            decorated_sample['filtered_mz'] = filtered['z']
        except Exception as e:
            # Filtering failed, skip decoration
            pass

        decorated.append(decorated_sample)

    return decorated
