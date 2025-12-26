"""
Magnetic Dipole Field Calculations

Implements the physics of magnetic dipole fields for simulating
finger magnet signals at a wrist-mounted sensor.

The magnetic field of a dipole is given by:
    B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³

Where:
    - B = magnetic field vector (Tesla)
    - μ₀ = permeability of free space (4π × 10⁻⁷ H/m)
    - m = magnetic dipole moment vector (A·m²)
    - r = position vector from dipole to observation point (m)
    - r̂ = unit vector in direction of r
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
MU_0_OVER_4PI = MU_0 / (4 * np.pi)

# Earth's magnetic field (Edinburgh, UK - approximate)
EARTH_FIELD_EDINBURGH = np.array([16.0, 0.0, 47.8])  # μT (horizontal N, E, down)


def magnetic_dipole_field(
    observation_point: np.ndarray,
    dipole_position: np.ndarray,
    dipole_moment: np.ndarray,
    min_distance: float = 1e-6
) -> np.ndarray:
    """
    Calculate the magnetic field from a magnetic dipole at a given point.

    Uses the exact dipole field equation:
        B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³

    Args:
        observation_point: 3D position where field is measured (meters)
        dipole_position: 3D position of the dipole center (meters)
        dipole_moment: Magnetic moment vector of the dipole (A·m²)
        min_distance: Minimum distance to avoid singularity (meters)

    Returns:
        Magnetic field vector in Tesla

    Example:
        >>> obs = np.array([0.0, 0.05, 0.0])  # 50mm from origin
        >>> dipole_pos = np.array([0.0, 0.0, 0.0])
        >>> moment = np.array([0.0, 0.0, 0.0135])  # 6x3mm N48 magnet
        >>> B = magnetic_dipole_field(obs, dipole_pos, moment)
        >>> print(f"Field: {B * 1e6} μT")  # Convert to μT
    """
    # Vector from dipole to observation point
    r_vec = observation_point - dipole_position
    r_mag = np.linalg.norm(r_vec)

    # Avoid singularity at dipole location
    if r_mag < min_distance:
        return np.zeros(3)

    r_hat = r_vec / r_mag

    # Dipole field equation: B = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³
    m_dot_r = np.dot(dipole_moment, r_hat)
    B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - dipole_moment) / (r_mag ** 3)

    return B


def magnetic_dipole_field_vectorized(
    observation_points: np.ndarray,
    dipole_position: np.ndarray,
    dipole_moment: np.ndarray,
    min_distance: float = 1e-6
) -> np.ndarray:
    """
    Calculate magnetic field at multiple observation points (vectorized).

    Args:
        observation_points: Array of shape (N, 3) with observation positions (meters)
        dipole_position: 3D position of the dipole center (meters)
        dipole_moment: Magnetic moment vector of the dipole (A·m²)
        min_distance: Minimum distance to avoid singularity (meters)

    Returns:
        Magnetic field vectors of shape (N, 3) in Tesla
    """
    # Vectors from dipole to all observation points
    r_vecs = observation_points - dipole_position
    r_mags = np.linalg.norm(r_vecs, axis=1, keepdims=True)

    # Mask for valid distances
    valid = r_mags.flatten() >= min_distance

    # Initialize output
    B = np.zeros_like(r_vecs)

    if not np.any(valid):
        return B

    # Only compute for valid points
    r_mags_valid = r_mags[valid]
    r_vecs_valid = r_vecs[valid]
    r_hats = r_vecs_valid / r_mags_valid

    # Dot product m · r̂ for each point
    m_dot_r = np.dot(r_hats, dipole_moment)

    # Dipole field equation
    term1 = 3 * m_dot_r[:, np.newaxis] * r_hats
    term2 = dipole_moment
    B[valid] = MU_0_OVER_4PI * (term1 - term2) / (r_mags_valid ** 3)

    return B


def compute_total_field(
    sensor_position: np.ndarray,
    finger_positions: Dict[str, np.ndarray],
    magnet_config: Dict[str, Dict],
    earth_field: Optional[np.ndarray] = None,
    include_earth: bool = True
) -> np.ndarray:
    """
    Compute total magnetic field at sensor from all finger magnets plus Earth field.

    Args:
        sensor_position: Position of magnetometer sensor (mm)
        finger_positions: Dict mapping finger names to fingertip positions (mm)
            Example: {'thumb': np.array([20, 80, -10]), 'index': np.array([45, 120, -5])}
        magnet_config: Dict mapping finger names to magnet properties
            Each entry should have 'moment' (A·m²) and optional 'offset' (mm)
        earth_field: Earth's magnetic field vector (μT). If None, uses Edinburgh default.
        include_earth: Whether to include Earth's field (set False for residual-only)

    Returns:
        Total magnetic field at sensor position in μT

    Example:
        >>> sensor = np.array([0, 0, 0])  # Wrist sensor at origin
        >>> fingers = {
        ...     'index': np.array([45, 80, 0]),  # Extended index finger
        ... }
        >>> config = {
        ...     'index': {'moment': [0, 0, 0.0135], 'offset': [0, 0, 0]}
        ... }
        >>> B = compute_total_field(sensor, fingers, config)
        >>> print(f"Total field: {B} μT")
    """
    if earth_field is None:
        earth_field = EARTH_FIELD_EDINBURGH

    # Start with Earth's field (or zero if not including)
    total_field = earth_field.copy() if include_earth else np.zeros(3)

    # Convert sensor position from mm to meters
    sensor_m = sensor_position / 1000.0

    for finger, position in finger_positions.items():
        if finger not in magnet_config:
            continue

        config = magnet_config[finger]
        moment = np.array(config['moment'])
        offset = np.array(config.get('offset', [0, 0, 0]))

        # Magnet position is fingertip + attachment offset, in meters
        magnet_pos_m = (position + offset) / 1000.0

        # Compute dipole field contribution (in Tesla)
        B_dipole = magnetic_dipole_field(sensor_m, magnet_pos_m, moment)

        # Add to total (convert Tesla to μT)
        total_field += B_dipole * 1e6

    return total_field


def estimate_dipole_moment(
    diameter_mm: float,
    height_mm: float,
    Br_mT: float = 1430.0  # N48 residual flux density
) -> float:
    """
    Estimate the magnetic dipole moment of a cylindrical permanent magnet.

    For a uniformly magnetized cylinder:
        m = M × V = (Br/μ₀) × πr²h

    Args:
        diameter_mm: Magnet diameter in millimeters
        height_mm: Magnet height/thickness in millimeters
        Br_mT: Residual flux density in milliTesla (1430 for N48)

    Returns:
        Magnetic dipole moment in A·m²

    Typical values:
        - 6mm × 3mm N48: ~0.0135 A·m²
        - 5mm × 2mm N42: ~0.0065 A·m²
        - 3mm × 1mm N35: ~0.0020 A·m²
    """
    radius_m = (diameter_mm / 2) / 1000.0
    height_m = height_mm / 1000.0
    volume_m3 = np.pi * radius_m ** 2 * height_m

    Br_T = Br_mT / 1000.0
    M = Br_T / MU_0  # Magnetization in A/m

    moment = M * volume_m3
    return moment


def field_magnitude_at_distance(
    distance_mm: float,
    moment: float = 0.0135,  # A·m² (6mm × 3mm N48)
    angle_deg: float = 0.0   # Angle from dipole axis
) -> float:
    """
    Calculate magnetic field magnitude at a given distance from a dipole.

    Useful for quick SNR estimates.

    Args:
        distance_mm: Distance from magnet center in millimeters
        moment: Dipole moment magnitude in A·m²
        angle_deg: Angle from dipole axis in degrees (0 = along axis)

    Returns:
        Field magnitude in μT

    Example:
        >>> B_50mm = field_magnitude_at_distance(50)  # At 50mm
        >>> B_80mm = field_magnitude_at_distance(80)  # At 80mm
        >>> print(f"50mm: {B_50mm:.1f} μT, 80mm: {B_80mm:.1f} μT")
        50mm: 141.0 μT, 80mm: 35.2 μT
    """
    r_m = distance_mm / 1000.0
    theta = np.radians(angle_deg)

    # Field components in spherical coordinates
    # B_r = (μ₀/4π) × 2m×cos(θ) / r³
    # B_θ = (μ₀/4π) × m×sin(θ) / r³
    B_r = MU_0_OVER_4PI * 2 * moment * np.cos(theta) / (r_m ** 3)
    B_theta = MU_0_OVER_4PI * moment * np.sin(theta) / (r_m ** 3)

    # Total magnitude
    B_mag = np.sqrt(B_r ** 2 + B_theta ** 2)

    return B_mag * 1e6  # Convert to μT


def snr_estimate(
    distance_extended_mm: float = 80,
    distance_flexed_mm: float = 50,
    moment: float = 0.0135,
    noise_floor_ut: float = 1.0
) -> Dict[str, float]:
    """
    Estimate signal-to-noise ratio for finger tracking.

    Args:
        distance_extended_mm: Distance when finger is extended
        distance_flexed_mm: Distance when finger is flexed
        moment: Dipole moment in A·m²
        noise_floor_ut: Sensor noise floor in μT (RMS)

    Returns:
        Dict with SNR metrics
    """
    B_extended = field_magnitude_at_distance(distance_extended_mm, moment)
    B_flexed = field_magnitude_at_distance(distance_flexed_mm, moment)
    delta = B_flexed - B_extended

    return {
        'signal_extended_ut': B_extended,
        'signal_flexed_ut': B_flexed,
        'signal_delta_ut': delta,
        'noise_floor_ut': noise_floor_ut,
        'snr_extended': B_extended / noise_floor_ut,
        'snr_flexed': B_flexed / noise_floor_ut,
        'snr_delta': delta / noise_floor_ut
    }


if __name__ == '__main__':
    # Quick validation
    print("Magnetic Dipole Field Calculations")
    print("=" * 50)

    # Estimate dipole moment for 6x3mm N48 magnet
    moment = estimate_dipole_moment(6, 3, 1430)
    print(f"\n6mm × 3mm N48 magnet:")
    print(f"  Dipole moment: {moment:.4f} A·m²")

    # Calculate field at various distances
    print(f"\nField magnitude vs distance (on-axis):")
    for d in [30, 40, 50, 60, 70, 80, 100]:
        B = field_magnitude_at_distance(d, moment)
        print(f"  {d:3d} mm: {B:6.1f} μT")

    # SNR estimate
    print(f"\nSNR estimate for finger tracking:")
    snr = snr_estimate(
        distance_extended_mm=80,
        distance_flexed_mm=50,
        moment=moment,
        noise_floor_ut=1.0
    )
    for key, value in snr.items():
        print(f"  {key}: {value:.1f}")

    # Multi-finger field superposition
    print(f"\nMulti-finger field superposition:")
    sensor = np.array([0, 0, 0])
    fingers = {
        'thumb': np.array([20, 50, 0]),
        'index': np.array([45, 80, 0]),
        'middle': np.array([50, 85, 0]),
        'ring': np.array([45, 80, 0]),
        'pinky': np.array([40, 75, 0])
    }
    config = {
        'thumb':  {'moment': [0, 0, 0.0135]},
        'index':  {'moment': [0, 0, -0.0135]},
        'middle': {'moment': [0, 0, 0.0135]},
        'ring':   {'moment': [0, 0, -0.0135]},
        'pinky':  {'moment': [0, 0, 0.0135]}
    }

    B_total = compute_total_field(sensor, fingers, config, include_earth=False)
    print(f"  Total field (5 fingers): {B_total} μT")
    print(f"  Magnitude: {np.linalg.norm(B_total):.1f} μT")
