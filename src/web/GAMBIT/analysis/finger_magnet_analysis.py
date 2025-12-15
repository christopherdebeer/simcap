#!/usr/bin/env python3
"""
GAMBIT Finger Magnet Analysis Script (Alternating Polarity)

Deep analysis of raw and computed sensor values for sessions with
finger magnets using alternating polarity configuration.

Analyzes:
- Raw magnetometer values and statistical properties
- Magnetic field magnitude and direction
- Alternating polarity patterns for finger detection
- Correlation with motion (accelerometer/gyroscope)
- Computed orientation and calibration values
- Spectral analysis of magnetic field variations
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math
from collections import defaultdict


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def mean(arr: List[float]) -> float:
    """Calculate mean of list."""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def std(arr: List[float]) -> float:
    """Calculate standard deviation."""
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)


def min_max(arr: List[float]) -> Tuple[float, float]:
    """Get min and max of list."""
    if not arr:
        return (0.0, 0.0)
    return (min(arr), max(arr))


def percentile(arr: List[float], p: float) -> float:
    """Calculate percentile (0-100)."""
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    k = (len(sorted_arr) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_arr[int(k)]
    return sorted_arr[int(f)] * (c - k) + sorted_arr[int(c)] * (k - f)


def correlation(a: List[float], b: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0
    a_mean = mean(a)
    b_mean = mean(b)
    num = sum((ai - a_mean) * (bi - b_mean) for ai, bi in zip(a, b))
    denom_a = sum((ai - a_mean)**2 for ai in a)
    denom_b = sum((bi - b_mean)**2 for bi in b)
    denom = math.sqrt(denom_a * denom_b)
    if denom < 1e-10:
        return 0.0
    return num / denom


def derivative(arr: List[float], dt_arr: List[float]) -> List[float]:
    """Calculate derivative using finite differences."""
    if len(arr) < 2:
        return []
    result = []
    for i in range(1, len(arr)):
        dt = dt_arr[i] if i < len(dt_arr) else 0.02  # default 50Hz
        if dt > 0:
            result.append((arr[i] - arr[i-1]) / dt)
        else:
            result.append(0.0)
    return result


def moving_average(arr: List[float], window: int = 10) -> List[float]:
    """Calculate moving average."""
    if len(arr) < window:
        return arr.copy()
    result = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result.append(mean(arr[start:i+1]))
    return result


# ============================================================================
# MAGNETIC FIELD ANALYSIS
# ============================================================================

def magnitude_3d(x: float, y: float, z: float) -> float:
    """Calculate 3D vector magnitude."""
    return math.sqrt(x*x + y*y + z*z)


def analyze_magnetic_field(samples: List[Dict]) -> Dict:
    """Comprehensive magnetic field analysis."""
    if not samples:
        return {'error': 'No samples'}

    # Extract raw magnetometer values
    mx_raw = [s.get('mx', 0) for s in samples]
    my_raw = [s.get('my', 0) for s in samples]
    mz_raw = [s.get('mz', 0) for s in samples]

    # Extract converted values (microTesla)
    mx_ut = [s.get('mx_ut', s.get('mx', 0) * 0.09765625) for s in samples]
    my_ut = [s.get('my_ut', s.get('my', 0) * 0.09765625) for s in samples]
    mz_ut = [s.get('mz_ut', s.get('mz', 0) * 0.09765625) for s in samples]

    # Extract filtered values (Kalman filtered)
    filtered_mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    filtered_my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    filtered_mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]

    # Calculate magnitudes
    raw_magnitudes = [magnitude_3d(mx_raw[i], my_raw[i], mz_raw[i]) for i in range(len(samples))]
    ut_magnitudes = [magnitude_3d(mx_ut[i], my_ut[i], mz_ut[i]) for i in range(len(samples))]
    filtered_magnitudes = [magnitude_3d(filtered_mx[i], filtered_my[i], filtered_mz[i]) for i in range(len(samples))]

    # Residual magnitudes (from calibration)
    residual_mags = [s.get('residual_magnitude', 0) for s in samples]

    return {
        'raw': {
            'mx': {'mean': mean(mx_raw), 'std': std(mx_raw), 'min': min(mx_raw), 'max': max(mx_raw)},
            'my': {'mean': mean(my_raw), 'std': std(my_raw), 'min': min(my_raw), 'max': max(my_raw)},
            'mz': {'mean': mean(mz_raw), 'std': std(mz_raw), 'min': min(mz_raw), 'max': max(mz_raw)},
            'magnitude': {'mean': mean(raw_magnitudes), 'std': std(raw_magnitudes), 'min': min(raw_magnitudes), 'max': max(raw_magnitudes)},
        },
        'microTesla': {
            'mx': {'mean': mean(mx_ut), 'std': std(mx_ut), 'min': min(mx_ut), 'max': max(mx_ut)},
            'my': {'mean': mean(my_ut), 'std': std(my_ut), 'min': min(my_ut), 'max': max(my_ut)},
            'mz': {'mean': mean(mz_ut), 'std': std(mz_ut), 'min': min(mz_ut), 'max': max(mz_ut)},
            'magnitude': {'mean': mean(ut_magnitudes), 'std': std(ut_magnitudes), 'min': min(ut_magnitudes), 'max': max(ut_magnitudes)},
        },
        'filtered': {
            'mx': {'mean': mean(filtered_mx), 'std': std(filtered_mx), 'min': min(filtered_mx), 'max': max(filtered_mx)},
            'my': {'mean': mean(filtered_my), 'std': std(filtered_my), 'min': min(filtered_my), 'max': max(filtered_my)},
            'mz': {'mean': mean(filtered_mz), 'std': std(filtered_mz), 'min': min(filtered_mz), 'max': max(filtered_mz)},
            'magnitude': {'mean': mean(filtered_magnitudes), 'std': std(filtered_magnitudes), 'min': min(filtered_magnitudes), 'max': max(filtered_magnitudes)},
        },
        'residual': {
            'mean': mean(residual_mags),
            'std': std(residual_mags),
            'min': min(residual_mags) if residual_mags else 0,
            'max': max(residual_mags) if residual_mags else 0,
        },
        'series': {
            'mx_ut': mx_ut,
            'my_ut': my_ut,
            'mz_ut': mz_ut,
            'filtered_mx': filtered_mx,
            'filtered_my': filtered_my,
            'filtered_mz': filtered_mz,
            'magnitude': ut_magnitudes,
        }
    }


def detect_polarity_changes(samples: List[Dict], threshold: float = 50.0) -> Dict:
    """
    Detect polarity changes in magnetic field (for alternating polarity magnets).

    Finger magnets with alternating N/S polarity create distinct patterns:
    - North pole approaching: positive field direction
    - South pole approaching: negative field direction
    """
    if len(samples) < 10:
        return {'error': 'Insufficient samples'}

    # Use filtered magnetometer values
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]

    # Smooth the data
    mx_smooth = moving_average(mx, 5)
    my_smooth = moving_average(my, 5)
    mz_smooth = moving_average(mz, 5)

    # Detect sign changes (polarity flips)
    polarity_events = []

    for axis_name, axis_data in [('mx', mx_smooth), ('my', my_smooth), ('mz', mz_smooth)]:
        sign_changes = 0
        change_indices = []

        for i in range(1, len(axis_data)):
            if abs(axis_data[i]) > threshold and abs(axis_data[i-1]) > threshold:
                if (axis_data[i] > 0) != (axis_data[i-1] > 0):
                    sign_changes += 1
                    change_indices.append(i)

        polarity_events.append({
            'axis': axis_name,
            'sign_changes': sign_changes,
            'change_indices': change_indices[:20],  # First 20
        })

    # Calculate polarity pattern metrics
    total_changes = sum(e['sign_changes'] for e in polarity_events)

    # Dominant axis for polarity detection
    dominant_axis = max(polarity_events, key=lambda x: x['sign_changes'])

    return {
        'polarity_events': polarity_events,
        'total_sign_changes': total_changes,
        'dominant_axis': dominant_axis['axis'],
        'dominant_axis_changes': dominant_axis['sign_changes'],
        'changes_per_second': total_changes / (len(samples) / 50.0) if samples else 0,  # Assuming 50Hz
    }


def analyze_magnetic_clusters(samples: List[Dict], n_clusters: int = 5) -> Dict:
    """
    Cluster magnetic field readings to identify distinct finger positions.
    Uses simple k-means-like clustering without external dependencies.
    """
    if len(samples) < n_clusters * 10:
        return {'error': 'Insufficient samples for clustering'}

    # Extract 3D magnetometer data
    points = []
    for s in samples:
        mx = s.get('filtered_mx', s.get('mx_ut', 0))
        my = s.get('filtered_my', s.get('my_ut', 0))
        mz = s.get('filtered_mz', s.get('mz_ut', 0))
        points.append((mx, my, mz))

    # Simple clustering: divide into magnitude bins
    magnitudes = [magnitude_3d(p[0], p[1], p[2]) for p in points]
    min_mag, max_mag = min(magnitudes), max(magnitudes)

    if max_mag - min_mag < 10:
        return {'error': 'Low magnetic field variation'}

    # Create bins
    bin_size = (max_mag - min_mag) / n_clusters
    clusters = defaultdict(list)

    for i, (mag, point) in enumerate(zip(magnitudes, points)):
        bin_idx = min(int((mag - min_mag) / bin_size), n_clusters - 1)
        clusters[bin_idx].append({
            'index': i,
            'mx': point[0],
            'my': point[1],
            'mz': point[2],
            'magnitude': mag
        })

    # Analyze each cluster
    cluster_stats = []
    for bin_idx in sorted(clusters.keys()):
        pts = clusters[bin_idx]
        if not pts:
            continue

        mx_vals = [p['mx'] for p in pts]
        my_vals = [p['my'] for p in pts]
        mz_vals = [p['mz'] for p in pts]
        mag_vals = [p['magnitude'] for p in pts]

        cluster_stats.append({
            'cluster_id': bin_idx,
            'count': len(pts),
            'percentage': 100.0 * len(pts) / len(points),
            'mx': {'mean': mean(mx_vals), 'std': std(mx_vals)},
            'my': {'mean': mean(my_vals), 'std': std(my_vals)},
            'mz': {'mean': mean(mz_vals), 'std': std(mz_vals)},
            'magnitude': {'mean': mean(mag_vals), 'std': std(mag_vals)},
        })

    return {
        'n_clusters': len(cluster_stats),
        'clusters': cluster_stats,
        'magnitude_range': max_mag - min_mag,
    }


# ============================================================================
# MOTION ANALYSIS
# ============================================================================

def analyze_motion(samples: List[Dict]) -> Dict:
    """Analyze accelerometer and gyroscope data."""
    if not samples:
        return {'error': 'No samples'}

    # Accelerometer (raw)
    ax_raw = [s.get('ax', 0) for s in samples]
    ay_raw = [s.get('ay', 0) for s in samples]
    az_raw = [s.get('az', 0) for s in samples]

    # Accelerometer (G's)
    ax_g = [s.get('ax_g', s.get('ax', 0) / 8192.0) for s in samples]
    ay_g = [s.get('ay_g', s.get('ay', 0) / 8192.0) for s in samples]
    az_g = [s.get('az_g', s.get('az', 0) / 8192.0) for s in samples]

    # Gyroscope (raw)
    gx_raw = [s.get('gx', 0) for s in samples]
    gy_raw = [s.get('gy', 0) for s in samples]
    gz_raw = [s.get('gz', 0) for s in samples]

    # Gyroscope (dps)
    gx_dps = [s.get('gx_dps', s.get('gx', 0) * 0.00875) for s in samples]
    gy_dps = [s.get('gy_dps', s.get('gy', 0) * 0.00875) for s in samples]
    gz_dps = [s.get('gz_dps', s.get('gz', 0) * 0.00875) for s in samples]

    # Acceleration magnitude
    accel_mag = [magnitude_3d(ax_g[i], ay_g[i], az_g[i]) for i in range(len(samples))]

    # Gyroscope magnitude (rotation rate)
    gyro_mag = [magnitude_3d(gx_dps[i], gy_dps[i], gz_dps[i]) for i in range(len(samples))]

    # Motion detection metrics
    accel_std_vals = [s.get('accelStd', 0) for s in samples]
    gyro_std_vals = [s.get('gyroStd', 0) for s in samples]
    is_moving = [s.get('isMoving', False) for s in samples]
    moving_percentage = 100.0 * sum(1 for m in is_moving if m) / len(is_moving) if is_moving else 0

    return {
        'accelerometer': {
            'raw': {
                'ax': {'mean': mean(ax_raw), 'std': std(ax_raw), 'min': min(ax_raw), 'max': max(ax_raw)},
                'ay': {'mean': mean(ay_raw), 'std': std(ay_raw), 'min': min(ay_raw), 'max': max(ay_raw)},
                'az': {'mean': mean(az_raw), 'std': std(az_raw), 'min': min(az_raw), 'max': max(az_raw)},
            },
            'g': {
                'ax': {'mean': mean(ax_g), 'std': std(ax_g), 'min': min(ax_g), 'max': max(ax_g)},
                'ay': {'mean': mean(ay_g), 'std': std(ay_g), 'min': min(ay_g), 'max': max(ay_g)},
                'az': {'mean': mean(az_g), 'std': std(az_g), 'min': min(az_g), 'max': max(az_g)},
            },
            'magnitude': {'mean': mean(accel_mag), 'std': std(accel_mag), 'min': min(accel_mag), 'max': max(accel_mag)},
        },
        'gyroscope': {
            'raw': {
                'gx': {'mean': mean(gx_raw), 'std': std(gx_raw), 'min': min(gx_raw), 'max': max(gx_raw)},
                'gy': {'mean': mean(gy_raw), 'std': std(gy_raw), 'min': min(gy_raw), 'max': max(gy_raw)},
                'gz': {'mean': mean(gz_raw), 'std': std(gz_raw), 'min': min(gz_raw), 'max': max(gz_raw)},
            },
            'dps': {
                'gx': {'mean': mean(gx_dps), 'std': std(gx_dps), 'min': min(gx_dps), 'max': max(gx_dps)},
                'gy': {'mean': mean(gy_dps), 'std': std(gy_dps), 'min': min(gy_dps), 'max': max(gy_dps)},
                'gz': {'mean': mean(gz_dps), 'std': std(gz_dps), 'min': min(gz_dps), 'max': max(gz_dps)},
            },
            'magnitude': {'mean': mean(gyro_mag), 'std': std(gyro_mag), 'min': min(gyro_mag), 'max': max(gyro_mag)},
        },
        'motion_detection': {
            'accelStd_mean': mean(accel_std_vals),
            'gyroStd_mean': mean(gyro_std_vals),
            'moving_percentage': moving_percentage,
        },
        'series': {
            'ax_g': ax_g,
            'ay_g': ay_g,
            'az_g': az_g,
            'gx_dps': gx_dps,
            'gy_dps': gy_dps,
            'gz_dps': gz_dps,
            'accel_mag': accel_mag,
            'gyro_mag': gyro_mag,
        }
    }


# ============================================================================
# ORIENTATION ANALYSIS
# ============================================================================

def analyze_orientation(samples: List[Dict]) -> Dict:
    """Analyze orientation (quaternion and Euler angles) data."""
    if not samples:
        return {'error': 'No samples'}

    # Check if orientation data exists
    has_orientation = any('orientation_w' in s for s in samples)
    if not has_orientation:
        return {'error': 'No orientation data'}

    # Extract quaternion components
    qw = [s.get('orientation_w', 0) for s in samples]
    qx = [s.get('orientation_x', 0) for s in samples]
    qy = [s.get('orientation_y', 0) for s in samples]
    qz = [s.get('orientation_z', 0) for s in samples]

    # Extract Euler angles
    roll = [s.get('euler_roll', 0) for s in samples]
    pitch = [s.get('euler_pitch', 0) for s in samples]
    yaw = [s.get('euler_yaw', 0) for s in samples]

    # AHRS magnetometer residuals
    ahrs_residual_x = [s.get('ahrs_mag_residual_x', None) for s in samples]
    ahrs_residual_y = [s.get('ahrs_mag_residual_y', None) for s in samples]
    ahrs_residual_z = [s.get('ahrs_mag_residual_z', None) for s in samples]
    ahrs_residual_mag = [s.get('ahrs_mag_residual_magnitude', None) for s in samples]

    # Filter out None values for residuals
    ahrs_residual_mag_valid = [r for r in ahrs_residual_mag if r is not None]

    return {
        'quaternion': {
            'w': {'mean': mean(qw), 'std': std(qw), 'min': min(qw), 'max': max(qw)},
            'x': {'mean': mean(qx), 'std': std(qx), 'min': min(qx), 'max': max(qx)},
            'y': {'mean': mean(qy), 'std': std(qy), 'min': min(qy), 'max': max(qy)},
            'z': {'mean': mean(qz), 'std': std(qz), 'min': min(qz), 'max': max(qz)},
        },
        'euler': {
            'roll': {'mean': mean(roll), 'std': std(roll), 'min': min(roll), 'max': max(roll)},
            'pitch': {'mean': mean(pitch), 'std': std(pitch), 'min': min(pitch), 'max': max(pitch)},
            'yaw': {'mean': mean(yaw), 'std': std(yaw), 'min': min(yaw), 'max': max(yaw)},
        },
        'ahrs_residuals': {
            'magnitude_mean': mean(ahrs_residual_mag_valid) if ahrs_residual_mag_valid else None,
            'magnitude_std': std(ahrs_residual_mag_valid) if ahrs_residual_mag_valid else None,
            'valid_count': len(ahrs_residual_mag_valid),
        },
        'series': {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        }
    }


# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================

def analyze_calibration(samples: List[Dict]) -> Dict:
    """Analyze calibration state and quality."""
    if not samples:
        return {'error': 'No samples'}

    # Fused magnetometer values
    fused_mx = [s.get('fused_mx', 0) for s in samples]
    fused_my = [s.get('fused_my', 0) for s in samples]
    fused_mz = [s.get('fused_mz', 0) for s in samples]

    # Calibration state flags
    fused_incomplete = sum(1 for s in samples if s.get('fused_incomplete', False))
    fused_uncalibrated = sum(1 for s in samples if s.get('fused_uncalibrated', False))
    gyro_bias_calibrated = sum(1 for s in samples if s.get('gyroBiasCalibrated', False))

    # Residual and confidence
    residual_mags = [s.get('residual_magnitude', 0) for s in samples]
    confidence = [s.get('incremental_cal_confidence', 0) for s in samples]
    earth_magnitude = [s.get('incremental_cal_earth_magnitude', 0) for s in samples]

    return {
        'fused_magnetometer': {
            'mx': {'mean': mean(fused_mx), 'std': std(fused_mx)},
            'my': {'mean': mean(fused_my), 'std': std(fused_my)},
            'mz': {'mean': mean(fused_mz), 'std': std(fused_mz)},
        },
        'calibration_state': {
            'incomplete_samples': fused_incomplete,
            'incomplete_percentage': 100.0 * fused_incomplete / len(samples),
            'uncalibrated_samples': fused_uncalibrated,
            'uncalibrated_percentage': 100.0 * fused_uncalibrated / len(samples),
            'gyro_calibrated_samples': gyro_bias_calibrated,
            'gyro_calibrated_percentage': 100.0 * gyro_bias_calibrated / len(samples),
        },
        'quality': {
            'residual_magnitude': {'mean': mean(residual_mags), 'std': std(residual_mags)},
            'confidence': {'mean': mean(confidence), 'max': max(confidence) if confidence else 0},
            'earth_magnitude': {'mean': mean(earth_magnitude), 'max': max(earth_magnitude) if earth_magnitude else 0},
        }
    }


# ============================================================================
# CORRELATION ANALYSIS (Motion <-> Magnetic Field)
# ============================================================================

def analyze_correlations(samples: List[Dict]) -> Dict:
    """Analyze correlations between motion and magnetic field changes."""
    if len(samples) < 10:
        return {'error': 'Insufficient samples'}

    # Extract time series
    dt = [s.get('dt', 0.02) for s in samples]

    # Magnetometer
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]
    mag = [magnitude_3d(mx[i], my[i], mz[i]) for i in range(len(samples))]

    # Gyroscope
    gx = [s.get('gx_dps', 0) for s in samples]
    gy = [s.get('gy_dps', 0) for s in samples]
    gz = [s.get('gz_dps', 0) for s in samples]
    gyro_mag = [magnitude_3d(gx[i], gy[i], gz[i]) for i in range(len(samples))]

    # Accelerometer
    ax = [s.get('ax_g', 0) for s in samples]
    ay = [s.get('ay_g', 0) for s in samples]
    az = [s.get('az_g', 0) for s in samples]
    accel_mag = [magnitude_3d(ax[i], ay[i], az[i]) for i in range(len(samples))]

    # Calculate derivatives (rate of change)
    dmag = derivative(mag, dt)
    dgyro = derivative(gyro_mag, dt)
    daccel = derivative(accel_mag, dt)

    # Align arrays (derivative is 1 shorter)
    min_len = min(len(dmag), len(dgyro), len(daccel))
    if min_len < 5:
        return {'error': 'Not enough data points for correlation'}

    dmag = dmag[:min_len]
    dgyro = dgyro[:min_len]
    daccel = daccel[:min_len]
    gyro_aligned = gyro_mag[1:min_len+1]
    accel_aligned = accel_mag[1:min_len+1]

    # Calculate correlations
    correlations = {
        'mag_change_vs_gyro': correlation(dmag, gyro_aligned),
        'mag_change_vs_accel': correlation(dmag, accel_aligned),
        'mag_change_vs_gyro_change': correlation(dmag, dgyro),
        'mag_change_vs_accel_change': correlation(dmag, daccel),
    }

    # Analyze by axis
    dmx = derivative(mx, dt)[:min_len]
    dmy = derivative(my, dt)[:min_len]
    dmz = derivative(mz, dt)[:min_len]

    axis_correlations = {
        'mx_change_vs_gx': correlation(dmx, gx[1:min_len+1]),
        'mx_change_vs_gy': correlation(dmx, gy[1:min_len+1]),
        'mx_change_vs_gz': correlation(dmx, gz[1:min_len+1]),
        'my_change_vs_gx': correlation(dmy, gx[1:min_len+1]),
        'my_change_vs_gy': correlation(dmy, gy[1:min_len+1]),
        'my_change_vs_gz': correlation(dmy, gz[1:min_len+1]),
        'mz_change_vs_gx': correlation(dmz, gx[1:min_len+1]),
        'mz_change_vs_gy': correlation(dmz, gy[1:min_len+1]),
        'mz_change_vs_gz': correlation(dmz, gz[1:min_len+1]),
    }

    return {
        'global_correlations': correlations,
        'axis_correlations': axis_correlations,
        'interpretation': {
            'high_gyro_correlation': abs(correlations['mag_change_vs_gyro']) > 0.5,
            'high_accel_correlation': abs(correlations['mag_change_vs_accel']) > 0.5,
        }
    }


# ============================================================================
# TIMING ANALYSIS
# ============================================================================

def analyze_timing(samples: List[Dict]) -> Dict:
    """Analyze timing and sample rate."""
    if not samples:
        return {'error': 'No samples'}

    dt_vals = [s.get('dt', 0) for s in samples]
    t_vals = [s.get('t', 0) for s in samples]

    # Filter valid dt values (> 0)
    valid_dt = [d for d in dt_vals if d > 0]

    # Calculate effective sample rate
    if valid_dt:
        avg_dt_ms = mean(valid_dt)
        effective_hz = 1000.0 / avg_dt_ms if avg_dt_ms > 0 else 0
    else:
        avg_dt_ms = 0
        effective_hz = 0

    # Duration
    total_duration_s = sum(valid_dt) / 1000.0 if valid_dt else 0

    return {
        'sample_count': len(samples),
        'dt': {
            'mean_ms': avg_dt_ms,
            'std_ms': std(valid_dt) if valid_dt else 0,
            'min_ms': min(valid_dt) if valid_dt else 0,
            'max_ms': max(valid_dt) if valid_dt else 0,
        },
        'effective_sample_rate_hz': effective_hz,
        'total_duration_seconds': total_duration_s,
        'battery': {
            'start': samples[0].get('b', 0) if samples else 0,
            'end': samples[-1].get('b', 0) if samples else 0,
        }
    }


# ============================================================================
# FINGER MAGNET SPECIFIC ANALYSIS
# ============================================================================

def analyze_finger_magnets(samples: List[Dict]) -> Dict:
    """
    Analyze magnetometer data specifically for finger magnet detection.

    Alternating polarity configuration:
    - Adjacent fingers have opposite magnetic poles facing sensor
    - Movement creates characteristic signature as different fingers approach
    - Can detect: finger proximity, finger identity, gesture patterns
    """
    if len(samples) < 50:
        return {'error': 'Insufficient samples for finger analysis'}

    # Extract filtered magnetometer data
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]

    # Calculate total field magnitude
    mag = [magnitude_3d(mx[i], my[i], mz[i]) for i in range(len(samples))]

    # Find field extremes (potential magnet approaches)
    sorted_indices = sorted(range(len(mag)), key=lambda i: mag[i])

    # Lowest magnitude samples (furthest from magnets / neutral)
    low_mag_indices = sorted_indices[:50]
    low_mag_mx = mean([mx[i] for i in low_mag_indices])
    low_mag_my = mean([my[i] for i in low_mag_indices])
    low_mag_mz = mean([mz[i] for i in low_mag_indices])
    baseline_mag = mean([mag[i] for i in low_mag_indices])

    # Highest magnitude samples (closest to magnets)
    high_mag_indices = sorted_indices[-50:]
    high_mag_mx = mean([mx[i] for i in high_mag_indices])
    high_mag_my = mean([my[i] for i in high_mag_indices])
    high_mag_mz = mean([mz[i] for i in high_mag_indices])
    peak_mag = mean([mag[i] for i in high_mag_indices])

    # Analyze field direction at peaks (indicates which magnet pole)
    # Positive vs negative dominant components suggest N vs S pole
    peak_direction = {
        'mx_bias': 'positive' if high_mag_mx > 0 else 'negative',
        'my_bias': 'positive' if high_mag_my > 0 else 'negative',
        'mz_bias': 'positive' if high_mag_mz > 0 else 'negative',
    }

    # Find peak events (local maxima in magnitude)
    peak_events = []
    window = 10
    for i in range(window, len(mag) - window):
        is_peak = all(mag[i] > mag[i-j] for j in range(1, window+1)) and \
                  all(mag[i] > mag[i+j] for j in range(1, window+1))
        if is_peak and mag[i] > baseline_mag * 1.2:  # 20% above baseline
            peak_events.append({
                'index': i,
                'magnitude': mag[i],
                'mx': mx[i],
                'my': my[i],
                'mz': mz[i],
                'pole_signature': 'N' if mz[i] > 0 else 'S',  # Simplified
            })

    # Analyze alternating pattern in peaks
    if len(peak_events) >= 2:
        alternating_count = 0
        for i in range(1, len(peak_events)):
            if peak_events[i]['pole_signature'] != peak_events[i-1]['pole_signature']:
                alternating_count += 1
        alternating_ratio = alternating_count / (len(peak_events) - 1)
    else:
        alternating_ratio = 0.0

    # Field variation analysis
    field_range = {
        'mx_range': max(mx) - min(mx),
        'my_range': max(my) - min(my),
        'mz_range': max(mz) - min(mz),
        'magnitude_range': max(mag) - min(mag),
    }

    # Dominant axis for magnet detection
    ranges = [('mx', field_range['mx_range']),
              ('my', field_range['my_range']),
              ('mz', field_range['mz_range'])]
    dominant_axis = max(ranges, key=lambda x: x[1])

    return {
        'baseline': {
            'mx': low_mag_mx,
            'my': low_mag_my,
            'mz': low_mag_mz,
            'magnitude': baseline_mag,
        },
        'peak': {
            'mx': high_mag_mx,
            'my': high_mag_my,
            'mz': high_mag_mz,
            'magnitude': peak_mag,
            'direction': peak_direction,
        },
        'field_range': field_range,
        'dominant_axis': dominant_axis[0],
        'dominant_axis_range': dominant_axis[1],
        'peak_events_count': len(peak_events),
        'alternating_ratio': alternating_ratio,
        'alternating_pattern_detected': alternating_ratio > 0.3,
        'signal_to_baseline_ratio': peak_mag / baseline_mag if baseline_mag > 0 else 0,
    }


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def load_session(filepath: str) -> Tuple[Dict, List[Dict]]:
    """Load a GAMBIT session file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = {
        'version': data.get('version', 'unknown'),
        'timestamp': data.get('timestamp', ''),
        'filepath': filepath,
    }

    samples = data.get('samples', [])
    return metadata, samples


def full_analysis(filepath: str) -> Dict:
    """Perform complete analysis on a session file."""
    metadata, samples = load_session(filepath)

    if not samples:
        return {'error': 'No samples in file', 'metadata': metadata}

    return {
        'metadata': metadata,
        'timing': analyze_timing(samples),
        'magnetic_field': analyze_magnetic_field(samples),
        'motion': analyze_motion(samples),
        'orientation': analyze_orientation(samples),
        'calibration': analyze_calibration(samples),
        'correlations': analyze_correlations(samples),
        'polarity': detect_polarity_changes(samples),
        'clusters': analyze_magnetic_clusters(samples),
        'finger_magnets': analyze_finger_magnets(samples),
    }


def print_report(analysis: Dict) -> None:
    """Print a formatted analysis report."""
    print("=" * 80)
    print("GAMBIT FINGER MAGNET SESSION ANALYSIS")
    print("=" * 80)

    # Metadata
    meta = analysis.get('metadata', {})
    print(f"\nFile: {Path(meta.get('filepath', '')).name}")
    print(f"Version: {meta.get('version')}")
    print(f"Timestamp: {meta.get('timestamp')}")

    # Timing
    timing = analysis.get('timing', {})
    print(f"\n--- TIMING ---")
    print(f"Samples: {timing.get('sample_count', 0):,}")
    print(f"Duration: {timing.get('total_duration_seconds', 0):.1f} seconds")
    print(f"Sample Rate: {timing.get('effective_sample_rate_hz', 0):.1f} Hz")
    dt = timing.get('dt', {})
    print(f"Delta-t: {dt.get('mean_ms', 0):.2f} ms (std: {dt.get('std_ms', 0):.2f})")
    batt = timing.get('battery', {})
    print(f"Battery: {batt.get('start', 0)}% -> {batt.get('end', 0)}%")

    # Magnetic Field
    mag = analysis.get('magnetic_field', {})
    print(f"\n--- RAW MAGNETOMETER (LSB) ---")
    raw = mag.get('raw', {})
    for axis in ['mx', 'my', 'mz']:
        a = raw.get(axis, {})
        print(f"  {axis}: mean={a.get('mean', 0):.0f}, std={a.get('std', 0):.0f}, range=[{a.get('min', 0):.0f}, {a.get('max', 0):.0f}]")
    raw_mag_stats = raw.get('magnitude', {})
    print(f"  magnitude: mean={raw_mag_stats.get('mean', 0):.0f}, std={raw_mag_stats.get('std', 0):.0f}")

    print(f"\n--- MAGNETOMETER (microTesla) ---")
    ut = mag.get('microTesla', {})
    for axis in ['mx', 'my', 'mz']:
        a = ut.get(axis, {})
        print(f"  {axis}: mean={a.get('mean', 0):.1f}, std={a.get('std', 0):.1f}, range=[{a.get('min', 0):.1f}, {a.get('max', 0):.1f}]")
    ut_mag_stats = ut.get('magnitude', {})
    print(f"  magnitude: mean={ut_mag_stats.get('mean', 0):.1f}, std={ut_mag_stats.get('std', 0):.1f}")

    print(f"\n--- FILTERED MAGNETOMETER (Kalman) ---")
    filt = mag.get('filtered', {})
    for axis in ['mx', 'my', 'mz']:
        a = filt.get(axis, {})
        print(f"  {axis}: mean={a.get('mean', 0):.1f}, std={a.get('std', 0):.1f}")

    # Motion
    motion = analysis.get('motion', {})
    print(f"\n--- ACCELEROMETER (G) ---")
    accel_g = motion.get('accelerometer', {}).get('g', {})
    for axis in ['ax', 'ay', 'az']:
        a = accel_g.get(axis, {})
        print(f"  {axis}: mean={a.get('mean', 0):.3f}, std={a.get('std', 0):.3f}")

    print(f"\n--- GYROSCOPE (deg/s) ---")
    gyro_dps = motion.get('gyroscope', {}).get('dps', {})
    for axis in ['gx', 'gy', 'gz']:
        a = gyro_dps.get(axis, {})
        print(f"  {axis}: mean={a.get('mean', 0):.2f}, std={a.get('std', 0):.2f}")

    motion_det = motion.get('motion_detection', {})
    print(f"\n  Motion Detection:")
    print(f"    Moving: {motion_det.get('moving_percentage', 0):.1f}% of samples")
    print(f"    Accel Std: {motion_det.get('accelStd_mean', 0):.1f}")
    print(f"    Gyro Std: {motion_det.get('gyroStd_mean', 0):.1f}")

    # Orientation
    orient = analysis.get('orientation', {})
    print(f"\n--- ORIENTATION (Euler angles, degrees) ---")
    euler = orient.get('euler', {})
    for angle in ['roll', 'pitch', 'yaw']:
        a = euler.get(angle, {})
        print(f"  {angle}: mean={a.get('mean', 0):.1f}, std={a.get('std', 0):.1f}, range=[{a.get('min', 0):.1f}, {a.get('max', 0):.1f}]")

    # Calibration
    cal = analysis.get('calibration', {})
    cal_state = cal.get('calibration_state', {})
    cal_qual = cal.get('quality', {})
    print(f"\n--- CALIBRATION STATE ---")
    print(f"  Incomplete: {cal_state.get('incomplete_percentage', 0):.1f}%")
    print(f"  Uncalibrated: {cal_state.get('uncalibrated_percentage', 0):.1f}%")
    print(f"  Gyro Calibrated: {cal_state.get('gyro_calibrated_percentage', 0):.1f}%")
    res = cal_qual.get('residual_magnitude', {})
    print(f"  Residual Magnitude: mean={res.get('mean', 0):.1f}, std={res.get('std', 0):.1f}")

    # Correlations
    corr = analysis.get('correlations', {})
    print(f"\n--- MOTION <-> MAGNETIC CORRELATIONS ---")
    global_corr = corr.get('global_correlations', {})
    for k, v in global_corr.items():
        indicator = " ***" if abs(v) > 0.5 else ""
        print(f"  {k}: {v:+.3f}{indicator}")

    interp = corr.get('interpretation', {})
    if interp.get('high_gyro_correlation'):
        print("  -> Strong correlation with rotation detected!")
    if interp.get('high_accel_correlation'):
        print("  -> Strong correlation with acceleration detected!")

    # Polarity Analysis
    pol = analysis.get('polarity', {})
    print(f"\n--- POLARITY ANALYSIS (Alternating Magnets) ---")
    print(f"  Total Sign Changes: {pol.get('total_sign_changes', 0)}")
    print(f"  Dominant Axis: {pol.get('dominant_axis', 'N/A')}")
    print(f"  Changes per Second: {pol.get('changes_per_second', 0):.2f}")

    # Finger Magnet Analysis
    fm = analysis.get('finger_magnets', {})
    print(f"\n--- FINGER MAGNET DETECTION ---")
    baseline = fm.get('baseline', {})
    print(f"  Baseline Field: {baseline.get('magnitude', 0):.1f} uT")
    print(f"    (mx={baseline.get('mx', 0):.1f}, my={baseline.get('my', 0):.1f}, mz={baseline.get('mz', 0):.1f})")

    peak = fm.get('peak', {})
    print(f"  Peak Field: {peak.get('magnitude', 0):.1f} uT")
    print(f"    (mx={peak.get('mx', 0):.1f}, my={peak.get('my', 0):.1f}, mz={peak.get('mz', 0):.1f})")

    peak_dir = peak.get('direction', {})
    print(f"    Direction Bias: mx={peak_dir.get('mx_bias', '?')}, my={peak_dir.get('my_bias', '?')}, mz={peak_dir.get('mz_bias', '?')}")

    fr = fm.get('field_range', {})
    print(f"  Field Ranges:")
    print(f"    mx: {fr.get('mx_range', 0):.1f} uT")
    print(f"    my: {fr.get('my_range', 0):.1f} uT")
    print(f"    mz: {fr.get('mz_range', 0):.1f} uT")
    print(f"    total: {fr.get('magnitude_range', 0):.1f} uT")

    print(f"  Dominant Axis: {fm.get('dominant_axis', 'N/A')} (range: {fm.get('dominant_axis_range', 0):.1f} uT)")
    print(f"  Peak Events Detected: {fm.get('peak_events_count', 0)}")
    print(f"  Signal-to-Baseline Ratio: {fm.get('signal_to_baseline_ratio', 0):.2f}x")

    alt_detected = fm.get('alternating_pattern_detected', False)
    alt_ratio = fm.get('alternating_ratio', 0)
    print(f"  Alternating Pattern: {'YES' if alt_detected else 'NO'} (ratio: {alt_ratio:.2f})")

    # Clusters
    clust = analysis.get('clusters', {})
    if 'error' not in clust:
        print(f"\n--- MAGNETIC FIELD CLUSTERS ---")
        print(f"  Number of Clusters: {clust.get('n_clusters', 0)}")
        print(f"  Magnitude Range: {clust.get('magnitude_range', 0):.1f} uT")
        for c in clust.get('clusters', []):
            m = c.get('magnitude', {})
            print(f"    Cluster {c.get('cluster_id')}: {c.get('count')} samples ({c.get('percentage', 0):.1f}%), mag={m.get('mean', 0):.1f}+/-{m.get('std', 0):.1f}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find today's sessions after 22:00
    target_date = "2025-12-15"
    target_hour = 22

    json_files = list(data_dir.glob(f'{target_date}*.json'))

    # Filter by time (after 22:00)
    filtered_files = []
    for f in json_files:
        # Parse timestamp from filename
        name = f.name
        # Handle both formats: T22:40:44 and T22_40_44
        try:
            # Try ISO format
            ts_str = name.replace('.json', '').replace('_', ':')
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if ts.hour >= target_hour:
                filtered_files.append((f, ts))
        except ValueError:
            continue

    if not filtered_files:
        print(f"No sessions found for {target_date} after {target_hour}:00")
        return

    # Sort by timestamp
    filtered_files.sort(key=lambda x: x[1])

    print(f"\nFound {len(filtered_files)} session(s) from {target_date} after {target_hour}:00:")
    for f, ts in filtered_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    # Analyze each session
    all_analyses = []
    for filepath, ts in filtered_files:
        print(f"\n{'#' * 80}")
        print(f"ANALYZING SESSION: {filepath.name}")
        print(f"{'#' * 80}")

        analysis = full_analysis(str(filepath))
        all_analyses.append(analysis)
        print_report(analysis)

    # Summary across sessions
    if len(all_analyses) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-SESSION SUMMARY")
        print(f"{'=' * 80}")

        total_samples = sum(a.get('timing', {}).get('sample_count', 0) for a in all_analyses)
        total_duration = sum(a.get('timing', {}).get('total_duration_seconds', 0) for a in all_analyses)

        print(f"Total Samples: {total_samples:,}")
        print(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")

        # Average magnetic field statistics
        avg_peak_mag = mean([a.get('finger_magnets', {}).get('peak', {}).get('magnitude', 0) for a in all_analyses])
        avg_baseline_mag = mean([a.get('finger_magnets', {}).get('baseline', {}).get('magnitude', 0) for a in all_analyses])

        print(f"Average Peak Field: {avg_peak_mag:.1f} uT")
        print(f"Average Baseline Field: {avg_baseline_mag:.1f} uT")

    return all_analyses


if __name__ == '__main__':
    main()
