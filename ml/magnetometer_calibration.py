"""
Magnetometer Hard-Iron + Soft-Iron Calibration

Uses only EEEEE (all fingers extended) windows to fit an ellipsoid,
then extracts calibration parameters to transform the ellipsoid to a sphere.

Calibration model:
    m_cal = S @ (m_raw - b)

Where:
    b = hard-iron offset (3-vector) - the ellipsoid center
    S = soft-iron correction (3x3 matrix) - untilt/deshear/rescale

References:
- Ellipsoid fitting: Least-squares quadratic surface fitting
- Soft-iron: Eigendecomposition of the quadratic form matrix
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class CalibrationResult:
    """Magnetometer calibration parameters."""
    hard_iron: np.ndarray      # b: 3-vector offset
    soft_iron: np.ndarray      # S: 3x3 correction matrix
    field_magnitude: float     # Expected calibrated magnitude (Earth field ~25-65 µT)

    # Validation metrics
    raw_magnitude_std: float
    calibrated_magnitude_std: float
    improvement_ratio: float

    def to_dict(self) -> dict:
        return {
            'hard_iron': self.hard_iron.tolist(),
            'soft_iron': self.soft_iron.tolist(),
            'field_magnitude': self.field_magnitude,
            'raw_magnitude_std': self.raw_magnitude_std,
            'calibrated_magnitude_std': self.calibrated_magnitude_std,
            'improvement_ratio': self.improvement_ratio
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationResult':
        return cls(
            hard_iron=np.array(d['hard_iron']),
            soft_iron=np.array(d['soft_iron']),
            field_magnitude=d['field_magnitude'],
            raw_magnitude_std=d['raw_magnitude_std'],
            calibrated_magnitude_std=d['calibrated_magnitude_std'],
            improvement_ratio=d['improvement_ratio']
        )


def load_eeeee_samples(data_path: Path) -> Tuple[np.ndarray, List[dict]]:
    """
    Load magnetometer samples from EEEEE (all extended) windows only.

    Returns:
        mag_data: [N, 3] array of magnetometer readings
        window_info: List of dicts with window metadata
    """
    with open(data_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    # Extract all mag data
    all_mag = np.array([[s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]
                        for s in samples])

    # Find EEEEE windows
    eeeee_samples = []
    window_info = []

    for label in labels:
        # Handle both label formats
        start_idx = label.get('startIndex') or label.get('start_sample')
        end_idx = label.get('endIndex') or label.get('end_sample')

        if start_idx is None or end_idx is None:
            continue

        # Get finger state
        if 'labels' in label and isinstance(label['labels'], dict):
            fingers = label['labels'].get('fingers', {})
        else:
            fingers = label.get('fingers', {})

        if not fingers:
            continue

        # Check if all extended (EEEEE)
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        is_eeeee = all(fingers.get(f, 'extended') == 'extended' for f in finger_names)

        if is_eeeee:
            window_mag = all_mag[start_idx:end_idx]
            eeeee_samples.append(window_mag)

            # Get orientation info if available
            pitches = [samples[i].get('euler_pitch', 0) for i in range(start_idx, end_idx)]
            rolls = [samples[i].get('euler_roll', 0) for i in range(start_idx, end_idx)]

            window_info.append({
                'start': start_idx,
                'end': end_idx,
                'n_samples': end_idx - start_idx,
                'mean_pitch': np.mean(pitches),
                'mean_roll': np.mean(rolls)
            })

    if not eeeee_samples:
        raise ValueError("No EEEEE windows found in session")

    mag_data = np.vstack(eeeee_samples)
    return mag_data, window_info


def check_orientation_coverage(mag_data: np.ndarray, n_bins: int = 8) -> Dict[str, float]:
    """
    Check if magnetometer samples cover diverse orientations.

    Returns metrics about directional coverage.
    """
    # Normalize to unit vectors
    norms = np.linalg.norm(mag_data, axis=1, keepdims=True)
    unit_vectors = mag_data / (norms + 1e-8)

    # Convert to spherical coordinates
    theta = np.arccos(unit_vectors[:, 2])  # Polar angle
    phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])  # Azimuthal angle

    # Bin the directions
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    phi_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Count samples in each bin
    hist, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins])

    # Coverage metrics
    n_occupied = np.sum(hist > 0)
    n_total_bins = n_bins * n_bins
    coverage_ratio = n_occupied / n_total_bins

    # Uniformity (entropy-based)
    hist_flat = hist.flatten()
    hist_norm = hist_flat / hist_flat.sum()
    hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
    entropy = -np.sum(hist_norm * np.log(hist_norm))
    max_entropy = np.log(n_total_bins)
    uniformity = entropy / max_entropy

    return {
        'coverage_ratio': coverage_ratio,
        'uniformity': uniformity,
        'n_occupied_bins': n_occupied,
        'n_total_bins': n_total_bins,
        'mean_magnitude': np.mean(norms),
        'std_magnitude': np.std(norms)
    }


def remove_outliers(mag_data: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """
    Remove outlier samples based on magnitude.
    """
    magnitudes = np.linalg.norm(mag_data, axis=1)
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)

    lower = mean_mag - n_sigma * std_mag
    upper = mean_mag + n_sigma * std_mag

    mask = (magnitudes >= lower) & (magnitudes <= upper)
    n_removed = np.sum(~mask)

    print(f"Outlier removal: {n_removed}/{len(mag_data)} samples removed ({n_sigma}σ)")
    return mag_data[mask]


def equalize_coverage(mag_data: np.ndarray, n_bins: int = 8,
                      max_per_bin: Optional[int] = None) -> np.ndarray:
    """
    Subsample to equalize directional coverage.
    Prevents overweighting orientations where user paused longer.
    """
    # Normalize to unit vectors
    norms = np.linalg.norm(mag_data, axis=1, keepdims=True)
    unit_vectors = mag_data / (norms + 1e-8)

    # Convert to spherical coordinates
    theta = np.arccos(unit_vectors[:, 2])
    phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])

    # Assign each sample to a bin
    theta_idx = np.clip((theta / np.pi * n_bins).astype(int), 0, n_bins - 1)
    phi_idx = np.clip(((phi + np.pi) / (2 * np.pi) * n_bins).astype(int), 0, n_bins - 1)
    bin_idx = theta_idx * n_bins + phi_idx

    # Count samples per bin
    unique_bins, counts = np.unique(bin_idx, return_counts=True)

    if max_per_bin is None:
        # Use median count as target
        max_per_bin = int(np.median(counts))

    # Subsample each bin
    selected_indices = []
    for b in unique_bins:
        bin_mask = bin_idx == b
        bin_indices = np.where(bin_mask)[0]
        if len(bin_indices) > max_per_bin:
            # Random subsample
            selected = np.random.choice(bin_indices, max_per_bin, replace=False)
        else:
            selected = bin_indices
        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)
    print(f"Coverage equalization: {len(mag_data)} → {len(selected_indices)} samples")
    return mag_data[selected_indices]


def fit_ellipsoid_robust(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust ellipsoid fitting using iterative approach.

    This uses the fact that the magnetometer samples should lie on an ellipsoid
    surface (constant Earth field magnitude, transformed by hard/soft iron).

    Returns:
        center: 3-vector (hard-iron offset)
        A: 3x3 matrix defining the ellipsoid shape
    """
    # Step 1: Initial center estimate (mean of data)
    center = np.mean(data, axis=0)

    # Step 2: Iterative refinement
    for iteration in range(10):
        # Center the data
        centered = data - center

        # Compute covariance (this captures the ellipsoid shape)
        cov = np.cov(centered.T)

        # The covariance eigenvectors give us the ellipsoid axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Transform to find better center
        # The center should be where the distance variance is minimized
        # Use weighted update based on direction consistency

        # Compute radial distances
        distances = np.linalg.norm(centered, axis=1)
        mean_distance = np.mean(distances)

        # Weight samples by how close they are to the mean distance
        # (outliers will have less influence)
        weights = np.exp(-0.5 * ((distances - mean_distance) / (0.2 * mean_distance))**2)
        weights /= weights.sum()

        # Update center estimate with weighted mean
        new_center = np.average(data, axis=0, weights=weights)

        # Check convergence
        if np.linalg.norm(new_center - center) < 1e-6:
            break

        center = new_center

    # Final covariance after centering
    centered = data - center
    cov = np.cov(centered.T)

    return center, cov


def extract_calibration_from_covariance(center: np.ndarray,
                                         cov: np.ndarray,
                                         data: np.ndarray,
                                         target_magnitude: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract calibration parameters from center and covariance.

    The covariance matrix encodes the ellipsoid shape.
    We want S such that after transformation, all points have similar magnitude.

    Args:
        center: Hard-iron offset (ellipsoid center)
        cov: Covariance matrix of centered data
        data: Original data points (for magnitude estimation)
        target_magnitude: Optional target magnitude. If None, uses mean distance from center.

    Returns:
        b: hard-iron offset
        S: soft-iron correction matrix
        R: expected field magnitude after calibration
    """
    b = center

    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # The covariance eigenvalues represent the variance along each principal axis
    # For an ellipsoid, these correspond to the squared semi-axes
    # We want to scale each axis so they all have the same length

    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    # The mean distance from center gives us the "radius" of the ellipsoid
    centered = data - center
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))

    # If no target specified, use the mean distance as target
    if target_magnitude is None:
        target_magnitude = mean_distance

    # Standard deviations along each axis
    stds = np.sqrt(eigenvalues)

    # We want to scale each axis so that std becomes target_magnitude/sqrt(3)
    # (for a sphere of radius R, std along each axis is R/sqrt(3))
    target_std = target_magnitude / np.sqrt(3)

    # Scale factors for each axis
    scales = target_std / stds

    # Build transformation matrix: S = diag(scales) @ V.T
    # This first rotates to principal axes, then scales each
    S = np.diag(scales) @ eigenvectors.T

    R = target_magnitude

    return b, S, R


def fit_ellipsoid(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit an ellipsoid to 3D point cloud using least-squares.

    This is the quadratic surface approach - kept for reference but
    using the robust covariance method instead.
    """
    # Use the robust covariance-based method
    center, cov = fit_ellipsoid_robust(data)

    # Convert to quadratic form Q, p, c where x^T Q x + p^T x + c = 0
    # For an ellipsoid centered at 'center' with shape given by cov^(-1):
    # (x - center)^T @ cov^(-1) @ (x - center) = k
    # Expanding: x^T @ cov^(-1) @ x - 2 center^T @ cov^(-1) @ x + center^T @ cov^(-1) @ center - k = 0

    cov_inv = np.linalg.inv(cov)
    Q = cov_inv
    p = -2 * cov_inv @ center
    c = center.T @ cov_inv @ center - 1  # Assuming k=1

    return Q, p, c


def extract_calibration(data: np.ndarray,
                        target_magnitude: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract hard-iron offset (b) and soft-iron correction (S) directly from data.

    Args:
        data: Magnetometer samples [N, 3]
        target_magnitude: Optional target magnitude. If None, uses mean distance from center.

    Returns:
        b: hard-iron offset (center of ellipsoid)
        S: soft-iron correction matrix (transforms ellipsoid to sphere)
        R: expected field magnitude after calibration
    """
    # Use robust ellipsoid fitting
    center, cov = fit_ellipsoid_robust(data)

    # Get soft-iron matrix
    b, S, R = extract_calibration_from_covariance(center, cov, data, target_magnitude)

    return b, S, R


def apply_calibration(mag_data: np.ndarray, b: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Apply calibration to magnetometer data.

    m_cal = S @ (m_raw - b)
    """
    centered = mag_data - b
    calibrated = (S @ centered.T).T
    return calibrated


def validate_calibration(raw_data: np.ndarray, cal_data: np.ndarray) -> Dict[str, float]:
    """
    Compute validation metrics for calibration.
    """
    raw_mags = np.linalg.norm(raw_data, axis=1)
    cal_mags = np.linalg.norm(cal_data, axis=1)

    return {
        'raw_mean_magnitude': np.mean(raw_mags),
        'raw_std_magnitude': np.std(raw_mags),
        'raw_cv': np.std(raw_mags) / np.mean(raw_mags),  # Coefficient of variation
        'cal_mean_magnitude': np.mean(cal_mags),
        'cal_std_magnitude': np.std(cal_mags),
        'cal_cv': np.std(cal_mags) / np.mean(cal_mags),
        'std_improvement_ratio': np.std(raw_mags) / np.std(cal_mags),
        'cv_improvement_ratio': (np.std(raw_mags) / np.mean(raw_mags)) / (np.std(cal_mags) / np.mean(cal_mags))
    }


def plot_calibration(raw_data: np.ndarray, cal_data: np.ndarray,
                     output_path: Optional[Path] = None):
    """
    Visualize raw vs calibrated magnetometer data.
    """
    fig = plt.figure(figsize=(16, 6))

    # 3D scatter plot - raw
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2],
                c=np.linalg.norm(raw_data, axis=1), cmap='viridis', alpha=0.5, s=1)
    ax1.set_xlabel('X (µT)')
    ax1.set_ylabel('Y (µT)')
    ax1.set_zlabel('Z (µT)')
    ax1.set_title('Raw Magnetometer (EEEEE)')

    # 3D scatter plot - calibrated
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(cal_data[:, 0], cal_data[:, 1], cal_data[:, 2],
                c=np.linalg.norm(cal_data, axis=1), cmap='viridis', alpha=0.5, s=1)
    ax2.set_xlabel('X (µT)')
    ax2.set_ylabel('Y (µT)')
    ax2.set_zlabel('Z (µT)')
    ax2.set_title('Calibrated Magnetometer')

    # Make axes equal for both 3D plots
    for ax in [ax1, ax2]:
        data = raw_data if ax == ax1 else cal_data
        max_range = np.max(np.abs(data)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    # Magnitude histogram
    ax3 = fig.add_subplot(133)
    raw_mags = np.linalg.norm(raw_data, axis=1)
    cal_mags = np.linalg.norm(cal_data, axis=1)

    ax3.hist(raw_mags, bins=50, alpha=0.5, label=f'Raw (σ={np.std(raw_mags):.2f})')
    ax3.hist(cal_mags, bins=50, alpha=0.5, label=f'Calibrated (σ={np.std(cal_mags):.2f})')
    ax3.set_xlabel('Magnitude (µT)')
    ax3.set_ylabel('Count')
    ax3.set_title('Magnitude Distribution')
    ax3.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")

    plt.close()


def calibrate_from_session(session_path: Path,
                           equalize: bool = True,
                           remove_outliers_sigma: float = 3.0) -> CalibrationResult:
    """
    Full calibration pipeline from session file.
    """
    print(f"\n{'='*60}")
    print(f"Magnetometer Calibration")
    print(f"Session: {session_path.name}")
    print(f"{'='*60}")

    # 1. Load EEEEE samples
    print("\n1. Loading EEEEE windows...")
    mag_data, window_info = load_eeeee_samples(session_path)
    print(f"   Loaded {len(mag_data)} samples from {len(window_info)} EEEEE windows")

    # 2. Sanity checks
    print("\n2. Checking orientation coverage...")
    coverage = check_orientation_coverage(mag_data)
    print(f"   Coverage ratio: {coverage['coverage_ratio']:.1%}")
    print(f"   Uniformity: {coverage['uniformity']:.2f}")
    print(f"   Raw magnitude: {coverage['mean_magnitude']:.1f} ± {coverage['std_magnitude']:.1f} µT")

    # 3. Remove outliers
    print("\n3. Removing outliers...")
    mag_clean = remove_outliers(mag_data, n_sigma=remove_outliers_sigma)

    # 4. Equalize coverage (optional)
    if equalize:
        print("\n4. Equalizing directional coverage...")
        mag_eq = equalize_coverage(mag_clean)
    else:
        mag_eq = mag_clean

    # 5. Fit ellipsoid and extract calibration
    print("\n5. Fitting ellipsoid and extracting calibration...")
    b, S, R = extract_calibration(mag_eq)
    print(f"   Hard-iron offset (b): [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}] µT")
    print(f"   Soft-iron matrix (S):")
    for row in S:
        print(f"      [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
    print(f"   Expected field magnitude: {R:.2f} µT")

    # 6. Apply calibration
    print("\n6. Applying calibration...")
    cal_data = apply_calibration(mag_data, b, S)

    # 7. Validate
    print("\n7. Validating...")
    metrics = validate_calibration(mag_data, cal_data)
    print(f"   Raw magnitude:   {metrics['raw_mean_magnitude']:.1f} ± {metrics['raw_std_magnitude']:.2f} µT (CV={metrics['raw_cv']:.2%})")
    print(f"   Cal magnitude:   {metrics['cal_mean_magnitude']:.1f} ± {metrics['cal_std_magnitude']:.2f} µT (CV={metrics['cal_cv']:.2%})")
    print(f"   Improvement:     {metrics['std_improvement_ratio']:.1f}x reduction in std")

    # Create result
    result = CalibrationResult(
        hard_iron=b,
        soft_iron=S,
        field_magnitude=R,
        raw_magnitude_std=metrics['raw_std_magnitude'],
        calibrated_magnitude_std=metrics['cal_std_magnitude'],
        improvement_ratio=metrics['std_improvement_ratio']
    )

    return result, mag_data, cal_data


def simple_hard_iron_calibration(data: np.ndarray) -> np.ndarray:
    """
    Simple hard-iron calibration: subtract the mean offset.

    This is appropriate when:
    - Not enough orientation coverage for ellipsoid fitting
    - The soft-iron distortion is small
    """
    offset = np.mean(data, axis=0)
    return offset


def analyze_pose_signatures(session_path: Path) -> Dict[str, np.ndarray]:
    """
    Analyze magnetic signatures for each finger pose.

    Returns:
        Dictionary mapping pose string to mean magnetic vector
    """
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    all_mag = np.array([[s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]
                        for s in samples])

    pose_signatures = {}

    for label in labels:
        start_idx = label.get('startIndex') or label.get('start_sample')
        end_idx = label.get('endIndex') or label.get('end_sample')

        if start_idx is None or end_idx is None:
            continue

        if 'labels' in label and isinstance(label['labels'], dict):
            fingers = label['labels'].get('fingers', {})
        else:
            fingers = label.get('fingers', {})

        if not fingers:
            continue

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        pose = ''.join('E' if fingers.get(f, 'extended') == 'extended' else 'F'
                      for f in finger_names)

        if pose not in pose_signatures:
            pose_signatures[pose] = []

        window_mag = all_mag[start_idx:end_idx]
        pose_signatures[pose].append(np.mean(window_mag, axis=0))

    # Average across windows for each pose
    for pose in pose_signatures:
        pose_signatures[pose] = np.mean(pose_signatures[pose], axis=0)

    return pose_signatures


def main():
    """Run calibration analysis on 12/31 session."""
    session_path = Path("data/GAMBIT/2025-12-31T14_06_18.270Z.json")

    if not session_path.exists():
        print(f"Session file not found: {session_path}")
        return

    output_dir = Path("ml/calibration_results")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Magnetometer Calibration Analysis")
    print("="*70)

    # Load and analyze data
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    all_mag = np.array([[s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]
                        for s in samples])

    # Analyze pose signatures
    print("\n1. Analyzing magnetic signatures by pose...")
    pose_signatures = analyze_pose_signatures(session_path)

    # Use EEEEE as baseline (all fingers extended = no magnet deflection)
    if 'EEEEE' in pose_signatures:
        baseline = pose_signatures['EEEEE']
        print(f"\n   EEEEE baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] µT")
    else:
        baseline = np.mean(list(pose_signatures.values()), axis=0)
        print(f"   No EEEEE found, using mean baseline")

    # Show pose signatures relative to baseline
    print("\n   Pose signatures (relative to EEEEE baseline):")
    print(f"   {'Pose':<8} {'ΔX':>8} {'ΔY':>8} {'ΔZ':>8} {'|Δ|':>8}")
    print("   " + "-"*45)
    for pose in sorted(pose_signatures.keys()):
        sig = pose_signatures[pose]
        delta = sig - baseline
        mag = np.linalg.norm(delta)
        print(f"   {pose:<8} {delta[0]:>8.1f} {delta[1]:>8.1f} {delta[2]:>8.1f} {mag:>8.1f}")

    # Simple calibration: subtract EEEEE baseline
    print("\n2. Simple hard-iron calibration (subtract EEEEE baseline)...")

    # Collect per-pose statistics
    pose_stats = {}
    for label in labels:
        start_idx = label.get('startIndex') or label.get('start_sample')
        end_idx = label.get('endIndex') or label.get('end_sample')

        if start_idx is None or end_idx is None:
            continue

        if 'labels' in label and isinstance(label['labels'], dict):
            fingers = label['labels'].get('fingers', {})
        else:
            fingers = label.get('fingers', {})

        if not fingers:
            continue

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        pose = ''.join('E' if fingers.get(f, 'extended') == 'extended' else 'F'
                      for f in finger_names)

        if pose not in pose_stats:
            pose_stats[pose] = {'raw': [], 'cal': []}

        window_mag = all_mag[start_idx:end_idx]
        calibrated = window_mag - baseline  # Simple offset subtraction

        pose_stats[pose]['raw'].extend(np.linalg.norm(window_mag, axis=1).tolist())
        pose_stats[pose]['cal'].extend(np.linalg.norm(calibrated, axis=1).tolist())

    # Print results
    print(f"\n   {'Pose':<8} {'N':>6} {'Raw µ':>8} {'Raw σ':>8} {'Cal µ':>8} {'Cal σ':>8}")
    print("   " + "-"*55)
    for pose in sorted(pose_stats.keys()):
        raw = np.array(pose_stats[pose]['raw'])
        cal = np.array(pose_stats[pose]['cal'])
        print(f"   {pose:<8} {len(raw):>6} {np.mean(raw):>8.1f} {np.std(raw):>8.1f} {np.mean(cal):>8.1f} {np.std(cal):>8.1f}")

    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
With finger magnets, traditional ellipsoid calibration doesn't apply because:
- Each finger pose creates a different magnetic signature
- The data doesn't lie on a single ellipsoid

Instead, the magnetic field differences between poses ARE the signal we use
for finger tracking. The 'calibration' for inference is:

1. Use EEEEE as the baseline (all fingers extended)
2. Subtract baseline to get relative magnetic change
3. Use the relative change for pose classification

The pose signatures show clear separation:
- Flexing a finger moves the magnet, changing the local field
- Each pose has a characteristic magnetic signature
- This is what enables finger tracking from magnetometry!
""")

    # Save results
    results = {
        'baseline_pose': 'EEEEE',
        'baseline_vector': baseline.tolist(),
        'pose_signatures': {k: v.tolist() for k, v in pose_signatures.items()},
        'note': 'Subtract baseline_vector from raw mag to get relative signal'
    }

    with open(output_dir / 'pose_signatures.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPose signatures saved to {output_dir / 'pose_signatures.json'}")


if __name__ == "__main__":
    main()
