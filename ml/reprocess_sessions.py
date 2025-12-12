#!/usr/bin/env python3
"""
Reprocess Session Data with Corrected Earth Field Algorithm

This script reprocesses all session data using the corrected earth field
subtraction algorithm (R.T instead of R) and regenerates visualizations
to show the improvement.

The key fix:
- OLD (buggy): rotatedEarth = R @ earthField
- NEW (correct): rotatedEarth = R.T @ earthField (world→sensor transform)
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA_DIR = Path('/home/user/simcap/data/GAMBIT')
OUTPUT_DIR = Path('/home/user/simcap/visualizations')
CALIBRATION_FILE = DATA_DIR / 'gambit_calibration.json'


def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix (sensor → world)."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def apply_iron_correction(mx, my, mz, calibration):
    """Apply hard/soft iron calibration."""
    hi = calibration['hardIronOffset']
    si = np.array(calibration['softIronMatrix']).reshape(3, 3)
    corrected = np.array([mx - hi['x'], my - hi['y'], mz - hi['z']])
    return si @ corrected


def old_earth_subtraction(corrected_mag, qw, qx, qy, qz, earth_field):
    """OLD (BUGGY) algorithm: R @ earthField."""
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    ef = np.array([earth_field['x'], earth_field['y'], earth_field['z']])
    rotated_earth = R @ ef
    return corrected_mag - rotated_earth


def new_earth_subtraction(corrected_mag, qw, qx, qy, qz, earth_field_world):
    """NEW (CORRECT) algorithm: R.T @ earthField (world→sensor)."""
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    ef = np.array([earth_field_world['x'], earth_field_world['y'], earth_field_world['z']])
    rotated_earth = R.T @ ef
    return corrected_mag - rotated_earth


def estimate_world_frame_earth_field(samples, calibration):
    """
    Estimate earth field in world frame from samples with diverse orientations.
    This is the heading-informed approach.
    """
    world_estimates = []

    for s in samples:
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        # Iron-corrected sensor reading
        b_sensor = apply_iron_correction(
            s['mx'], s['my'], s['mz'], calibration
        )

        # Rotation matrix from quaternion
        R = quaternion_to_rotation_matrix(
            s['orientation_w'], s['orientation_x'],
            s['orientation_y'], s['orientation_z']
        )

        # Transform to world frame: B_world = R @ B_sensor
        b_world = R @ b_sensor
        world_estimates.append(b_world)

    if len(world_estimates) < 10:
        return None

    world_estimates = np.array(world_estimates)
    mean_world = np.mean(world_estimates, axis=0)

    return {
        'x': mean_world[0],
        'y': mean_world[1],
        'z': mean_world[2]
    }


def load_session(filepath):
    """Load session data."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
        metadata = data.get('metadata', {})
    elif isinstance(data, list):
        samples = data
        metadata = {}
    else:
        return None, None

    return samples, metadata


def load_calibration():
    """Load the checked-in calibration file."""
    with open(CALIBRATION_FILE, 'r') as f:
        return json.load(f)


def reprocess_session(session_path, calibration):
    """
    Reprocess a session with both old and new algorithms for comparison.
    Returns arrays of: time, yaw, pitch, old_mag, new_mag, iron_corrected_mag
    """
    samples, metadata = load_session(session_path)
    if samples is None:
        return None

    # Check if session has zeroed calibration - use checked-in calibration
    session_cal = metadata.get('calibration', {})
    if not session_cal.get('hardIronCalibrated', False):
        use_cal = calibration
        print(f"  Using checked-in calibration")
    else:
        use_cal = session_cal
        print(f"  Using embedded calibration")

    # Estimate world-frame earth field from the session's orientation diversity
    earth_field_world = estimate_world_frame_earth_field(samples, use_cal)
    if earth_field_world is None:
        print(f"  Could not estimate world-frame earth field")
        return None

    print(f"  Estimated world-frame earth field: [{earth_field_world['x']:.1f}, {earth_field_world['y']:.1f}, {earth_field_world['z']:.1f}]")

    # Process each sample
    results = {
        'time': [],
        'yaw': [],
        'pitch': [],
        'roll': [],
        'iron_corrected_mag': [],
        'old_mag': [],
        'new_mag': [],
        'iron_corrected_mx': [],
        'iron_corrected_my': [],
        'iron_corrected_mz': [],
        'new_mx': [],
        'new_my': [],
        'new_mz': [],
    }

    for i, s in enumerate(samples):
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']

        # Iron correction
        iron_corrected = apply_iron_correction(s['mx'], s['my'], s['mz'], use_cal)

        # Old algorithm (using stored earth field in sensor frame)
        old_result = old_earth_subtraction(iron_corrected, qw, qx, qy, qz, use_cal['earthField'])

        # New algorithm (using world-frame earth field)
        new_result = new_earth_subtraction(iron_corrected, qw, qx, qy, qz, earth_field_world)

        results['time'].append(i / 50.0)
        results['yaw'].append(s.get('euler_yaw', 0))
        results['pitch'].append(s.get('euler_pitch', 0))
        results['roll'].append(s.get('euler_roll', 0))
        results['iron_corrected_mag'].append(np.linalg.norm(iron_corrected))
        results['old_mag'].append(np.linalg.norm(old_result))
        results['new_mag'].append(np.linalg.norm(new_result))
        results['iron_corrected_mx'].append(iron_corrected[0])
        results['iron_corrected_my'].append(iron_corrected[1])
        results['iron_corrected_mz'].append(iron_corrected[2])
        results['new_mx'].append(new_result[0])
        results['new_my'].append(new_result[1])
        results['new_mz'].append(new_result[2])

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def generate_comparison_visualization(results, session_name, output_path):
    """Generate a comparison visualization showing old vs new algorithm."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

    time = results['time']
    yaw = results['yaw']
    pitch = results['pitch']

    # Calculate statistics
    old_mean = np.mean(results['old_mag'])
    old_std = np.std(results['old_mag'])
    new_mean = np.mean(results['new_mag'])
    new_std = np.std(results['new_mag'])
    iron_mean = np.mean(results['iron_corrected_mag'])
    iron_std = np.std(results['iron_corrected_mag'])

    old_cv = old_std / old_mean * 100 if old_mean > 0 else 0
    new_cv = new_std / new_mean * 100 if new_mean > 0 else 0
    iron_cv = iron_std / iron_mean * 100 if iron_mean > 0 else 0

    improvement = (old_cv - new_cv) / old_cv * 100 if old_cv > new_cv else 0

    # Title with stats
    fig.suptitle(f'Earth Field Compensation Comparison: {session_name}\n'
                 f'Old CV: {old_cv:.1f}% → New CV: {new_cv:.1f}% '
                 f'(Iron-only CV: {iron_cv:.1f}%) | Improvement: {improvement:.1f}%',
                 fontsize=12, fontweight='bold')

    # Plot 1: Magnitude over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, results['old_mag'], 'r-', alpha=0.7, linewidth=0.8, label=f'Old Algorithm (CV={old_cv:.1f}%)')
    ax1.plot(time, results['new_mag'], 'g-', alpha=0.7, linewidth=0.8, label=f'New Algorithm (CV={new_cv:.1f}%)')
    ax1.plot(time, results['iron_corrected_mag'], 'b-', alpha=0.4, linewidth=0.5, label=f'Iron-only (CV={iron_cv:.1f}%)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('|B| (LSB)')
    ax1.set_title('Magnetic Field Magnitude Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Old vs Yaw scatter
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(yaw, results['old_mag'], c=pitch, cmap='coolwarm', alpha=0.5, s=10)
    ax2.set_xlabel('Yaw (°)')
    ax2.set_ylabel('|B| (LSB)')
    ax2.set_title(f'OLD: |B| vs Yaw (corr={np.corrcoef(yaw, results["old_mag"])[0,1]:.3f})')
    plt.colorbar(scatter, ax=ax2, label='Pitch (°)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: New vs Yaw scatter
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(yaw, results['new_mag'], c=pitch, cmap='coolwarm', alpha=0.5, s=10)
    ax3.set_xlabel('Yaw (°)')
    ax3.set_ylabel('|B| (LSB)')
    ax3.set_title(f'NEW: |B| vs Yaw (corr={np.corrcoef(yaw, results["new_mag"])[0,1]:.3f})')
    plt.colorbar(scatter, ax=ax3, label='Pitch (°)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histogram comparison
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(results['old_mag'], bins=50, alpha=0.7, color='red', label='Old', density=True)
    ax4.hist(results['new_mag'], bins=50, alpha=0.7, color='green', label='New', density=True)
    ax4.axvline(old_mean, color='red', linestyle='--', linewidth=2, label=f'Old mean: {old_mean:.0f}')
    ax4.axvline(new_mean, color='green', linestyle='--', linewidth=2, label=f'New mean: {new_mean:.0f}')
    ax4.set_xlabel('|B| (LSB)')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: 3D trajectory comparison
    ax5 = fig.add_subplot(gs[2, 1], projection='3d')
    # Subsample for clarity
    step = max(1, len(results['new_mx']) // 500)
    ax5.scatter(results['iron_corrected_mx'][::step], results['iron_corrected_my'][::step],
                results['iron_corrected_mz'][::step], c='blue', alpha=0.3, s=5, label='Iron-only')
    ax5.scatter(results['new_mx'][::step], results['new_my'][::step],
                results['new_mz'][::step], c='green', alpha=0.5, s=5, label='New')
    ax5.set_xlabel('Bx')
    ax5.set_ylabel('By')
    ax5.set_zlabel('Bz')
    ax5.set_title('3D Magnetic Field Trajectory')
    ax5.legend()

    # Plot 6: Per-axis comparison over time
    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(time, results['new_mx'], 'r-', alpha=0.7, linewidth=0.5, label='Bx (new)')
    ax6.plot(time, results['new_my'], 'g-', alpha=0.7, linewidth=0.5, label='By (new)')
    ax6.plot(time, results['new_mz'], 'b-', alpha=0.7, linewidth=0.5, label='Bz (new)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Field (LSB)')
    ax6.set_title('Per-Axis Compensated Field (New Algorithm)')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'old_cv': old_cv,
        'new_cv': new_cv,
        'iron_cv': iron_cv,
        'improvement': improvement,
        'old_mean': old_mean,
        'new_mean': new_mean,
        'old_std': old_std,
        'new_std': new_std
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("REPROCESSING SESSIONS WITH CORRECTED EARTH FIELD ALGORITHM")
    print("=" * 70)

    # Load calibration
    calibration = load_calibration()
    print(f"\nLoaded calibration from {CALIBRATION_FILE}")
    print(f"  Earth field: {calibration['earthField']}")
    print(f"  Magnitude: {calibration['earthFieldMagnitude']:.1f} LSB")

    # Find all session files
    session_files = sorted(DATA_DIR.glob('2025-*.json'))
    print(f"\nFound {len(session_files)} session files")

    # Process each session
    all_stats = []
    for session_path in session_files:
        print(f"\n{'='*60}")
        print(f"Processing: {session_path.name}")
        print(f"{'='*60}")

        results = reprocess_session(session_path, calibration)
        if results is None:
            print(f"  Skipped (no valid data)")
            continue

        # Generate visualization
        output_path = OUTPUT_DIR / f'heading_compensation_{session_path.stem}.png'
        stats = generate_comparison_visualization(results, session_path.stem, output_path)

        print(f"\n  Results:")
        print(f"    Old algorithm: mean={stats['old_mean']:.1f}, std={stats['old_std']:.1f}, CV={stats['old_cv']:.1f}%")
        print(f"    New algorithm: mean={stats['new_mean']:.1f}, std={stats['new_std']:.1f}, CV={stats['new_cv']:.1f}%")
        print(f"    Improvement: {stats['improvement']:.1f}%")
        print(f"  Saved: {output_path}")

        all_stats.append({
            'session': session_path.stem,
            **stats
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_stats:
        avg_improvement = np.mean([s['improvement'] for s in all_stats])
        avg_old_cv = np.mean([s['old_cv'] for s in all_stats])
        avg_new_cv = np.mean([s['new_cv'] for s in all_stats])

        print(f"\nAcross {len(all_stats)} sessions:")
        print(f"  Average Old CV: {avg_old_cv:.1f}%")
        print(f"  Average New CV: {avg_new_cv:.1f}%")
        print(f"  Average Improvement: {avg_improvement:.1f}%")

        print("\nPer-session results:")
        for s in all_stats:
            print(f"  {s['session']}: {s['old_cv']:.1f}% → {s['new_cv']:.1f}% ({s['improvement']:.1f}% improvement)")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
