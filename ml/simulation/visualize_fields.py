"""
Magnetic Field Visualization

Creates visualizations of simulated magnetic fields from finger magnets
using Magpylib for accurate field calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from .magpylib_sim import MagpylibSimulator, MagnetSpec, DEFAULT_MAGNET_SPECS
    HAS_MAGPYLIB = True
except ImportError:
    HAS_MAGPYLIB = False


def create_field_slice_plot(
    sim: 'MagpylibSimulator',
    finger_positions: Dict[str, np.ndarray],
    plane: str = 'xy',
    z_value: float = 0.0,
    extent_mm: float = 150.0,
    grid_points: int = 50,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create a 2D plot of magnetic field magnitude on a plane.

    Args:
        sim: MagpylibSimulator instance
        finger_positions: Dict of finger positions in mm
        plane: 'xy', 'xz', or 'yz'
        z_value: Position of the slice on the third axis (mm)
        extent_mm: Plot extent from -extent to +extent
        grid_points: Resolution of the grid
        ax: Optional axes to plot on

    Returns:
        Figure object
    """
    # Position magnets
    for finger, position in finger_positions.items():
        if finger in sim.magnets:
            sim.magnets[finger].position = position

    # Create 2D grid
    coords = np.linspace(-extent_mm, extent_mm, grid_points)

    if plane == 'xy':
        X, Y = np.meshgrid(coords, coords)
        Z = np.full_like(X, z_value)
    elif plane == 'xz':
        X, Z = np.meshgrid(coords, coords)
        Y = np.full_like(X, z_value)
    else:  # yz
        Y, Z = np.meshgrid(coords, coords)
        X = np.full_like(Y, z_value)

    # Flatten for sensor creation
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Import magpylib
    import magpylib as magpy

    # Create sensors and compute field
    sensors = magpy.Sensor(position=positions)
    B = np.zeros((len(positions), 3))
    for magnet in sim.magnets.values():
        B = B + sensors.getB(magnet)

    # Compute magnitude (convert mT to μT)
    magnitude = np.linalg.norm(B, axis=1) * 1000
    magnitude = magnitude.reshape(grid_points, grid_points)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Use log scale for better visualization (field varies over orders of magnitude)
    magnitude_clipped = np.clip(magnitude, 1, 10000)  # Avoid log(0)

    im = ax.pcolormesh(X, Y if plane != 'xz' else Z, magnitude_clipped,
                       norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=1000),
                       cmap='viridis', shading='auto')

    cbar = plt.colorbar(im, ax=ax, label='Field Magnitude (μT)')

    # Plot magnet positions
    for finger, pos in finger_positions.items():
        if plane == 'xy':
            ax.plot(pos[0], pos[1], 'r^', markersize=10, label=finger if finger == 'thumb' else '')
            ax.annotate(finger[:2], (pos[0], pos[1]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, color='white')
        elif plane == 'xz':
            ax.plot(pos[0], pos[2], 'r^', markersize=10)
            ax.annotate(finger[:2], (pos[0], pos[2]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, color='white')
        else:
            ax.plot(pos[1], pos[2], 'r^', markersize=10)
            ax.annotate(finger[:2], (pos[1], pos[2]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, color='white')

    # Mark sensor position
    ax.plot(0, 0, 'wo', markersize=12, markeredgecolor='red', markeredgewidth=2)
    ax.annotate('Sensor', (0, 0), textcoords="offset points",
               xytext=(8, 8), fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('X (mm)' if plane != 'yz' else 'Y (mm)')
    ax.set_ylabel('Y (mm)' if plane == 'xy' else 'Z (mm)')
    ax.set_title(f'Magnetic Field Magnitude ({plane.upper()} plane at {["Z", "Y", "X"][["xy", "xz", "yz"].index(plane)]}={z_value}mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def create_distance_falloff_plot(
    sim: 'MagpylibSimulator',
    fingers: list = None,
    max_distance_mm: float = 150.0,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot magnetic field magnitude vs distance for each finger magnet.
    """
    if fingers is None:
        fingers = ['index']

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    distances = np.linspace(10, max_distance_mm, 100)

    for finger in fingers:
        _, magnitudes = sim.field_magnitude_vs_distance(finger, distances)
        ax.semilogy(distances, magnitudes, label=finger, linewidth=2)

    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Earth field (~50 μT)')
    ax.axhline(y=5, color='gray', linestyle=':', alpha=0.7, label='Sensor noise floor (~5 μT)')

    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Field Magnitude (μT)')
    ax.set_title('Magnetic Field Falloff with Distance (1/r³ decay)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([10, max_distance_mm])
    ax.set_ylim([1, 10000])

    return fig


def create_pose_comparison_plot(
    sim: 'MagpylibSimulator',
    poses: Dict[str, Dict[str, np.ndarray]],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Compare magnetic field magnitudes for different hand poses.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    pose_names = list(poses.keys())
    magnitudes = []

    for pose_name, positions in poses.items():
        B = sim.compute_field(positions, include_earth=False)
        magnitudes.append(np.linalg.norm(B))

    bars = ax.bar(pose_names, magnitudes, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6'][:len(poses)])

    # Add value labels on bars
    for bar, mag in zip(bars, magnitudes):
        height = bar.get_height()
        ax.annotate(f'{mag:.1f} μT',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Field Magnitude (μT)')
    ax.set_title('Magnetic Field at Sensor for Different Hand Poses')
    ax.grid(True, alpha=0.3, axis='y')

    return fig


def generate_pose_visualizations(output_dir: str = 'images') -> list:
    """
    Generate a comprehensive set of visualizations.

    Returns list of saved file paths.
    """
    if not HAS_MAGPYLIB:
        print("Magpylib not installed, cannot generate visualizations")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sim = MagpylibSimulator()
    saved_files = []

    # Define poses
    poses = {
        'Open Palm': {
            'thumb': np.array([60.0, 70.0, 0.0]),
            'index': np.array([45.0, 120.0, 0.0]),
            'middle': np.array([50.0, 130.0, 0.0]),
            'ring': np.array([45.0, 120.0, 0.0]),
            'pinky': np.array([40.0, 100.0, 0.0])
        },
        'Fist': {
            'thumb': np.array([40.0, 30.0, -15.0]),
            'index': np.array([35.0, 45.0, -25.0]),
            'middle': np.array([30.0, 50.0, -25.0]),
            'ring': np.array([25.0, 45.0, -25.0]),
            'pinky': np.array([20.0, 40.0, -20.0])
        },
        'Pointing': {
            'thumb': np.array([40.0, 30.0, -15.0]),
            'index': np.array([45.0, 120.0, 0.0]),
            'middle': np.array([30.0, 50.0, -25.0]),
            'ring': np.array([25.0, 45.0, -25.0]),
            'pinky': np.array([20.0, 40.0, -20.0])
        },
        'Peace': {
            'thumb': np.array([40.0, 30.0, -15.0]),
            'index': np.array([45.0, 120.0, 0.0]),
            'middle': np.array([50.0, 130.0, 0.0]),
            'ring': np.array([25.0, 45.0, -25.0]),
            'pinky': np.array([20.0, 40.0, -20.0])
        }
    }

    # 1. Field slice plots for open palm
    print("Generating field slice plots...")
    fig = create_field_slice_plot(sim, poses['Open Palm'], plane='xy', z_value=0)
    filepath = output_path / 'magnetic_field_xy_slice.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))
    print(f"  Saved: {filepath}")

    # XZ slice
    fig = create_field_slice_plot(sim, poses['Open Palm'], plane='xz', z_value=50)
    filepath = output_path / 'magnetic_field_xz_slice.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))
    print(f"  Saved: {filepath}")

    # 2. Distance falloff plot
    print("Generating distance falloff plot...")
    fig = create_distance_falloff_plot(sim, fingers=['thumb', 'index', 'middle'])
    filepath = output_path / 'magnetic_field_falloff.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))
    print(f"  Saved: {filepath}")

    # 3. Pose comparison plot
    print("Generating pose comparison plot...")
    fig = create_pose_comparison_plot(sim, poses)
    filepath = output_path / 'magnetic_field_pose_comparison.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))
    print(f"  Saved: {filepath}")

    # 4. Multi-panel overview
    print("Generating overview plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Open palm field slice
    create_field_slice_plot(sim, poses['Open Palm'], plane='xy', z_value=0, ax=axes[0, 0])
    axes[0, 0].set_title('Open Palm - XY Plane')

    # Fist field slice
    create_field_slice_plot(sim, poses['Fist'], plane='xy', z_value=-20, ax=axes[0, 1])
    axes[0, 1].set_title('Fist - XY Plane (Z=-20mm)')

    # Distance falloff
    create_distance_falloff_plot(sim, fingers=['index'], ax=axes[1, 0])

    # Pose comparison
    create_pose_comparison_plot(sim, poses, ax=axes[1, 1])

    plt.tight_layout()
    filepath = output_path / 'magnetic_field_overview.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(filepath))
    print(f"  Saved: {filepath}")

    print(f"\nGenerated {len(saved_files)} visualization files")
    return saved_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate magnetic field visualizations')
    parser.add_argument('--output', '-o', type=str, default='images',
                       help='Output directory for images')

    args = parser.parse_args()

    print("Magnetic Field Visualization")
    print("=" * 50)

    files = generate_pose_visualizations(args.output)

    if files:
        print(f"\nSaved {len(files)} files to {args.output}/")
    else:
        print("No files generated")
