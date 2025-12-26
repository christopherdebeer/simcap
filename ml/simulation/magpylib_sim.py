"""
Magpylib-based Magnetic Field Simulation

Uses the Magpylib library for accurate magnetic field calculations with
real magnet geometries (cylinders) instead of point dipoles.

Magpylib provides:
- Exact solutions for common magnet shapes
- Fast vectorized numpy operations
- Support for arbitrary magnet positions and orientations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import magpylib as magpy
    HAS_MAGPYLIB = True
except ImportError:
    HAS_MAGPYLIB = False
    magpy = None


@dataclass
class MagnetSpec:
    """Specification for a finger magnet."""
    diameter_mm: float = 6.0      # Magnet diameter
    height_mm: float = 3.0        # Magnet thickness
    Br_mT: float = 1400.0         # N48 neodymium remanence (~1.4 T = 1400 mT)
    polarity: str = 'north_up'    # 'north_up' or 'north_down'


# Default magnet configuration (6mm × 3mm N48 neodymium)
DEFAULT_MAGNET_SPECS = {
    'thumb': MagnetSpec(polarity='north_up'),
    'index': MagnetSpec(polarity='north_down'),  # Alternating
    'middle': MagnetSpec(polarity='north_up'),
    'ring': MagnetSpec(polarity='north_down'),
    'pinky': MagnetSpec(polarity='north_up')
}


class MagpylibSimulator:
    """
    High-fidelity magnetic field simulation using Magpylib.

    Creates actual cylindrical magnet objects and computes exact
    magnetic fields at sensor positions.
    """

    def __init__(
        self,
        magnet_specs: Optional[Dict[str, MagnetSpec]] = None,
        sensor_position_mm: Tuple[float, float, float] = (0, 0, 0)
    ):
        """
        Initialize the Magpylib simulator.

        Args:
            magnet_specs: Dict mapping finger names to MagnetSpec objects
            sensor_position_mm: Position of the magnetometer sensor (mm)
        """
        if not HAS_MAGPYLIB:
            raise ImportError("Magpylib not installed. Run: pip install magpylib")

        self.magnet_specs = magnet_specs or DEFAULT_MAGNET_SPECS
        self.sensor_position_mm = np.array(sensor_position_mm)

        # Create sensor object (Magpylib 5.x uses mm by default)
        self.sensor = magpy.Sensor(position=self.sensor_position_mm)

        # Create magnet objects (will be positioned later)
        self.magnets = self._create_magnets()

    def _create_magnets(self) -> Dict[str, 'magpy.magnet.Cylinder']:
        """Create Magpylib cylinder magnets for each finger."""
        magnets = {}

        for finger, spec in self.magnet_specs.items():
            # Polarization direction based on polarity (Br in mT)
            # Magpylib 5.x uses polarization (J = Br) in mT, not magnetization
            if spec.polarity == 'north_up':
                polarization = (0, 0, spec.Br_mT)  # +Z direction
            else:
                polarization = (0, 0, -spec.Br_mT)  # -Z direction

            # Create cylinder magnet (dimensions in mm for Magpylib 5.x)
            mag = magpy.magnet.Cylinder(
                polarization=polarization,
                dimension=(spec.diameter_mm, spec.height_mm),  # (diameter, height) in mm
                position=(0, 0, 0)  # Will be set during simulation
            )
            magnets[finger] = mag

        return magnets

    def compute_field(
        self,
        finger_positions_mm: Dict[str, np.ndarray],
        include_earth: bool = True,
        earth_field_ut: np.ndarray = np.array([16.0, 0.0, 47.8])
    ) -> np.ndarray:
        """
        Compute total magnetic field at sensor from all finger magnets.

        Args:
            finger_positions_mm: Dict mapping finger names to positions (mm)
            include_earth: Include Earth's magnetic field
            earth_field_ut: Earth field vector in μT

        Returns:
            Total magnetic field at sensor in μT
        """
        # Position all magnets (Magpylib 5.x uses mm)
        for finger, position in finger_positions_mm.items():
            if finger in self.magnets:
                self.magnets[finger].position = position

        # Compute field from all magnets (sum individually to avoid collection parent issues)
        B_mT = np.zeros(3)
        for magnet in self.magnets.values():
            B_mT = B_mT + self.sensor.getB(magnet)

        # Convert mT to μT
        B_ut = B_mT * 1000.0

        # Add Earth field if requested
        if include_earth:
            B_ut = B_ut + earth_field_ut

        return B_ut

    def compute_field_grid(
        self,
        finger_positions_mm: Dict[str, np.ndarray],
        grid_extent_mm: float = 100,
        grid_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute magnetic field on a 3D grid for visualization.

        Args:
            finger_positions_mm: Dict mapping finger names to positions (mm)
            grid_extent_mm: Grid extends from -extent to +extent
            grid_points: Number of points per dimension

        Returns:
            (X, Y, Z, B) where B has shape (grid_points, grid_points, grid_points, 3)
        """
        # Position magnets (all in mm)
        for finger, position in finger_positions_mm.items():
            if finger in self.magnets:
                self.magnets[finger].position = position

        # Create grid in mm
        x = np.linspace(-grid_extent_mm, grid_extent_mm, grid_points)
        y = np.linspace(-grid_extent_mm, grid_extent_mm, grid_points)
        z = np.linspace(-grid_extent_mm, grid_extent_mm, grid_points)

        X, Y, Z = np.meshgrid(x, y, z)
        positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Create sensors at all grid points
        sensors = magpy.Sensor(position=positions)

        # Compute field from all magnets (sum individually)
        B = np.zeros((len(positions), 3))
        for magnet in self.magnets.values():
            B = B + sensors.getB(magnet)

        # Reshape back to grid and convert mT to μT
        B = B.reshape(grid_points, grid_points, grid_points, 3) * 1000

        return X, Y, Z, B  # All in mm, field in μT

    def field_magnitude_vs_distance(
        self,
        finger: str = 'index',
        distances_mm: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute field magnitude as a function of distance from a single magnet.

        Useful for validating the 1/r³ falloff and comparing with dipole approximation.

        Args:
            finger: Which finger's magnet to use
            distances_mm: Array of distances to compute (default: 20-150mm)

        Returns:
            (distances_mm, field_magnitudes_ut)
        """
        if distances_mm is None:
            distances_mm = np.linspace(20, 150, 50)

        # Create positions along the magnet axis (all in mm)
        positions = np.zeros((len(distances_mm), 3))
        positions[:, 1] = distances_mm  # Y axis (toward fingers)

        # Create sensors
        sensors = magpy.Sensor(position=positions)

        # Compute field from single magnet at origin
        self.magnets[finger].position = (0, 0, 0)
        B = sensors.getB(self.magnets[finger])

        # Compute magnitudes and convert mT to μT
        magnitudes = np.linalg.norm(B, axis=1) * 1000

        return distances_mm, magnitudes


def compare_dipole_vs_magpylib():
    """
    Compare our dipole approximation with Magpylib's exact solution.
    """
    if not HAS_MAGPYLIB:
        print("Magpylib not installed")
        return

    from .dipole import field_magnitude_at_distance, estimate_dipole_moment

    # Create simulator
    sim = MagpylibSimulator()

    # Compute field vs distance
    distances, magpylib_fields = sim.field_magnitude_vs_distance('index')

    # Compute dipole approximation
    moment = estimate_dipole_moment(6, 3, 1430)  # 6x3mm N48
    dipole_fields = np.array([
        field_magnitude_at_distance(d, moment) for d in distances
    ])

    print("Distance (mm) | Magpylib (μT) | Dipole (μT) | Ratio")
    print("-" * 55)
    for d, mp, dp in zip(distances[::5], magpylib_fields[::5], dipole_fields[::5]):
        ratio = mp / dp if dp > 0 else 0
        print(f"  {d:5.0f}       |    {mp:6.1f}     |   {dp:6.1f}    | {ratio:.2f}")


if __name__ == '__main__':
    print("Magpylib Simulation Test")
    print("=" * 50)

    if not HAS_MAGPYLIB:
        print("ERROR: Magpylib not installed")
        exit(1)

    # Create simulator
    sim = MagpylibSimulator()

    # Test positions (fingertips when hand is open)
    positions = {
        'thumb': np.array([60, 70, 0]),
        'index': np.array([45, 120, 0]),
        'middle': np.array([50, 130, 0]),
        'ring': np.array([45, 120, 0]),
        'pinky': np.array([40, 100, 0])
    }

    print("\nFinger positions (mm):")
    for finger, pos in positions.items():
        print(f"  {finger}: {pos}")

    # Compute field
    B = sim.compute_field(positions, include_earth=False)
    print(f"\nTotal field at sensor (no Earth): {B} μT")
    print(f"Magnitude: {np.linalg.norm(B):.1f} μT")

    B_with_earth = sim.compute_field(positions, include_earth=True)
    print(f"\nWith Earth field: {B_with_earth} μT")
    print(f"Magnitude: {np.linalg.norm(B_with_earth):.1f} μT")

    # Compare with dipole
    print("\n" + "=" * 50)
    print("Dipole vs Magpylib Comparison")
    print("=" * 50)
    compare_dipole_vs_magpylib()
