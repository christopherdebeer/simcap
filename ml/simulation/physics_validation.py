#!/usr/bin/env python3
"""
Physics Simulation Validation and Parameterized Data Generation

This module validates the magnetic dipole simulation against observed sensor data
and provides tools for generating synthetic training data with different magnet
configurations (size, strength, grade).

Key Features:
1. Validate simulation physics against theoretical dipole equations
2. Compare simulation output to observed data distributions
3. Parameterized simulation for different magnet sizes/strengths
4. Support for planning reduced magnet size experiments

Usage:
    python -m ml.simulation.physics_validation --validate
    python -m ml.simulation.physics_validation --compare-real
    python -m ml.simulation.physics_validation --magnet-sweep
    python -m ml.simulation.physics_validation --generate --magnet-size 5x2 --grade N35

Author: Physics Simulation Data Generation Task
Date: 2025-01-01
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import glob


# ============================================================================
# PHYSICAL CONSTANTS AND MAGNET SPECIFICATIONS
# ============================================================================

MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
MU_0_OVER_4PI = MU_0 / (4 * np.pi)

# Earth's magnetic field for different locations (μT)
EARTH_FIELDS = {
    'edinburgh': np.array([16.0, 0.0, 47.8]),   # Horizontal N, E, Down
    'london': np.array([17.0, 0.0, 46.0]),
    'san_francisco': np.array([23.0, 5.0, 42.0]),
    'equator': np.array([30.0, 0.0, 0.0]),      # Horizontal, no vertical
}

# Residual flux density (Br) for different neodymium grades (mT)
MAGNET_GRADES = {
    'N35': 1170,   # mT - lower grade, smaller/weaker magnets
    'N38': 1250,
    'N42': 1320,
    'N45': 1370,
    'N48': 1430,   # Common grade for current setup
    'N50': 1450,
    'N52': 1480,   # Highest common grade
}


@dataclass
class MagnetSpec:
    """Specification for a cylindrical neodymium magnet."""
    diameter_mm: float
    height_mm: float  # Also called thickness
    grade: str = 'N48'

    @property
    def Br_mT(self) -> float:
        """Residual flux density in milliTesla."""
        return MAGNET_GRADES.get(self.grade, 1430)

    @property
    def volume_m3(self) -> float:
        """Volume in cubic meters."""
        radius_m = (self.diameter_mm / 2) / 1000.0
        height_m = self.height_mm / 1000.0
        return np.pi * radius_m**2 * height_m

    @property
    def dipole_moment(self) -> float:
        """
        Magnetic dipole moment in A·m².

        For a uniformly magnetized cylinder:
            m = M × V = (Br/μ₀) × πr²h
        """
        Br_T = self.Br_mT / 1000.0
        M = Br_T / MU_0  # Magnetization in A/m
        return M * self.volume_m3

    def __str__(self) -> str:
        return f"{self.diameter_mm}×{self.height_mm}mm {self.grade} (m={self.dipole_moment:.4f} A·m²)"


# Standard magnet specifications for reference
MAGNET_SPECS = {
    'current':  MagnetSpec(6, 3, 'N48'),   # Current setup: 6mm×3mm N48
    'reduced_1': MagnetSpec(5, 2, 'N42'),  # Smaller: 5mm×2mm N42
    'reduced_2': MagnetSpec(4, 2, 'N38'),  # Even smaller: 4mm×2mm N38
    'reduced_3': MagnetSpec(3, 1, 'N35'),  # Minimal: 3mm×1mm N35
    'tiny':      MagnetSpec(2, 1, 'N35'),  # Very small: 2mm×1mm N35
}


# ============================================================================
# DIPOLE PHYSICS CALCULATIONS
# ============================================================================

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
    """
    r_vec = observation_point - dipole_position
    r_mag = np.linalg.norm(r_vec)

    if r_mag < min_distance:
        return np.zeros(3)

    r_hat = r_vec / r_mag
    m_dot_r = np.dot(dipole_moment, r_hat)
    B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - dipole_moment) / (r_mag ** 3)

    return B


def field_magnitude_at_distance(
    distance_mm: float,
    magnet: MagnetSpec,
    angle_deg: float = 0.0
) -> float:
    """
    Calculate field magnitude at a given distance from a dipole.

    Args:
        distance_mm: Distance from magnet center in mm
        magnet: Magnet specification
        angle_deg: Angle from dipole axis (0 = along axis)

    Returns:
        Field magnitude in μT
    """
    r_m = distance_mm / 1000.0
    theta = np.radians(angle_deg)
    moment = magnet.dipole_moment

    # Field components in spherical coordinates
    B_r = MU_0_OVER_4PI * 2 * moment * np.cos(theta) / (r_m ** 3)
    B_theta = MU_0_OVER_4PI * moment * np.sin(theta) / (r_m ** 3)

    B_mag = np.sqrt(B_r**2 + B_theta**2)
    return B_mag * 1e6  # Convert to μT


def compute_snr(
    magnet: MagnetSpec,
    distance_extended_mm: float = 80,
    distance_flexed_mm: float = 50,
    noise_floor_ut: float = 1.0
) -> Dict[str, float]:
    """
    Estimate signal-to-noise ratio for finger tracking with given magnet.

    Args:
        magnet: Magnet specification
        distance_extended_mm: Distance when finger is extended
        distance_flexed_mm: Distance when finger is flexed
        noise_floor_ut: Sensor noise floor in μT (RMS)

    Returns:
        Dict with SNR metrics
    """
    B_extended = field_magnitude_at_distance(distance_extended_mm, magnet)
    B_flexed = field_magnitude_at_distance(distance_flexed_mm, magnet)
    delta = B_flexed - B_extended

    return {
        'magnet': str(magnet),
        'signal_extended_ut': B_extended,
        'signal_flexed_ut': B_flexed,
        'signal_delta_ut': delta,
        'noise_floor_ut': noise_floor_ut,
        'snr_extended': B_extended / noise_floor_ut,
        'snr_flexed': B_flexed / noise_floor_ut,
        'snr_delta': delta / noise_floor_ut,
        'detectable': delta > 3 * noise_floor_ut  # 3σ criterion
    }


# ============================================================================
# REAL DATA LOADING AND ANALYSIS
# ============================================================================

def load_real_sessions(data_dir: str = 'data/GAMBIT') -> List[Dict]:
    """
    Load real sensor data sessions from disk.

    Args:
        data_dir: Path to GAMBIT session directory

    Returns:
        List of session data dicts with parsed magnetometer values
    """
    # Handle both absolute and relative paths
    data_path = Path(data_dir)
    if not data_path.exists():
        # Try relative to script location
        data_path = Path(__file__).parent.parent.parent / data_dir

    session_files = list(data_path.glob('*.json'))
    sessions = []

    for f in sorted(session_files):
        try:
            with open(f, 'r') as file:
                data = json.load(file)

            if 'samples' not in data or len(data['samples']) < 50:
                continue

            samples = data['samples']
            mag_vectors = []

            for s in samples:
                if 'mx_ut' in s:
                    mx, my, mz = s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)
                elif 'mx' in s:
                    # Convert from LSB: 1024 LSB/Gauss = 10.24 LSB/μT
                    mx = s.get('mx', 0) / 10.24
                    my = s.get('my', 0) / 10.24
                    mz = s.get('mz', 0) / 10.24
                else:
                    continue

                mag_vectors.append([mx, my, mz])

            if len(mag_vectors) > 50:
                mag_array = np.array(mag_vectors)
                sessions.append({
                    'filename': f.name,
                    'n_samples': len(mag_vectors),
                    'mag_vectors': mag_array,
                    'magnitudes': np.linalg.norm(mag_array, axis=1),
                    'labels': data.get('labels', []),
                    'metadata': data.get('metadata', {})
                })
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            continue

    return sessions


def analyze_real_data(sessions: List[Dict]) -> Dict:
    """
    Compute statistics from real sensor data.

    Args:
        sessions: List of session dicts from load_real_sessions()

    Returns:
        Dict with statistics for comparison with simulation
    """
    all_vectors = []
    all_magnitudes = []

    for s in sessions:
        all_vectors.extend(s['mag_vectors'].tolist())
        all_magnitudes.extend(s['magnitudes'].tolist())

    vectors = np.array(all_vectors)
    magnitudes = np.array(all_magnitudes)

    # Filter to reasonable range (remove outliers)
    valid_mask = (magnitudes > 20) & (magnitudes < 300)
    filtered_mags = magnitudes[valid_mask]

    earth_magnitude = np.linalg.norm(EARTH_FIELDS['edinburgh'])

    return {
        'n_sessions': len(sessions),
        'n_samples_total': len(magnitudes),
        'n_samples_valid': len(filtered_mags),

        # Raw statistics
        'raw': {
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes)),
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'p5': float(np.percentile(magnitudes, 5)),
            'p50': float(np.percentile(magnitudes, 50)),
            'p95': float(np.percentile(magnitudes, 95)),
        },

        # Filtered statistics (20-300 μT)
        'filtered': {
            'mean': float(np.mean(filtered_mags)),
            'std': float(np.std(filtered_mags)),
            'min': float(np.min(filtered_mags)),
            'max': float(np.max(filtered_mags)),
            'p5': float(np.percentile(filtered_mags, 5)),
            'p25': float(np.percentile(filtered_mags, 25)),
            'p50': float(np.percentile(filtered_mags, 50)),
            'p75': float(np.percentile(filtered_mags, 75)),
            'p95': float(np.percentile(filtered_mags, 95)),
        },

        # Deviation from Earth field
        'residual': {
            'mean_deviation': float(np.mean(filtered_mags) - earth_magnitude),
            'p50_deviation': float(np.percentile(filtered_mags, 50) - earth_magnitude),
            'p95_deviation': float(np.percentile(filtered_mags, 95) - earth_magnitude),
        },

        # Per-axis statistics
        'axes': {
            'x': {'mean': float(np.mean(vectors[:, 0])), 'std': float(np.std(vectors[:, 0]))},
            'y': {'mean': float(np.mean(vectors[:, 1])), 'std': float(np.std(vectors[:, 1]))},
            'z': {'mean': float(np.mean(vectors[:, 2])), 'std': float(np.std(vectors[:, 2]))},
        }
    }


# ============================================================================
# SIMULATION VALIDATION
# ============================================================================

def validate_dipole_physics() -> Dict:
    """
    Validate dipole calculations against known physics.

    Returns:
        Validation results with computed vs expected values
    """
    results = {}

    # Test 1: 6x3mm N48 magnet at various distances
    magnet = MAGNET_SPECS['current']
    results['magnet_specs'] = {
        'size': f"{magnet.diameter_mm}×{magnet.height_mm}mm",
        'grade': magnet.grade,
        'Br_mT': magnet.Br_mT,
        'dipole_moment': magnet.dipole_moment,
    }

    # Expected: ~0.0135 A·m² for 6x3mm N48
    expected_moment = 0.0135
    moment_error = abs(magnet.dipole_moment - expected_moment) / expected_moment * 100
    results['moment_validation'] = {
        'computed': magnet.dipole_moment,
        'expected': expected_moment,
        'error_percent': moment_error,
        'passed': moment_error < 5  # <5% error acceptable
    }

    # Test 2: Field at various distances
    distances = [30, 40, 50, 60, 70, 80, 100]
    field_vs_distance = {}

    # Expected values (μT) from known dipole physics
    for d in distances:
        B = field_magnitude_at_distance(d, magnet)
        field_vs_distance[d] = B

    # Verify 1/r³ falloff
    B_50 = field_vs_distance[50]
    B_100 = field_vs_distance[100]
    expected_ratio = (100/50)**3  # Should be 8
    actual_ratio = B_50 / B_100

    results['field_vs_distance'] = {
        'values_ut': field_vs_distance,
        'r_cubed_ratio': {
            'expected': expected_ratio,
            'actual': actual_ratio,
            'error_percent': abs(actual_ratio - expected_ratio) / expected_ratio * 100,
            'passed': abs(actual_ratio - expected_ratio) < 0.1
        }
    }

    # Test 3: SNR for finger tracking
    snr = compute_snr(magnet)
    results['snr'] = snr

    return results


def compare_simulation_to_real(
    sessions: List[Dict],
    magnet: MagnetSpec = None
) -> Dict:
    """
    Compare simulation output to real data distributions.

    Args:
        sessions: Real data sessions
        magnet: Magnet spec to use (default: current setup)

    Returns:
        Comparison metrics
    """
    if magnet is None:
        magnet = MAGNET_SPECS['current']

    # Analyze real data
    real_stats = analyze_real_data(sessions)

    # Simulate expected field magnitudes for typical finger poses
    # Extended: 70-90mm from sensor, Flexed: 40-60mm
    simulated_mags = []

    for _ in range(1000):
        # Random finger distances
        distances = {
            'thumb': np.random.uniform(40, 80),
            'index': np.random.uniform(50, 90),
            'middle': np.random.uniform(55, 95),
            'ring': np.random.uniform(50, 90),
            'pinky': np.random.uniform(45, 85),
        }

        # Compute field from each finger (simplified: just magnitude sum)
        total_field = EARTH_FIELDS['edinburgh'].copy()

        for finger, dist in distances.items():
            B = field_magnitude_at_distance(dist, magnet)
            # Add random direction component
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            total_field += B * direction * 0.5  # Attenuate for direction

        simulated_mags.append(np.linalg.norm(total_field))

    sim_mags = np.array(simulated_mags)

    return {
        'magnet': str(magnet),
        'real_data': real_stats['filtered'],
        'simulated': {
            'mean': float(np.mean(sim_mags)),
            'std': float(np.std(sim_mags)),
            'p5': float(np.percentile(sim_mags, 5)),
            'p50': float(np.percentile(sim_mags, 50)),
            'p95': float(np.percentile(sim_mags, 95)),
        },
        'comparison': {
            'mean_diff': float(np.mean(sim_mags) - real_stats['filtered']['mean']),
            'std_diff': float(np.std(sim_mags) - real_stats['filtered']['std']),
            'p50_diff': float(np.percentile(sim_mags, 50) - real_stats['filtered']['p50']),
        }
    }


# ============================================================================
# MAGNET SIZE SWEEP FOR PLANNING
# ============================================================================

def magnet_sweep_analysis(
    noise_floor_ut: float = 1.0,
    distance_extended_mm: float = 80,
    distance_flexed_mm: float = 50
) -> Dict:
    """
    Analyze performance across different magnet sizes for planning.

    Args:
        noise_floor_ut: Sensor noise floor
        distance_extended_mm: Distance when finger extended
        distance_flexed_mm: Distance when finger flexed

    Returns:
        Comparison of different magnet configurations
    """
    results = []

    for name, magnet in MAGNET_SPECS.items():
        snr = compute_snr(
            magnet,
            distance_extended_mm=distance_extended_mm,
            distance_flexed_mm=distance_flexed_mm,
            noise_floor_ut=noise_floor_ut
        )

        results.append({
            'name': name,
            'size': f"{magnet.diameter_mm}×{magnet.height_mm}mm",
            'grade': magnet.grade,
            'dipole_moment': magnet.dipole_moment,
            **snr
        })

    return {'magnet_comparison': results}


def custom_magnet_analysis(
    diameter_mm: float,
    height_mm: float,
    grade: str = 'N48'
) -> Dict:
    """
    Analyze a custom magnet configuration.

    Args:
        diameter_mm: Magnet diameter
        height_mm: Magnet height/thickness
        grade: Magnet grade

    Returns:
        Analysis results for the custom magnet
    """
    magnet = MagnetSpec(diameter_mm, height_mm, grade)

    # Field at various distances
    distances = [30, 40, 50, 60, 70, 80, 100, 120]
    fields = {d: field_magnitude_at_distance(d, magnet) for d in distances}

    # SNR analysis
    snr = compute_snr(magnet)

    # Compare to current setup
    current = MAGNET_SPECS['current']
    moment_ratio = magnet.dipole_moment / current.dipole_moment

    return {
        'magnet': str(magnet),
        'specs': {
            'diameter_mm': diameter_mm,
            'height_mm': height_mm,
            'grade': grade,
            'Br_mT': magnet.Br_mT,
            'volume_mm3': magnet.volume_m3 * 1e9,
            'dipole_moment': magnet.dipole_moment,
        },
        'field_vs_distance_ut': fields,
        'snr': snr,
        'vs_current': {
            'moment_ratio': moment_ratio,
            'field_ratio': moment_ratio,  # Linear with moment
            'note': f'This magnet has {moment_ratio*100:.1f}% of current magnet strength'
        }
    }


# ============================================================================
# MAIN CLI
# ============================================================================

def print_validation_results():
    """Print physics validation results."""
    print("=" * 70)
    print("PHYSICS VALIDATION")
    print("=" * 70)

    results = validate_dipole_physics()

    print(f"\n1. Magnet Specifications ({results['magnet_specs']['size']} {results['magnet_specs']['grade']})")
    print(f"   Dipole moment: {results['magnet_specs']['dipole_moment']:.4f} A·m²")

    print(f"\n2. Moment Validation")
    mv = results['moment_validation']
    print(f"   Computed: {mv['computed']:.4f} A·m²")
    print(f"   Expected: {mv['expected']:.4f} A·m²")
    print(f"   Error: {mv['error_percent']:.1f}% {'✓' if mv['passed'] else '✗'}")

    print(f"\n3. Field vs Distance (on-axis, μT)")
    for d, B in results['field_vs_distance']['values_ut'].items():
        print(f"   {d:3d} mm: {B:7.1f} μT")

    r3 = results['field_vs_distance']['r_cubed_ratio']
    print(f"\n   1/r³ validation: ratio={r3['actual']:.2f} (expected {r3['expected']:.1f}) {'✓' if r3['passed'] else '✗'}")

    print(f"\n4. SNR for Finger Tracking")
    snr = results['snr']
    print(f"   Extended (80mm): {snr['signal_extended_ut']:.1f} μT, SNR={snr['snr_extended']:.0f}")
    print(f"   Flexed (50mm):   {snr['signal_flexed_ut']:.1f} μT, SNR={snr['snr_flexed']:.0f}")
    print(f"   Delta:           {snr['signal_delta_ut']:.1f} μT, SNR={snr['snr_delta']:.0f}")
    print(f"   Detectable (3σ): {'Yes ✓' if snr['detectable'] else 'No ✗'}")


def print_real_data_comparison():
    """Print comparison with real data."""
    print("=" * 70)
    print("REAL DATA COMPARISON")
    print("=" * 70)

    try:
        sessions = load_real_sessions()
        if not sessions:
            print("\nNo real data sessions found. Check data/GAMBIT directory.")
            return

        stats = analyze_real_data(sessions)

        print(f"\n1. Data Overview")
        print(f"   Sessions: {stats['n_sessions']}")
        print(f"   Total samples: {stats['n_samples_total']}")
        print(f"   Valid samples (20-300 μT): {stats['n_samples_valid']}")

        print(f"\n2. Magnitude Statistics (filtered)")
        f = stats['filtered']
        print(f"   Mean: {f['mean']:.1f} μT")
        print(f"   Std:  {f['std']:.1f} μT")
        print(f"   P5:   {f['p5']:.1f} μT")
        print(f"   P25:  {f['p25']:.1f} μT")
        print(f"   P50:  {f['p50']:.1f} μT (median)")
        print(f"   P75:  {f['p75']:.1f} μT")
        print(f"   P95:  {f['p95']:.1f} μT")

        print(f"\n3. Residual vs Earth Field (~50.4 μT)")
        r = stats['residual']
        print(f"   Mean deviation: {r['mean_deviation']:.1f} μT")
        print(f"   P50 deviation:  {r['p50_deviation']:.1f} μT")
        print(f"   P95 deviation:  {r['p95_deviation']:.1f} μT")

        # Simulation comparison
        comparison = compare_simulation_to_real(sessions)
        print(f"\n4. Simulation Comparison ({comparison['magnet']})")
        sim = comparison['simulated']
        print(f"   Simulated mean: {sim['mean']:.1f} μT")
        print(f"   Simulated P50:  {sim['p50']:.1f} μT")

        c = comparison['comparison']
        print(f"   Mean diff: {c['mean_diff']:+.1f} μT")
        print(f"   P50 diff:  {c['p50_diff']:+.1f} μT")

    except Exception as e:
        print(f"\nError: {e}")


def print_magnet_sweep():
    """Print magnet size sweep analysis."""
    print("=" * 70)
    print("MAGNET SIZE SWEEP ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing different magnet configurations for reduced-size planning...\n")

    sweep = magnet_sweep_analysis()

    print(f"{'Name':12s} {'Size':10s} {'Grade':5s} {'Moment':10s} {'B@50mm':8s} {'B@80mm':8s} {'Delta':8s} {'SNR':5s} {'Det?':4s}")
    print("-" * 80)

    for m in sweep['magnet_comparison']:
        print(f"{m['name']:12s} {m['size']:10s} {m['grade']:5s} "
              f"{m['dipole_moment']:.4f} A·m² "
              f"{m['signal_flexed_ut']:6.1f} μT "
              f"{m['signal_extended_ut']:6.1f} μT "
              f"{m['signal_delta_ut']:6.1f} μT "
              f"{m['snr_delta']:5.0f} "
              f"{'✓' if m['detectable'] else '✗':4s}")

    print("\nNotes:")
    print("- 'Det?' = Detectable at 3σ threshold (delta > 3 × noise floor)")
    print("- Noise floor assumed: 1.0 μT RMS")
    print("- Distances: Extended=80mm, Flexed=50mm")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Physics Simulation Validation and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml.simulation.physics_validation --validate
  python -m ml.simulation.physics_validation --compare-real
  python -m ml.simulation.physics_validation --magnet-sweep
  python -m ml.simulation.physics_validation --custom 5 2 N42
        """
    )

    parser.add_argument('--validate', action='store_true',
                        help='Validate dipole physics calculations')
    parser.add_argument('--compare-real', action='store_true',
                        help='Compare simulation to real data')
    parser.add_argument('--magnet-sweep', action='store_true',
                        help='Analyze different magnet sizes')
    parser.add_argument('--custom', nargs=3, metavar=('DIAM', 'HEIGHT', 'GRADE'),
                        help='Analyze custom magnet (e.g., --custom 5 2 N42)')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    if args.all or (not any([args.validate, args.compare_real, args.magnet_sweep, args.custom])):
        args.validate = True
        args.compare_real = True
        args.magnet_sweep = True

    if args.json:
        results = {}
        if args.validate:
            results['validation'] = validate_dipole_physics()
        if args.magnet_sweep:
            results['magnet_sweep'] = magnet_sweep_analysis()
        if args.custom:
            results['custom'] = custom_magnet_analysis(
                float(args.custom[0]), float(args.custom[1]), args.custom[2]
            )
        print(json.dumps(results, indent=2))
    else:
        if args.validate:
            print_validation_results()
            print()
        if args.compare_real:
            print_real_data_comparison()
            print()
        if args.magnet_sweep:
            print_magnet_sweep()
            print()
        if args.custom:
            print("=" * 70)
            print("CUSTOM MAGNET ANALYSIS")
            print("=" * 70)
            result = custom_magnet_analysis(
                float(args.custom[0]), float(args.custom[1]), args.custom[2]
            )
            print(f"\nMagnet: {result['magnet']}")
            print(f"\nField vs Distance:")
            for d, B in result['field_vs_distance_ut'].items():
                print(f"  {d:3d} mm: {B:6.1f} μT")
            print(f"\n{result['vs_current']['note']}")


if __name__ == '__main__':
    main()
