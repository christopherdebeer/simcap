#!/usr/bin/env python3
"""
Analyze magnetometer axis alignment in session data.

Goal: Determine if axis alignment fixes have been applied correctly,
and identify the proper single-location fix.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import sys


def load_session(filepath: Path) -> list:
    """Load session JSON data."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get('samples', [])


def analyze_axis_alignment(samples: list, session_name: str):
    """Analyze axis alignment between raw and processed magnetometer values."""

    # Extract raw magnetometer (LSB) and processed values
    raw_mx = np.array([s.get('mx', 0) for s in samples])
    raw_my = np.array([s.get('my', 0) for s in samples])
    raw_mz = np.array([s.get('mz', 0) for s in samples])

    # Extract processed magnetometer (µT)
    mx_ut = np.array([s.get('mx_ut', 0) for s in samples])
    my_ut = np.array([s.get('my_ut', 0) for s in samples])
    mz_ut = np.array([s.get('mz_ut', 0) for s in samples])

    # Extract accelerometer (g)
    ax_g = np.array([s.get('ax_g', 0) for s in samples])
    ay_g = np.array([s.get('ay_g', 0) for s in samples])
    az_g = np.array([s.get('az_g', 0) for s in samples])

    # Convert raw to µT for comparison
    MAG_LSB_TO_UT = 0.09765625  # 10 µT / 1024 LSB
    raw_mx_ut = raw_mx * MAG_LSB_TO_UT
    raw_my_ut = raw_my * MAG_LSB_TO_UT
    raw_mz_ut = raw_mz * MAG_LSB_TO_UT

    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}")

    # 1. Check what axis transformation was applied
    print("\n--- Axis Transformation Detection ---")

    # Check correlation between raw and processed axes
    transforms = []
    for raw_label, raw_data in [('raw_mx', raw_mx_ut), ('raw_my', raw_my_ut), ('raw_mz', raw_mz_ut)]:
        for proc_label, proc_data in [('mx_ut', mx_ut), ('my_ut', my_ut), ('mz_ut', mz_ut)]:
            corr, _ = stats.pearsonr(raw_data, proc_data) if np.std(raw_data) > 0.1 else (0, 1)
            if abs(corr) > 0.9:
                sign = '+' if corr > 0 else '-'
                transforms.append((raw_label, proc_label, sign, corr))

    print("\nDetected mappings (|correlation| > 0.9):")
    for raw_label, proc_label, sign, corr in transforms:
        print(f"  {proc_label} = {sign}{raw_label} (r={corr:.3f})")

    # Expected transformation according to current code:
    #   mx_ut = my_ut_raw (swap X/Y)
    #   my_ut = -mx_ut_raw (swap X/Y, negate)
    #   mz_ut = mz_ut_raw (no change)

    # 2. Verify against expected transformation
    print("\n--- Expected vs Actual Transformation ---")
    print("Expected (from code):")
    print("  mx_ut = raw_my  (swap)")
    print("  my_ut = -raw_mx (swap + negate)")
    print("  mz_ut = raw_mz  (unchanged)")

    # Direct comparison
    expected_mx = raw_my_ut
    expected_my_negated = -raw_mx_ut
    expected_my_unnegated = raw_mx_ut
    expected_mz = raw_mz_ut

    corr_mx, _ = stats.pearsonr(expected_mx, mx_ut)
    corr_my_neg, _ = stats.pearsonr(expected_my_negated, my_ut)
    corr_my_unneg, _ = stats.pearsonr(expected_my_unnegated, my_ut)
    corr_mz, _ = stats.pearsonr(expected_mz, mz_ut)

    print(f"\nActual correlations:")
    print(f"  mx_ut vs raw_my: r={corr_mx:.4f} {'✓' if corr_mx > 0.99 else '✗'}")
    print(f"  my_ut vs -raw_mx: r={corr_my_neg:.4f} {'✓' if corr_my_neg > 0.99 else '✗'}")
    print(f"  my_ut vs +raw_mx: r={corr_my_unneg:.4f} {'(old/unnegated)' if corr_my_unneg > 0.99 else ''}")
    print(f"  mz_ut vs raw_mz: r={corr_mz:.4f} {'✓' if corr_mz > 0.99 else '✗'}")

    # 3. Analyze accel-mag correlation (should all be positive when aligned correctly)
    print("\n--- Accelerometer-Magnetometer Correlations ---")
    print("(Should all be POSITIVE for correctly aligned axes)")

    corr_ax_mx, _ = stats.pearsonr(ax_g, mx_ut)
    corr_ay_my, _ = stats.pearsonr(ay_g, my_ut)
    corr_az_mz, _ = stats.pearsonr(az_g, mz_ut)

    print(f"  ax vs mx: r={corr_ax_mx:.3f} {'✓' if corr_ax_mx > 0 else '✗ INVERTED'}")
    print(f"  ay vs my: r={corr_ay_my:.3f} {'✓' if corr_ay_my > 0 else '✗ INVERTED'}")
    print(f"  az vs mz: r={corr_az_mz:.3f} {'✓' if corr_az_mz > 0 else '✗ INVERTED'}")

    # 4. Cross-axis correlations (should be near zero if axes are independent)
    print("\n--- Cross-Axis Correlations ---")
    print("(Should be near zero for orthogonal axes)")

    cross_corrs = [
        ('ax vs my', stats.pearsonr(ax_g, my_ut)[0]),
        ('ax vs mz', stats.pearsonr(ax_g, mz_ut)[0]),
        ('ay vs mx', stats.pearsonr(ay_g, mx_ut)[0]),
        ('ay vs mz', stats.pearsonr(ay_g, mz_ut)[0]),
        ('az vs mx', stats.pearsonr(az_g, mx_ut)[0]),
        ('az vs my', stats.pearsonr(az_g, my_ut)[0]),
    ]

    for label, corr in cross_corrs:
        flag = '⚠' if abs(corr) > 0.5 else ''
        print(f"  {label}: r={corr:.3f} {flag}")

    # 5. Check what the RAW accel-mag correlations are (before any processing)
    print("\n--- RAW Accel-Mag Correlations (no processing) ---")
    print("This tells us what transformation is needed at source.")

    raw_ax = np.array([s.get('ax', 0) for s in samples])
    raw_ay = np.array([s.get('ay', 0) for s in samples])
    raw_az = np.array([s.get('az', 0) for s in samples])

    # Check all raw correlations
    print("\nRaw accel vs raw mag:")
    raw_correlations = []
    for a_label, a_data in [('ax', raw_ax), ('ay', raw_ay), ('az', raw_az)]:
        for m_label, m_data in [('mx', raw_mx), ('my', raw_my), ('mz', raw_mz)]:
            corr, _ = stats.pearsonr(a_data, m_data)
            raw_correlations.append((a_label, m_label, corr))

    # Sort by absolute correlation to see strongest relationships
    raw_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    for a_label, m_label, corr in raw_correlations:
        flag = '**' if abs(corr) > 0.8 else ''
        print(f"  {a_label} vs {m_label}: r={corr:.3f} {flag}")

    # 6. Determine the correct mapping
    print("\n--- Recommended Axis Mapping ---")

    # For each accel axis, find the most correlated mag axis
    best_mappings = {}
    for a_label, a_data in [('ax', raw_ax), ('ay', raw_ay), ('az', raw_az)]:
        best_corr = 0
        best_m = None
        best_sign = '+'
        for m_label, m_data in [('mx', raw_mx), ('my', raw_my), ('mz', raw_mz)]:
            corr, _ = stats.pearsonr(a_data, m_data)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_m = m_label
                best_sign = '+' if corr > 0 else '-'
        best_mappings[a_label] = (best_m, best_sign, best_corr)

    print("\nBest mapping (raw mag axis for each accel axis):")
    for a_label, (m_label, sign, corr) in best_mappings.items():
        proc_m = {'ax': 'mx_ut', 'ay': 'my_ut', 'az': 'mz_ut'}[a_label]
        print(f"  {proc_m} should be {sign}{m_label}_ut (correlation: {corr:.3f})")

    # 7. Sample value check
    print("\n--- Sample Values (first 3 samples) ---")
    for i in range(min(3, len(samples))):
        s = samples[i]
        print(f"\nSample {i}:")
        print(f"  Raw mag: mx={s.get('mx')}, my={s.get('my')}, mz={s.get('mz')}")
        print(f"  Processed: mx_ut={s.get('mx_ut'):.2f}, my_ut={s.get('my_ut'):.2f}, mz_ut={s.get('mz_ut'):.2f}")
        print(f"  Expected (swap+negate): mx_ut={raw_my_ut[i]:.2f}, my_ut={-raw_mx_ut[i]:.2f}, mz_ut={raw_mz_ut[i]:.2f}")

    return {
        'session': session_name,
        'y_negated': corr_my_neg > 0.99,
        'accel_mag_aligned': all([corr_ax_mx > 0, corr_ay_my > 0, corr_az_mz > 0]),
        'corr_ax_mx': corr_ax_mx,
        'corr_ay_my': corr_ay_my,
        'corr_az_mz': corr_az_mz,
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Get all session files, sorted by date
    session_files = sorted(data_dir.glob('2025-12-19*.json'))

    if not session_files:
        print("No session files found!")
        return

    print(f"Found {len(session_files)} sessions from 2025-12-19")

    # Analyze latest session in detail
    latest = session_files[-1]
    samples = load_session(latest)
    result = analyze_axis_alignment(samples, latest.name)

    # Summary across all recent sessions
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SESSIONS")
    print("="*60)

    all_results = []
    for session_file in session_files:
        samples = load_session(session_file)
        if len(samples) < 10:
            continue
        r = analyze_axis_alignment(samples, session_file.name)
        all_results.append(r)

    print("\n--- Y-Axis Negation Status ---")
    for r in all_results:
        status = "✓ Negated" if r['y_negated'] else "✗ Not negated"
        print(f"  {r['session']}: {status}")

    print("\n--- Accel-Mag Alignment Status ---")
    for r in all_results:
        status = "✓ Aligned" if r['accel_mag_aligned'] else "✗ Misaligned"
        details = f"ax-mx:{r['corr_ax_mx']:.2f} ay-my:{r['corr_ay_my']:.2f} az-mz:{r['corr_az_mz']:.2f}"
        print(f"  {r['session']}: {status} ({details})")


if __name__ == '__main__':
    main()
