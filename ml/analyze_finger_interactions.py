#!/usr/bin/env python3
"""
Analyze finger magnetic field interactions to understand non-additivity.

Key question: Why does thumb + index ≠ ffeee?
Hypothesis: Fingers physically interact when flexed together, changing field geometry.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

print("=" * 70)
print("FINGER MAGNETIC FIELD INTERACTION ANALYSIS")
print("=" * 70)


@dataclass
class ComboData:
    combo: str
    residuals: np.ndarray
    n_flexed: int


def load_data():
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    baseline_mags = []
    combo_raw = {}

    for lbl in data['labels']:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start, end = lbl.get('start_sample', 0), lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start, end = lbl.get('startIndex', 0), lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        combo = ''.join(['e' if fingers.get(f, '?') == 'extended' else 'f' if fingers.get(f, '?') == 'flexed' else '?'
                        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']])

        segment = data['samples'][start:end]
        if len(segment) < 5:
            continue

        mags = []
        for s in segment:
            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx, my, mz = s.get('mx', 0)/10.24, s.get('my', 0)/10.24, s.get('mz', 0)/10.24
            mags.append([mx, my, mz])

        if combo not in combo_raw:
            combo_raw[combo] = []
        combo_raw[combo].extend(mags)

        if combo == 'eeeee':
            baseline_mags.extend(mags)

    baseline = np.mean(baseline_mags, axis=0)

    result = {}
    for combo, mags in combo_raw.items():
        mags = np.array(mags)
        residuals = mags - baseline
        n_flexed = combo.count('f')
        result[combo] = ComboData(combo=combo, residuals=residuals, n_flexed=n_flexed)

    return result, baseline


def analyze_pairwise_interactions(data: Dict[str, ComboData]):
    """Test all available pairs for additivity."""
    print("\n" + "=" * 70)
    print("PAIRWISE ADDITIVITY ANALYSIS")
    print("=" * 70)

    single = {'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee', 'ring': 'eeefe', 'pinky': 'eeeef'}
    pairs = {
        ('thumb', 'index'): 'ffeee',
        ('middle', 'ring', 'pinky'): 'eefff',
    }

    # Get single finger effects
    single_effects = {}
    for finger, combo in single.items():
        if combo in data:
            single_effects[finger] = data[combo].residuals.mean(axis=0)
            print(f"{finger}: [{single_effects[finger][0]:+.0f}, {single_effects[finger][1]:+.0f}, {single_effects[finger][2]:+.0f}]")

    print("\n" + "-" * 70)
    print("ADDITIVITY TESTS")
    print("-" * 70)

    # Test pairs we have data for
    tests = [
        (('thumb', 'index'), 'ffeee'),
        (('middle', 'ring', 'pinky'), 'eefff'),
    ]

    for fingers, combo in tests:
        if combo not in data:
            continue
        if not all(f in single_effects for f in fingers):
            continue

        predicted = sum(single_effects[f] for f in fingers)
        actual = data[combo].residuals.mean(axis=0)
        error = np.linalg.norm(actual - predicted)
        actual_norm = np.linalg.norm(actual)

        print(f"\n{' + '.join(fingers)} = {combo}:")
        print(f"  Predicted: [{predicted[0]:+.0f}, {predicted[1]:+.0f}, {predicted[2]:+.0f}]")
        print(f"  Actual:    [{actual[0]:+.0f}, {actual[1]:+.0f}, {actual[2]:+.0f}]")
        print(f"  Error: {error:.0f} μT ({error/actual_norm*100:.0f}% of actual)")


def analyze_scaling_patterns(data: Dict[str, ComboData]):
    """Look for scaling/saturation patterns."""
    print("\n" + "=" * 70)
    print("MAGNITUDE vs NUMBER OF FLEXED FINGERS")
    print("=" * 70)

    by_n_flexed = {}
    for combo, d in data.items():
        n = d.n_flexed
        if n not in by_n_flexed:
            by_n_flexed[n] = []
        mag = np.linalg.norm(d.residuals.mean(axis=0))
        by_n_flexed[n].append((combo, mag))

    print(f"\n{'N Flexed':>10} {'Combos':>10} {'Mag Mean':>12} {'Mag Std':>12} {'Examples'}")
    print("-" * 70)

    for n in sorted(by_n_flexed.keys()):
        combos = by_n_flexed[n]
        mags = [m for _, m in combos]
        examples = ', '.join([c for c, _ in combos[:3]])
        print(f"{n:>10} {len(combos):>10} {np.mean(mags):>12.0f} {np.std(mags):>12.0f} {examples}")


def analyze_direction_patterns(data: Dict[str, ComboData]):
    """Analyze how direction changes with different combos."""
    print("\n" + "=" * 70)
    print("DIRECTION ANALYSIS")
    print("=" * 70)

    print(f"\n{'Combo':<8} {'Dx':>8} {'Dy':>8} {'Dz':>8} {'Dominant Axis':<15}")
    print("-" * 55)

    for combo in sorted(data.keys()):
        if combo == 'eeeee':
            continue

        mean = data[combo].residuals.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 50:
            continue

        direction = mean / norm
        dominant_idx = np.argmax(np.abs(direction))
        dominant = ['X', 'Y', 'Z'][dominant_idx]
        sign = '+' if direction[dominant_idx] > 0 else '-'

        print(f"{combo:<8} {direction[0]:>+7.2f} {direction[1]:>+7.2f} {direction[2]:>+7.2f} {sign}{dominant}")


def propose_better_model(data: Dict[str, ComboData]):
    """Propose a non-additive model for synthetic generation."""
    print("\n" + "=" * 70)
    print("PROPOSED NON-ADDITIVE MODEL")
    print("=" * 70)

    # Key observation: Rather than sum finger effects, we should:
    # 1. Use lookup table for observed combos
    # 2. For unseen combos, interpolate between similar observed combos

    print("""
STRATEGY: Instead of additive synthesis, use:

1. DIRECT LOOKUP: If combo was observed, sample from its distribution
2. INTERPOLATION: For unseen combos, find nearest observed combos and interpolate

Key insight: The magnetic field depends on the PHYSICAL GEOMETRY of flexed
fingers together, not a simple superposition of individual effects.

Example: thumb+index flexed together brings magnets closer to each other,
creating field interactions that don't exist when measured separately.
    """)

    # Build interpolation model
    observed = list(data.keys())
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    missing = [c for c in all_combos if c not in observed]

    print(f"\nObserved combos: {len(observed)}")
    print(f"Missing combos: {len(missing)}")

    # For each missing combo, find closest observed combo (by Hamming distance)
    def hamming(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    print(f"\nMissing combo interpolation (nearest observed):")
    print(f"{'Missing':<10} {'Nearest':<10} {'Distance':>10}")
    print("-" * 35)

    for missing_combo in missing[:10]:  # Show first 10
        distances = [(c, hamming(missing_combo, c)) for c in observed]
        nearest = min(distances, key=lambda x: x[1])
        print(f"{missing_combo:<10} {nearest[0]:<10} {nearest[1]:>10}")


def main():
    data, baseline = load_data()
    print(f"Loaded {len(data)} combos, baseline: [{baseline[0]:.0f}, {baseline[1]:.0f}, {baseline[2]:.0f}]")

    analyze_pairwise_interactions(data)
    analyze_scaling_patterns(data)
    analyze_direction_patterns(data)
    propose_better_model(data)

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. ADDITIVITY FAILS: Finger effects are NOT additive (76% error for thumb+index)

2. PHYSICAL REASON: When multiple fingers flex, the magnets move into different
   positions relative to each other, creating field INTERACTIONS that can't be
   predicted from individual finger measurements.

3. BETTER APPROACH: Use nearest-neighbor interpolation instead of additive model:
   - For observed combos: sample directly from real distribution
   - For unseen combos: find closest observed combo and perturb its distribution

4. IMPLICATION: With only 10 observed combos, we can't reliably synthesize all 32.
   The synthetic data will have a domain gap until we collect more real combos.
    """)


if __name__ == '__main__':
    main()
