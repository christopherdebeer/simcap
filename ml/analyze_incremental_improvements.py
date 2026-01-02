#!/usr/bin/env python3
"""
Incremental Physics Calibration.

Instead of fitting from scratch, calibrate per-finger based on:
1. Known fitted effects from per_finger_fit_results.json
2. Observed combo validation from physics_synthetic_training_results.json

Goal: Improve synthetic data quality to close the 98.6% → 34.8% gap.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

print("=" * 70)
print("INCREMENTAL PHYSICS CALIBRATION")
print("Analyzing prediction errors to improve synthetic generation")
print("=" * 70)


def load_prior_results():
    """Load all prior analysis results."""
    ml_dir = Path(__file__).parent

    # Load fitted interaction model
    with open(ml_dir / 'per_finger_fit_results.json') as f:
        per_finger = json.load(f)

    # Load training results
    with open(ml_dir / 'physics_synthetic_training_results.json') as f:
        training = json.load(f)

    # Load generalization analysis
    gen_path = ml_dir / 'generalization_analysis.json'
    if gen_path.exists():
        with open(gen_path) as f:
            generalization = json.load(f)
    else:
        generalization = None

    return per_finger, training, generalization


def analyze_prediction_errors(per_finger: Dict, training: Dict):
    """Analyze where predictions fail vs succeed."""
    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    # Fitted effects
    fitted = per_finger['interaction_model']['fitted_effects']
    interaction = per_finger['interaction_model']['interaction_strength']

    # Original single-finger effects
    original = per_finger['finger_effects']

    print(f"\nInteraction strength: {interaction:.3f}")
    print(f"Per additional finger: {1 + interaction:.2f}x scaling")

    # Compare fitted vs original
    print(f"\n{'Finger':<10} {'Original Effect':<35} {'Fitted Effect':<35} {'Delta %'}")
    print("-" * 90)

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    # Handle both dict and list formats
    if isinstance(fitted, list):
        fitted_dict = {name: fitted[i] for i, name in enumerate(finger_names)}
    else:
        fitted_dict = fitted

    for finger in finger_names:
        orig = np.array(original[finger])
        fit = np.array(fitted_dict[finger])
        delta = np.linalg.norm(fit - orig) / (np.linalg.norm(orig) + 1e-6) * 100

        orig_str = f"[{orig[0]:+7.0f}, {orig[1]:+7.0f}, {orig[2]:+7.0f}]"
        fit_str = f"[{fit[0]:+7.0f}, {fit[1]:+7.0f}, {fit[2]:+7.0f}]"

        print(f"{finger:<10} {orig_str:<35} {fit_str:<35} {delta:5.1f}%")

    # Per-combo accuracy analysis
    print("\n" + "=" * 70)
    print("PER-COMBO PERFORMANCE")
    print("=" * 70)

    combo_acc = training['per_combo_accuracy']

    # Group by accuracy
    perfect = []
    good = []
    poor = []

    for combo, data in combo_acc.items():
        acc = data['accuracy']
        if acc >= 0.99:
            perfect.append((combo, acc, data['n_flexed']))
        elif acc >= 0.90:
            good.append((combo, acc, data['n_flexed']))
        else:
            poor.append((combo, acc, data['n_flexed']))

    print(f"\nPerfect (≥99%): {len(perfect)} combos")
    for combo, acc, n in perfect:
        print(f"  {combo}: {acc*100:.1f}% (n_flexed={n})")

    print(f"\nGood (90-99%): {len(good)} combos")
    for combo, acc, n in good:
        print(f"  {combo}: {acc*100:.1f}% (n_flexed={n})")

    print(f"\nPoor (<90%): {len(poor)} combos")
    for combo, acc, n in poor:
        print(f"  {combo}: {acc*100:.1f}% (n_flexed={n})")


def analyze_generalization(generalization: Dict):
    """Analyze where generalization works/fails."""
    if not generalization:
        print("\nNo generalization analysis found.")
        return

    print("\n" + "=" * 70)
    print("GENERALIZATION ANALYSIS (Leave-One-Out)")
    print("=" * 70)

    real_only = generalization['real_only']
    with_synth = generalization['with_synthetic']

    # Find where synthetic helps
    helps = []
    no_help = []

    for combo in real_only:
        r = real_only[combo]['accuracy']
        s = with_synth[combo]['accuracy']
        delta = s - r

        if delta > 0.1:  # >10% improvement
            helps.append((combo, r, s, delta))
        else:
            no_help.append((combo, r, s, delta))

    print(f"\nSynthetic HELPS (>10% improvement): {len(helps)} combos")
    for combo, r, s, d in helps:
        print(f"  {combo}: {r*100:.0f}% → {s*100:.0f}% (+{d*100:.0f}%)")

    print(f"\nSynthetic NO HELP: {len(no_help)} combos")
    for combo, r, s, d in no_help:
        print(f"  {combo}: {r*100:.0f}% → {s*100:.0f}% ({d*100:+.0f}%)")

    # Analyze pattern
    print("\n" + "-" * 50)
    print("PATTERN ANALYSIS:")

    # What do the successful ones have in common?
    if helps:
        success_flexed = [sum(1 for c in combo if c == 'f') for combo, *_ in helps]
        print(f"  Successful combos n_flexed: {success_flexed}")

        # Which fingers?
        success_fingers = {}
        for combo, *_ in helps:
            for i, (c, name) in enumerate(zip(combo, ['thumb', 'index', 'middle', 'ring', 'pinky'])):
                if c == 'f':
                    success_fingers[name] = success_fingers.get(name, 0) + 1
        print(f"  Fingers in successful combos: {success_fingers}")


def propose_improvements():
    """Propose specific improvements based on analysis."""
    print("\n" + "=" * 70)
    print("PROPOSED IMPROVEMENTS")
    print("=" * 70)

    print("""
FINDINGS:
1. Real data alone: 98.6% on observed combos (excellent)
2. Physics model: 39% prediction error (needs improvement)
3. Synthetic training: 34.8% (domain gap too large)
4. Generalization: Only ring (100%) and thumb (45%) benefit from synthetic

ROOT CAUSES:
1. Interaction model is linear approximation of nonlinear physics
2. Noise model doesn't match real sensor noise distribution
3. Per-finger effects fitted globally, not per-combo

INCREMENTAL IMPROVEMENTS:

1. CALIBRATE NOISE MODEL:
   - Current: Gaussian noise with magnitude-proportional std
   - Better: Match actual per-combo std from observed data
   - Use observed std[combo] instead of synthetic approximation

2. PER-COMBO CORRECTIONS:
   - For observed combos: Use real mean/std directly
   - For unseen combos: Interpolate from nearest observed

3. AUGMENTATION STRATEGY:
   - Keep real data for observed combos (98.6% is hard to beat)
   - Only use synthetic for the 22 missing combos
   - Weight real data higher than synthetic in training

4. FOCUS ON NEAREST NEIGHBORS:
   - 22 missing combos, but many are 1-bit away from observed
   - Use observed combo stats to generate realistic neighbors

5. COLLECT MORE DATA:
   - Priority combos: multi-finger combinations
   - Missing pairs: (thumb+middle), (index+ring), etc.
   - 2-3 additional combos would dramatically help
""")


def main():
    per_finger, training, generalization = load_prior_results()

    analyze_prediction_errors(per_finger, training)
    analyze_generalization(generalization)
    propose_improvements()

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    experiments = training['experiments']
    print(f"\n{'Approach':<35} {'Accuracy':>10}")
    print("-" * 50)
    for exp in experiments:
        print(f"{exp['name']:<35} {exp['overall_accuracy']*100:>9.1f}%")

    print(f"\nKey metrics:")
    print(f"  Observed combos: 10/32")
    print(f"  Missing combos: 22/32")
    print(f"  Best observed accuracy: 98.6% (real data only)")
    print(f"  Best with synthetic: 89.0% (hybrid)")
    print(f"  Generalization gap: 9.6%")


if __name__ == '__main__':
    main()
