#!/usr/bin/env python3
"""
Leave-One-Out Generalization Analysis.

Tests whether synthetic data helps models generalize to UNSEEN combos
by holding out each observed combo in turn.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("LEAVE-ONE-OUT GENERALIZATION ANALYSIS")
print("=" * 70)


def load_session_data() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load session data and compute per-combo residuals."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    baseline_mags = []
    combo_samples = {}

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

        if combo not in combo_samples:
            combo_samples[combo] = []
        combo_samples[combo].extend(mags)

        if combo == 'eeeee':
            baseline_mags.extend(mags)

    baseline = np.mean(baseline_mags, axis=0)

    result = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        result[combo] = mags - baseline

    return result, baseline


def load_interaction_model() -> Tuple[Dict[str, np.ndarray], float]:
    """Load fitted interaction model."""
    model_path = Path(__file__).parent / 'per_finger_fit_results.json'
    with open(model_path, 'r') as f:
        data = json.load(f)

    fitted = np.array(data['interaction_model']['fitted_effects'])
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    effects = {name: fitted[i] for i, name in enumerate(finger_names)}
    interaction = data['interaction_model']['interaction_strength']

    return effects, interaction


def generate_synthetic(combo: str, effects: Dict[str, np.ndarray],
                      interaction: float, n_samples: int = 200) -> np.ndarray:
    """Generate synthetic samples using interaction model."""
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    if combo == 'eeeee':
        mean = np.zeros(3)
    else:
        flexed = [finger_names[i] for i, c in enumerate(combo) if c == 'f']
        n_flexed = len(flexed)
        linear = sum(effects[f] for f in flexed)
        scale = 1.0 + interaction * (n_flexed - 1) / 4 if n_flexed > 1 else 1.0
        mean = linear * scale

    noise_std = (np.linalg.norm(mean) + 50) * 0.15
    return np.random.normal(mean, noise_std, size=(n_samples, 3))


def combo_to_labels(combo: str) -> np.ndarray:
    return np.array([1.0 if c == 'f' else 0.0 for c in combo])


def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def leave_one_out_analysis(combo_samples: Dict[str, np.ndarray],
                          effects: Dict[str, np.ndarray],
                          interaction: float,
                          use_synthetic: bool) -> Dict:
    """
    Leave-one-out cross-validation.

    For each observed combo:
    1. Hold it out as test set
    2. Train on remaining combos (+ synthetic if use_synthetic=True)
    3. Evaluate on held-out combo
    """
    results = {}
    observed_combos = list(combo_samples.keys())

    for held_out in observed_combos:
        if held_out == 'eeeee':
            continue  # Skip baseline

        # Prepare training data
        X_train, y_train = [], []

        for combo, samples in combo_samples.items():
            if combo == held_out:
                continue

            labels = combo_to_labels(combo)
            X_train.append(samples)
            y_train.append(np.tile(labels, (len(samples), 1)))

        if use_synthetic:
            # Add synthetic data for held-out combo and all unobserved combos
            all_combos = [f"{t}{i}{m}{r}{p}"
                          for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

            for combo in all_combos:
                if combo in combo_samples and combo != held_out:
                    continue  # Already have real data

                synth = generate_synthetic(combo, effects, interaction, n_samples=100)
                labels = combo_to_labels(combo)
                X_train.append(synth)
                y_train.append(np.tile(labels, (len(synth), 1)))

        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)

        # Prepare test data (held-out combo)
        X_test = combo_samples[held_out]
        y_test = np.tile(combo_to_labels(held_out), (len(X_test), 1))

        # Normalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train_norm = X_train_norm[idx]
        y_train = y_train[idx]

        # Train
        model = build_model()
        model.fit(
            X_train_norm, y_train,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )

        # Evaluate on held-out combo
        pred = model.predict(X_test_norm, verbose=0)
        pred_binary = (pred > 0.5).astype(int)
        acc = np.mean((pred_binary == y_test).all(axis=1))

        results[held_out] = {
            'accuracy': float(acc),
            'n_test': len(X_test),
            'n_train': len(X_train),
            'n_flexed': sum(1 for c in held_out if c == 'f')
        }

        print(f"  {held_out}: {acc*100:.1f}% (held out, n={len(X_test)})")

    return results


def main():
    # Load data
    combo_samples, baseline = load_session_data()
    effects, interaction = load_interaction_model()

    print(f"\nLoaded {len(combo_samples)} observed combinations")
    print(f"Interaction strength: {interaction:.3f}")

    # Analysis 1: Real data only (no synthetic)
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT: Real Data Only")
    print("=" * 70)
    results_real = leave_one_out_analysis(combo_samples, effects, interaction, use_synthetic=False)

    # Analysis 2: With synthetic data
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT: Real + Synthetic (Interaction Model)")
    print("=" * 70)
    results_synth = leave_one_out_analysis(combo_samples, effects, interaction, use_synthetic=True)

    # Summary
    print("\n" + "=" * 70)
    print("GENERALIZATION COMPARISON")
    print("=" * 70)

    print(f"\n{'Held-Out':<10} {'N Flexed':>10} {'Real Only':>12} {'With Synth':>12} {'Delta':>10}")
    print("-" * 60)

    for combo in sorted(results_real.keys()):
        r = results_real[combo]
        s = results_synth[combo]
        delta = s['accuracy'] - r['accuracy']
        delta_str = f"{delta*100:+.1f}%" if delta != 0 else "0.0%"

        print(f"{combo:<10} {r['n_flexed']:>10} {r['accuracy']*100:>11.1f}% {s['accuracy']*100:>11.1f}% {delta_str:>10}")

    # Aggregate stats
    real_accs = [r['accuracy'] for r in results_real.values()]
    synth_accs = [r['accuracy'] for r in results_synth.values()]

    print("-" * 60)
    print(f"{'Mean':<10} {'':<10} {np.mean(real_accs)*100:>11.1f}% {np.mean(synth_accs)*100:>11.1f}% "
          f"{(np.mean(synth_accs)-np.mean(real_accs))*100:>+9.1f}%")

    # Analysis by number of flexed fingers
    print("\n" + "=" * 70)
    print("GENERALIZATION BY N FLEXED")
    print("=" * 70)

    flex_groups = {}
    for combo, r in results_real.items():
        n = r['n_flexed']
        if n not in flex_groups:
            flex_groups[n] = {'real': [], 'synth': []}
        flex_groups[n]['real'].append(r['accuracy'])
        flex_groups[n]['synth'].append(results_synth[combo]['accuracy'])

    print(f"\n{'N Flexed':>10} {'Real Mean':>12} {'Synth Mean':>12} {'Delta':>10}")
    print("-" * 50)

    for n in sorted(flex_groups.keys()):
        rm = np.mean(flex_groups[n]['real'])
        sm = np.mean(flex_groups[n]['synth'])
        delta = sm - rm
        print(f"{n:>10} {rm*100:>11.1f}% {sm*100:>11.1f}% {delta*100:>+9.1f}%")

    # Save results
    output = {
        'real_only': results_real,
        'with_synthetic': results_synth,
        'mean_accuracy': {
            'real_only': float(np.mean(real_accs)),
            'with_synthetic': float(np.mean(synth_accs)),
        }
    }

    output_path = Path(__file__).parent / 'generalization_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
