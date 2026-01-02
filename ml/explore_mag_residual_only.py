#!/usr/bin/env python3
"""
Explore training on ONLY magnetic residual features (3 features) vs full 9-DoF.

Questions:
1. Do we need accel/gyro at all for finger state detection?
2. Can 3 residual mag features match or beat 9 raw features?
3. What's the minimal feature set for good accuracy?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("MAGNETIC RESIDUAL ONLY - MINIMAL FEATURE EXPLORATION")
print("=" * 70)


@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray
    magnitudes: np.ndarray
    mag_vectors: np.ndarray


def load_dec31_session() -> Dict[str, FingerStateData]:
    """Load December 31 session data."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    labels = data['labels']
    combo_data = {}

    for lbl in labels:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        segment_samples = samples[start:end]
        if len(segment_samples) < 5:
            continue

        sensor_data = []
        mag_data = []
        mag_vectors = []

        for s in segment_samples:
            ax = s.get('ax', 0) / 8192.0
            ay = s.get('ay', 0) / 8192.0
            az = s.get('az', 0) / 8192.0
            gx = s.get('gx', 0) / 114.28
            gy = s.get('gy', 0) / 114.28
            gz = s.get('gz', 0) / 114.28

            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx = s.get('mx', 0) / 10.24
                my = s.get('my', 0) / 10.24
                mz = s.get('mz', 0) / 10.24

            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            mag_data.append(np.sqrt(mx**2 + my**2 + mz**2))
            mag_vectors.append([mx, my, mz])

        if not sensor_data:
            continue

        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.array(sensor_data),
                magnitudes=np.array(mag_data),
                mag_vectors=np.array(mag_vectors)
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                magnitudes=np.concatenate([existing.magnitudes, mag_data]),
                mag_vectors=np.vstack([existing.mag_vectors, mag_vectors])
            )

    return combo_data


def compute_baseline(real_data: Dict[str, FingerStateData]) -> np.ndarray:
    """Compute baseline from open palm state."""
    if 'eeeee' in real_data:
        return real_data['eeeee'].mag_vectors.mean(axis=0)
    all_mags = np.vstack([d.mag_vectors for d in real_data.values()])
    return all_mags.mean(axis=0)


class SyntheticGenerator:
    """Generate synthetic samples."""

    def __init__(self, real_data: Dict[str, FingerStateData], baseline: np.ndarray):
        self.real_data = real_data
        self.baseline = baseline
        self._compute_effects()

    def _compute_effects(self):
        self.finger_effects = {}
        single = {'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee', 'ring': 'eeefe', 'pinky': 'eeeef'}
        for finger, combo in single.items():
            if combo in self.real_data:
                d = self.real_data[combo]
                self.finger_effects[finger] = {
                    'delta': d.mag_vectors.mean(axis=0) - self.baseline,
                    'std': d.mag_vectors.std(axis=0)
                }
            else:
                self.finger_effects[finger] = {'delta': np.array([200, 200, 200]), 'std': np.array([50, 50, 50])}

    def generate_combo(self, combo: str, n: int) -> np.ndarray:
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        base_std = self.real_data['eeeee'].mag_vectors.std(axis=0) if 'eeeee' in self.real_data else np.array([20, 20, 20])

        samples = []
        for _ in range(n):
            delta = np.zeros(3)
            std = base_std.copy()
            for i, state in enumerate(combo):
                if state == 'f':
                    f = fingers[i]
                    if f in self.finger_effects:
                        delta += self.finger_effects[f]['delta']
                        std = np.sqrt(std**2 + self.finger_effects[f]['std']**2)
            mag = self.baseline + delta + np.random.normal(0, std)
            samples.append([0, 0, -1, 0, 0, 0, mag[0], mag[1], mag[2]])  # Static IMU
        return np.array(samples)


def extract_features(samples: np.ndarray, baseline: np.ndarray, mode: str) -> np.ndarray:
    """Extract features based on mode.

    Modes:
    - 'raw_9': All 9 raw features
    - 'raw_3': Only raw mag (3 features)
    - 'residual_9': 9 features with mag as residual
    - 'residual_3': Only residual mag (3 features)
    - 'residual_4': Residual mag + magnitude (4 features)
    """
    if mode == 'raw_9':
        return samples
    elif mode == 'raw_3':
        return samples[:, 6:9]  # mx, my, mz
    elif mode == 'residual_9':
        result = samples.copy()
        result[:, 6:9] -= baseline
        return result
    elif mode == 'residual_3':
        return samples[:, 6:9] - baseline
    elif mode == 'residual_4':
        residual = samples[:, 6:9] - baseline
        magnitude = np.linalg.norm(residual, axis=1, keepdims=True)
        return np.hstack([residual, magnitude])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def prepare_dataset(real_data, baseline, mode: str, samples_per_combo: int = 500):
    """Prepare dataset with specified feature extraction mode."""
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    generator = SyntheticGenerator(real_data, baseline)

    all_samples = []
    all_labels = []

    for combo in all_combos:
        if combo in real_data:
            real_samples = real_data[combo].samples
            n_synth = max(0, samples_per_combo - len(real_samples))
            if n_synth > 0:
                combined = np.vstack([real_samples, generator.generate_combo(combo, n_synth)])
            else:
                combined = real_samples[:samples_per_combo]
        else:
            combined = generator.generate_combo(combo, samples_per_combo)

        features = extract_features(combined, baseline, mode)
        label = np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)

        for f in features:
            all_samples.append(f)
            all_labels.append(label)

    X = np.array(all_samples)
    y = np.array(all_labels)

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1

    X = (X - mean) / std

    # Shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    train_end = int(0.7 * len(X))
    val_end = int(0.85 * len(X))

    X_train, y_train = X[idx[:train_end]], y[idx[:train_end]]
    X_val, y_val = X[idx[train_end:val_end]], y[idx[train_end:val_end]]
    X_test, y_test = X[idx[val_end:]], y[idx[val_end:]]

    return X_train, y_train, X_val, y_val, X_test, y_test, {'mean': mean.tolist(), 'std': std.tolist()}


def build_model(n_features: int):
    """Build simple dense model for single-sample prediction."""
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
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


def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, name: str):
    """Train and evaluate."""
    print(f"\n--- {name} (features={X_train.shape[1]}) ---")

    model = build_model(X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=50, batch_size=32, callbacks=callbacks, verbose=0)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: float(np.mean(y_pred[:, i] == y_test[:, i])) for i, f in enumerate(fingers)}
    overall = float(np.mean(y_pred == y_test))

    print(f"  Overall: {overall:.1%}")
    for f in fingers:
        print(f"    {f}: {per_finger[f]:.1%}")

    return {
        'name': name,
        'n_features': X_train.shape[1],
        'overall': overall,
        'per_finger': per_finger,
        'epochs': len(history.history['loss']),
        'val_loss': min(history.history['val_loss'])
    }


def main():
    # Load data
    print("\n--- Loading Data ---")
    real_data = load_dec31_session()
    print(f"Loaded {len(real_data)} combos")

    baseline = compute_baseline(real_data)
    print(f"Baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Test different feature modes
    modes = [
        ('raw_9', 'Raw 9-DoF (ax,ay,az,gx,gy,gz,mx,my,mz)'),
        ('raw_3', 'Raw Mag Only (mx,my,mz)'),
        ('residual_9', 'Residual 9-DoF (with mag-baseline)'),
        ('residual_3', 'Residual Mag Only (mx-bx,my-by,mz-bz)'),
        ('residual_4', 'Residual Mag + Magnitude (4 features)'),
    ]

    results = []
    for mode, desc in modes:
        X_train, y_train, X_val, y_val, X_test, y_test, stats = \
            prepare_dataset(real_data, baseline, mode, samples_per_combo=500)
        r = train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, desc)
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Mode':<45} {'Features':>8} {'Accuracy':>10}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: x['overall'], reverse=True):
        print(f"{r['name']:<45} {r['n_features']:>8} {r['overall']:>9.1%}")

    # Best minimal model
    best = max(results, key=lambda x: x['overall'])
    minimal = [r for r in results if r['n_features'] <= 4]
    best_minimal = max(minimal, key=lambda x: x['overall']) if minimal else None

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print(f"\nBest overall: {best['name']} ({best['overall']:.1%})")
    if best_minimal:
        print(f"Best minimal (≤4 features): {best_minimal['name']} ({best_minimal['overall']:.1%})")

    # Feature importance analysis
    print("\n--- Can we use ONLY magnetic residual? ---")
    res3 = next((r for r in results if r['n_features'] == 3 and 'Residual' in r['name']), None)
    raw9 = next((r for r in results if r['n_features'] == 9 and 'Raw' in r['name']), None)

    if res3 and raw9:
        diff = res3['overall'] - raw9['overall']
        if diff >= 0:
            print(f"✓ YES! 3-feature residual ({res3['overall']:.1%}) matches 9-feature raw ({raw9['overall']:.1%})")
            print("  Accel/gyro appear to add NO value for static finger detection")
        else:
            print(f"✗ NO. 3-feature residual ({res3['overall']:.1%}) is worse than 9-feature ({raw9['overall']:.1%})")
            print(f"  Difference: {diff:.1%}")


if __name__ == '__main__':
    main()
