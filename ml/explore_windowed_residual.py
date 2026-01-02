#!/usr/bin/env python3
"""
Compare windowed models: Raw 9-DoF vs 3-feature Residual Mag

Key question: Can we simplify the model to only use 3 magnetic residual features
with windowed inference, achieving similar accuracy to the full 9-DoF model?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("WINDOWED MODEL: 9-DoF RAW vs 3-FEATURE RESIDUAL")
print("=" * 70)


@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray
    mag_vectors: np.ndarray


def load_session():
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    combo_data = {}
    for lbl in data['labels']:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start, end = lbl.get('start_sample', 0), lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start, end = lbl.get('startIndex', 0), lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        segment = data['samples'][start:end]
        if len(segment) < 5:
            continue

        sensor_data, mag_vectors = [], []
        for s in segment:
            ax, ay, az = s.get('ax', 0)/8192.0, s.get('ay', 0)/8192.0, s.get('az', 0)/8192.0
            gx, gy, gz = s.get('gx', 0)/114.28, s.get('gy', 0)/114.28, s.get('gz', 0)/114.28
            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx, my, mz = s.get('mx', 0)/10.24, s.get('my', 0)/10.24, s.get('mz', 0)/10.24
            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            mag_vectors.append([mx, my, mz])

        combo = ''.join(['e' if fingers.get(f, '?') == 'extended' else 'f' if fingers.get(f, '?') == 'flexed' else '?'
                        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']])

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(combo=combo, samples=np.array(sensor_data), mag_vectors=np.array(mag_vectors))
        else:
            e = combo_data[combo]
            combo_data[combo] = FingerStateData(combo=combo, samples=np.vstack([e.samples, sensor_data]), mag_vectors=np.vstack([e.mag_vectors, mag_vectors]))

    return combo_data


class SyntheticGen:
    def __init__(self, real_data, baseline):
        self.real_data = real_data
        self.baseline = baseline
        self.effects = {}
        for f, c in [('thumb', 'feeee'), ('index', 'efeee'), ('middle', 'eefee'), ('ring', 'eeefe'), ('pinky', 'eeeef')]:
            if c in real_data:
                d = real_data[c]
                self.effects[f] = {'delta': d.mag_vectors.mean(axis=0) - baseline, 'std': d.mag_vectors.std(axis=0)}
            else:
                self.effects[f] = {'delta': np.array([200, 200, 200]), 'std': np.array([50, 50, 50])}

    def generate(self, combo, n):
        base_std = self.real_data['eeeee'].mag_vectors.std(axis=0) if 'eeeee' in self.real_data else np.array([20, 20, 20])
        samples = []
        for _ in range(n):
            delta, std = np.zeros(3), base_std.copy()
            for i, s in enumerate(combo):
                if s == 'f':
                    f = ['thumb', 'index', 'middle', 'ring', 'pinky'][i]
                    delta += self.effects[f]['delta']
                    std = np.sqrt(std**2 + self.effects[f]['std']**2)
            mag = self.baseline + delta + np.random.normal(0, std)
            samples.append([0, 0, -1, 0, 0, 0, mag[0], mag[1], mag[2]])
        return np.array(samples)


def create_windows(samples, window_size=50, stride=25):
    n = len(samples)
    if n < window_size:
        samples = np.vstack([samples, np.zeros((window_size - n, samples.shape[1]))])
        n = window_size
    windows = [samples[i:i+window_size] for i in range(0, n - window_size + 1, stride)]
    return np.array(windows) if windows else np.array([samples[:window_size]])


def prepare_dataset(real_data, baseline, use_residual_3=False, samples_per_combo=1000, window_size=50):
    """Prepare windowed dataset.

    Args:
        use_residual_3: If True, use only 3-feature residual (mx-bx, my-by, mz-bz).
                       If False, use full 9-DoF raw features.
    """
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    gen = SyntheticGen(real_data, baseline)

    all_windows, all_labels = [], []

    for combo in all_combos:
        if combo in real_data:
            real = real_data[combo].samples
            n_synth = max(0, samples_per_combo - len(real))
            combined = np.vstack([real, gen.generate(combo, n_synth)]) if n_synth > 0 else real[:samples_per_combo]
        else:
            combined = gen.generate(combo, samples_per_combo)

        if use_residual_3:
            # Extract only mag residual (3 features)
            combined = combined[:, 6:9] - baseline

        windows = create_windows(combined, window_size)
        label = np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)

        for w in windows:
            all_windows.append(w)
            all_labels.append(label)

    X = np.array(all_windows)
    y = np.array(all_labels)

    # Normalize
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1

    X = (X - mean) / std

    # Split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    t1, t2 = int(0.7 * len(X)), int(0.85 * len(X))

    return (X[idx[:t1]], y[idx[:t1]], X[idx[t1:t2]], y[idx[t1:t2]],
            X[idx[t2:]], y[idx[t2:]], {'mean': mean.tolist(), 'std': std.tolist()})


def build_cnn_lstm(input_shape):
    """CNN-LSTM hybrid model."""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"\n--- {name} ---")
    print(f"  Input shape: {X_train.shape}")

    model = build_cnn_lstm(X_train.shape[1:])

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=50, batch_size=32, callbacks=callbacks, verbose=0)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: float(np.mean(y_pred[:, i] == y_test[:, i])) for i, f in enumerate(fingers)}
    overall = float(np.mean(y_pred == y_test))

    print(f"  Overall: {overall:.1%}")
    print(f"  Per-finger: {', '.join(f'{f}:{per_finger[f]:.0%}' for f in fingers)}")
    print(f"  Epochs: {len(history.history['loss'])}, Val Loss: {min(history.history['val_loss']):.4f}")

    return {'name': name, 'shape': X_train.shape[1:], 'overall': overall, 'per_finger': per_finger,
            'val_loss': min(history.history['val_loss']), 'params': model.count_params()}


def main():
    print("\n--- Loading Data ---")
    real_data = load_session()
    baseline = real_data['eeeee'].mag_vectors.mean(axis=0) if 'eeeee' in real_data else np.zeros(3)
    print(f"Loaded {len(real_data)} combos, baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Run 3 trials for each to get stable results
    n_trials = 3
    results = {'raw_9': [], 'residual_3': []}

    for trial in range(n_trials):
        print(f"\n{'='*70}\nTRIAL {trial+1}/{n_trials}\n{'='*70}")

        # Raw 9-DoF
        X_train, y_train, X_val, y_val, X_test, y_test, _ = prepare_dataset(
            real_data, baseline, use_residual_3=False, samples_per_combo=1000)
        r1 = train_eval(X_train, y_train, X_val, y_val, X_test, y_test, "Raw 9-DoF (50×9)")
        results['raw_9'].append(r1)

        # Residual 3-feature
        X_train, y_train, X_val, y_val, X_test, y_test, stats = prepare_dataset(
            real_data, baseline, use_residual_3=True, samples_per_combo=1000)
        r2 = train_eval(X_train, y_train, X_val, y_val, X_test, y_test, "Residual Mag (50×3)")
        results['residual_3'].append(r2)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (averaged over", n_trials, "trials)")
    print("=" * 70)

    for key, trials in results.items():
        accs = [t['overall'] for t in trials]
        per_finger_means = {}
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            per_finger_means[f] = np.mean([t['per_finger'][f] for t in trials])

        print(f"\n{trials[0]['name']}:")
        print(f"  Overall: {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"  Per-finger: {', '.join(f'{f}:{per_finger_means[f]:.0%}' for f in per_finger_means)}")
        print(f"  Model params: {trials[0]['params']:,}")

    # Comparison
    raw_mean = np.mean([t['overall'] for t in results['raw_9']])
    res_mean = np.mean([t['overall'] for t in results['residual_3']])
    diff = res_mean - raw_mean

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(diff) < 0.01:
        print(f"\n≈ EQUIVALENT: 3-feature residual ({res_mean:.1%}) ≈ 9-feature raw ({raw_mean:.1%})")
        print("  → Can use 3-feature residual with 66% fewer features!")
        print(f"  → Model size reduction: ~{1 - results['residual_3'][0]['params']/results['raw_9'][0]['params']:.0%}")
    elif diff > 0:
        print(f"\n✓ RESIDUAL BETTER: 3-feature ({res_mean:.1%}) > 9-feature ({raw_mean:.1%})")
    else:
        print(f"\n✗ RAW BETTER: 9-feature ({raw_mean:.1%}) > 3-feature ({res_mean:.1%})")
        print(f"  Difference: {diff:.1%}")


if __name__ == '__main__':
    main()
