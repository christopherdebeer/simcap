#!/usr/bin/env python3
"""
Baseline Subtraction Experiment

Apply learnings from magnetometer calibration analysis:
- Use EEEEE pose as baseline reference
- Subtract baseline from raw magnetometer data
- Compare model performance with/without baseline subtraction
- Try orientation-aware normalization using quaternions

Key insight: Each finger pose creates a distinct magnetic signature.
Subtracting the EEEEE baseline normalizes the signal to show pose-relative changes.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.transform import Rotation as R


@dataclass
class LabeledSegment:
    """A labeled segment of magnetometer data."""
    session_id: str
    start_idx: int
    end_idx: int
    finger_binary: np.ndarray  # [5] binary: 0=extended, 1=flexed
    mag_data: np.ndarray       # [N, 3] magnetometer readings
    pitch: float               # Mean pitch during segment
    quaternions: np.ndarray    # [N, 4] orientation quaternions (w, x, y, z)


def load_baseline() -> np.ndarray:
    """Load the EEEEE baseline vector from calibration results."""
    baseline_path = Path("ml/calibration_results/pose_signatures.json")

    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
        return np.array(data['baseline_vector'])
    else:
        # Fallback: compute from data
        print("Warning: No saved baseline found, will compute from EEEEE windows")
        return None


def load_labeled_data(data_dir: Path = Path("data/GAMBIT"),
                      session_filter: str = "2025-12-31") -> List[LabeledSegment]:
    """Load labeled segments from session files."""
    segments = []

    for fpath in data_dir.glob("*.json"):
        if fpath.name == "manifest.json":
            continue
        if fpath.name.endswith(".py"):
            continue
        if session_filter and session_filter not in fpath.name:
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Skipping {fpath.name}: {e}")
            continue

        labels = data.get('labels', [])
        samples = data.get('samples', [])

        if not labels or not samples:
            continue

        # Extract mag data, pitch, and quaternions
        all_mag = np.array([[s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]
                           for s in samples])
        all_pitch = np.array([s.get('euler_pitch', 0) for s in samples])
        all_quats = np.array([[s.get('orientation_w', 1), s.get('orientation_x', 0),
                               s.get('orientation_y', 0), s.get('orientation_z', 0)]
                              for s in samples])

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for label in labels:
            start_idx = label.get('startIndex') or label.get('start_sample')
            end_idx = label.get('endIndex') or label.get('end_sample')

            if start_idx is None or end_idx is None:
                continue

            # Get finger states
            if 'labels' in label and isinstance(label['labels'], dict):
                fingers = label['labels'].get('fingers', {})
            else:
                fingers = label.get('fingers', {})

            if not fingers:
                continue

            # Convert to binary: 0=extended, 1=flexed
            finger_binary = np.array([
                0 if fingers.get(f, 'extended') == 'extended' else 1
                for f in finger_names
            ])

            mag_window = all_mag[start_idx:end_idx]
            pitch_window = all_pitch[start_idx:end_idx]
            quat_window = all_quats[start_idx:end_idx]

            if len(mag_window) < 5:
                continue

            segments.append(LabeledSegment(
                session_id=fpath.stem,
                start_idx=start_idx,
                end_idx=end_idx,
                finger_binary=finger_binary,
                mag_data=mag_window,
                pitch=np.mean(pitch_window),
                quaternions=quat_window
            ))

    return segments


def compute_baseline_from_segments(segments: List[LabeledSegment]) -> np.ndarray:
    """Compute baseline from EEEEE (all extended) segments."""
    eeeee_mags = []

    for seg in segments:
        if np.all(seg.finger_binary == 0):  # All extended
            eeeee_mags.append(np.mean(seg.mag_data, axis=0))

    if eeeee_mags:
        baseline = np.mean(eeeee_mags, axis=0)
        print(f"Computed baseline from {len(eeeee_mags)} EEEEE windows: {baseline}")
        return baseline
    else:
        print("Warning: No EEEEE windows found, using zero baseline")
        return np.zeros(3)


def apply_baseline_subtraction(segments: List[LabeledSegment],
                                baseline: np.ndarray) -> List[LabeledSegment]:
    """Apply baseline subtraction to all segments."""
    subtracted = []

    for seg in segments:
        new_seg = LabeledSegment(
            session_id=seg.session_id,
            start_idx=seg.start_idx,
            end_idx=seg.end_idx,
            finger_binary=seg.finger_binary,
            mag_data=seg.mag_data - baseline,  # Subtract baseline
            pitch=seg.pitch,
            quaternions=seg.quaternions
        )
        subtracted.append(new_seg)

    return subtracted


def rotate_to_world_frame(mag_data: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """
    Rotate magnetometer readings from sensor frame to world frame.

    The quaternion represents sensor orientation in world frame.
    To get world-frame magnetic field: m_world = quat.apply(m_sensor)
    """
    rotated = np.zeros_like(mag_data)

    for i in range(len(mag_data)):
        # Create rotation from quaternion (scipy uses scalar-last by default)
        # Our quaternions are stored as [w, x, y, z], so reorder to [x, y, z, w]
        q = quaternions[i]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # xyzw format

        # Rotate sensor reading to world frame
        rotated[i] = rot.apply(mag_data[i])

    return rotated


def apply_orientation_normalization(segments: List[LabeledSegment]) -> List[LabeledSegment]:
    """
    Apply orientation-aware normalization: rotate all readings to world frame.

    This should make the Earth's field component consistent across orientations,
    leaving only the finger magnet contributions as pose-dependent.
    """
    normalized = []

    for seg in segments:
        # Rotate to world frame
        mag_world = rotate_to_world_frame(seg.mag_data, seg.quaternions)

        new_seg = LabeledSegment(
            session_id=seg.session_id,
            start_idx=seg.start_idx,
            end_idx=seg.end_idx,
            finger_binary=seg.finger_binary,
            mag_data=mag_world,
            pitch=seg.pitch,
            quaternions=seg.quaternions
        )
        normalized.append(new_seg)

    return normalized


def get_split_masks(segments: List[LabeledSegment],
                    strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get train/test masks based on pitch distribution strategy."""
    pitches = np.array([seg.pitch for seg in segments])

    print(f"   Pitch range: [{pitches.min():.1f}°, {pitches.max():.1f}°], std={pitches.std():.1f}°")

    if strategy == "STRICT":
        q1 = np.percentile(pitches, 25)
        q3 = np.percentile(pitches, 75)
        print(f"   Q1={q1:.1f}°, Q3={q3:.1f}° (gap={q3-q1:.1f}°)")
        train_mask = pitches >= q3
        test_mask = pitches <= q1
    elif strategy == "MODERATE":
        p40 = np.percentile(pitches, 40)
        p60 = np.percentile(pitches, 60)
        print(f"   P40={p40:.1f}°, P60={p60:.1f}° (gap={p60-p40:.1f}°)")
        train_mask = pitches >= p60
        test_mask = pitches <= p40
    elif strategy == "MEDIAN":
        median = np.median(pitches)
        print(f"   Median={median:.1f}°")
        train_mask = pitches >= median
        test_mask = pitches < median
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"   Masks: train={train_mask.sum()}, test={test_mask.sum()}, overlap={(train_mask & test_mask).sum()}")

    return train_mask, test_mask


def prepare_windows(segments: List[LabeledSegment],
                    window_size: int = 8,
                    use_derivatives: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare windowed data with optional derivatives."""
    X_list = []
    y_list = []

    for seg in segments:
        mag = seg.mag_data

        if len(mag) < window_size:
            continue

        # Take center window
        start = (len(mag) - window_size) // 2
        window = mag[start:start + window_size]

        if use_derivatives:
            # Compute velocity and acceleration
            features = []
            for t in range(window_size):
                m = window[t]

                # Velocity
                if t == 0:
                    vel = np.zeros(3)
                else:
                    vel = window[t] - window[t-1]

                # Acceleration
                if t <= 1:
                    acc = np.zeros(3)
                else:
                    prev_vel = window[t-1] - window[t-2]
                    acc = vel - prev_vel

                features.append(np.concatenate([m, vel, acc]))

            X_list.append(np.array(features))
        else:
            X_list.append(window)

        y_list.append(seg.finger_binary)

    return np.array(X_list), np.array(y_list)


def create_v6_physics_model(window_size: int, n_features: int,
                            physics_weight: float = 0.1) -> keras.Model:
    """V6 physics-constrained model."""
    from ml.inverse_magnetometry_physics_constrained import PhysicsConstrainedModel

    model = PhysicsConstrainedModel(
        window_size=window_size,
        n_features=n_features,
        hidden_dim=64,
        encoder_type='lstm',
        physics_loss_weight=physics_weight,
        learnable_physics=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    dummy = tf.zeros((1, window_size, n_features))
    _ = model(dummy)

    return model


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_type: str = "v6_physics") -> Dict:
    """Train model and return evaluation metrics."""

    # Normalize
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    if model_type == "v6_physics":
        model = create_v6_physics_model(window_size, n_features, physics_weight=0.1)
    else:
        # Simple baseline model
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(window_size, n_features)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(5, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(
        X_train_norm, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=0
    )

    # Evaluate
    y_train_pred = model.predict(X_train_norm, verbose=0)
    y_test_pred = model.predict(X_test_norm, verbose=0)

    y_train_binary = (y_train_pred > 0.5).astype(int)
    y_test_binary = (y_test_pred > 0.5).astype(int)

    train_acc = np.mean(y_train_binary == y_train)
    test_acc = np.mean(y_test_binary == y_test)

    # Per-finger accuracy
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(finger_names):
        acc = np.mean(y_test_binary[:, i] == y_test[:, i])
        per_finger[finger] = acc

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'per_finger': per_finger,
        'norm_stats': {'mean': mean.tolist(), 'std': std.tolist()}
    }


def run_experiment():
    """Run the baseline subtraction and orientation normalization experiment."""
    print("=" * 70)
    print("CALIBRATION LEARNINGS EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\n1. Loading labeled data from 12/31 session...")
    segments = load_labeled_data()
    print(f"   Loaded {len(segments)} labeled segments")

    # Load or compute baseline
    print("\n2. Loading EEEEE baseline...")
    baseline = load_baseline()
    if baseline is None:
        baseline = compute_baseline_from_segments(segments)
    print(f"   Baseline vector: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] µT")

    # Create different preprocessing versions
    print("\n3. Creating preprocessed data versions...")

    # Version 1: Baseline-subtracted (sensor frame)
    segments_subtracted = apply_baseline_subtraction(segments, baseline)
    print("   ✓ Baseline-subtracted (sensor frame)")

    # Version 2: Orientation-normalized (world frame)
    segments_world = apply_orientation_normalization(segments)
    print("   ✓ Orientation-normalized (world frame)")

    # Version 3: World frame + baseline subtraction
    # First compute world-frame baseline from EEEEE windows
    eeeee_world_mags = []
    for seg in segments_world:
        if np.all(seg.finger_binary == 0):  # EEEEE
            eeeee_world_mags.append(np.mean(seg.mag_data, axis=0))
    if eeeee_world_mags:
        baseline_world = np.mean(eeeee_world_mags, axis=0)
    else:
        baseline_world = np.zeros(3)
    segments_world_sub = apply_baseline_subtraction(segments_world, baseline_world)
    print(f"   ✓ World frame + baseline subtraction")
    print(f"     World baseline: [{baseline_world[0]:.1f}, {baseline_world[1]:.1f}, {baseline_world[2]:.1f}] µT")

    # Show pose signature comparison across preprocessing methods
    print("\n4. Pose magnitudes by preprocessing method:")
    print(f"   {'Pose':<8} {'Raw':>10} {'Sub':>10} {'World':>10} {'W+Sub':>10}")
    print("   " + "-" * 55)

    pose_mags = {'raw': {}, 'sub': {}, 'world': {}, 'world_sub': {}}

    for seg, seg_sub, seg_w, seg_ws in zip(segments, segments_subtracted, segments_world, segments_world_sub):
        pose = ''.join('E' if f == 0 else 'F' for f in seg.finger_binary)
        for key, s in [('raw', seg), ('sub', seg_sub), ('world', seg_w), ('world_sub', seg_ws)]:
            if pose not in pose_mags[key]:
                pose_mags[key][pose] = []
            pose_mags[key][pose].append(np.mean(np.linalg.norm(s.mag_data, axis=1)))

    for pose in sorted(pose_mags['raw'].keys()):
        raw = np.mean(pose_mags['raw'][pose])
        sub = np.mean(pose_mags['sub'][pose])
        world = np.mean(pose_mags['world'][pose])
        world_sub = np.mean(pose_mags['world_sub'][pose])
        print(f"   {pose:<8} {raw:>10.1f} {sub:>10.1f} {world:>10.1f} {world_sub:>10.1f}")

    # Run experiments across split strategies
    strategies = ["STRICT", "MODERATE", "MEDIAN"]
    results = {}

    preprocessing_methods = [
        ('raw', segments, 'RAW (sensor frame)'),
        ('baseline_sub', segments_subtracted, 'BASELINE-SUBTRACTED'),
        ('world_frame', segments_world, 'WORLD FRAME'),
        ('world_sub', segments_world_sub, 'WORLD + BASELINE')
    ]

    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"SPLIT STRATEGY: {strategy}")
        print("=" * 70)

        train_mask, test_mask = get_split_masks(segments, strategy)

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"\n   Train: {n_train} segments, Test: {n_test} segments")

        if n_train == 0 or n_test == 0:
            print("   Skipping - insufficient data")
            continue

        strategy_results = {}

        for key, segs, name in preprocessing_methods:
            train_segs = [s for s, m in zip(segs, train_mask) if m]
            test_segs = [s for s, m in zip(segs, test_mask) if m]

            print(f"\n   Training V6 Physics ({name})...")
            X_train, y_train = prepare_windows(train_segs)
            X_test, y_test = prepare_windows(test_segs)

            if len(X_train) > 0 and len(X_test) > 0:
                result = train_and_evaluate(X_train, y_train, X_test, y_test, "v6_physics")
                strategy_results[key] = result
                print(f"   → Train: {result['train_acc']:.1%}, Test: {result['test_acc']:.1%}")

        results[strategy] = strategy_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impact of Preprocessing on Cross-Orientation Generalization")
    print("=" * 70)
    print(f"\n{'Strategy':<10} {'Raw':>10} {'Sub':>10} {'World':>10} {'W+Sub':>10}")
    print("-" * 55)

    for strategy in strategies:
        if strategy in results:
            raw = results[strategy].get('raw', {}).get('test_acc', 0)
            sub = results[strategy].get('baseline_sub', {}).get('test_acc', 0)
            world = results[strategy].get('world_frame', {}).get('test_acc', 0)
            world_sub = results[strategy].get('world_sub', {}).get('test_acc', 0)
            print(f"{strategy:<10} {raw:>10.1%} {sub:>10.1%} {world:>10.1%} {world_sub:>10.1%}")

    # Best method analysis
    print("\n" + "-" * 55)
    print("Best preprocessing method per split:")
    for strategy in strategies:
        if strategy in results and results[strategy]:
            best_key = max(results[strategy].keys(),
                          key=lambda k: results[strategy][k].get('test_acc', 0))
            best_acc = results[strategy][best_key]['test_acc']
            method_names = {'raw': 'Raw', 'baseline_sub': 'Baseline-Sub',
                           'world_frame': 'World Frame', 'world_sub': 'World+Sub'}
            print(f"   {strategy}: {method_names.get(best_key, best_key)} ({best_acc:.1%})")

    # Per-finger breakdown for best STRICT result
    if 'STRICT' in results and results['STRICT']:
        best_key = max(results['STRICT'].keys(),
                      key=lambda k: results['STRICT'][k].get('test_acc', 0))
        print(f"\nPer-finger accuracy (STRICT split, {best_key}):")
        per_finger = results['STRICT'][best_key]['per_finger']
        for finger, acc in per_finger.items():
            print(f"   {finger:<10}: {acc:.1%}")

    # Save results
    output_path = Path("ml/calibration_preprocessing_results.json")

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
