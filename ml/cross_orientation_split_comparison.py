"""
Cross-Orientation Split Comparison

Compares V6 physics-constrained model vs V4-style baseline across different
test split strategies:
- STRICT: Train on pitch ≥ Q3, test on pitch ≤ Q1 (42° gap, harshest)
- MODERATE: Train on pitch ≥ P60, test on pitch ≤ P40 (4° gap, balanced)
- MEDIAN: Train on pitch ≥ median, test on pitch < median (no gap, fairest)

This experiment helps understand how model generalization changes with
different levels of train/test orientation separation.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf


# ============================================================================
# Data Loading (from inverse_magnetometry_physics_constrained.py)
# ============================================================================

@dataclass
class LabeledSegment:
    """A labeled segment with magnetometer data and finger state."""
    session_id: str
    start_idx: int
    end_idx: int
    finger_binary: np.ndarray  # [5] binary: 0=extended, 1=flexed
    mag_data: np.ndarray       # [N, 3] magnetometer readings
    pitch: float               # Mean pitch during segment


def load_labeled_data(data_dir: Path = Path("data/GAMBIT"),
                      session_filter: str = "2025-12-31") -> List[LabeledSegment]:
    """Load labeled segments from session files.

    Args:
        data_dir: Directory containing session JSON files
        session_filter: Only load sessions containing this string in filename.
                       Set to None to load all sessions.
    """
    segments = []

    for fpath in data_dir.glob("*.json"):
        if fpath.name == "manifest.json":
            continue
        if fpath.name.endswith(".py"):
            continue
        # Filter to specific session(s)
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

        # Extract mag data and pitch
        mag_data = np.array([[s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]
                            for s in samples])
        pitches = np.array([s.get('euler_pitch', 0) for s in samples])

        for label in labels:
            # Handle both label formats
            start_idx = label.get('startIndex') or label.get('start_sample')
            end_idx = label.get('endIndex') or label.get('end_sample')

            if start_idx is None or end_idx is None:
                continue

            # Get fingers - handle both formats
            if 'labels' in label and isinstance(label['labels'], dict):
                fingers = label['labels'].get('fingers', {})
            else:
                fingers = label.get('fingers', {})

            if not fingers:
                continue
            if not any(v not in ['unknown', None, ''] for v in fingers.values()):
                continue

            # Convert to binary
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            finger_binary = np.array([
                1 if fingers.get(f, 'extended') == 'flexed' else 0
                for f in finger_names
            ])

            segment = LabeledSegment(
                session_id=fpath.stem,
                start_idx=start_idx,
                end_idx=end_idx,
                finger_binary=finger_binary,
                mag_data=mag_data[start_idx:end_idx],
                pitch=np.mean(pitches[start_idx:end_idx])
            )
            segments.append(segment)

    return segments


def add_temporal_derivatives(windows: np.ndarray) -> np.ndarray:
    """Add velocity and acceleration features to windows.

    Input: [N, T, 3] raw magnetometer
    Output: [N, T, 9] with [mag, velocity, acceleration]
    """
    # Velocity: first-order finite difference
    velocity = np.diff(windows, axis=1, prepend=windows[:, :1, :])

    # Acceleration: second-order finite difference
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])

    return np.concatenate([windows, velocity, acceleration], axis=-1)


def create_windows(segments: List[LabeledSegment],
                   window_size: int = 8,
                   use_derivatives: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windowed samples from segments.

    Returns: (windows, labels, segment_indices)
    """
    windows = []
    labels = []
    seg_indices = []

    for seg_idx, seg in enumerate(segments):
        n_samples = len(seg.mag_data)
        if n_samples < window_size:
            continue

        # Create overlapping windows
        for start in range(0, n_samples - window_size + 1, window_size // 2):
            window = seg.mag_data[start:start + window_size]
            windows.append(window)
            labels.append(seg.finger_binary)
            seg_indices.append(seg_idx)

    windows = np.array(windows)
    labels = np.array(labels)
    seg_indices = np.array(seg_indices)

    if use_derivatives:
        windows = add_temporal_derivatives(windows)

    return windows, labels, seg_indices


# ============================================================================
# Split Strategies
# ============================================================================

def get_split_masks(segments: List[LabeledSegment],
                    strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get train/test masks for different split strategies."""
    pitches = np.array([seg.pitch for seg in segments])

    if strategy == "STRICT":
        # Q3/Q1 - largest gap
        q1 = np.percentile(pitches, 25)
        q3 = np.percentile(pitches, 75)
        train_mask = pitches >= q3
        test_mask = pitches <= q1

    elif strategy == "MODERATE":
        # P60/P40 - moderate gap
        p40 = np.percentile(pitches, 40)
        p60 = np.percentile(pitches, 60)
        train_mask = pitches >= p60
        test_mask = pitches <= p40

    elif strategy == "MEDIAN":
        # 50/50 split, no gap
        median = np.percentile(pitches, 50)
        train_mask = pitches >= median
        test_mask = pitches < median

    elif strategy == "SOFT":
        # Median/P35
        p35 = np.percentile(pitches, 35)
        median = np.percentile(pitches, 50)
        train_mask = pitches >= median
        test_mask = pitches <= p35

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return train_mask, test_mask


def split_data(windows: np.ndarray,
               labels: np.ndarray,
               seg_indices: np.ndarray,
               segments: List[LabeledSegment],
               strategy: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data according to strategy."""
    train_mask, test_mask = get_split_masks(segments, strategy)

    # Map segment masks to window masks
    train_seg_set = set(np.where(train_mask)[0])
    test_seg_set = set(np.where(test_mask)[0])

    train_window_mask = np.array([i in train_seg_set for i in seg_indices])
    test_window_mask = np.array([i in test_seg_set for i in seg_indices])

    X_train = windows[train_window_mask]
    y_train = labels[train_window_mask]
    X_test = windows[test_window_mask]
    y_test = labels[test_window_mask]

    return X_train, y_train, X_test, y_test


# ============================================================================
# Models
# ============================================================================

def build_v4_style_model(window_size: int, n_features: int) -> keras.Model:
    """V4-style baseline (no physics constraint)."""
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, n_features)),
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(5, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_v6_physics_model(window_size: int, n_features: int,
                           physics_weight: float = 0.01) -> keras.Model:
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

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Build by calling with dummy input
    dummy = tf.zeros((1, window_size, n_features))
    _ = model(dummy)

    return model


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_and_evaluate(model: keras.Model,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       epochs: int = 100,
                       model_name: str = "model") -> Dict[str, float]:
    """Train model and return metrics."""
    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Train
    history = model.fit(
        X_train_norm, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=0
    )

    # Compute accuracy from predictions (more reliable across model types)
    y_train_pred = model.predict(X_train_norm, verbose=0)
    y_test_pred = model.predict(X_test_norm, verbose=0)

    # Compute binary accuracy
    y_train_binary = (y_train_pred > 0.5).astype(int)
    y_test_binary = (y_test_pred > 0.5).astype(int)

    train_acc = np.mean(y_train_binary == y_train)
    test_acc = np.mean(y_test_binary == y_test)

    # Use test predictions for per-finger analysis
    y_pred = y_test_pred
    y_pred_binary = y_test_binary

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(finger_names):
        acc = np.mean(y_pred_binary[:, i] == y_test[:, i])
        per_finger[finger] = acc

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_test_gap': train_acc - test_acc,
        'per_finger': per_finger
    }


def run_experiment(strategy: str,
                   window_size: int = 8,
                   use_derivatives: bool = True,
                   epochs: int = 100) -> Dict[str, Any]:
    """Run full experiment with given split strategy."""
    print(f"\n{'='*60}")
    print(f"Split Strategy: {strategy}")
    print(f"{'='*60}")

    # Load data
    segments = load_labeled_data()
    print(f"Loaded {len(segments)} labeled segments")

    # Create windows
    n_features = 9 if use_derivatives else 3
    windows, labels, seg_indices = create_windows(
        segments, window_size=window_size, use_derivatives=use_derivatives
    )
    print(f"Created {len(windows)} windows, shape: {windows.shape}")

    # Split
    X_train, y_train, X_test, y_test = split_data(
        windows, labels, seg_indices, segments, strategy
    )
    print(f"Train: {len(X_train)} windows, Test: {len(X_test)} windows")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Empty train or test set!")
        return None

    results = {
        'strategy': strategy,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'models': {}
    }

    # V4-style baseline
    print("\n--- V4-style Baseline ---")
    v4_model = build_v4_style_model(window_size, n_features)
    v4_results = train_and_evaluate(
        v4_model, X_train, y_train, X_test, y_test,
        epochs=epochs, model_name="V4-style"
    )
    results['models']['v4_style'] = v4_results
    print(f"Train: {v4_results['train_acc']:.1%}, Test: {v4_results['test_acc']:.1%}")

    # V6 physics-constrained
    print("\n--- V6 Physics-Constrained ---")
    v6_model = build_v6_physics_model(window_size, n_features, physics_weight=0.01)
    v6_results = train_and_evaluate(
        v6_model, X_train, y_train, X_test, y_test,
        epochs=epochs, model_name="V6-physics"
    )
    results['models']['v6_physics'] = v6_results
    print(f"Train: {v6_results['train_acc']:.1%}, Test: {v6_results['test_acc']:.1%}")

    # V6 no physics (ablation)
    print("\n--- V6 No Physics (Ablation) ---")
    v6_nophys = build_v6_physics_model(window_size, n_features, physics_weight=0.0)
    v6_nophys_results = train_and_evaluate(
        v6_nophys, X_train, y_train, X_test, y_test,
        epochs=epochs, model_name="V6-no-physics"
    )
    results['models']['v6_no_physics'] = v6_nophys_results
    print(f"Train: {v6_nophys_results['train_acc']:.1%}, Test: {v6_nophys_results['test_acc']:.1%}")

    return results


def main():
    """Run all experiments."""
    all_results = {}

    for strategy in ["STRICT", "MODERATE", "MEDIAN"]:
        results = run_experiment(strategy)
        if results:
            all_results[strategy] = results

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Cross-Orientation Split Comparison")
    print("="*80)

    print(f"\n{'Strategy':<12} {'Model':<18} {'Train':<10} {'Test':<10} {'Gap':<10}")
    print("-"*60)

    for strategy in ["STRICT", "MODERATE", "MEDIAN"]:
        if strategy not in all_results:
            continue
        results = all_results[strategy]
        for model_name in ['v4_style', 'v6_physics', 'v6_no_physics']:
            m = results['models'][model_name]
            print(f"{strategy:<12} {model_name:<18} {m['train_acc']:.1%}     {m['test_acc']:.1%}     {m['train_test_gap']:.1%}")
        print()

    # Save results
    output_path = Path("ml/cross_orientation_split_results.json")

    # Convert numpy types to Python types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
