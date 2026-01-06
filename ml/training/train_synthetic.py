#!/usr/bin/env python3
"""
Synthetic Data Training Pipeline

Generates synthetic magnetic field training data and trains a finger tracking model.

Usage:
    python -m ml.train_synthetic --output-dir ml/synthetic_output --num-sessions 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

from ml.simulation import (
    MagneticFieldSimulator, DEFAULT_MAGNET_CONFIG,
    generate_synthetic_session
)
from ml.simulation.hand_model import POSE_TEMPLATES


def generate_synthetic_dataset(
    output_dir: Path,
    num_sessions: int = 100,
    samples_per_pose: int = 100,
    poses_per_session: int = 5,
    randomize: bool = True
) -> dict:
    """
    Generate a complete synthetic training dataset.

    Args:
        output_dir: Directory to save generated sessions
        num_sessions: Number of sessions to generate
        samples_per_pose: Samples per pose
        poses_per_session: Poses per session
        randomize: Apply domain randomization

    Returns:
        Summary dict with dataset statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_poses = list(POSE_TEMPLATES.keys())
    total_samples = 0
    generated_files = []

    print(f"Generating {num_sessions} synthetic sessions...")
    print(f"  Poses available: {all_poses}")
    print(f"  Samples per pose: {samples_per_pose}")

    for i in range(num_sessions):
        # Select poses for this session
        if poses_per_session >= len(all_poses):
            poses = all_poses.copy()
        else:
            poses = list(np.random.choice(all_poses, size=poses_per_session, replace=False))

        # Create simulator with randomization
        sim = MagneticFieldSimulator(
            magnet_config=DEFAULT_MAGNET_CONFIG,
            randomize_geometry=randomize,
            randomize_sensor=randomize
        )

        # Generate session
        session = sim.generate_session(
            poses=poses,
            samples_per_pose=samples_per_pose,
            include_transitions=True,
            transition_samples=20
        )

        # Save session
        filename = f"synthetic_{i:04d}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(session, f)

        generated_files.append(str(filepath))
        total_samples += len(session['samples'])

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_sessions} sessions ({total_samples} samples)")

    print(f"  Total: {num_sessions} sessions, {total_samples} samples")

    # Generate dataset stats file (needed by data loader)
    print("\nComputing dataset statistics...")
    all_data = []
    for filepath in generated_files:
        with open(filepath, 'r') as f:
            session = json.load(f)
        for sample in session['samples']:
            row = [
                sample['ax'], sample['ay'], sample['az'],
                sample['gx'], sample['gy'], sample['gz'],
                sample['mx_ut'], sample['my_ut'], sample['mz_ut']
            ]
            all_data.append(row)

    data = np.array(all_data, dtype=np.float32)
    stats = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0) + 1e-8,
        'min_val': np.min(data, axis=0),
        'max_val': np.max(data, axis=0)
    }

    stats_path = output_dir / 'dataset_stats.npz'
    np.savez(stats_path, **stats)
    print(f"  Saved stats to {stats_path}")

    return {
        'num_sessions': num_sessions,
        'total_samples': total_samples,
        'generated_files': generated_files,
        'output_dir': str(output_dir)
    }


def prepare_synthetic_data_for_training(
    data_dir: Path,
    window_size: int = 50,
    stride: int = 25,
    normalize: bool = True
) -> tuple:
    """
    Load synthetic data and prepare for training.

    Returns:
        (X_train, y_train, X_val, y_val) tuple for finger tracking
    """
    from .schema import NUM_FEATURES

    all_windows = []
    all_labels = []

    # Load stats for normalization
    stats_path = data_dir / 'dataset_stats.npz'
    if stats_path.exists():
        stats_data = np.load(stats_path)
        mean = stats_data['mean']
        std = stats_data['std']
    else:
        mean = None
        std = None
        normalize = False

    # Finger name to index mapping
    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}
    state_map = {'extended': 0, 'partial': 1, 'flexed': 2}

    # Load all sessions
    session_files = sorted(data_dir.glob('synthetic_*.json'))
    print(f"Loading {len(session_files)} synthetic sessions...")

    for filepath in session_files:
        with open(filepath, 'r') as f:
            session = json.load(f)

        samples = session['samples']
        labels = session['labels']

        # Extract feature array
        data = []
        for sample in samples:
            row = [
                sample['ax'], sample['ay'], sample['az'],
                sample['gx'], sample['gy'], sample['gz'],
                sample['mx_ut'], sample['my_ut'], sample['mz_ut']
            ]
            data.append(row)
        data = np.array(data, dtype=np.float32)

        # Normalize
        if normalize and mean is not None:
            data = (data - mean) / std

        # Create per-sample label array (5 fingers)
        sample_labels = np.full((len(samples), 5), -1, dtype=np.int32)

        for label in labels:
            start = label['start_sample']
            end = label['end_sample']
            finger_states = label['labels'].get('fingers', {})

            for finger, state in finger_states.items():
                if finger in finger_map and state in state_map:
                    sample_labels[start:end, finger_map[finger]] = state_map[state]

        # Create windows
        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            window_data = data[start:end]
            window_labels = sample_labels[start:end]

            # Check if all samples have valid labels
            if np.all(window_labels[0] >= 0) and np.all(window_labels == window_labels[0]):
                all_windows.append(window_data)
                all_labels.append(window_labels[0])

    if not all_windows:
        return None, None, None, None

    X = np.array(all_windows)
    y = np.array(all_labels)

    # Shuffle and split
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * 0.2)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  Training windows: {len(X_train)}")
    print(f"  Validation windows: {len(X_val)}")
    print(f"  Window shape: {X_train.shape}")

    return X_train, y_train, X_val, y_val


def train_on_synthetic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32
) -> dict:
    """
    Train a finger tracking model on synthetic data.
    """
    from .model import (
        create_finger_tracking_model_keras,
        train_finger_tracking_model_keras,
        evaluate_finger_tracking_model,
        save_model_for_inference,
        HAS_TF
    )

    if not HAS_TF:
        print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
        return {}

    print("\nCreating finger tracking model...")
    model = create_finger_tracking_model_keras(window_size=X_train.shape[1])
    model.summary()

    print(f"\nTraining for up to {epochs} epochs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_dir / 'best_finger_model.keras')

    history = train_finger_tracking_model_keras(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path
    )

    print("\nEvaluating on validation set...")
    metrics = evaluate_finger_tracking_model(model, X_val, y_val)

    print(f"  Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"\n  Per-finger accuracy:")
    for finger, acc in metrics['per_finger_accuracy'].items():
        print(f"    {finger}: {acc*100:.1f}%")

    # Save model
    print("\nSaving model...")
    saved = save_model_for_inference(model, str(output_dir), 'synthetic_finger_model')
    for fmt, path in saved.items():
        print(f"  {fmt}: {path}")

    return {
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'metrics': metrics,
        'saved_models': saved
    }


def main():
    parser = argparse.ArgumentParser(description='Train on synthetic magnetic field data')
    parser.add_argument('--output-dir', type=str, default='ml/synthetic_output',
                        help='Directory for generated data and models')
    parser.add_argument('--num-sessions', type=int, default=100,
                        help='Number of synthetic sessions to generate')
    parser.add_argument('--samples-per-pose', type=int, default=100,
                        help='Samples per pose')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip data generation (use existing)')
    parser.add_argument('--no-randomize', action='store_true',
                        help='Disable domain randomization')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / 'data'
    models_dir = output_dir / 'models'

    print("=" * 60)
    print("Synthetic Data Training Pipeline")
    print("=" * 60)

    # Step 1: Generate synthetic data
    if not args.skip_generation:
        print("\n[Step 1] Generating synthetic data...")
        gen_summary = generate_synthetic_dataset(
            output_dir=data_dir,
            num_sessions=args.num_sessions,
            samples_per_pose=args.samples_per_pose,
            randomize=not args.no_randomize
        )
        print(f"  Generated {gen_summary['total_samples']} samples")
    else:
        print("\n[Step 1] Skipping generation (using existing data)")

    # Step 2: Prepare data for training
    print("\n[Step 2] Preparing data for training...")
    X_train, y_train, X_val, y_val = prepare_synthetic_data_for_training(
        data_dir=data_dir,
        window_size=50,
        stride=25
    )

    if X_train is None or len(X_train) == 0:
        print("ERROR: No training data prepared!")
        sys.exit(1)

    # Step 3: Train model
    print("\n[Step 3] Training finger tracking model...")
    results = train_on_synthetic(
        X_train, y_train, X_val, y_val,
        output_dir=models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            **results
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
