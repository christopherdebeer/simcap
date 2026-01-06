"""
Export Finger Tracking Model to TensorFlow.js Format

Trains a contrastive pre-trained model on synthetic data and exports it
for client-side inference in the GAMBIT web app.

Usage:
    python -m ml.export_tfjs_model --output apps/gambit/models/finger_contrastive_v1
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass, asdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class NormalizationStats:
    """Normalization statistics for model input."""
    mean: list
    std: list


def build_encoder(input_dim: int = 6, latent_dim: int = 32) -> keras.Model:
    """Build feature encoder network."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(48, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(latent_dim, activation='relu')(x)
    return keras.Model(inputs, x, name='encoder')


def build_classifier(encoder: keras.Model, n_fingers: int = 5, n_states: int = 3) -> keras.Model:
    """
    Build multi-output classifier for finger state prediction.

    Returns a model with 5 output heads (one per finger), each outputting
    probabilities for 3 states (extended, partial, flexed).
    """
    inputs = keras.Input(shape=(encoder.input_shape[1],))
    features = encoder(inputs)

    outputs = []
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for i, finger in enumerate(finger_names):
        x = layers.Dense(16, activation='relu', name=f'{finger}_hidden')(features)
        out = layers.Dense(n_states, activation='softmax', name=f'{finger}_output')(x)
        outputs.append(out)

    return keras.Model(inputs, outputs, name='finger_classifier')


def build_contrastive_model(encoder: keras.Model, projection_dim: int = 16) -> keras.Model:
    """Build contrastive learning model with projection head."""
    inputs = keras.Input(shape=(encoder.input_shape[1],))
    features = encoder(inputs)
    x = layers.Dense(projection_dim, activation='relu')(features)
    projections = layers.Dense(projection_dim)(x)  # No activation - raw for contrastive loss
    return keras.Model(inputs, projections, name='contrastive_model')


def load_real_data(data_dir: str = 'data/GAMBIT') -> Tuple[np.ndarray, NormalizationStats]:
    """Load real sensor data for contrastive pre-training."""
    data_path = Path(data_dir)
    all_samples = []

    for f in sorted(data_path.glob('*.json')):
        if f.name in ['manifest.json']:
            continue
        try:
            with open(f) as fp:
                content = fp.read()
                if not content.strip():
                    continue
                data = json.loads(content)

            for sample in data.get('samples', []):
                mx = sample.get('mx_ut', 0)
                my = sample.get('my_ut', 0)
                mz = sample.get('mz_ut', 0)
                ax = sample.get('ax_g', 0)
                ay = sample.get('ay_g', 0)
                az = sample.get('az_g', 0)

                if mx and my and mz:
                    all_samples.append([mx, my, mz, ax, ay, az])
        except Exception:
            continue

    samples = np.array(all_samples, dtype=np.float32)

    stats = NormalizationStats(
        mean=np.mean(samples, axis=0).tolist(),
        std=np.std(samples, axis=0).tolist()
    )

    return samples, stats


def generate_synthetic_data(n_sessions: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate labeled synthetic data for fine-tuning.

    Returns:
        (samples, labels) where labels is (N, 5) array of states per finger
    """
    from ml.simulation import MagneticFieldSimulator, DEFAULT_MAGNET_CONFIG
    from ml.simulation.hand_model import POSE_TEMPLATES, FingerState

    all_samples = []
    all_labels = []
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    state_map = {FingerState.EXTENDED: 0, FingerState.PARTIAL: 1, FingerState.FLEXED: 2}

    poses = list(POSE_TEMPLATES.keys())
    samples_per_pose = 500 // len(poses)

    print(f"Generating synthetic data: {n_sessions} sessions...")

    for i in range(n_sessions):
        if (i + 1) % 10 == 0:
            print(f"  Session {i + 1}/{n_sessions}")

        sim = MagneticFieldSimulator(
            magnet_config=DEFAULT_MAGNET_CONFIG,
            use_magpylib=True,
            randomize_geometry=True,
            randomize_sensor=True
        )

        session = sim.generate_session(
            poses=poses,
            samples_per_pose=samples_per_pose,
            include_transitions=True
        )

        # Build sample-to-label mapping
        sample_labels = {}
        for label in session.get('labels', []):
            start_idx = label['start_sample']
            end_idx = label['end_sample']
            fingers = label.get('labels', {}).get('fingers', {})

            label_vec = []
            for finger in finger_names:
                state_str = fingers.get(finger, 'unknown')
                if state_str == 'unknown':
                    state = 1
                else:
                    state = state_map.get(FingerState(state_str), 1)
                label_vec.append(state)

            for idx in range(start_idx, end_idx):
                sample_labels[idx] = label_vec

        # Extract samples with labels
        for idx, sample in enumerate(session['samples']):
            if idx in sample_labels:
                features = [
                    sample['mx_ut'], sample['my_ut'], sample['mz_ut'],
                    sample.get('ax_g', 0), sample.get('ay_g', 0), sample.get('az_g', 0)
                ]
                all_samples.append(features)
                all_labels.append(sample_labels[idx])

    return np.array(all_samples, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def contrastive_pretrain(
    encoder: keras.Model,
    real_samples: np.ndarray,
    stats: NormalizationStats,
    epochs: int = 50,
    batch_size: int = 256,
    temperature: float = 0.1
) -> None:
    """
    Pre-train encoder using contrastive learning on real data.

    Uses NT-Xent loss to learn representations where temporally close
    samples are similar and distant samples are different.
    """
    print(f"\nContrastive pre-training on {len(real_samples)} real samples...")

    # Normalize data
    mean = np.array(stats.mean)
    std = np.array(stats.std)
    normalized = (real_samples - mean) / (std + 1e-8)

    contrastive_model = build_contrastive_model(encoder, projection_dim=16)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    n_samples = len(normalized)

    for epoch in range(epochs):
        # Sample batch
        idx = np.random.choice(n_samples, batch_size, replace=False)
        batch = normalized[idx]

        # Create positive pairs (nearby samples)
        pos_offsets = np.random.randint(-5, 6, size=batch_size)
        pos_idx = np.clip(idx + pos_offsets, 0, n_samples - 1)
        pos_batch = normalized[pos_idx]

        with tf.GradientTape() as tape:
            # Get projections
            z_a = contrastive_model(batch, training=True)
            z_b = contrastive_model(pos_batch, training=True)

            # Normalize
            z_a = tf.nn.l2_normalize(z_a, axis=1)
            z_b = tf.nn.l2_normalize(z_b, axis=1)

            # NT-Xent loss
            sim_matrix = tf.matmul(z_a, z_b, transpose_b=True) / temperature
            labels = tf.range(batch_size)
            loss_a = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels, sim_matrix)
            )
            loss_b = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels, tf.transpose(sim_matrix))
            )
            loss = (loss_a + loss_b) / 2

        grads = tape.gradient(loss, contrastive_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, contrastive_model.trainable_variables))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")

    print("  Pre-training complete!")


def finetune_classifier(
    encoder: keras.Model,
    syn_samples: np.ndarray,
    syn_labels: np.ndarray,
    stats: NormalizationStats,
    epochs: int = 30,
    freeze_encoder: bool = False
) -> keras.Model:
    """
    Fine-tune classifier on synthetic labeled data.

    Returns the trained classifier model.
    """
    print(f"\nFine-tuning classifier on {len(syn_samples)} synthetic samples...")

    if freeze_encoder:
        encoder.trainable = False
        print("  Encoder frozen - training classifier heads only")

    classifier = build_classifier(encoder)

    # Normalize data
    mean = np.array(stats.mean)
    std = np.array(stats.std)
    normalized = (syn_samples - mean) / (std + 1e-8)

    # Prepare labels for multi-output (list of arrays)
    labels_list = [syn_labels[:, i] for i in range(5)]

    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics={f'{finger}_output': 'accuracy' for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']}
    )

    classifier.fit(
        normalized,
        labels_list,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return classifier


def export_to_tfjs(
    model: keras.Model,
    stats: NormalizationStats,
    output_dir: str
) -> None:
    """
    Export model to TensorFlow.js format.

    Creates:
        - model.json (model architecture)
        - group1-shard*.bin (model weights)
        - config.json (normalization stats and metadata)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting model to {output_dir}...")

    # Save as Keras format first
    keras_path = output_path / 'model.keras'
    model.save(keras_path)
    print(f"  Keras model saved: {keras_path}")

    # Convert to TensorFlow.js using subprocess
    import subprocess
    result = subprocess.run([
        'tensorflowjs_converter',
        '--input_format=keras',
        str(keras_path),
        str(output_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Warning: TensorFlow.js conversion failed: {result.stderr}")
        # Try exporting as SavedModel and converting
        saved_model_path = output_path / 'saved_model'
        model.export(saved_model_path)
        print(f"  Exported SavedModel: {saved_model_path}")

        result = subprocess.run([
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            '--output_format=tfjs_graph_model',
            str(saved_model_path),
            str(output_path)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Warning: Graph model conversion also failed")
            print(f"  Keras model available at: {keras_path}")
        else:
            print(f"  Model exported as graph model")
    else:
        print(f"  Model exported: {output_path / 'model.json'}")

    # Export config
    config = {
        'stats': asdict(stats),
        'inputShape': [None, 6],
        'fingerNames': ['thumb', 'index', 'middle', 'ring', 'pinky'],
        'stateNames': ['extended', 'partial', 'flexed'],
        'description': 'Contrastive pre-trained finger tracking model',
        'version': 'contrastive_v1',
        'date': str(np.datetime64('today'))
    }

    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config exported: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Export finger tracking model to TensorFlow.js')
    parser.add_argument('--output', type=str, default='apps/gambit/models/finger_contrastive_v1',
                        help='Output directory for TensorFlow.js model')
    parser.add_argument('--contrastive-epochs', type=int, default=50,
                        help='Number of contrastive pre-training epochs')
    parser.add_argument('--finetune-epochs', type=int, default=30,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--synthetic-sessions', type=int, default=30,
                        help='Number of synthetic sessions to generate')
    parser.add_argument('--skip-contrastive', action='store_true',
                        help='Skip contrastive pre-training (train from scratch)')
    args = parser.parse_args()

    print("="*60)
    print("Finger Tracking Model Export")
    print("="*60)

    # Load real data for pre-training and stats
    print("\nLoading real data...")
    real_samples, stats = load_real_data()
    print(f"  Loaded {len(real_samples)} real samples")
    print(f"  Stats: mean={[f'{m:.1f}' for m in stats.mean]}")
    print(f"         std={[f'{s:.1f}' for s in stats.std]}")

    # Generate synthetic data for fine-tuning
    syn_samples, syn_labels = generate_synthetic_data(n_sessions=args.synthetic_sessions)
    print(f"\nGenerated {len(syn_samples)} synthetic samples")

    # Build encoder
    encoder = build_encoder(input_dim=6, latent_dim=32)

    # Contrastive pre-training
    if not args.skip_contrastive:
        contrastive_pretrain(
            encoder, real_samples, stats,
            epochs=args.contrastive_epochs
        )
    else:
        print("\nSkipping contrastive pre-training")

    # Fine-tune classifier
    classifier = finetune_classifier(
        encoder, syn_samples, syn_labels, stats,
        epochs=args.finetune_epochs,
        freeze_encoder=not args.skip_contrastive
    )

    # Export
    export_to_tfjs(classifier, stats, args.output)

    print("\n" + "="*60)
    print("Export complete!")
    print(f"Model saved to: {args.output}")
    print("\nTo use in GAMBIT app:")
    print("  1. Add model to FINGER_MODELS in gesture-inference.ts")
    print("  2. Use createFingerTrackingInference('contrastive_v1')")
    print("="*60)


if __name__ == '__main__':
    main()
