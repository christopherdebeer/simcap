#!/usr/bin/env python3
"""
UniMTS-Inspired Orientation Invariance Experiments

Implements techniques from recent papers to improve cross-orientation accuracy:
1. SO(3) Rotation Augmentation (UniMTS, NeurIPS 2024)
2. SVD-Based Orientation-Invariant Transform (PMC5579846)
3. Physics-Based Augmentation (PPDA, arXiv:2508.13284)
4. Contrastive Learning with rotation pairs
5. Heuristic OIT features (9-dimensional)

Compares all approaches against V3 baseline.

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING (reused from deploy_finger_model_v3.py)
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    pitch_angles: np.ndarray


def load_session_with_pitch() -> Dict[str, FingerStateData]:
    """Load session data with pitch angles for cross-orientation testing."""
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    if not session_path.exists():
        session_path = Path('.worktrees/data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    with open(session_path) as f:
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
        pitch_data = []

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

            pitch = s.get('euler_pitch', 0)
            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            pitch_data.append(pitch)

        if not sensor_data:
            continue

        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        if '?' in combo:
            continue

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.array(sensor_data),
                pitch_angles=np.array(pitch_data)
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                pitch_angles=np.concatenate([existing.pitch_angles, pitch_data])
            )

    return combo_data


# ============================================================================
# AUGMENTATION TECHNIQUES
# ============================================================================

def so3_rotation_augmentation(samples: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    UniMTS-style SO(3) rotation augmentation.

    Samples a random rotation matrix uniformly from SO(3) and applies it
    to all samples. This is done during training to make the model
    orientation-invariant.

    Args:
        samples: (n, 3) magnetometer data [mx, my, mz]
        seed: Random seed for reproducibility

    Returns:
        Rotated samples with same shape
    """
    R = Rotation.random(random_state=seed)
    return samples @ R.as_matrix().T


def so3_augment_window(window: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply SO(3) rotation to an entire window consistently.

    UniMTS applies the same rotation to all timesteps in a window.

    Args:
        window: (window_size, 3) magnetometer window
        seed: Random seed

    Returns:
        Rotated window
    """
    R = Rotation.random(random_state=seed)
    return window @ R.as_matrix().T


def svd_orientation_invariant(window: np.ndarray) -> np.ndarray:
    """
    SVD-based orientation-invariant transformation.

    From PMC5579846: Principal axes rotate with data constellation,
    so representation in principal axis frame is orientation-invariant.

    Args:
        window: (window_size, 3) magnetometer window

    Returns:
        Transformed window in principal axis frame
    """
    # Center the data
    centered = window - window.mean(axis=0)

    # SVD to get principal axes
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Transform to principal axis frame
        return centered @ Vt.T
    except np.linalg.LinAlgError:
        # Fallback if SVD fails
        return centered


def heuristic_oit_features(window: np.ndarray) -> np.ndarray:
    """
    Heuristic Orientation-Invariant Transformation (9 elements).

    From PMC5579846:
    - (1-3) Norms of signal and its 1st/2nd order differences
    - (4-6) Angles between successive time samples
    - (7-9) Angles between rotation axes (cross products)

    Args:
        window: (window_size, 3) magnetometer window

    Returns:
        (window_size, 9) orientation-invariant features
    """
    n = len(window)
    if n < 3:
        # Return zeros if window too small
        return np.zeros((n, 9))

    # Norms
    norms = np.linalg.norm(window, axis=1, keepdims=True)

    # First difference
    diff1 = np.diff(window, axis=0)
    diff1_norms = np.linalg.norm(diff1, axis=1, keepdims=True)
    diff1_norms = np.vstack([diff1_norms, diff1_norms[-1:]])  # Pad to match length

    # Second difference
    diff2 = np.diff(diff1, axis=0)
    diff2_norms = np.linalg.norm(diff2, axis=1, keepdims=True)
    diff2_norms = np.vstack([diff2_norms, diff2_norms[-1:], diff2_norms[-1:]])  # Pad

    # Angles between successive samples
    angles = []
    for i in range(n):
        if i < n - 1:
            n1 = norms[i, 0]
            n2 = norms[i + 1, 0]
            if n1 > 1e-8 and n2 > 1e-8:
                cos_angle = np.dot(window[i], window[i + 1]) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angles.append(np.arccos(cos_angle))
            else:
                angles.append(0)
        else:
            angles.append(angles[-1] if angles else 0)
    angles = np.array(angles).reshape(-1, 1)

    # Angles between first differences
    diff_angles = []
    for i in range(n - 1):
        if i < len(diff1) - 1:
            n1 = diff1_norms[i, 0]
            n2 = diff1_norms[i + 1, 0]
            if n1 > 1e-8 and n2 > 1e-8:
                cos_angle = np.dot(diff1[i], diff1[i + 1]) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                diff_angles.append(np.arccos(cos_angle))
            else:
                diff_angles.append(0)
        else:
            diff_angles.append(diff_angles[-1] if diff_angles else 0)
    diff_angles.append(diff_angles[-1] if diff_angles else 0)
    diff_angles = np.array(diff_angles).reshape(-1, 1)

    # Cross product axis angles
    cross_angles = []
    for i in range(n - 2):
        cross1 = np.cross(window[i], window[i + 1])
        cross2 = np.cross(window[i + 1], window[i + 2])
        n1 = np.linalg.norm(cross1)
        n2 = np.linalg.norm(cross2)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_angle = np.dot(cross1, cross2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            cross_angles.append(np.arccos(cos_angle))
        else:
            cross_angles.append(0)
    # Pad to match length
    while len(cross_angles) < n:
        cross_angles.append(cross_angles[-1] if cross_angles else 0)
    cross_angles = np.array(cross_angles).reshape(-1, 1)

    # Combine all 9 features
    features = np.hstack([
        norms, diff1_norms, diff2_norms,
        angles, diff_angles, cross_angles,
        np.zeros((n, 3))  # Padding for additional features if needed
    ])

    return features[:, :9]


def physics_augmentation(
    samples: np.ndarray,
    seed: Optional[int] = None,
    bias_range: float = 1.0,
    rotation_range: float = np.pi / 7.2,  # ±25 degrees
    scale_range: Tuple[float, float] = (0.95, 1.05)
) -> np.ndarray:
    """
    PPDA-inspired physics-based augmentation.

    Models real-world physical variations:
    - Hardware bias: Per-axis magnetometer bias
    - Placement rotation: Sensor attachment variation
    - Calibration scale: Per-axis calibration error

    Args:
        samples: (n, 3) magnetometer data
        seed: Random seed
        bias_range: Max bias in μT
        rotation_range: Max rotation in radians
        scale_range: Min/max scale factors

    Returns:
        Augmented samples
    """
    rng = np.random.default_rng(seed)

    # Hardware bias (per-axis)
    bias = rng.uniform(-bias_range, bias_range, size=3)

    # Placement rotation (small rotation around each axis)
    angles = rng.uniform(-rotation_range, rotation_range, size=3)
    R = Rotation.from_euler('xyz', angles).as_matrix()

    # Calibration scale (per-axis)
    scale = rng.uniform(scale_range[0], scale_range[1], size=3)

    # Apply: rotate, scale, add bias
    augmented = (samples @ R.T) * scale + bias

    return augmented


# ============================================================================
# WINDOWING AND DATA PREPARATION
# ============================================================================

def create_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """Create sliding windows."""
    if stride is None:
        stride = max(1, window_size // 2)

    n_samples = len(samples)
    if n_samples < window_size:
        padding = np.zeros((window_size - n_samples, samples.shape[1]))
        samples = np.vstack([samples, padding])
        n_samples = window_size

    windows = []
    for i in range(0, n_samples - window_size + 1, stride):
        windows.append(samples[i:i + window_size])

    if not windows:
        windows.append(samples[:window_size])

    return np.array(windows)


def combo_to_label(combo: str) -> np.ndarray:
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


class SyntheticGenerator:
    """Generate synthetic samples with optional augmentation."""

    def __init__(self, real_data: Dict[str, FingerStateData]):
        self.real_data = real_data
        self.baseline = real_data.get('eeeee')
        self._compute_finger_effects()

    def _compute_finger_effects(self):
        self.finger_effects = {}
        if not self.baseline:
            return

        baseline_mag = self.baseline.samples[:, 6:9].mean(axis=0)
        single_finger_combos = {
            'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
            'ring': 'eeefe', 'pinky': 'eeeef'
        }

        for finger, combo in single_finger_combos.items():
            if combo in self.real_data:
                data = self.real_data[combo]
                self.finger_effects[finger] = {
                    'mag_delta': data.samples[:, 6:9].mean(axis=0) - baseline_mag,
                    'mag_std': data.samples[:, 6:9].std(axis=0),
                }
            else:
                self.finger_effects[finger] = {
                    'mag_delta': np.array([200, 200, 200]),
                    'mag_std': np.array([50, 50, 50]),
                }

    def generate_combo(self, combo: str, n_samples: int, augmentation: str = 'none') -> np.ndarray:
        """
        Generate synthetic samples with optional augmentation.

        Args:
            combo: Finger state combination (e.g., 'eefff')
            n_samples: Number of samples to generate
            augmentation: 'none', 'so3', 'physics', or 'both'
        """
        if combo in self.real_data:
            real = self.real_data[combo]
            mag_mean = real.samples[:, 6:9].mean(axis=0)
            mag_std = real.samples[:, 6:9].std(axis=0)
        else:
            if self.baseline:
                mag_mean = self.baseline.samples[:, 6:9].mean(axis=0)
                mag_std = self.baseline.samples[:, 6:9].std(axis=0)
            else:
                mag_mean = np.array([46, -46, 31])
                mag_std = np.array([25, 40, 50])

            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, state in enumerate(combo):
                if state == 'f':
                    finger = fingers[i]
                    if finger in self.finger_effects:
                        mag_mean = mag_mean + self.finger_effects[finger]['mag_delta']

        # Generate base samples (mag only)
        samples = []
        for i in range(n_samples):
            mag_sample = mag_mean + np.random.randn(3) * mag_std
            samples.append(mag_sample)

        samples = np.array(samples)

        # Apply augmentation
        if augmentation in ('so3', 'both'):
            samples = so3_rotation_augmentation(samples, seed=np.random.randint(0, 10000))

        if augmentation in ('physics', 'both'):
            samples = physics_augmentation(samples, seed=np.random.randint(0, 10000))

        return samples


# ============================================================================
# DATA PREPARATION WITH AUGMENTATION
# ============================================================================

def prepare_data_with_augmentation(
    real_data: Dict[str, FingerStateData],
    window_size: int,
    synthetic_ratio: float,
    augmentation_method: str = 'none',
    feature_transform: str = 'none',
    aug_multiplier: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data with various augmentation and feature transform methods.

    Args:
        real_data: Raw finger state data
        window_size: Size of sliding windows
        synthetic_ratio: Ratio of synthetic data (0.0 to 1.0)
        augmentation_method: 'none', 'so3', 'physics', 'both'
        feature_transform: 'none', 'svd', 'oit'
        aug_multiplier: Number of augmented copies per sample

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, info
    """
    # Calculate pitch quartiles for cross-orientation split
    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data) if synthetic_ratio > 0 else None

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)

        # Extract mag only features
        features = combo_data.samples[:, 6:9]

        # Split by pitch
        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = features[high_pitch_mask]
        low_pitch_samples = features[low_pitch_mask]

        # Add synthetic data to training
        if synthetic_ratio > 0 and generator:
            n_synth = int(300 * synthetic_ratio)
            synth_samples = generator.generate_combo(combo, n_synth, augmentation=augmentation_method)

            if len(high_pitch_samples) > 0:
                high_pitch_samples = np.vstack([high_pitch_samples, synth_samples])
            else:
                high_pitch_samples = synth_samples

        # Create training windows with augmentation
        if len(high_pitch_samples) >= window_size:
            windows = create_windows(high_pitch_samples, window_size)

            for w in windows:
                # Apply feature transform
                if feature_transform == 'svd':
                    w = svd_orientation_invariant(w)
                elif feature_transform == 'oit':
                    w = heuristic_oit_features(w)

                train_windows.append(w)
                train_labels.append(label)

                # Add augmented copies during training
                if augmentation_method in ('so3', 'both') and aug_multiplier > 1:
                    for _ in range(aug_multiplier - 1):
                        aug_w = so3_augment_window(w.copy())
                        if feature_transform == 'svd':
                            aug_w = svd_orientation_invariant(aug_w)
                        elif feature_transform == 'oit':
                            aug_w = heuristic_oit_features(aug_w)
                        train_windows.append(aug_w)
                        train_labels.append(label)

        # Test windows (no augmentation)
        if len(low_pitch_samples) >= window_size:
            windows = create_windows(low_pitch_samples, window_size)
            for w in windows:
                if feature_transform == 'svd':
                    w = svd_orientation_invariant(w)
                elif feature_transform == 'oit':
                    w = heuristic_oit_features(w)
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    # Compute normalization stats from training data
    n_features = X_train.shape[-1]
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8

    # Apply global z-score normalization
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Split train into train/val
    indices = np.random.permutation(len(X_train))
    val_size = int(0.15 * len(X_train))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    info = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': n_features,
        'window_size': window_size,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'q1_pitch': q1,
        'q3_pitch': q3,
        'augmentation': augmentation_method,
        'feature_transform': feature_transform,
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, info


# ============================================================================
# MODELS
# ============================================================================

def build_standard_model(window_size: int, n_features: int) -> keras.Model:
    """Standard CNN-LSTM model (same as V3)."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    if window_size <= 5:
        x = keras.layers.Conv1D(32, min(3, window_size), activation='relu', padding='same')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.LSTM(32)(x)

    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_contrastive_encoder(window_size: int, n_features: int, embedding_dim: int = 64) -> keras.Model:
    """Encoder for contrastive pre-training."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)

    # Projection head for contrastive learning
    x = keras.layers.Dense(64, activation='relu')(x)
    embeddings = keras.layers.Dense(embedding_dim, activation=None)(x)
    # L2 normalize embeddings
    embeddings = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embeddings)

    return keras.Model(inputs, embeddings)


def build_classifier_from_encoder(encoder: keras.Model, freeze_encoder: bool = False) -> keras.Model:
    """Build classifier on top of pre-trained encoder."""
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    inputs = encoder.input
    embeddings = encoder.output

    x = keras.layers.Dense(32, activation='relu')(embeddings)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# CONTRASTIVE LEARNING
# ============================================================================

class ContrastiveDataGenerator:
    """Generate positive/negative pairs for contrastive learning."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, batch_size: int = 32):
        self.windows = windows
        self.labels = labels
        self.batch_size = batch_size

        # Group windows by label
        self.label_to_indices = {}
        for i, label in enumerate(labels):
            key = tuple(label.astype(int))
            if key not in self.label_to_indices:
                self.label_to_indices[key] = []
            self.label_to_indices[key].append(i)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch with positive pairs (same class, different augmentations).

        Returns:
            anchor, positive, negative windows
        """
        indices = np.random.choice(len(self.windows), self.batch_size, replace=True)

        anchors = []
        positives = []

        for idx in indices:
            anchor = self.windows[idx]

            # Create positive by rotating the anchor (same gesture, different orientation)
            positive = so3_augment_window(anchor.copy())

            anchors.append(anchor)
            positives.append(positive)

        return np.array(anchors), np.array(positives)


def nt_xent_loss(z_i: tf.Tensor, z_j: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    From SimCLR: positive pairs are (i, i+batch_size) and (i+batch_size, i).
    """
    batch_size = tf.shape(z_i)[0]

    # Concatenate embeddings
    z = tf.concat([z_i, z_j], axis=0)  # (2*batch_size, embedding_dim)

    # Similarity matrix
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature  # (2*batch_size, 2*batch_size)

    # Mask out self-similarity
    mask = tf.eye(2 * batch_size) * 1e9
    sim_matrix = sim_matrix - mask

    # Labels: positive pairs are at offset batch_size
    labels_1 = tf.range(batch_size) + batch_size  # [batch_size, batch_size+1, ...]
    labels_2 = tf.range(batch_size)  # [0, 1, 2, ...]
    labels = tf.concat([labels_1, labels_2], axis=0)

    # Cross entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.int32),
        logits=sim_matrix
    )

    return tf.reduce_mean(loss)


def pretrain_contrastive(
    encoder: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    temperature: float = 0.5
) -> keras.Model:
    """Pre-train encoder with contrastive learning."""
    optimizer = keras.optimizers.Adam(0.001)

    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0

        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            anchors = X_shuffled[start:end]

            # Create positives via SO(3) rotation
            positives = np.array([so3_augment_window(w.copy()) for w in anchors])

            with tf.GradientTape() as tape:
                z_i = encoder(anchors, training=True)
                z_j = encoder(positives, training=True)
                loss = nt_xent_loss(z_i, z_j, temperature)

            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"  Contrastive epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}")

    return encoder


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_fn, epochs: int = 30
) -> Tuple[keras.Model, Dict]:
    """Train model and evaluate."""
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    model = model_fn(window_size, n_features)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)

    train_acc = np.mean((y_pred_train > 0.5).astype(int) == y_train)
    test_acc = np.mean((y_pred_test > 0.5).astype(int) == y_test)

    # Per-finger accuracy
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    y_pred_bin = (y_pred_test > 0.5).astype(int)
    for i, f in enumerate(fingers):
        per_finger[f] = float(np.mean(y_pred_bin[:, i] == y_test[:, i]))

    metrics = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'gap': float(train_acc - test_acc),
        'per_finger': per_finger,
    }

    return model, metrics


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiment(
    name: str,
    real_data: Dict[str, FingerStateData],
    window_size: int,
    synthetic_ratio: float,
    augmentation_method: str,
    feature_transform: str,
    aug_multiplier: int = 1,
    use_contrastive: bool = False,
    contrastive_epochs: int = 20
) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"  Window: {window_size}, Synth: {synthetic_ratio:.0%}, Aug: {augmentation_method}")
    print(f"  Transform: {feature_transform}, Aug mult: {aug_multiplier}, Contrastive: {use_contrastive}")

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data_with_augmentation(
        real_data,
        window_size=window_size,
        synthetic_ratio=synthetic_ratio,
        augmentation_method=augmentation_method,
        feature_transform=feature_transform,
        aug_multiplier=aug_multiplier
    )

    print(f"  Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
    print(f"  Features: {info['n_features']}")

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    if use_contrastive:
        # Pre-train encoder with contrastive learning
        print("  Pre-training with contrastive learning...")
        encoder = build_contrastive_encoder(window_size, n_features)
        encoder = pretrain_contrastive(
            encoder, X_train, y_train,
            epochs=contrastive_epochs
        )

        # Build classifier on top
        model = build_classifier_from_encoder(encoder, freeze_encoder=False)

        # Fine-tune classifier
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)

        train_acc = np.mean((y_pred_train > 0.5).astype(int) == y_train)
        test_acc = np.mean((y_pred_test > 0.5).astype(int) == y_test)

        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        per_finger = {}
        y_pred_bin = (y_pred_test > 0.5).astype(int)
        for i, f in enumerate(fingers):
            per_finger[f] = float(np.mean(y_pred_bin[:, i] == y_test[:, i]))

        metrics = {
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'gap': float(train_acc - test_acc),
            'per_finger': per_finger,
        }
    else:
        # Standard training
        _, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            build_standard_model
        )

    print(f"  Train: {metrics['train_acc']:.1%}, Test: {metrics['test_acc']:.1%}, Gap: {metrics['gap']:.1%}")

    tf.keras.backend.clear_session()

    return {
        'name': name,
        **metrics,
        **info,
        'use_contrastive': use_contrastive,
    }


def main():
    print("=" * 80)
    print("UNIMTS-INSPIRED ORIENTATION INVARIANCE EXPERIMENTS")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    results = []

    # =========================================================================
    # EXPERIMENT 1: V3 Baseline (for comparison)
    # =========================================================================
    results.append(run_experiment(
        name="V3_baseline",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='none',
        feature_transform='none'
    ))

    # =========================================================================
    # EXPERIMENT 2: SO(3) Rotation Augmentation (UniMTS-style)
    # =========================================================================
    results.append(run_experiment(
        name="SO3_aug_x1",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='so3',
        feature_transform='none',
        aug_multiplier=1
    ))

    results.append(run_experiment(
        name="SO3_aug_x3",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='so3',
        feature_transform='none',
        aug_multiplier=3
    ))

    results.append(run_experiment(
        name="SO3_aug_x5",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='so3',
        feature_transform='none',
        aug_multiplier=5
    ))

    # =========================================================================
    # EXPERIMENT 3: SVD-Based Orientation-Invariant Transform
    # =========================================================================
    results.append(run_experiment(
        name="SVD_transform",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='none',
        feature_transform='svd'
    ))

    results.append(run_experiment(
        name="SVD_with_SO3_aug",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='so3',
        feature_transform='svd',
        aug_multiplier=3
    ))

    # =========================================================================
    # EXPERIMENT 4: Heuristic OIT Features
    # =========================================================================
    results.append(run_experiment(
        name="OIT_features",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='none',
        feature_transform='oit'
    ))

    # =========================================================================
    # EXPERIMENT 5: Physics-Based Augmentation (PPDA-style)
    # =========================================================================
    results.append(run_experiment(
        name="Physics_aug",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='physics',
        feature_transform='none'
    ))

    results.append(run_experiment(
        name="Physics_and_SO3",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='both',
        feature_transform='none',
        aug_multiplier=2
    ))

    # =========================================================================
    # EXPERIMENT 6: Contrastive Pre-training (UniMTS-style)
    # =========================================================================
    results.append(run_experiment(
        name="Contrastive_pretrain",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='none',
        feature_transform='none',
        use_contrastive=True,
        contrastive_epochs=20
    ))

    results.append(run_experiment(
        name="Contrastive_with_SO3",
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5,
        augmentation_method='so3',
        feature_transform='none',
        aug_multiplier=3,
        use_contrastive=True,
        contrastive_epochs=20
    ))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (Cross-Orientation Test)")
    print("=" * 80)

    print(f"\n{'Experiment':<25} {'Train':>8} {'Test':>8} {'Gap':>8} {'vs V3':>8}")
    print("-" * 60)

    baseline_test = results[0]['test_acc']

    # Sort by test accuracy
    sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

    for r in sorted_results:
        delta = r['test_acc'] - baseline_test
        delta_str = f"{delta:+.1%}" if delta != 0 else "baseline"
        print(f"{r['name']:<25} {r['train_acc']:>7.1%} {r['test_acc']:>7.1%} {r['gap']:>7.1%} {delta_str:>8}")

    # Best result
    best = sorted_results[0]
    print(f"\n*** Best: {best['name']} with {best['test_acc']:.1%} cross-orientation accuracy ***")

    if best['test_acc'] > baseline_test:
        print(f"*** Improvement over V3 baseline: {(best['test_acc'] - baseline_test)*100:.1f}% ***")

    # Per-finger accuracy for best model
    print(f"\nPer-finger accuracy ({best['name']}):")
    for finger, acc in best['per_finger'].items():
        print(f"  {finger}: {acc:.1%}")

    # Save results
    results_path = Path("ml/unimts_experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Compare augmentation methods
    aug_methods = {
        'none': [r for r in results if r['augmentation'] == 'none' and r['feature_transform'] == 'none' and not r.get('use_contrastive')],
        'so3': [r for r in results if 'SO3' in r['name'] and not r.get('use_contrastive')],
        'physics': [r for r in results if r['augmentation'] == 'physics'],
        'svd': [r for r in results if r['feature_transform'] == 'svd'],
        'oit': [r for r in results if r['feature_transform'] == 'oit'],
        'contrastive': [r for r in results if r.get('use_contrastive')],
    }

    print("\nMethod effectiveness (best test accuracy per category):")
    for method, method_results in aug_methods.items():
        if method_results:
            best_in_category = max(method_results, key=lambda x: x['test_acc'])
            delta = best_in_category['test_acc'] - baseline_test
            print(f"  {method:<15}: {best_in_category['test_acc']:.1%} ({delta:+.1%} vs baseline)")

    return sorted_results


if __name__ == '__main__':
    main()
