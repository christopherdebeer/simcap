#!/usr/bin/env python3
"""
Inverse Magnetometry as Learned Field - Temporal Learning Experiment

The core insight: Five dipoles superpose into a single 3-vector at the sensor.
The forward problem (positions → reading) is closed-form but the inverse is
massively underdetermined—15 DOF from 3 measurements.

What breaks the degeneracy:
1. TEMPORAL STRUCTURE - The trajectory through field-space is far more informative
   than any single reading. The manifold of plausible configurations is much smaller.
2. LEARNED PRIORS ON MOTION - A hand is a kinematic chain with strong covariance
   structure. The network learns the submanifold magnets actually traverse.

Architecture: Temporal model operating on windowed magnetometer sequences.
The window gives implicit derivatives—rate of field change carries position
information that instantaneous readings don't.

Training signals:
- Direct position regression
- Physics consistency: predicted positions → simulated field → match input
- Smoothness prior: penalize jerk in predicted trajectories
- Contrastive structure: similar readings distinguishable by temporal context

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from scipy.spatial.transform import Rotation


# =============================================================================
# PHYSICAL CONSTANTS AND DIPOLE MODEL
# =============================================================================

MU_0_OVER_4PI = 1e-7  # T·m/A (μ₀/4π)

# Earth field approximation (Edinburgh, UK)
EARTH_FIELD = np.array([16.0, 0.0, 47.8])  # μT

# Finger names in canonical order
FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']


def dipole_field(r_vec: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """
    Compute magnetic dipole field.

    B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³

    Args:
        r_vec: Position vector from dipole to observation point (meters)
        moment: Magnetic dipole moment (A·m²)

    Returns:
        Magnetic field in Tesla
    """
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-6:
        return np.zeros(3)

    r_hat = r_vec / r_mag
    m_dot_r = np.dot(moment, r_hat)
    B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - moment) / (r_mag ** 3)
    return B


def compute_total_field_batch(
    positions: np.ndarray,  # [batch, 5, 3] finger positions in mm
    moments: np.ndarray,    # [5, 3] dipole moments
    sensor_pos: np.ndarray = None,  # Sensor position (default origin)
    include_earth: bool = False
) -> np.ndarray:
    """
    Compute total magnetic field at sensor from all finger magnets.

    Args:
        positions: Finger positions [batch, 5, 3] in millimeters
        moments: Dipole moments [5, 3] in A·m²
        sensor_pos: Sensor position in mm (default origin)
        include_earth: Include Earth's field

    Returns:
        Total field [batch, 3] in μT
    """
    if sensor_pos is None:
        sensor_pos = np.zeros(3)

    batch_size = positions.shape[0]
    total_field = np.zeros((batch_size, 3))

    # Convert positions from mm to meters
    positions_m = positions / 1000.0
    sensor_pos_m = sensor_pos / 1000.0

    for b in range(batch_size):
        B = np.zeros(3)
        for f in range(5):
            r_vec = sensor_pos_m - positions_m[b, f]
            B_dipole = dipole_field(r_vec, moments[f])
            B += B_dipole * 1e6  # Convert T to μT

        if include_earth:
            B += EARTH_FIELD

        total_field[b] = B

    return total_field


# Vectorized version using TensorFlow for training
@tf.function
def compute_total_field_tf(
    positions: tf.Tensor,  # [batch, 5, 3] in mm
    moments: tf.Tensor,    # [5, 3] dipole moments
) -> tf.Tensor:
    """TensorFlow version of field computation for differentiable training."""
    # Convert to meters
    positions_m = positions / 1000.0

    # Position vectors from magnets to sensor (at origin)
    r_vecs = -positions_m  # [batch, 5, 3]
    r_mags = tf.norm(r_vecs, axis=-1, keepdims=True)  # [batch, 5, 1]
    r_mags = tf.maximum(r_mags, 1e-6)  # Avoid singularity
    r_hats = r_vecs / r_mags  # [batch, 5, 3]

    # Dipole field for each magnet
    # m·r̂ for each sample and magnet
    m_dot_r = tf.reduce_sum(r_hats * moments[None, :, :], axis=-1, keepdims=True)

    # B = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³
    B_magnets = MU_0_OVER_4PI * (3 * m_dot_r * r_hats - moments[None, :, :]) / (r_mags ** 3)
    B_magnets = B_magnets * 1e6  # T to μT

    # Sum over all magnets
    B_total = tf.reduce_sum(B_magnets, axis=1)  # [batch, 3]

    return B_total


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class LabeledSegment:
    """A labeled segment from session data."""
    samples: np.ndarray      # [n, features] sensor data
    finger_states: str       # e.g., "eefff" (extended/flexed)
    pitch_angles: np.ndarray # Euler pitch for each sample
    timestamps: np.ndarray   # Time series

    @property
    def finger_binary(self) -> np.ndarray:
        """Convert string states to binary [5] (0=extended, 1=flexed)."""
        return np.array([0 if c == 'e' else 1 for c in self.finger_states])


def load_labeled_sessions(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all sessions with labels."""
    sessions = []

    for path in sorted(data_dir.glob('*.json')):
        try:
            with open(path) as f:
                data = json.load(f)

            if 'labels' in data and data['labels']:
                sessions.append({
                    'path': path,
                    'samples': data.get('samples', []),
                    'labels': data.get('labels', []),
                    'version': data.get('version', '1.0')
                })
        except Exception:
            continue

    return sessions


def extract_segments(sessions: List[Dict]) -> List[LabeledSegment]:
    """Extract labeled segments with magnetometer data."""
    segments = []

    for session in sessions:
        samples = session['samples']
        labels = session['labels']

        for label in labels:
            # Handle both v1 and v2 label formats
            if 'labels' in label and isinstance(label['labels'], dict):
                fingers = label['labels'].get('fingers', {})
                start = label.get('start_sample', 0)
                end = label.get('end_sample', 0)
            else:
                fingers = label.get('fingers', {})
                start = label.get('startIndex', 0)
                end = label.get('endIndex', 0)

            if not fingers or end <= start:
                continue

            # Build finger state string
            combo = ''
            valid = True
            for f in FINGER_ORDER:
                state = fingers.get(f, 'unknown')
                if state == 'extended':
                    combo += 'e'
                elif state == 'flexed':
                    combo += 'f'
                else:
                    valid = False
                    break

            if not valid:
                continue

            # Extract sensor data
            segment_samples = samples[start:min(end, len(samples))]
            if len(segment_samples) < 5:
                continue

            sensor_data = []
            pitch_data = []
            time_data = []

            for s in segment_samples:
                # Magnetometer (prefer calibrated)
                if 'iron_mx' in s:
                    mx, my, mz = s['iron_mx'], s['iron_my'], s['iron_mz']
                elif 'mx_ut' in s:
                    mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
                else:
                    mx = s.get('mx', 0) / 10.24
                    my = s.get('my', 0) / 10.24
                    mz = s.get('mz', 0) / 10.24

                # Also extract accelerometer/gyro for full 9-DOF
                ax = s.get('ax', 0) / 8192.0
                ay = s.get('ay', 0) / 8192.0
                az = s.get('az', 0) / 8192.0
                gx = s.get('gx', 0) / 114.28
                gy = s.get('gy', 0) / 114.28
                gz = s.get('gz', 0) / 114.28

                sensor_data.append([mx, my, mz, ax, ay, az, gx, gy, gz])
                pitch_data.append(s.get('euler_pitch', 0))
                time_data.append(s.get('t', 0))

            segments.append(LabeledSegment(
                samples=np.array(sensor_data),
                finger_states=combo,
                pitch_angles=np.array(pitch_data),
                timestamps=np.array(time_data)
            ))

    return segments


# =============================================================================
# WINDOWING AND TEMPORAL FEATURES
# =============================================================================

def create_temporal_windows(
    segments: List[LabeledSegment],
    window_size: int = 16,
    stride: int = 4,
    mag_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed data for temporal learning.

    Key insight: Windows capture implicit derivatives (rate of change).
    This is how temporal structure breaks the inverse problem degeneracy.

    Args:
        segments: List of labeled segments
        window_size: Number of samples per window
        stride: Stride between windows
        mag_only: Use only magnetometer (3-ch) or full 9-DOF

    Returns:
        X: [N, window_size, channels] windowed data
        y: [N, 5] binary finger states
        pitch: [N] mean pitch angle per window
    """
    windows = []
    labels = []
    pitches = []

    for seg in segments:
        n_samples = len(seg.samples)
        if n_samples < window_size:
            continue

        # Slice features
        features = seg.samples[:, :3] if mag_only else seg.samples

        # Sliding windows
        for i in range(0, n_samples - window_size + 1, stride):
            window = features[i:i + window_size]
            windows.append(window)
            labels.append(seg.finger_binary)
            pitches.append(np.mean(seg.pitch_angles[i:i + window_size]))

    return np.array(windows), np.array(labels), np.array(pitches)


def add_temporal_derivatives(windows: np.ndarray) -> np.ndarray:
    """
    Augment windows with explicit temporal derivatives.

    The raw magnetometer window implicitly contains derivative information,
    but making it explicit can help the network.

    Args:
        windows: [N, T, C] raw windows

    Returns:
        [N, T, C*3] windows with value, velocity, acceleration
    """
    # First derivative (velocity)
    velocity = np.diff(windows, axis=1, prepend=windows[:, :1, :])

    # Second derivative (acceleration)
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])

    return np.concatenate([windows, velocity, acceleration], axis=-1)


# =============================================================================
# MODELS
# =============================================================================

def build_temporal_transformer(
    window_size: int,
    n_features: int,
    n_outputs: int = 5,
    d_model: int = 64,
    n_heads: int = 4,
    ff_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1
) -> keras.Model:
    """
    Transformer model for temporal magnetometer sequences.

    Self-attention allows the model to weight which timesteps within
    the window are most informative for the current prediction.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # Project to model dimension
    x = keras.layers.Dense(d_model)(inputs)

    # Positional encoding (learned)
    positions = keras.layers.Embedding(window_size, d_model)(
        tf.range(window_size)
    )
    x = x + positions

    # Transformer blocks
    for _ in range(n_layers):
        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization()(x + attn_output)

        # Feed-forward
        ff_output = keras.layers.Dense(ff_dim, activation='relu')(x)
        ff_output = keras.layers.Dense(d_model)(ff_output)
        ff_output = keras.layers.Dropout(dropout)(ff_output)
        x = keras.layers.LayerNormalization()(x + ff_output)

    # Pool across time dimension
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Output heads
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(n_outputs, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model


def build_state_space_model(
    window_size: int,
    n_features: int,
    n_outputs: int = 5,
    hidden_dim: int = 64,
    n_layers: int = 2
) -> keras.Model:
    """
    State-space inspired model using bidirectional LSTM.

    State-space models excel at capturing temporal dynamics,
    which is key for inverse magnetometry where derivative
    information disambiguates configurations.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = inputs
    for i in range(n_layers):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                hidden_dim,
                return_sequences=(i < n_layers - 1)
            )
        )(x)
        if i < n_layers - 1:
            x = keras.layers.LayerNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(n_outputs, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model


def build_temporal_cnn(
    window_size: int,
    n_features: int,
    n_outputs: int = 5
) -> keras.Model:
    """
    Temporal CNN with dilated convolutions.

    Dilated convolutions efficiently capture multi-scale temporal patterns
    without the sequential dependency of RNNs.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # Multi-scale temporal convolutions
    x = inputs
    filters = [32, 64, 64]
    dilations = [1, 2, 4]

    for f, d in zip(filters, dilations):
        x = keras.layers.Conv1D(
            f, kernel_size=3, dilation_rate=d,
            padding='causal', activation='relu'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_outputs, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model


# =============================================================================
# PHYSICS-CONSISTENT LOSS FUNCTIONS
# =============================================================================

def physics_consistency_loss(
    y_pred_positions: tf.Tensor,  # Predicted positions [batch, 5, 3]
    x_observed: tf.Tensor,        # Observed magnetometer [batch, T, 3]
    dipole_moments: tf.Tensor,    # Magnet moments [5, 3]
    weight: float = 1.0
) -> tf.Tensor:
    """
    Physics consistency loss: predicted positions should produce
    fields that match observations.

    This is the key insight - use the known forward model as a constraint.
    """
    # Predict field from positions
    B_predicted = compute_total_field_tf(y_pred_positions, dipole_moments)

    # Use mean of observed window as target
    B_observed = tf.reduce_mean(x_observed, axis=1)  # [batch, 3]

    # MSE between predicted and observed fields
    loss = tf.reduce_mean(tf.square(B_predicted - B_observed))

    return weight * loss


def smoothness_loss(
    positions_sequence: tf.Tensor,  # [batch, T, 5, 3] position trajectory
    weight: float = 0.1
) -> tf.Tensor:
    """
    Smoothness prior: penalize jerk (third derivative) in trajectories.

    Real finger movements are smooth - this prior regularizes
    the learned inverse towards physically plausible solutions.
    """
    # First derivative (velocity)
    velocity = positions_sequence[:, 1:] - positions_sequence[:, :-1]

    # Second derivative (acceleration)
    acceleration = velocity[:, 1:] - velocity[:, :-1]

    # Third derivative (jerk)
    jerk = acceleration[:, 1:] - acceleration[:, :-1]

    # Penalize jerk magnitude
    loss = tf.reduce_mean(tf.square(jerk))

    return weight * loss


class ContrastiveTemporalLoss(keras.losses.Loss):
    """
    Contrastive loss for temporal disambiguation.

    Key insight: Similar instantaneous readings from different configurations
    should be distinguishable by their temporal context.
    """

    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, z_anchor: tf.Tensor, z_positive: tf.Tensor) -> tf.Tensor:
        """NT-Xent loss for contrastive learning."""
        batch_size = tf.shape(z_anchor)[0]

        # L2 normalize embeddings
        z_anchor = tf.math.l2_normalize(z_anchor, axis=-1)
        z_positive = tf.math.l2_normalize(z_positive, axis=-1)

        # Concatenate
        z = tf.concat([z_anchor, z_positive], axis=0)

        # Similarity matrix
        sim_matrix = tf.matmul(z, z, transpose_b=True) / self.temperature

        # Mask self-similarity
        mask = tf.eye(2 * batch_size) * 1e9
        sim_matrix = sim_matrix - mask

        # Labels: positive pairs are at offset batch_size
        labels = tf.concat([
            tf.range(batch_size) + batch_size,
            tf.range(batch_size)
        ], axis=0)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.int32),
            logits=sim_matrix
        )

        return tf.reduce_mean(loss)


# =============================================================================
# FULL INVERSE MAGNETOMETRY MODEL
# =============================================================================

class InverseMagnetometryModel(keras.Model):
    """
    Full model for learned inverse magnetometry.

    Combines:
    1. Temporal encoder (transformer/LSTM/CNN)
    2. Position decoder
    3. Physics-consistent training
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        encoder_type: str = 'transformer',  # 'transformer', 'lstm', 'cnn'
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.n_features = n_features

        # Build encoder
        if encoder_type == 'transformer':
            self.encoder = self._build_transformer_encoder(hidden_dim)
        elif encoder_type == 'lstm':
            self.encoder = self._build_lstm_encoder(hidden_dim)
        else:
            self.encoder = self._build_cnn_encoder(hidden_dim)

        # Classification head (finger states)
        self.classifier = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='sigmoid')
        ])

        # Position regression head (optional, for physics consistency)
        self.position_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(15)  # 5 fingers × 3 coordinates
        ])

    def _build_transformer_encoder(self, hidden_dim: int):
        """Build transformer encoder."""
        return keras.Sequential([
            keras.layers.Dense(hidden_dim),
            # Simplified transformer block
            keras.layers.MultiHeadAttention(num_heads=4, key_dim=hidden_dim // 4),
            keras.layers.LayerNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_dim, activation='relu')
        ])

    def _build_lstm_encoder(self, hidden_dim: int):
        """Build LSTM encoder."""
        return keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(hidden_dim)),
            keras.layers.Dense(hidden_dim, activation='relu')
        ])

    def _build_cnn_encoder(self, hidden_dim: int):
        """Build CNN encoder."""
        return keras.Sequential([
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_dim, activation='relu')
        ])

    def call(self, inputs, training=False):
        """Forward pass returning finger state predictions."""
        features = self.encoder(inputs, training=training)
        return self.classifier(features, training=training)

    def predict_positions(self, inputs, training=False):
        """Predict finger positions (for physics consistency)."""
        features = self.encoder(inputs, training=training)
        positions_flat = self.position_head(features, training=training)
        return tf.reshape(positions_flat, (-1, 5, 3))


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_temporal_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'transformer',
    epochs: int = 50,
    batch_size: int = 32,
    use_derivatives: bool = True
) -> Tuple[keras.Model, Dict]:
    """
    Train temporal model for inverse magnetometry.

    Args:
        X_train: [N, T, C] training windows
        y_train: [N, 5] binary finger states
        X_val: Validation windows
        y_val: Validation labels
        model_type: 'transformer', 'lstm', or 'cnn'
        epochs: Training epochs
        batch_size: Batch size
        use_derivatives: Add explicit temporal derivatives

    Returns:
        Trained model and training history
    """
    # Optionally add derivatives
    if use_derivatives:
        X_train = add_temporal_derivatives(X_train)
        X_val = add_temporal_derivatives(X_val)

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    # Build model
    if model_type == 'transformer':
        model = build_temporal_transformer(window_size, n_features)
    elif model_type == 'lstm':
        model = build_state_space_model(window_size, n_features)
    else:
        model = build_temporal_cnn(window_size, n_features)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return model, history.history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_derivatives: bool = True
) -> Dict:
    """Evaluate model on test set."""
    if use_derivatives:
        X_test = add_temporal_derivatives(X_test)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Exact match (all 5 fingers correct)
    exact_match = np.all(y_pred_binary == y_test, axis=1)
    exact_accuracy = np.mean(exact_match)

    # Per-finger accuracy
    per_finger = {
        FINGER_ORDER[i]: float(np.mean(y_pred_binary[:, i] == y_test[:, i]))
        for i in range(5)
    }

    # Mean finger accuracy
    mean_finger_acc = np.mean([per_finger[f] for f in FINGER_ORDER])

    # Hamming distance
    hamming = np.sum(y_pred_binary != y_test, axis=1)
    mean_hamming = np.mean(hamming)

    return {
        'exact_match_accuracy': float(exact_accuracy),
        'mean_finger_accuracy': float(mean_finger_acc),
        'per_finger_accuracy': per_finger,
        'mean_hamming_distance': float(mean_hamming)
    }


# =============================================================================
# CROSS-ORIENTATION EVALUATION
# =============================================================================

def prepare_cross_orientation_split(
    segments: List[LabeledSegment],
    window_size: int = 16,
    stride: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by orientation for cross-orientation testing.

    Train on high-pitch, test on low-pitch orientations.
    This tests whether temporal features generalize across device orientation.
    """
    # Calculate pitch quartiles
    all_pitches = np.concatenate([seg.pitch_angles for seg in segments])
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    for seg in segments:
        n_samples = len(seg.samples)
        if n_samples < window_size:
            continue

        features = seg.samples[:, :3]  # Magnetometer only

        for i in range(0, n_samples - window_size + 1, stride):
            window = features[i:i + window_size]
            mean_pitch = np.mean(seg.pitch_angles[i:i + window_size])

            if mean_pitch >= q3:  # High pitch → training
                train_windows.append(window)
                train_labels.append(seg.finger_binary)
            elif mean_pitch <= q1:  # Low pitch → testing
                test_windows.append(window)
                test_labels.append(seg.finger_binary)

    return (
        np.array(train_windows), np.array(train_labels),
        np.array(test_windows), np.array(test_labels)
    )


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiment(
    name: str,
    segments: List[LabeledSegment],
    window_size: int,
    model_type: str,
    use_derivatives: bool,
    cross_orientation: bool = True
) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"  Window: {window_size}, Model: {model_type}, Derivatives: {use_derivatives}")

    if cross_orientation:
        # Split by orientation
        X_train, y_train, X_test, y_test = prepare_cross_orientation_split(
            segments, window_size=window_size
        )
    else:
        # Random split
        X, y, _ = create_temporal_windows(segments, window_size=window_size)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        X_train, y_train = X[indices[:split]], y[indices[:split]]
        X_test, y_test = X[indices[split:]], y[indices[split:]]

    # Further split training into train/val
    n_train = len(X_train)
    val_split = int(0.85 * n_train)
    indices = np.random.permutation(n_train)

    X_val = X_train[indices[val_split:]]
    y_val = y_train[indices[val_split:]]
    X_train = X_train[indices[:val_split]]
    y_train = y_train[indices[:val_split]]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    if len(X_train) < 10 or len(X_test) < 5:
        print("  SKIPPED: Insufficient data")
        return {'name': name, 'error': 'insufficient_data'}

    # Normalize
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    # Train
    model, history = train_temporal_model(
        X_train_norm, y_train,
        X_val_norm, y_val,
        model_type=model_type,
        use_derivatives=use_derivatives,
        epochs=50
    )

    # Evaluate
    train_metrics = evaluate_model(model, X_train_norm, y_train, use_derivatives)
    test_metrics = evaluate_model(model, X_test_norm, y_test, use_derivatives)

    print(f"  Train Accuracy: {train_metrics['exact_match_accuracy']:.1%}")
    print(f"  Test Accuracy:  {test_metrics['exact_match_accuracy']:.1%}")
    print(f"  Gap: {train_metrics['exact_match_accuracy'] - test_metrics['exact_match_accuracy']:.1%}")

    tf.keras.backend.clear_session()

    return {
        'name': name,
        'window_size': window_size,
        'model_type': model_type,
        'use_derivatives': use_derivatives,
        'cross_orientation': cross_orientation,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }


def main():
    """Run inverse magnetometry temporal learning experiments."""
    print("="*80)
    print("INVERSE MAGNETOMETRY AS LEARNED FIELD")
    print("Temporal Learning Experiments")
    print("="*80)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    data_dir = Path('data/GAMBIT')
    if not data_dir.exists():
        data_dir = Path('.worktrees/data/GAMBIT')

    sessions = load_labeled_sessions(data_dir)
    print(f"Found {len(sessions)} sessions with labels")

    segments = extract_segments(sessions)
    print(f"Extracted {len(segments)} labeled segments")

    if not segments:
        print("ERROR: No labeled segments found!")
        return

    # Show segment distribution
    combo_counts = defaultdict(int)
    total_samples = 0
    for seg in segments:
        combo_counts[seg.finger_states] += 1
        total_samples += len(seg.samples)

    print(f"Total samples: {total_samples}")
    print(f"Unique finger states: {len(combo_counts)}")
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {combo}: {count} segments")

    results = []

    # =========================================================================
    # EXPERIMENT SUITE
    # =========================================================================

    # 1. Baseline: Standard CNN-LSTM (existing approach)
    results.append(run_experiment(
        name="Baseline_CNN_w16",
        segments=segments,
        window_size=16,
        model_type='cnn',
        use_derivatives=False
    ))

    # 2. Temporal CNN with derivatives (key innovation: explicit temporal info)
    results.append(run_experiment(
        name="CNN_derivatives_w16",
        segments=segments,
        window_size=16,
        model_type='cnn',
        use_derivatives=True
    ))

    # 3. Transformer - can learn which timesteps matter most
    results.append(run_experiment(
        name="Transformer_w16",
        segments=segments,
        window_size=16,
        model_type='transformer',
        use_derivatives=False
    ))

    results.append(run_experiment(
        name="Transformer_derivatives_w16",
        segments=segments,
        window_size=16,
        model_type='transformer',
        use_derivatives=True
    ))

    # 4. LSTM - state-space modeling for temporal dynamics
    results.append(run_experiment(
        name="LSTM_w16",
        segments=segments,
        window_size=16,
        model_type='lstm',
        use_derivatives=False
    ))

    results.append(run_experiment(
        name="LSTM_derivatives_w16",
        segments=segments,
        window_size=16,
        model_type='lstm',
        use_derivatives=True
    ))

    # 5. Window size ablation (how much temporal context helps?)
    for ws in [8, 24, 32]:
        results.append(run_experiment(
            name=f"LSTM_derivatives_w{ws}",
            segments=segments,
            window_size=ws,
            model_type='lstm',
            use_derivatives=True
        ))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Filter out failed experiments
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("No valid results to summarize.")
        return

    print(f"\n{'Experiment':<35} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print("-"*70)

    # Sort by test accuracy
    sorted_results = sorted(
        valid_results,
        key=lambda x: x['test_metrics']['exact_match_accuracy'],
        reverse=True
    )

    baseline_test = valid_results[0]['test_metrics']['exact_match_accuracy']

    for r in sorted_results:
        train_acc = r['train_metrics']['exact_match_accuracy']
        test_acc = r['test_metrics']['exact_match_accuracy']
        gap = train_acc - test_acc
        print(f"{r['name']:<35} {train_acc:>9.1%} {test_acc:>9.1%} {gap:>9.1%}")

    # Best result analysis
    best = sorted_results[0]
    print(f"\n*** Best: {best['name']} with {best['test_metrics']['exact_match_accuracy']:.1%} test accuracy ***")

    print(f"\nPer-finger accuracy ({best['name']}):")
    for finger, acc in best['test_metrics']['per_finger_accuracy'].items():
        print(f"  {finger}: {acc:.1%}")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Compare derivatives vs no derivatives
    deriv_results = [r for r in valid_results if r['use_derivatives']]
    no_deriv_results = [r for r in valid_results if not r['use_derivatives']]

    if deriv_results and no_deriv_results:
        avg_deriv = np.mean([r['test_metrics']['exact_match_accuracy'] for r in deriv_results])
        avg_no_deriv = np.mean([r['test_metrics']['exact_match_accuracy'] for r in no_deriv_results])

        print(f"\n  Effect of explicit temporal derivatives:")
        print(f"    Without derivatives: {avg_no_deriv:.1%} avg test accuracy")
        print(f"    With derivatives:    {avg_deriv:.1%} avg test accuracy")
        print(f"    Improvement:         {avg_deriv - avg_no_deriv:+.1%}")

    # Compare architectures
    for arch in ['cnn', 'transformer', 'lstm']:
        arch_results = [r for r in valid_results if r['model_type'] == arch]
        if arch_results:
            best_arch = max(arch_results, key=lambda x: x['test_metrics']['exact_match_accuracy'])
            print(f"\n  Best {arch.upper()}: {best_arch['test_metrics']['exact_match_accuracy']:.1%}")

    # Save results
    output_path = Path("ml/inverse_magnetometry_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64, np.int32, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(to_serializable(sorted_results), f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Theoretical discussion
    print("\n" + "="*80)
    print("THEORETICAL DISCUSSION: INVERSE MAGNETOMETRY AS LEARNED FIELD")
    print("="*80)
    print("""
    PROBLEM: 5 dipoles → 3-vector at sensor. Forward model closed-form,
    inverse is 15-DOF from 3 measurements (massively underdetermined).

    WHAT BREAKS THE DEGENERACY:

    1. TEMPORAL STRUCTURE
       - Single readings are ambiguous
       - Trajectories through field-space are far more informative
       - The manifold of plausible configurations is much smaller than ℝ¹⁵
       - Rate of field change (implicit in windows) carries position info

    2. LEARNED PRIORS ON MOTION
       - Hand is kinematic chain with ~20 DOF but strong covariance structure
       - Network learns the submanifold magnets actually traverse
       - Not solving general inverse magnetometry—learning "configurations
         that actually happen"

    ARCHITECTURE INSIGHTS:
    - Transformers: Learn which timesteps in window are most discriminative
    - LSTMs: Model state evolution (velocity, acceleration implicitly)
    - Explicit derivatives: Make temporal information explicit for the network

    PHYSICS CONSISTENCY (future work):
    - Use forward model as constraint during training
    - Predicted positions → simulated field → match observation
    - Smoothness priors: penalize jerk in predicted trajectories

    The network becomes a REGULARIZER encoding "configurations that actually
    happen" as implicit structure. This is learned inverse physics.
    """)

    return sorted_results


if __name__ == '__main__':
    main()
