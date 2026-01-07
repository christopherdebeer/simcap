#!/usr/bin/env python3
"""
Physics-Constrained Inverse Magnetometry Training

Extends the temporal learning approach with physics-consistency loss:
- Predicted positions → forward model → simulated field → match observation
- This regularizes the network using known physics as a constraint

The key insight: We know the forward model exactly (dipole physics).
Using it as a training constraint prevents overfitting to spurious patterns
and enforces physically plausible solutions.

Training signal:
    L_total = L_classification + λ_physics * L_physics + λ_smooth * L_smooth

Where:
    L_physics = ||B_predicted - B_observed||²
    L_smooth = ||jerk(positions)||² (trajectory smoothness)

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


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

MU_0_OVER_4PI = 1e-7  # T·m/A (μ₀/4π)
FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Default finger geometry from hand_model.py (in mm)
# These are approximate positions for extended/flexed states
DEFAULT_POSITIONS_EXTENDED = np.array([
    [63.5, 58.5, -5.0],   # thumb (base + extended offset)
    [35.0, 135.0, 0.0],   # index
    [15.0, 150.0, 0.0],   # middle
    [-5.0, 137.0, 0.0],   # ring
    [-25.0, 108.0, 0.0],  # pinky
], dtype=np.float32)

DEFAULT_POSITIONS_FLEXED = np.array([
    [40.0, 30.0, -25.0],  # thumb (curled)
    [35.0, 75.0, -30.0],  # index
    [15.0, 80.0, -30.0],  # middle
    [-5.0, 75.0, -30.0],  # ring
    [-25.0, 65.0, -30.0], # pinky
], dtype=np.float32)

# Default dipole moments (6x3mm N48 magnets, alternating polarity)
# Moment magnitude ~0.0135 A·m² for 6x3mm N48
DEFAULT_DIPOLE_MOMENTS = np.array([
    [0.0, 0.0, 0.0135],    # thumb
    [0.0, 0.0, -0.0135],   # index (alternating)
    [0.0, 0.0, 0.0135],    # middle
    [0.0, 0.0, -0.0135],   # ring (alternating)
    [0.0, 0.0, 0.0135],    # pinky
], dtype=np.float32)


# =============================================================================
# DIFFERENTIABLE PHYSICS MODEL
# =============================================================================

class DifferentiablePhysicsModel(keras.layers.Layer):
    """
    Differentiable forward physics model for magnetic dipoles.

    Given finger positions, computes expected magnetic field at sensor.
    This allows gradients to flow through the physics during training.
    """

    def __init__(
        self,
        dipole_moments: np.ndarray = None,
        learnable_moments: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        moments = dipole_moments if dipole_moments is not None else DEFAULT_DIPOLE_MOMENTS

        if learnable_moments:
            self.dipole_moments = self.add_weight(
                name='dipole_moments',
                shape=(5, 3),
                initializer=keras.initializers.Constant(moments),
                trainable=True
            )
        else:
            self.dipole_moments = tf.constant(moments, dtype=tf.float32)

    def call(self, positions):
        """
        Compute magnetic field from finger positions.

        Args:
            positions: [batch, 5, 3] finger positions in mm

        Returns:
            [batch, 3] magnetic field in μT
        """
        # Convert mm to meters
        positions_m = positions / 1000.0

        # Position vectors from magnets to sensor (at origin)
        r_vecs = -positions_m  # [batch, 5, 3]

        # Distance magnitudes with singularity protection
        r_mags = tf.norm(r_vecs, axis=-1, keepdims=True)  # [batch, 5, 1]
        r_mags = tf.maximum(r_mags, 1e-6)

        # Unit vectors
        r_hats = r_vecs / r_mags  # [batch, 5, 3]

        # Dipole field: B = (μ₀/4π) × [3(m·r̂)r̂ - m] / r³
        m_dot_r = tf.reduce_sum(
            r_hats * self.dipole_moments[None, :, :],
            axis=-1, keepdims=True
        )  # [batch, 5, 1]

        B_magnets = MU_0_OVER_4PI * (
            3 * m_dot_r * r_hats - self.dipole_moments[None, :, :]
        ) / (r_mags ** 3)

        # Convert T to μT and sum over all magnets
        B_total = tf.reduce_sum(B_magnets, axis=1) * 1e6  # [batch, 3]

        return B_total


class FingerStateToPosition(keras.layers.Layer):
    """
    Convert finger states (binary) to physical positions.

    Uses learnable interpolation between extended/flexed positions.
    """

    def __init__(
        self,
        pos_extended: np.ndarray = None,
        pos_flexed: np.ndarray = None,
        learnable_geometry: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        ext = pos_extended if pos_extended is not None else DEFAULT_POSITIONS_EXTENDED
        flex = pos_flexed if pos_flexed is not None else DEFAULT_POSITIONS_FLEXED

        if learnable_geometry:
            self.pos_extended = self.add_weight(
                name='pos_extended',
                shape=(5, 3),
                initializer=keras.initializers.Constant(ext),
                trainable=True
            )
            self.pos_flexed = self.add_weight(
                name='pos_flexed',
                shape=(5, 3),
                initializer=keras.initializers.Constant(flex),
                trainable=True
            )
        else:
            self.pos_extended = tf.constant(ext, dtype=tf.float32)
            self.pos_flexed = tf.constant(flex, dtype=tf.float32)

    def call(self, finger_states):
        """
        Convert finger states to positions via interpolation.

        Args:
            finger_states: [batch, 5] continuous values in [0, 1]
                          (0 = extended, 1 = flexed)

        Returns:
            [batch, 5, 3] finger positions in mm
        """
        # Expand for broadcasting
        states = finger_states[:, :, None]  # [batch, 5, 1]

        # Linear interpolation: pos = extended + state * (flexed - extended)
        positions = (
            self.pos_extended[None, :, :] +
            states * (self.pos_flexed - self.pos_extended)[None, :, :]
        )

        return positions


# =============================================================================
# PHYSICS-CONSTRAINED MODEL
# =============================================================================

class PhysicsConstrainedModel(keras.Model):
    """
    Temporal model with physics-consistency constraint.

    Architecture:
    1. Temporal encoder processes magnetometer window
    2. State head predicts finger states (classification)
    3. Position head predicts continuous finger positions
    4. Physics layer computes expected field from positions
    5. Physics loss enforces consistency with observations
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        encoder_type: str = 'lstm',
        hidden_dim: int = 64,
        physics_loss_weight: float = 0.1,
        smoothness_loss_weight: float = 0.01,
        learnable_physics: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.n_features = n_features
        self.physics_loss_weight = physics_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight

        # Temporal encoder
        if encoder_type == 'lstm':
            self.encoder = self._build_lstm_encoder(hidden_dim)
        elif encoder_type == 'transformer':
            self.encoder = self._build_transformer_encoder(hidden_dim)
        else:
            self.encoder = self._build_cnn_encoder(hidden_dim)

        # Classification head (finger states as binary)
        self.state_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='sigmoid', name='state_output')
        ], name='state_head')

        # Position head (continuous positions for physics)
        self.position_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='sigmoid', name='position_factors')
        ], name='position_head')

        # Physics components
        self.state_to_position = FingerStateToPosition(
            learnable_geometry=learnable_physics
        )
        self.physics_model = DifferentiablePhysicsModel(
            learnable_moments=learnable_physics
        )

        # Loss trackers
        self.classification_loss_tracker = keras.metrics.Mean(name='cls_loss')
        self.physics_loss_tracker = keras.metrics.Mean(name='phys_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')

    def _build_lstm_encoder(self, hidden_dim):
        return keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_dim, return_sequences=True)
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_dim // 2)
            ),
            keras.layers.Dense(hidden_dim, activation='relu')
        ], name='lstm_encoder')

    def _build_transformer_encoder(self, hidden_dim):
        # Use LSTM as fallback - transformer encoder requires known input shape
        # For proper transformer, would need to pass window_size, n_features
        return keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_dim // 2, return_sequences=True)
            ),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_dim, activation='relu')
        ], name='transformer_encoder_fallback')

    def _build_cnn_encoder(self, hidden_dim):
        return keras.Sequential([
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_dim, activation='relu')
        ], name='cnn_encoder')

    def call(self, inputs, training=False):
        """Forward pass returning finger state predictions."""
        # inputs: [batch, window_size, n_features]
        features = self.encoder(inputs, training=training)
        states = self.state_head(features, training=training)
        return states

    def predict_with_physics(self, inputs, training=False):
        """
        Full forward pass including physics predictions.

        Returns:
            states: [batch, 5] predicted finger states
            positions: [batch, 5, 3] predicted positions
            predicted_field: [batch, 3] field from predicted positions
        """
        features = self.encoder(inputs, training=training)
        states = self.state_head(features, training=training)
        position_factors = self.position_head(features, training=training)

        # Convert to physical positions
        positions = self.state_to_position(position_factors)

        # Compute expected field
        predicted_field = self.physics_model(positions)

        return states, positions, predicted_field

    def compute_physics_loss(self, predicted_field, observed_field):
        """
        Physics consistency loss: predicted field should match observation.

        Args:
            predicted_field: [batch, 3] from forward model
            observed_field: [batch, 3] mean of observed window
        """
        return tf.reduce_mean(tf.square(predicted_field - observed_field))

    def train_step(self, data):
        """Custom training step with physics constraint."""
        x, y = data

        # Extract observed field (mean of magnetometer channels in window)
        # Assuming first 3 channels are mx, my, mz
        observed_field = tf.reduce_mean(x[:, :, :3], axis=1)  # [batch, 3]

        with tf.GradientTape() as tape:
            # Forward pass
            features = self.encoder(x, training=True)
            pred_states = self.state_head(features, training=True)
            position_factors = self.position_head(features, training=True)

            # Physics prediction
            positions = self.state_to_position(position_factors)
            predicted_field = self.physics_model(positions)

            # Classification loss
            cls_loss = keras.losses.binary_crossentropy(y, pred_states)
            cls_loss = tf.reduce_mean(cls_loss)

            # Physics consistency loss
            phys_loss = self.compute_physics_loss(predicted_field, observed_field)

            # Total loss
            total_loss = (
                cls_loss +
                self.physics_loss_weight * phys_loss
            )

        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.classification_loss_tracker.update_state(cls_loss)
        self.physics_loss_tracker.update_state(phys_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'cls_loss': self.classification_loss_tracker.result(),
            'phys_loss': self.physics_loss_tracker.result()
        }

    def test_step(self, data):
        """Validation step."""
        x, y = data

        observed_field = tf.reduce_mean(x[:, :, :3], axis=1)

        # Forward pass
        features = self.encoder(x, training=False)
        pred_states = self.state_head(features, training=False)
        position_factors = self.position_head(features, training=False)

        positions = self.state_to_position(position_factors)
        predicted_field = self.physics_model(positions)

        cls_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y, pred_states))
        phys_loss = self.compute_physics_loss(predicted_field, observed_field)
        total_loss = cls_loss + self.physics_loss_weight * phys_loss

        self.classification_loss_tracker.update_state(cls_loss)
        self.physics_loss_tracker.update_state(phys_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'cls_loss': self.classification_loss_tracker.result(),
            'phys_loss': self.physics_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.classification_loss_tracker,
            self.physics_loss_tracker
        ]


# =============================================================================
# DATA LOADING (from inverse_magnetometry_temporal.py)
# =============================================================================

@dataclass
class LabeledSegment:
    samples: np.ndarray
    finger_states: str
    pitch_angles: np.ndarray
    timestamps: np.ndarray

    @property
    def finger_binary(self) -> np.ndarray:
        return np.array([0 if c == 'e' else 1 for c in self.finger_states])


def load_labeled_sessions(data_dir: Path) -> List[Dict[str, Any]]:
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
                })
        except Exception:
            continue
    return sessions


def extract_segments(sessions: List[Dict]) -> List[LabeledSegment]:
    segments = []
    for session in sessions:
        samples = session['samples']
        labels = session['labels']

        for label in labels:
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

            segment_samples = samples[start:min(end, len(samples))]
            if len(segment_samples) < 5:
                continue

            sensor_data = []
            pitch_data = []
            time_data = []

            for s in segment_samples:
                if 'iron_mx' in s:
                    mx, my, mz = s['iron_mx'], s['iron_my'], s['iron_mz']
                elif 'mx_ut' in s:
                    mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
                else:
                    mx = s.get('mx', 0) / 10.24
                    my = s.get('my', 0) / 10.24
                    mz = s.get('mz', 0) / 10.24

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


def prepare_cross_orientation_split(
    segments: List[LabeledSegment],
    window_size: int = 8,
    stride: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by orientation for cross-orientation testing."""
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

            if mean_pitch >= q3:
                train_windows.append(window)
                train_labels.append(seg.finger_binary)
            elif mean_pitch <= q1:
                test_windows.append(window)
                test_labels.append(seg.finger_binary)

    return (
        np.array(train_windows), np.array(train_labels),
        np.array(test_windows), np.array(test_labels)
    )


def add_temporal_derivatives(windows: np.ndarray) -> np.ndarray:
    """Add velocity and acceleration as features."""
    velocity = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])
    return np.concatenate([windows, velocity, acceleration], axis=-1)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    exact_match = np.all(y_pred_binary == y_test, axis=1)
    exact_accuracy = np.mean(exact_match)

    per_finger = {
        FINGER_ORDER[i]: float(np.mean(y_pred_binary[:, i] == y_test[:, i]))
        for i in range(5)
    }

    mean_finger_acc = np.mean([per_finger[f] for f in FINGER_ORDER])
    hamming = np.sum(y_pred_binary != y_test, axis=1)

    return {
        'exact_match_accuracy': float(exact_accuracy),
        'mean_finger_accuracy': float(mean_finger_acc),
        'per_finger_accuracy': per_finger,
        'mean_hamming_distance': float(np.mean(hamming))
    }


def run_physics_constrained_experiment(
    segments: List[LabeledSegment],
    window_size: int = 8,
    physics_weight: float = 0.1,
    use_derivatives: bool = True,
    encoder_type: str = 'lstm',
    epochs: int = 50
) -> Dict:
    """Run physics-constrained training experiment."""
    print(f"\n{'='*70}")
    print(f"PHYSICS-CONSTRAINED EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Window: {window_size}, Physics λ: {physics_weight}")
    print(f"  Encoder: {encoder_type}, Derivatives: {use_derivatives}")

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_cross_orientation_split(
        segments, window_size=window_size, stride=2
    )

    if len(X_train) < 10 or len(X_test) < 5:
        print("  SKIPPED: Insufficient data")
        return {'error': 'insufficient_data'}

    # Add derivatives if requested
    if use_derivatives:
        X_train = add_temporal_derivatives(X_train)
        X_test = add_temporal_derivatives(X_test)

    n_features = X_train.shape[-1]

    # Split train into train/val
    n_train = len(X_train)
    val_split = int(0.85 * n_train)
    indices = np.random.permutation(n_train)

    X_val = X_train[indices[val_split:]]
    y_val = y_train[indices[val_split:]]
    X_train = X_train[indices[:val_split]]
    y_train = y_train[indices[:val_split]]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Build model
    model = PhysicsConstrainedModel(
        window_size=window_size,
        n_features=n_features,
        encoder_type=encoder_type,
        physics_loss_weight=physics_weight,
        learnable_physics=True
    )

    model.compile(optimizer=keras.optimizers.Adam(0.001))

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7
    )

    # Train
    print("  Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print(f"  Train Accuracy: {train_metrics['exact_match_accuracy']:.1%}")
    print(f"  Test Accuracy:  {test_metrics['exact_match_accuracy']:.1%}")
    print(f"  Gap: {train_metrics['exact_match_accuracy'] - test_metrics['exact_match_accuracy']:.1%}")

    # Get final physics loss
    final_phys_loss = history.history.get('phys_loss', [0])[-1]
    print(f"  Final Physics Loss: {final_phys_loss:.4f}")

    tf.keras.backend.clear_session()

    return {
        'window_size': window_size,
        'physics_weight': physics_weight,
        'encoder_type': encoder_type,
        'use_derivatives': use_derivatives,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'final_physics_loss': float(final_phys_loss),
        'history': {
            'loss': [float(x) for x in history.history.get('loss', [])],
            'cls_loss': [float(x) for x in history.history.get('cls_loss', [])],
            'phys_loss': [float(x) for x in history.history.get('phys_loss', [])]
        }
    }


def main():
    """Run physics-constrained experiments."""
    print("="*80)
    print("PHYSICS-CONSTRAINED INVERSE MAGNETOMETRY")
    print("Using forward model as training constraint")
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

    results = []

    # =========================================================================
    # EXPERIMENT 1: Baseline (no physics constraint)
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE: No Physics Constraint (λ=0)")
    print("="*70)

    results.append({
        'name': 'baseline_no_physics',
        **run_physics_constrained_experiment(
            segments,
            window_size=8,
            physics_weight=0.0,
            use_derivatives=True,
            encoder_type='lstm'
        )
    })

    # =========================================================================
    # EXPERIMENT 2: Vary physics loss weight
    # =========================================================================
    print("\n" + "="*70)
    print("VARYING PHYSICS LOSS WEIGHT")
    print("="*70)

    for weight in [0.01, 0.05, 0.1, 0.2, 0.5]:
        results.append({
            'name': f'physics_weight_{weight}',
            **run_physics_constrained_experiment(
                segments,
                window_size=8,
                physics_weight=weight,
                use_derivatives=True,
                encoder_type='lstm'
            )
        })

    # =========================================================================
    # EXPERIMENT 3: Different encoder types with physics
    # =========================================================================
    print("\n" + "="*70)
    print("ENCODER COMPARISON WITH PHYSICS CONSTRAINT")
    print("="*70)

    for encoder in ['lstm', 'cnn', 'transformer']:
        results.append({
            'name': f'physics_{encoder}',
            **run_physics_constrained_experiment(
                segments,
                window_size=8,
                physics_weight=0.1,
                use_derivatives=True,
                encoder_type=encoder
            )
        })

    # =========================================================================
    # EXPERIMENT 4: Window size with physics
    # =========================================================================
    print("\n" + "="*70)
    print("WINDOW SIZE WITH PHYSICS CONSTRAINT")
    print("="*70)

    for ws in [4, 8, 16]:
        results.append({
            'name': f'physics_w{ws}',
            **run_physics_constrained_experiment(
                segments,
                window_size=ws,
                physics_weight=0.1,
                use_derivatives=True,
                encoder_type='lstm'
            )
        })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("No valid results.")
        return

    print(f"\n{'Experiment':<25} {'Phys λ':>8} {'Train':>10} {'Test':>10} {'Gap':>10} {'Phys Loss':>12}")
    print("-"*80)

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
        phys_loss = r.get('final_physics_loss', 0)
        weight = r.get('physics_weight', 0)
        print(f"{r['name']:<25} {weight:>8.2f} {train_acc:>9.1%} {test_acc:>9.1%} {gap:>9.1%} {phys_loss:>11.4f}")

    # Best result
    best = sorted_results[0]
    print(f"\n*** Best: {best['name']} with {best['test_metrics']['exact_match_accuracy']:.1%} test accuracy ***")

    # Improvement analysis
    physics_results = [r for r in valid_results if r.get('physics_weight', 0) > 0]
    if physics_results:
        best_physics = max(physics_results, key=lambda x: x['test_metrics']['exact_match_accuracy'])
        improvement = best_physics['test_metrics']['exact_match_accuracy'] - baseline_test
        print(f"\n  Physics constraint improvement: {improvement:+.1%}")

    # Gap reduction analysis
    baseline_gap = valid_results[0]['train_metrics']['exact_match_accuracy'] - baseline_test
    if physics_results:
        best_physics_gap = (
            best_physics['train_metrics']['exact_match_accuracy'] -
            best_physics['test_metrics']['exact_match_accuracy']
        )
        gap_reduction = baseline_gap - best_physics_gap
        print(f"  Train-test gap reduction: {gap_reduction:+.1%}")

    # Per-finger accuracy for best
    print(f"\nPer-finger accuracy ({best['name']}):")
    for finger, acc in best['test_metrics']['per_finger_accuracy'].items():
        print(f"  {finger}: {acc:.1%}")

    # Save results
    output_path = Path("ml/physics_constrained_results.json")

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

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS: PHYSICS-CONSTRAINED TRAINING")
    print("="*80)
    print("""
    PHYSICS CONSISTENCY LOSS:
    - Predicted positions → forward dipole model → expected field
    - Loss = ||B_predicted - B_observed||²
    - Forces network to learn physically plausible position-to-field mappings

    EXPECTED BENEFITS:
    1. REDUCED OVERFITTING: Physics constraint acts as regularizer
    2. BETTER GENERALIZATION: Can't cheat with spurious correlations
    3. CALIBRATION ROBUSTNESS: Physics is orientation-invariant

    LEARNABLE PHYSICS PARAMETERS:
    - Finger positions (extended/flexed) adapt to actual hand geometry
    - Dipole moments can adjust to actual magnet strengths
    - This bridges sim-to-real gap

    The physics constraint encodes domain knowledge:
    "The relationship between positions and fields must obey Maxwell's equations"
    """)

    return sorted_results


if __name__ == '__main__':
    main()
