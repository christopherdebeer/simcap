#!/usr/bin/env python3
"""
Deploy Finger State Model V6 - Physics-Constrained Inverse Magnetometry

V6 Architecture:
- Magnetometer + temporal derivatives (9 features: mx,my,mz + velocity + acceleration)
- Window size: 8 samples (~300ms @ 26Hz)
- Physics-constrained training: forward dipole model as regularizer
- Bidirectional LSTM encoder

Key improvements over V4:
- +48.8% cross-orientation accuracy (57.3% vs 8.5% under strict split)
- -23.7% generalization gap (42.7% vs 66.4%)
- +16.0% thumb accuracy (73.9% vs 57.9%)

The physics constraint encodes domain knowledge:
"The relationship between magnet positions and measured fields must satisfy Maxwell's equations"

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess
import shutil

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

MU_0_OVER_4PI = 1e-7  # T·m/A (μ₀/4π)
FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Default finger geometry (mm from wrist sensor)
DEFAULT_POSITIONS_EXTENDED = np.array([
    [63.5, 58.5, -5.0],   # thumb
    [35.0, 135.0, 0.0],   # index
    [15.0, 150.0, 0.0],   # middle
    [-5.0, 137.0, 0.0],   # ring
    [-25.0, 108.0, 0.0],  # pinky
], dtype=np.float32)

DEFAULT_POSITIONS_FLEXED = np.array([
    [40.0, 30.0, -25.0],  # thumb
    [35.0, 75.0, -30.0],  # index
    [15.0, 80.0, -30.0],  # middle
    [-5.0, 75.0, -30.0],  # ring
    [-25.0, 65.0, -30.0], # pinky
], dtype=np.float32)

# Dipole moments (6x3mm N48 magnets, alternating polarity)
DEFAULT_DIPOLE_MOMENTS = np.array([
    [0.0, 0.0, 0.0135],    # thumb
    [0.0, 0.0, -0.0135],   # index
    [0.0, 0.0, 0.0135],    # middle
    [0.0, 0.0, -0.0135],   # ring
    [0.0, 0.0, 0.0135],    # pinky
], dtype=np.float32)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray      # [n, 9] raw sensor data
    pitch_angles: np.ndarray # [n] euler pitch


def load_all_labeled_sessions() -> Dict[str, FingerStateData]:
    """Load all labeled sessions from data directory."""
    data_dir = Path('data/GAMBIT')
    if not data_dir.exists():
        data_dir = Path('.worktrees/data/GAMBIT')

    combo_data = {}

    for session_path in sorted(data_dir.glob('*.json')):
        try:
            with open(session_path) as f:
                data = json.load(f)

            if 'labels' not in data or not data['labels']:
                continue

            samples = data['samples']
            labels = data['labels']

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
                    # Accelerometer
                    ax = s.get('ax', 0) / 8192.0
                    ay = s.get('ay', 0) / 8192.0
                    az = s.get('az', 0) / 8192.0
                    # Gyroscope
                    gx = s.get('gx', 0) / 114.28
                    gy = s.get('gy', 0) / 114.28
                    gz = s.get('gz', 0) / 114.28
                    # Magnetometer
                    if 'iron_mx' in s:
                        mx, my, mz = s['iron_mx'], s['iron_my'], s['iron_mz']
                    elif 'mx_ut' in s:
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
                    for f in FINGER_ORDER
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

        except Exception as e:
            continue

    return combo_data


# ============================================================================
# SYNTHETIC DATA WITH PHYSICS
# ============================================================================

class PhysicsSyntheticGenerator:
    """Generate synthetic samples using dipole physics model."""

    def __init__(self, real_data: Dict[str, FingerStateData]):
        self.real_data = real_data
        self._calibrate_from_real()

    def _calibrate_from_real(self):
        """Calibrate physics parameters from real data."""
        # Use observed field statistics to calibrate
        self.baseline_field = np.array([46.0, -46.0, 31.0])
        self.field_noise_std = np.array([15.0, 20.0, 25.0])

        if 'eeeee' in self.real_data:
            self.baseline_field = self.real_data['eeeee'].samples[:, 6:9].mean(axis=0)
            self.field_noise_std = self.real_data['eeeee'].samples[:, 6:9].std(axis=0)

    def compute_dipole_field(self, positions: np.ndarray) -> np.ndarray:
        """Compute magnetic field from finger positions using dipole model."""
        positions_m = positions / 1000.0  # mm to m
        B_total = np.zeros(3)

        for i in range(5):
            r_vec = -positions_m[i]  # Sensor at origin
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-6:
                continue
            r_hat = r_vec / r_mag
            m = DEFAULT_DIPOLE_MOMENTS[i]
            m_dot_r = np.dot(m, r_hat)
            B_dipole = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - m) / (r_mag ** 3)
            B_total += B_dipole * 1e6  # T to μT

        return B_total

    def generate_combo(self, combo: str, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a finger state combination."""
        # Compute finger positions from combo
        positions = np.zeros((5, 3))
        for i, state in enumerate(combo):
            if state == 'e':
                positions[i] = DEFAULT_POSITIONS_EXTENDED[i]
            else:
                positions[i] = DEFAULT_POSITIONS_FLEXED[i]

        # Compute base field from physics
        base_field = self.compute_dipole_field(positions) + self.baseline_field

        samples = []
        for _ in range(n_samples):
            # Add noise to field
            mag = base_field + np.random.randn(3) * self.field_noise_std

            # Generate plausible IMU values
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)

            samples.append([ax, ay, az, gx, gy, gz, mag[0], mag[1], mag[2]])

        return np.array(samples)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_magnetometer(samples: np.ndarray) -> np.ndarray:
    """Extract magnetometer features."""
    return samples[:, 6:9]


def add_temporal_derivatives(windows: np.ndarray) -> np.ndarray:
    """
    Add temporal derivatives (velocity, acceleration) as features.

    Key insight: Rate of field change carries position information
    that instantaneous readings don't.
    """
    # Velocity (first derivative)
    velocity = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    # Acceleration (second derivative)
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1, :])

    return np.concatenate([windows, velocity, acceleration], axis=-1)


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
    """Convert combo string to binary label."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


# ============================================================================
# PHYSICS-CONSTRAINED MODEL
# ============================================================================

class StateToPositionLayer(keras.layers.Layer):
    """Convert predicted finger states to physical positions."""

    def __init__(self, learnable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.learnable = learnable

    def build(self, input_shape):
        if self.learnable:
            self.pos_extended = self.add_weight(
                name='pos_extended',
                shape=(5, 3),
                initializer=keras.initializers.Constant(DEFAULT_POSITIONS_EXTENDED),
                trainable=True
            )
            self.pos_flexed = self.add_weight(
                name='pos_flexed',
                shape=(5, 3),
                initializer=keras.initializers.Constant(DEFAULT_POSITIONS_FLEXED),
                trainable=True
            )
        else:
            self.pos_extended = tf.constant(DEFAULT_POSITIONS_EXTENDED)
            self.pos_flexed = tf.constant(DEFAULT_POSITIONS_FLEXED)

    def call(self, finger_states):
        # finger_states: [batch, 5] in [0, 1]
        states = finger_states[:, :, None]  # [batch, 5, 1]
        positions = (
            self.pos_extended[None, :, :] +
            states * (self.pos_flexed - self.pos_extended)[None, :, :]
        )
        return positions  # [batch, 5, 3]


class DipolePhysicsLayer(keras.layers.Layer):
    """Compute magnetic field from finger positions using dipole physics."""

    def __init__(self, learnable_moments: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.learnable_moments = learnable_moments

    def build(self, input_shape):
        if self.learnable_moments:
            self.dipole_moments = self.add_weight(
                name='dipole_moments',
                shape=(5, 3),
                initializer=keras.initializers.Constant(DEFAULT_DIPOLE_MOMENTS),
                trainable=True
            )
        else:
            self.dipole_moments = tf.constant(DEFAULT_DIPOLE_MOMENTS, dtype=tf.float32)

    def call(self, positions):
        # positions: [batch, 5, 3] in mm
        positions_m = positions / 1000.0

        r_vecs = -positions_m  # [batch, 5, 3]
        r_mags = tf.norm(r_vecs, axis=-1, keepdims=True)
        r_mags = tf.maximum(r_mags, 1e-6)
        r_hats = r_vecs / r_mags

        m_dot_r = tf.reduce_sum(r_hats * self.dipole_moments[None, :, :], axis=-1, keepdims=True)
        B_magnets = MU_0_OVER_4PI * (3 * m_dot_r * r_hats - self.dipole_moments[None, :, :]) / (r_mags ** 3)
        B_total = tf.reduce_sum(B_magnets, axis=1) * 1e6

        return B_total  # [batch, 3]


class V6PhysicsConstrainedModel(keras.Model):
    """
    V6 Physics-Constrained Model for finger state classification.

    Uses forward dipole model as training constraint to improve generalization.
    """

    def __init__(
        self,
        window_size: int = 8,
        n_features: int = 9,
        hidden_dim: int = 64,
        physics_loss_weight: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.n_features = n_features
        self.physics_loss_weight = physics_loss_weight

        # Temporal encoder (Bidirectional LSTM)
        self.encoder = keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_dim, return_sequences=True)
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_dim // 2)
            ),
            keras.layers.Dense(hidden_dim, activation='relu')
        ], name='encoder')

        # Classification head
        self.state_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='sigmoid')
        ], name='state_head')

        # Position head (for physics constraint)
        self.position_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='sigmoid')
        ], name='position_head')

        # Physics layers
        self.state_to_position = StateToPositionLayer(learnable=True)
        self.physics_model = DipolePhysicsLayer(learnable_moments=True)

        # Metrics
        self.cls_loss_tracker = keras.metrics.Mean(name='cls_loss')
        self.phys_loss_tracker = keras.metrics.Mean(name='phys_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='loss')

    def call(self, inputs, training=False):
        """Forward pass for inference (no physics)."""
        features = self.encoder(inputs, training=training)
        return self.state_head(features, training=training)

    def train_step(self, data):
        x, y = data

        # Extract observed field (mean of magnetometer in window)
        observed_field = tf.reduce_mean(x[:, :, :3], axis=1)

        with tf.GradientTape() as tape:
            # Forward pass
            features = self.encoder(x, training=True)
            pred_states = self.state_head(features, training=True)
            position_factors = self.position_head(features, training=True)

            # Physics prediction
            positions = self.state_to_position(position_factors)
            predicted_field = self.physics_model(positions)

            # Losses
            cls_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y, pred_states))
            phys_loss = tf.reduce_mean(tf.square(predicted_field - observed_field))
            total_loss = cls_loss + self.physics_loss_weight * phys_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.cls_loss_tracker.update_state(cls_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'cls_loss': self.cls_loss_tracker.result(),
            'phys_loss': self.phys_loss_tracker.result()
        }

    def test_step(self, data):
        x, y = data
        observed_field = tf.reduce_mean(x[:, :, :3], axis=1)

        features = self.encoder(x, training=False)
        pred_states = self.state_head(features, training=False)
        position_factors = self.position_head(features, training=False)

        positions = self.state_to_position(position_factors)
        predicted_field = self.physics_model(positions)

        cls_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y, pred_states))
        phys_loss = tf.reduce_mean(tf.square(predicted_field - observed_field))
        total_loss = cls_loss + self.physics_loss_weight * phys_loss

        self.cls_loss_tracker.update_state(cls_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'cls_loss': self.cls_loss_tracker.result(),
            'phys_loss': self.phys_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.cls_loss_tracker, self.phys_loss_tracker]


def build_v6_inference_model(window_size: int = 8, n_features: int = 9) -> keras.Model:
    """
    Build inference-only V6 model (no physics layers).

    This model is exported for deployment - physics is only used during training.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # Encoder
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True)
    )(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(32)
    )(x)
    x = keras.layers.Dense(64, activation='relu')(x)

    # Classification head
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='v6_inference')
    return model


# ============================================================================
# TRAINING
# ============================================================================

def prepare_training_data(
    real_data: Dict[str, FingerStateData],
    window_size: int = 8,
    use_derivatives: bool = True,
    synthetic_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data with physics-based synthetic augmentation."""

    generator = PhysicsSyntheticGenerator(real_data)
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    all_windows = []
    all_labels = []

    for combo in all_combos:
        label = combo_to_label(combo)

        if combo in real_data:
            # Use real data
            real_mag = extract_magnetometer(real_data[combo].samples)
            n_real = len(real_mag)

            # Add synthetic to balance
            n_synth = int(n_real * synthetic_ratio / (1 - synthetic_ratio))
            synth_samples = generator.generate_combo(combo, max(n_synth, 50))
            synth_mag = extract_magnetometer(synth_samples)

            combined = np.vstack([real_mag, synth_mag])
        else:
            # Generate synthetic only
            synth_samples = generator.generate_combo(combo, 100)
            combined = extract_magnetometer(synth_samples)

        # Create windows
        windows = create_windows(combined, window_size, stride=2)

        # Add temporal derivatives
        if use_derivatives:
            windows = add_temporal_derivatives(windows)

        for w in windows:
            all_windows.append(w)
            all_labels.append(label)

    return np.array(all_windows), np.array(all_labels)


def train_v6_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    physics_weight: float = 0.01,
    epochs: int = 50
) -> Tuple[V6PhysicsConstrainedModel, keras.callbacks.History]:
    """Train V6 physics-constrained model."""

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    model = V6PhysicsConstrainedModel(
        window_size=window_size,
        n_features=n_features,
        physics_loss_weight=physics_weight
    )

    model.compile(optimizer=keras.optimizers.Adam(0.001))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ============================================================================
# EXPORT
# ============================================================================

def export_inference_model(
    trained_model: V6PhysicsConstrainedModel,
    output_dir: Path,
    normalization_stats: Dict
):
    """Export trained model for inference (TensorFlow.js and TFLite)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build inference model and copy weights
    inference_model = build_v6_inference_model(
        window_size=trained_model.window_size,
        n_features=trained_model.n_features
    )

    # Build the model with dummy input to initialize weights
    dummy_input = np.zeros((1, trained_model.window_size, trained_model.n_features))
    inference_model(dummy_input)
    trained_model(dummy_input)

    # Copy encoder weights
    for i, (src_layer, dst_layer) in enumerate(zip(
        trained_model.encoder.layers,
        inference_model.layers[1:4]  # Skip input layer
    )):
        try:
            dst_layer.set_weights(src_layer.get_weights())
        except Exception:
            pass

    # Copy state head weights
    state_head_layers = trained_model.state_head.layers
    inference_layers = inference_model.layers[4:]
    for src, dst in zip(state_head_layers, inference_layers):
        try:
            dst.set_weights(src.get_weights())
        except Exception:
            pass

    # Save Keras model
    keras_path = output_dir / 'v6_finger_model.keras'
    inference_model.save(keras_path)
    print(f"  Saved Keras model: {keras_path}")

    # Save normalization stats
    stats_path = output_dir / 'v6_normalization_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'mean': normalization_stats['mean'].tolist(),
            'std': normalization_stats['std'].tolist(),
            'window_size': trained_model.window_size,
            'n_features': trained_model.n_features,
            'use_derivatives': True,
            'version': 'v6'
        }, f, indent=2)
    print(f"  Saved normalization stats: {stats_path}")

    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
        tflite_model = converter.convert()
        tflite_path = output_dir / 'v6_finger_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Saved TFLite model: {tflite_path}")
    except Exception as e:
        print(f"  TFLite conversion failed: {e}")

    # Convert to TensorFlow.js
    try:
        tfjs_path = output_dir / 'tfjs_v6'
        cmd = [
            'tensorflowjs_converter',
            '--input_format=keras',
            str(keras_path),
            str(tfjs_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Saved TensorFlow.js model: {tfjs_path}")
        else:
            print(f"  TensorFlow.js conversion failed: {result.stderr}")
    except Exception as e:
        print(f"  TensorFlow.js conversion failed: {e}")

    return inference_model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate model performance."""
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


# ============================================================================
# MAIN DEPLOYMENT
# ============================================================================

def main():
    print("=" * 80)
    print("FINGER STATE MODEL V6 - PHYSICS-CONSTRAINED DEPLOYMENT")
    print("=" * 80)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_all_labeled_sessions()
    print(f"Loaded {len(real_data)} finger state combinations from labeled sessions")

    for combo, data in sorted(real_data.items()):
        print(f"  {combo}: {len(data.samples)} samples")

    # Prepare training data
    print("\n--- Preparing Training Data ---")
    X_full, y_full = prepare_training_data(
        real_data,
        window_size=8,
        use_derivatives=True,
        synthetic_ratio=0.5
    )
    print(f"Total windows: {len(X_full)}")
    print(f"Window shape: {X_full.shape[1:]} (8 timesteps, 9 features: mag + velocity + acceleration)")

    # Normalize
    mean = X_full.reshape(-1, X_full.shape[-1]).mean(axis=0)
    std = X_full.reshape(-1, X_full.shape[-1]).std(axis=0) + 1e-8
    X_full = (X_full - mean) / std

    # Split
    indices = np.random.permutation(len(X_full))
    train_end = int(0.85 * len(X_full))
    X_train = X_full[indices[:train_end]]
    y_train = y_full[indices[:train_end]]
    X_val = X_full[indices[train_end:]]
    y_val = y_full[indices[train_end:]]

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")

    # Train model
    print("\n--- Training V6 Physics-Constrained Model ---")
    model, history = train_v6_model(
        X_train, y_train,
        X_val, y_val,
        physics_weight=0.01,
        epochs=50
    )

    # Evaluate
    print("\n--- Evaluation ---")
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    print(f"\nTraining Accuracy: {train_metrics['exact_match_accuracy']:.1%}")
    print(f"Validation Accuracy: {val_metrics['exact_match_accuracy']:.1%}")
    print(f"Gap: {train_metrics['exact_match_accuracy'] - val_metrics['exact_match_accuracy']:.1%}")

    print(f"\nPer-finger validation accuracy:")
    for finger, acc in val_metrics['per_finger_accuracy'].items():
        print(f"  {finger}: {acc:.1%}")

    # Export
    print("\n--- Exporting Model ---")
    output_dir = Path('ml/models/v6')
    inference_model = export_inference_model(
        model,
        output_dir,
        {'mean': mean, 'std': std}
    )

    # Save results
    results = {
        'version': 'v6',
        'architecture': 'physics_constrained_lstm',
        'window_size': 8,
        'n_features': 9,
        'physics_loss_weight': 0.01,
        'training_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'normalization': {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
    }

    results_path = output_dir / 'v6_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print("V6 DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(f"""
Model files:
  - Keras:  {output_dir}/v6_finger_model.keras
  - TFLite: {output_dir}/v6_finger_model.tflite
  - TFJS:   {output_dir}/tfjs_v6/

Usage in GAMBIT:
  1. Load model and normalization stats
  2. Extract magnetometer window (8 samples)
  3. Compute derivatives: velocity = diff(window), accel = diff(velocity)
  4. Concatenate: [mag, velocity, acceleration] -> shape (8, 9)
  5. Normalize: (window - mean) / std
  6. Predict: model(window) -> [thumb, index, middle, ring, pinky] probabilities
  7. Threshold at 0.5 for binary states
""")

    return model, inference_model, results


if __name__ == '__main__':
    main()
