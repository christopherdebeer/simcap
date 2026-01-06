#!/usr/bin/env python3
"""
Train and export TensorFlow.js model using ground truth aligned data.

Uses the AlignedGenerator to create training data grounded in measured signatures.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from ml.simulation.aligned_generator import AlignedGenerator


def build_model(input_dim: int = 3) -> 'keras.Model':
    """Build multi-output finger classifier model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    inputs = keras.Input(shape=(input_dim,), name='mag_input')
    
    # Shared encoder
    x = layers.Dense(64, activation='relu', name='enc_dense1')(inputs)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = layers.Dense(48, activation='relu', name='enc_dense2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = layers.Dense(32, activation='relu', name='enc_dense3')(x)
    
    # Per-finger output heads
    outputs = []
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for finger in finger_names:
        h = layers.Dense(16, activation='relu', name=f'{finger}_hidden')(x)
        out = layers.Dense(2, activation='softmax', name=f'{finger}_output')(h)  # Binary: 0=extended, 1=flexed
        outputs.append(out)
    
    model = keras.Model(inputs, outputs, name='finger_classifier_v2')
    return model


def train_model(X_train, y_train, X_val, y_val, epochs=50):
    """Train the model."""
    import tensorflow as tf
    from tensorflow import keras
    
    model = build_model(input_dim=X_train.shape[1])
    
    # Compile with per-output metrics
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={f'{f}_output': 'sparse_categorical_crossentropy' for f in finger_names},
        metrics={f'{f}_output': 'accuracy' for f in finger_names}
    )
    
    # Prepare labels for multi-output
    y_train_list = [y_train[:, i] for i in range(5)]
    y_val_list = [y_val[:, i] for i in range(5)]
    
    history = model.fit(
        X_train, y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    
    return model, history


def export_to_tfjs(model, stats: dict, output_dir: str):
    """Export model to TensorFlow.js format."""
    import subprocess
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as Keras format
    keras_path = output_path / 'model.keras'
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")
    
    # Convert to TensorFlow.js
    result = subprocess.run([
        'tensorflowjs_converter',
        '--input_format=keras',
        str(keras_path),
        str(output_path)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"TensorFlow.js conversion warning: {result.stderr}")
        # Try with tf_keras
        result = subprocess.run([
            'tensorflowjs_converter',
            '--input_format=tf_keras',
            str(keras_path),
            str(output_path)
        ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Exported TensorFlow.js model: {output_path / 'model.json'}")
    else:
        print(f"Warning: TF.js conversion failed. Manual conversion may be needed.")
        print(f"  Error: {result.stderr}")
    
    # Save config
    config = {
        'stats': stats,
        'inputShape': [None, 3],
        'fingerNames': ['thumb', 'index', 'middle', 'ring', 'pinky'],
        'stateNames': ['extended', 'flexed'],  # Binary classification
        'description': 'Ground truth aligned finger tracking model',
        'version': 'aligned_v1',
        'date': str(np.datetime64('today'))
    }
    
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")


def main():
    print("=" * 70)
    print("TRAINING GROUND TRUTH ALIGNED MODEL")
    print("=" * 70)
    
    # Load ground truth session
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    
    print(f"\nLoading ground truth from: {session_path.name}")
    gen = AlignedGenerator(session_path)
    
    # Generate training data
    print("\nGenerating aligned training data...")
    X, y = gen.generate_all_configurations(samples_per_config=500)
    print(f"Generated {len(X)} samples")
    
    # Compute normalization stats from training data
    stats = {
        'mean': X.mean(axis=0).tolist(),
        'std': X.std(axis=0).tolist()
    }
    print(f"Stats: mean={[f'{m:.0f}' for m in stats['mean']]}, std={[f'{s:.0f}' for s in stats['std']]}")
    
    # Normalize
    X_norm = (X - np.array(stats['mean'])) / (np.array(stats['std']) + 1e-8)
    
    # Split train/val
    n_val = int(len(X) * 0.15)
    indices = np.random.permutation(len(X))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_val, y_val = X_norm[val_idx], y[val_idx]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Train
    print("\nTraining model...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate
    print("\nFinal validation accuracy per finger:")
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for i, finger in enumerate(finger_names):
        val_acc = history.history.get(f'val_{finger}_output_accuracy', [0])[-1]
        print(f"  {finger}: {val_acc:.1%}")
    
    # Export
    output_dir = 'public/models/finger_aligned_v1'
    print(f"\nExporting to {output_dir}...")
    export_to_tfjs(model, stats, output_dir)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
