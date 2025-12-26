"""
SIMCAP Gesture Classification Models

1D CNN models for classifying static hand poses from IMU sensor windows.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Optional imports - gracefully handle missing dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .schema import Gesture, NUM_FEATURES


def create_cnn_model_keras(
    window_size: int = 50,
    num_features: int = NUM_FEATURES,
    num_classes: int = len(Gesture),
    filters: Tuple[int, ...] = (32, 64, 64),
    kernel_size: int = 5,
    dropout: float = 0.3
) -> 'keras.Model':
    """
    Create a 1D CNN model for gesture classification using Keras.

    Architecture:
        Input -> [Conv1D -> ReLU -> MaxPool] x 3 -> GlobalAvgPool -> Dense -> Softmax

    Args:
        window_size: Number of timesteps per window
        num_features: Number of input features (9 for IMU)
        num_classes: Number of gesture classes
        filters: Number of filters in each conv layer
        kernel_size: Convolution kernel size
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow")

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(window_size, num_features)),

        # Conv block 1
        layers.Conv1D(filters[0], kernel_size, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout),

        # Conv block 2
        layers.Conv1D(filters[1], kernel_size, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout),

        # Conv block 3
        layers.Conv1D(filters[2], kernel_size, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Global pooling and classification
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if HAS_TORCH:
    class GestureCNN(nn.Module):
        """
        1D CNN for gesture classification using PyTorch.

        Same architecture as the Keras version.
        """

        def __init__(
            self,
            window_size: int = 50,
            num_features: int = NUM_FEATURES,
            num_classes: int = len(Gesture),
            filters: Tuple[int, ...] = (32, 64, 64),
            kernel_size: int = 5,
            dropout: float = 0.3
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(num_features, filters[0], kernel_size, padding='same')
            self.bn1 = nn.BatchNorm1d(filters[0])
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, padding='same')
            self.bn2 = nn.BatchNorm1d(filters[1])
            self.pool2 = nn.MaxPool1d(2)

            self.conv3 = nn.Conv1d(filters[1], filters[2], kernel_size, padding='same')
            self.bn3 = nn.BatchNorm1d(filters[2])

            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(filters[2], 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Input: (batch, seq_len, features)
            # Conv1d expects: (batch, features, seq_len)
            x = x.transpose(1, 2)

            # Conv block 1
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.dropout(x)

            # Conv block 2
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.dropout(x)

            # Conv block 3
            x = F.relu(self.bn3(self.conv3(x)))

            # Global average pooling
            x = x.mean(dim=2)

            # Classification
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)

            return x
else:
    # Placeholder when PyTorch is not installed
    GestureCNN = None


def train_model_keras(
    model: 'keras.Model',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a Keras model with early stopping and optional checkpointing.

    Args:
        model: Compiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Maximum training epochs
        batch_size: Batch size
        early_stopping_patience: Epochs to wait before early stopping
        checkpoint_path: Path to save best model

    Returns:
        Training history dict
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
    ]

    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history.history


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance and return metrics.

    Args:
        model: Trained model (Keras or PyTorch)
        X_test: Test data
        y_test: Test labels
        class_names: Optional list of class names for report

    Returns:
        Dict with accuracy, per-class metrics, confusion matrix
    """
    if class_names is None:
        class_names = Gesture.names()

    # Get predictions
    if HAS_TF and isinstance(model, keras.Model):
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
    elif HAS_TORCH and isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            y_pred = model(X_tensor)
            y_pred_classes = torch.argmax(y_pred, dim=1).numpy()
    else:
        raise ValueError("Model must be Keras or PyTorch")

    # Compute metrics
    accuracy = np.mean(y_pred_classes == y_test)

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            per_class[name] = np.mean(y_pred_classes[mask] == i)
        else:
            per_class[name] = None

    # Confusion matrix
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_test, y_pred_classes):
        confusion[int(true), int(pred)] += 1

    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class,
        'confusion_matrix': confusion.tolist(),
        'class_names': class_names
    }


def save_model_for_inference(
    model,
    output_dir: str,
    model_name: str = 'gesture_model'
) -> Dict[str, str]:
    """
    Save model in formats suitable for deployment.

    Args:
        model: Trained model
        output_dir: Directory to save model files
        model_name: Base name for model files

    Returns:
        Dict of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    if HAS_TF and isinstance(model, keras.Model):
        # Save Keras model
        keras_path = output_dir / f'{model_name}.keras'
        model.save(keras_path)
        saved_paths['keras'] = str(keras_path)

        # Convert to TFLite for embedded deployment
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            tflite_path = output_dir / f'{model_name}.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            saved_paths['tflite'] = str(tflite_path)
        except Exception as e:
            print(f"TFLite conversion failed: {e}")

        # Convert to TensorFlow.js for web deployment
        try:
            import subprocess
            import shutil
            
            # Save as SavedModel format first (required for tfjs conversion)
            saved_model_path = output_dir / f'{model_name}_saved_model'
            model.save(saved_model_path, save_format='tf')
            
            # Convert to TensorFlow.js using tensorflowjs_converter
            tfjs_path = output_dir / f'{model_name}_tfjs'
            result = subprocess.run([
                'tensorflowjs_converter',
                '--input_format=tf_saved_model',
                '--output_format=tfjs_graph_model',
                str(saved_model_path),
                str(tfjs_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                saved_paths['tfjs'] = str(tfjs_path)
                print(f"TensorFlow.js model saved to: {tfjs_path}")
            else:
                print(f"TensorFlow.js conversion failed: {result.stderr}")
                # Fallback: save model config for manual conversion
                config_path = output_dir / f'{model_name}_config.json'
                import json
                with open(config_path, 'w') as f:
                    json.dump({
                        'model_type': 'finger_tracking' if len(model.outputs) > 1 else 'gesture',
                        'input_shape': list(model.input_shape),
                        'output_names': [o.name for o in model.outputs],
                        'conversion_command': f'tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {saved_model_path} {tfjs_path}'
                    }, f, indent=2)
                saved_paths['config'] = str(config_path)
                
            # Clean up SavedModel if tfjs conversion succeeded
            if 'tfjs' in saved_paths:
                shutil.rmtree(saved_model_path, ignore_errors=True)
                
        except FileNotFoundError:
            print("tensorflowjs_converter not found. Install with: pip install tensorflowjs")
        except Exception as e:
            print(f"TensorFlow.js conversion failed: {e}")

    elif HAS_TORCH and isinstance(model, nn.Module):
        # Save PyTorch model
        torch_path = output_dir / f'{model_name}.pt'
        torch.save(model.state_dict(), torch_path)
        saved_paths['pytorch'] = str(torch_path)

        # Export to ONNX for cross-platform deployment
        try:
            model.eval()
            dummy_input = torch.randn(1, 50, NUM_FEATURES)
            onnx_path = output_dir / f'{model_name}.onnx'
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            saved_paths['onnx'] = str(onnx_path)
        except Exception as e:
            print(f"ONNX export failed: {e}")

    return saved_paths


def create_finger_tracking_model_keras(
    window_size: int = 50,
    num_features: int = NUM_FEATURES,
    num_states: int = 3,  # extended, partial, flexed
    filters: Tuple[int, ...] = (32, 64, 64),
    kernel_size: int = 5,
    dropout: float = 0.3
) -> 'keras.Model':
    """
    Create a multi-output model for per-finger state prediction.
    
    Architecture:
        Input -> Shared CNN feature extraction -> 5 output heads (one per finger)
        Each head predicts 3-class state: extended(0), partial(1), flexed(2)
    
    Args:
        window_size: Number of timesteps per window
        num_features: Number of input features (9 for IMU)
        num_states: Number of states per finger (3: extended, partial, flexed)
        filters: Number of filters in each conv layer
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model with 5 outputs
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow")
    
    inputs = keras.Input(shape=(window_size, num_features))
    
    # Shared feature extraction
    x = layers.Conv1D(filters[0], kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Conv1D(filters[1], kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Conv1D(filters[2], kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    
    # Per-finger output heads - use consistent naming
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    outputs = []
    for finger in fingers:
        out = layers.Dense(16, activation='relu', name=f'{finger}_hidden')(x)
        out = layers.Dropout(dropout / 2)(out)
        out = layers.Dense(num_states, activation='softmax', name=f'{finger}_state')(out)
        outputs.append(out)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Multi-output loss and metrics - use output order matching
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=['sparse_categorical_crossentropy'] * 5,
        metrics=[['accuracy'] for _ in range(5)]
    )
    
    return model


def train_finger_tracking_model_keras(
    model: 'keras.Model',
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a multi-output finger tracking model.
    
    Args:
        model: Compiled multi-output Keras model
        X_train, X_val: Training and validation data (N, window_size, 9)
        y_train, y_val: Training and validation labels (N, 5) - one column per finger
        epochs: Maximum training epochs
        batch_size: Batch size
        early_stopping_patience: Epochs to wait before early stopping
        checkpoint_path: Path to save best model
    
    Returns:
        Training history dict
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    # Convert label array to list format for multi-output model
    # Model outputs are [thumb, index, middle, ring, pinky]
    y_train_list = [y_train[:, i] for i in range(5)]
    y_val_list = [y_val[:, i] for i in range(5)]
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
    ]
    
    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )
        )
    
    history = model.fit(
        X_train, y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history.history


def evaluate_finger_tracking_model(
    model: 'keras.Model',
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate finger tracking model performance.
    
    Args:
        model: Trained multi-output model
        X_test: Test data (N, window_size, 9)
        y_test: Test labels (N, 5) - one column per finger
    
    Returns:
        Dict with per-finger accuracy and confusion matrices
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")
    
    # Get predictions (returns list of arrays for multi-output model)
    predictions = model.predict(X_test, verbose=0)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    results = {
        'per_finger_accuracy': {},
        'per_finger_confusion': {},
        'overall_accuracy': 0.0
    }

    correct_total = 0
    total_predictions = 0

    for i, finger in enumerate(fingers):
        # predictions is a list of arrays [thumb_pred, index_pred, ...]
        y_pred = np.argmax(predictions[i], axis=1)
        y_true = y_test[:, i]
        
        # Accuracy
        accuracy = np.mean(y_pred == y_true)
        results['per_finger_accuracy'][finger] = float(accuracy)
        
        correct_total += np.sum(y_pred == y_true)
        total_predictions += len(y_true)
        
        # Confusion matrix (3x3 for extended/partial/flexed)
        confusion = np.zeros((3, 3), dtype=int)
        for true, pred in zip(y_true, y_pred):
            confusion[int(true), int(pred)] += 1
        results['per_finger_confusion'][finger] = confusion.tolist()
    
    results['overall_accuracy'] = float(correct_total / total_predictions)
    
    return results


if __name__ == '__main__':
    # Quick architecture test
    print("Testing model architectures...")

    if HAS_TF:
        print("\nKeras model:")
        keras_model = create_cnn_model_keras()
        keras_model.summary()

    if HAS_TORCH:
        print("\nPyTorch model:")
        torch_model = GestureCNN()
        print(torch_model)

        # Test forward pass
        x = torch.randn(4, 50, 9)
        y = torch_model(x)
        print(f"Input shape: {x.shape}, Output shape: {y.shape}")
