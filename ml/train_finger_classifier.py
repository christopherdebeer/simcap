"""
Finger State Classifier Training

Trains a classifier to predict finger states (5 fingers × 3 states each)
from magnetometer readings. Uses ground truth wizard data and optionally
augments with synthetic data.

The model predicts a 5-character code like "22000" where:
- 0 = extended
- 1 = partial (half-flexed)
- 2 = flexed

Usage:
    python -m ml.train_finger_classifier --real-only
    python -m ml.train_finger_classifier --with-synthetic
    python -m ml.train_finger_classifier --eval-only
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


def load_session(path: Path) -> Dict:
    """Load a session JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_features_and_labels(
    session: Dict,
    use_raw: bool = True,
    use_calibrated: bool = False,
    use_residual: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features and labels from a session.

    Args:
        session: Session dict with samples and labels
        use_raw: Include raw mx, my, mz
        use_calibrated: Include calibrated magnetometer (if available)
        use_residual: Include residual magnetometer (if available)

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: List of finger codes (e.g., "22000")
    """
    samples = session.get('samples', [])
    labels = session.get('labels', [])

    # Build feature arrays
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])

    # Start with magnetometer
    features = [mx, my, mz]

    # Add calibrated if available and requested
    if use_calibrated and 'calibrated_mx' in samples[0]:
        cal_mx = np.array([s.get('calibrated_mx', 0) for s in samples])
        cal_my = np.array([s.get('calibrated_my', 0) for s in samples])
        cal_mz = np.array([s.get('calibrated_mz', 0) for s in samples])
        features.extend([cal_mx, cal_my, cal_mz])

    # Add residual if available and requested
    if use_residual and 'residual_mx' in samples[0]:
        res_mx = np.array([s.get('residual_mx', 0) for s in samples])
        res_my = np.array([s.get('residual_my', 0) for s in samples])
        res_mz = np.array([s.get('residual_mz', 0) for s in samples])
        features.extend([res_mx, res_my, res_mz])

    # Compute magnitude as additional feature
    mag = np.sqrt(mx**2 + my**2 + mz**2)
    features.append(mag)

    X = np.column_stack(features)

    # Build label array
    y = [''] * len(samples)

    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        if not fingers:
            continue

        # Convert to code
        code = ''
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            state = fingers.get(f, 'unknown')
            if state == 'extended':
                code += '0'
            elif state == 'partial':
                code += '1'
            elif state == 'flexed':
                code += '2'
            else:
                code += '?'

        if '?' not in code:
            for i in range(start, min(end, len(y))):
                y[i] = code

    # Filter to labeled samples only
    labeled_mask = np.array([yi != '' for yi in y])
    X = X[labeled_mask]
    y = [yi for yi in y if yi != '']

    return X, y


def prepare_dataset(
    real_sessions: List[Path],
    synthetic_sessions: Optional[List[Path]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict]:
    """
    Prepare train/test datasets from sessions.

    Returns:
        X_train, X_test, y_train, y_test, scaler, class_info
    """
    all_X = []
    all_y = []

    # Load real sessions
    print("\nLoading real sessions...")
    for path in real_sessions:
        session = load_session(path)
        X, y = extract_features_and_labels(session)
        all_X.append(X)
        all_y.extend(y)
        print(f"  {path.name}: {len(y)} samples")

    # Load synthetic sessions if provided
    if synthetic_sessions:
        print("\nLoading synthetic sessions...")
        for path in synthetic_sessions:
            session = load_session(path)
            X, y = extract_features_and_labels(session)
            all_X.append(X)
            all_y.extend(y)
            print(f"  {path.name}: {len(y)} samples")

    X = np.vstack(all_X)
    y = np.array(all_y)

    print(f"\nTotal samples: {len(y)}")

    # Class distribution
    class_counts = defaultdict(int)
    for yi in y:
        class_counts[yi] += 1

    print("\nClass distribution:")
    for code, count in sorted(class_counts.items()):
        print(f"  {code}: {count} ({count/len(y)*100:.1f}%)")

    # Encode labels as integers
    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_encoded = np.array([class_to_idx[yi] for yi in y])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_info = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'n_classes': len(classes)
    }

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler, class_info


def train_sklearn_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_info: Dict
) -> Dict:
    """
    Train and evaluate sklearn classifiers.
    """
    results = {}

    # 1. K-Nearest Neighbors
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS")
    print("=" * 60)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_info['classes'],
        zero_division=0
    ))

    results['knn'] = {
        'model': knn,
        'accuracy': accuracy,
        'predictions': y_pred
    }

    # 2. Support Vector Machine
    print("\n" + "=" * 60)
    print("SUPPORT VECTOR MACHINE")
    print("=" * 60)

    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_info['classes'],
        zero_division=0
    ))

    results['svm'] = {
        'model': svm,
        'accuracy': accuracy,
        'predictions': y_pred
    }

    # 3. Random Forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_info['classes'],
        zero_division=0
    ))

    # Feature importance
    print("\nFeature Importance:")
    feature_names = ['mx', 'my', 'mz', 'magnitude']
    for name, importance in zip(feature_names, rf.feature_importances_):
        print(f"  {name}: {importance:.4f}")

    results['rf'] = {
        'model': rf,
        'accuracy': accuracy,
        'predictions': y_pred,
        'feature_importance': dict(zip(feature_names, rf.feature_importances_))
    }

    return results


def train_neural_network(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_info: Dict,
    epochs: int = 50
) -> Dict:
    """
    Train a simple neural network classifier.
    """
    if not HAS_TF:
        print("TensorFlow not available, skipping neural network")
        return {}

    print("\n" + "=" * 60)
    print("NEURAL NETWORK")
    print("=" * 60)

    n_features = X_train.shape[1]
    n_classes = class_info['n_classes']

    # Simple MLP
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_info['classes'],
        zero_division=0
    ))

    return {
        'model': model,
        'accuracy': accuracy,
        'history': history.history,
        'predictions': y_pred
    }


def save_best_model(
    results: Dict,
    scaler: StandardScaler,
    class_info: Dict,
    output_dir: Path
):
    """Save the best performing model."""
    import pickle

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
    best_result = results[best_name]

    print(f"\nBest model: {best_name} (accuracy: {best_result['accuracy']:.4f})")

    # Save model
    if best_name == 'nn' and HAS_TF:
        model_path = output_dir / 'finger_classifier.h5'
        best_result['model'].save(model_path)
        print(f"Saved Keras model to: {model_path}")
    else:
        model_path = output_dir / f'finger_classifier_{best_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_result['model'], f)
        print(f"Saved sklearn model to: {model_path}")

    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {scaler_path}")

    # Save class info
    info_path = output_dir / 'class_info.json'
    with open(info_path, 'w') as f:
        json.dump({
            'classes': class_info['classes'],
            'class_to_idx': class_info['class_to_idx'],
            'best_model': best_name,
            'accuracy': best_result['accuracy']
        }, f, indent=2)
    print(f"Saved class info to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Train finger state classifier')
    parser.add_argument('--real-only', action='store_true',
                       help='Use only real wizard data')
    parser.add_argument('--with-synthetic', action='store_true',
                       help='Include synthetic data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Neural network epochs')
    parser.add_argument('--output-dir', type=Path, default=Path('ml/models/finger_classifier'),
                       help='Output directory for models')

    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("Error: scikit-learn is required")
        return

    print("=" * 80)
    print("FINGER STATE CLASSIFIER TRAINING")
    print("=" * 80)

    # Find real sessions with wizard labels
    data_dir = Path('data/GAMBIT')
    real_sessions = []

    if data_dir.exists():
        for path in data_dir.glob('*.json'):
            session = load_session(path)
            labels = session.get('labels', [])
            # Check if has finger labels
            has_fingers = any(
                label.get('labels', label).get('fingers')
                for label in labels
            )
            if has_fingers and len(labels) > 5:
                real_sessions.append(path)

    print(f"\nFound {len(real_sessions)} real sessions with finger labels")

    if not real_sessions:
        print("No labeled sessions found. Please run the wizard first.")
        return

    # Find synthetic sessions
    synthetic_sessions = []
    if args.with_synthetic:
        synthetic_path = Path('ml/synthetic_balanced_dataset.json')
        if synthetic_path.exists():
            synthetic_sessions.append(synthetic_path)
            print(f"Using synthetic data: {synthetic_path}")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, class_info = prepare_dataset(
        real_sessions,
        synthetic_sessions if args.with_synthetic else None
    )

    # Train sklearn models
    results = train_sklearn_models(X_train, X_test, y_train, y_test, class_info)

    # Train neural network
    if HAS_TF:
        nn_result = train_neural_network(
            X_train, X_test, y_train, y_test, class_info,
            epochs=args.epochs
        )
        if nn_result:
            results['nn'] = nn_result

    # Save best model
    save_best_model(results, scaler, class_info, args.output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nModel Accuracies:")
    for name, result in sorted(results.items(), key=lambda x: -x[1].get('accuracy', 0)):
        print(f"  {name}: {result['accuracy']:.4f}")


if __name__ == '__main__':
    main()
