#!/usr/bin/env python3
"""
Physics-to-ML Pipeline: From Magnetic Dipole Model to Improved Classifier

This pipeline:
1. Evaluates physics model classification accuracy on observed data
2. Generates synthetic data for all 32 finger state combinations
3. Trains improved classifier on augmented dataset
4. Compares performance with original model

Author: Claude
Date: January 2026
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


# Physical constants
MU_0_OVER_4PI = 1e-7  # T·m/A


class PhysicsBasedClassifier:
    """
    Uses fitted physics model to classify finger states.

    Two approaches:
    1. Template matching: Find closest predicted field to observation
    2. Inverse model: Optimize finger states to match observation
    """

    def __init__(self, physics_params: np.ndarray):
        """
        Args:
            physics_params: Flattened physics model parameters from optimization
        """
        self.params = physics_params
        self._unpack_params()

    def _unpack_params(self):
        """Unpack flattened parameters."""
        self.pos_ext = self.params[0:15].reshape(5, 3)
        self.pos_flex = self.params[15:30].reshape(5, 3)
        self.dipoles = self.params[30:45].reshape(5, 3)
        self.baseline = self.params[45:48]

    def dipole_field_vectorized(self, r_batch: np.ndarray, m_batch: np.ndarray) -> np.ndarray:
        """Compute dipole fields."""
        r_mag = np.linalg.norm(r_batch, axis=-1, keepdims=True)
        r_mag = np.maximum(r_mag, 1e-6)
        r_hat = r_batch / r_mag
        m_dot_r = np.sum(r_hat * m_batch[None, :, :], axis=-1, keepdims=True)
        B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - m_batch[None, :, :]) / (r_mag ** 3)
        return B * 1e6

    def predict_field(self, finger_states: np.ndarray) -> np.ndarray:
        """
        Predict magnetic field for given finger states.

        Args:
            finger_states: [N_samples, 5] binary states (0=extended, 1=flexed)

        Returns:
            Predicted fields [N_samples, 3] in μT
        """
        N_samples = finger_states.shape[0]

        # Interpolate positions
        positions = (
            self.pos_ext[None, :, :] +
            finger_states[:, :, None] * (self.pos_flex - self.pos_ext)[None, :, :]
        )

        # Position vectors from magnets to sensor
        r_batch = -positions

        # Compute fields
        B_magnets = self.dipole_field_vectorized(r_batch, self.dipoles)

        # Sum over magnets and add baseline
        B_total = np.sum(B_magnets, axis=1) + self.baseline[None, :]

        return B_total

    def predict_class_template_matching(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify observations using template matching.

        For each observation, find the finger state combo whose predicted field
        is closest to the observation.

        Args:
            observations: [N_samples, 3] measured fields in μT

        Returns:
            predicted_states: [N_samples, 5] binary finger states
            distances: [N_samples] distances to closest template
        """
        N_obs = observations.shape[0]

        # Generate all 32 possible finger state combinations
        all_states = np.array([
            [(i >> (4-j)) & 1 for j in range(5)]
            for i in range(32)
        ], dtype=float)

        # Predict fields for all states
        all_predictions = self.predict_field(all_states)  # [32, 3]

        # For each observation, find closest prediction
        predicted_states = np.zeros((N_obs, 5))
        distances = np.zeros(N_obs)

        for i, obs in enumerate(observations):
            # Compute distances to all templates
            dists = np.linalg.norm(all_predictions - obs[None, :], axis=1)

            # Find closest
            best_idx = np.argmin(dists)
            predicted_states[i] = all_states[best_idx]
            distances[i] = dists[best_idx]

        return predicted_states, distances

    def evaluate_classification_accuracy(
        self,
        observations: np.ndarray,
        true_states: np.ndarray
    ) -> Dict:
        """
        Evaluate classification accuracy on labeled data.

        Args:
            observations: [N_samples, 3] measured fields
            true_states: [N_samples, 5] true binary finger states

        Returns:
            Dictionary with accuracy metrics
        """
        predicted_states, distances = self.predict_class_template_matching(observations)

        # Per-sample accuracy (all 5 fingers correct)
        exact_match = np.all(predicted_states == true_states, axis=1)
        exact_accuracy = np.mean(exact_match)

        # Per-finger accuracy
        finger_accuracy = np.mean(predicted_states == true_states, axis=0)

        # Hamming distance (how many fingers wrong on average)
        hamming_dist = np.sum(predicted_states != true_states, axis=1)
        mean_hamming = np.mean(hamming_dist)

        return {
            'exact_match_accuracy': float(exact_accuracy),
            'per_finger_accuracy': finger_accuracy.tolist(),
            'mean_hamming_distance': float(mean_hamming),
            'mean_template_distance_ut': float(np.mean(distances)),
            'predictions': predicted_states,
            'distances': distances
        }


class SyntheticDataGenerator:
    """
    Generates synthetic magnetometer data using fitted physics model.
    """

    def __init__(self, physics_params: np.ndarray):
        """
        Args:
            physics_params: Flattened physics model parameters
        """
        self.classifier = PhysicsBasedClassifier(physics_params)

    def generate_synthetic_dataset(
        self,
        n_samples_per_combo: int = 100,
        noise_std_ut: float = 10.0,
        position_variation_m: float = 0.005,
        include_combos: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic dataset for all or specific finger state combos.

        Args:
            n_samples_per_combo: Number of samples per combo
            noise_std_ut: Standard deviation of measurement noise (μT)
            position_variation_m: Variation in magnet positions (meters)
            include_combos: List of combo codes to generate (None = all 32)

        Returns:
            fields: [N_total, 3] synthetic magnetometer readings
            states: [N_total, 5] finger states
            combo_codes: [N_total] combo code strings
        """
        # Generate all 32 combos if not specified
        if include_combos is None:
            all_states = np.array([
                [(i >> (4-j)) & 1 for j in range(5)]
                for i in range(32)
            ], dtype=float)

            include_combos = [
                ''.join(['f' if (i >> (4-j)) & 1 else 'e' for j in range(5)])
                for i in range(32)
            ]
        else:
            # Convert combo codes to states
            all_states = []
            for combo in include_combos:
                state = [1.0 if c == 'f' else 0.0 for c in combo]
                all_states.append(state)
            all_states = np.array(all_states)

        n_combos = len(include_combos)
        n_total = n_combos * n_samples_per_combo

        fields = np.zeros((n_total, 3))
        states = np.zeros((n_total, 5))
        combo_codes = []

        idx = 0
        for combo_idx, state in enumerate(all_states):
            for sample_idx in range(n_samples_per_combo):
                # Generate state with slight variations (simulate measurement variability)
                state_varied = state.copy()

                # Add position variation by slightly modulating state (simulate hand tremor)
                if position_variation_m > 0:
                    # Small random offsets in the continuous state space
                    state_jitter = np.random.randn(5) * 0.05
                    state_varied = np.clip(state + state_jitter, 0, 1)

                # Predict field
                field = self.classifier.predict_field(state_varied[None, :])[0]

                # Add measurement noise
                if noise_std_ut > 0:
                    field += np.random.randn(3) * noise_std_ut

                fields[idx] = field
                states[idx] = state  # Store original discrete state
                combo_codes.append(include_combos[combo_idx])
                idx += 1

        return fields, states, combo_codes


class ImprovedMLClassifier:
    """
    Train improved ML classifier on augmented dataset.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest' or 'mlp'
        """
        self.model_type = model_type

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict:
        """
        Train classifier on augmented dataset.

        Args:
            X_train: [N_train, 3] magnetometer readings
            y_train: [N_train, 5] finger states (or [N_train] combo indices)
            X_val: Optional validation set
            y_val: Optional validation labels

        Returns:
            Training metrics
        """
        # Convert multi-label to single combo index for sklearn
        if len(y_train.shape) == 2 and y_train.shape[1] == 5:
            y_train_idx = self._states_to_indices(y_train)
        else:
            y_train_idx = y_train

        t0 = time.time()
        self.model.fit(X_train, y_train_idx)
        elapsed = time.time() - t0

        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train_idx, train_pred)

        metrics = {
            'training_time': elapsed,
            'train_accuracy': train_acc,
        }

        # Validation accuracy
        if X_val is not None and y_val is not None:
            if len(y_val.shape) == 2 and y_val.shape[1] == 5:
                y_val_idx = self._states_to_indices(y_val)
            else:
                y_val_idx = y_val

            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val_idx, val_pred)
            metrics['val_accuracy'] = val_acc

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict finger states.

        Args:
            X: [N, 3] magnetometer readings

        Returns:
            [N, 5] predicted finger states
        """
        indices = self.model.predict(X)
        return self._indices_to_states(indices)

    def _states_to_indices(self, states: np.ndarray) -> np.ndarray:
        """Convert [N, 5] binary states to [N] combo indices."""
        indices = np.zeros(len(states), dtype=int)
        for i, state in enumerate(states):
            idx = 0
            for j in range(5):
                if state[j] > 0.5:
                    idx |= (1 << (4-j))
            indices[i] = idx
        return indices

    def _indices_to_states(self, indices: np.ndarray) -> np.ndarray:
        """Convert [N] combo indices to [N, 5] binary states."""
        states = np.zeros((len(indices), 5))
        for i, idx in enumerate(indices):
            for j in range(5):
                states[i, j] = 1.0 if (idx >> (4-j)) & 1 else 0.0
        return states

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate on test set.

        Args:
            X_test: [N_test, 3] magnetometer readings
            y_test: [N_test, 5] true finger states

        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X_test)

        # Exact match accuracy
        exact_match = np.all(y_pred == y_test, axis=1)
        exact_acc = np.mean(exact_match)

        # Per-finger accuracy
        finger_acc = np.mean(y_pred == y_test, axis=0)

        # Hamming distance
        hamming = np.sum(y_pred != y_test, axis=1)
        mean_hamming = np.mean(hamming)

        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        return {
            'exact_match_accuracy': float(exact_acc),
            'per_finger_accuracy': {
                finger_names[i]: float(finger_acc[i])
                for i in range(5)
            },
            'mean_hamming_distance': float(mean_hamming),
            'predictions': y_pred,
        }


def main():
    """Run complete physics-to-ML pipeline."""
    print("="*70)
    print("PHYSICS-TO-ML PIPELINE")
    print("="*70)

    # ========================================================================
    # STEP 1: Load physics model parameters
    # ========================================================================
    print("\n[1/4] Loading physics model parameters...")

    physics_results_path = Path("ml/analysis/physics/gpu_physics_optimization_results.json")
    with open(physics_results_path) as f:
        physics_results = json.load(f)

    physics_params = np.array(physics_results['parameters'])
    print(f"  ✓ Loaded {len(physics_params)} parameters")

    # ========================================================================
    # STEP 2: Evaluate physics model classification accuracy
    # ========================================================================
    print("\n[2/4] Evaluating physics model classification accuracy...")

    # Load observed data
    data_path = Path(".worktrees/data/GAMBIT/2025-12-31T14_06_18.270Z.json")
    with open(data_path) as f:
        session = json.load(f)

    # Extract labeled observations
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    index_to_combo = {}
    for lbl in session.get('labels', []):
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        if not fingers:
            continue

        combo = ''.join([
            'e' if fingers.get(f) == 'extended' else 'f' if fingers.get(f) == 'flexed' else '?'
            for f in FINGER_ORDER
        ])

        if '?' in combo:
            continue

        for i in range(start, end):
            index_to_combo[i] = combo

    # Collect all labeled samples
    observations = []
    true_states = []

    for i, sample in enumerate(session.get('samples', [])):
        if i not in index_to_combo:
            continue

        combo = index_to_combo[i]

        # Magnetometer reading
        mag = np.array([
            sample.get('iron_mx', sample.get('mx_ut', 0)),
            sample.get('iron_my', sample.get('my_ut', 0)),
            sample.get('iron_mz', sample.get('mz_ut', 0))
        ])

        # Finger state
        state = np.array([1.0 if c == 'f' else 0.0 for c in combo])

        observations.append(mag)
        true_states.append(state)

    observations = np.array(observations)
    true_states = np.array(true_states)

    print(f"  ✓ Loaded {len(observations)} labeled samples")

    # Evaluate physics classifier
    physics_classifier = PhysicsBasedClassifier(physics_params)
    physics_metrics = physics_classifier.evaluate_classification_accuracy(
        observations, true_states
    )

    print(f"\n  Physics Model Classification Results:")
    print(f"    Exact match accuracy: {physics_metrics['exact_match_accuracy']*100:.1f}%")
    print(f"    Mean Hamming distance: {physics_metrics['mean_hamming_distance']:.2f} fingers")
    print(f"    Per-finger accuracy:")
    for i, acc in enumerate(physics_metrics['per_finger_accuracy']):
        print(f"      {FINGER_ORDER[i]:>8}: {acc*100:.1f}%")

    # ========================================================================
    # STEP 3: Generate synthetic data
    # ========================================================================
    print(f"\n[3/4] Generating synthetic data...")

    generator = SyntheticDataGenerator(physics_params)

    # Generate for all 32 combos
    synthetic_fields, synthetic_states, synthetic_combos = generator.generate_synthetic_dataset(
        n_samples_per_combo=200,
        noise_std_ut=15.0,  # Realistic sensor noise
        position_variation_m=0.003  # Small hand tremor
    )

    print(f"  ✓ Generated {len(synthetic_fields)} synthetic samples")
    print(f"    Combos: 32 (all possible)")
    print(f"    Samples per combo: 200")
    print(f"    Noise level: 15 μT")

    # Combine with real observations
    all_fields = np.vstack([observations, synthetic_fields])
    all_states = np.vstack([true_states, synthetic_states])

    # Mark which samples are real vs synthetic
    is_real = np.zeros(len(all_fields), dtype=bool)
    is_real[:len(observations)] = True

    print(f"\n  Combined Dataset:")
    print(f"    Real samples: {np.sum(is_real)}")
    print(f"    Synthetic samples: {np.sum(~is_real)}")
    print(f"    Total: {len(all_fields)}")

    # ========================================================================
    # STEP 4: Train improved classifier
    # ========================================================================
    print(f"\n[4/4] Training improved classifier...")

    # Split data: use real data for testing, train on mix of real + synthetic
    X_real = observations
    y_real = true_states

    # For training, use 80% real + all synthetic
    n_train_real = int(0.8 * len(X_real))
    indices = np.random.permutation(len(X_real))
    train_indices = indices[:n_train_real]
    test_indices = indices[n_train_real:]

    X_train_real = X_real[train_indices]
    y_train_real = y_real[train_indices]

    X_test = X_real[test_indices]
    y_test = y_real[test_indices]

    # Combine real + synthetic for training
    X_train = np.vstack([X_train_real, synthetic_fields])
    y_train = np.vstack([y_train_real, synthetic_states])

    print(f"\n  Dataset splits:")
    print(f"    Train (real): {len(X_train_real)}")
    print(f"    Train (synthetic): {len(synthetic_fields)}")
    print(f"    Train (total): {len(X_train)}")
    print(f"    Test: {len(X_test)}")

    # Train baseline model (real data only)
    print(f"\n  Training baseline model (real data only)...")
    baseline_model = ImprovedMLClassifier(model_type='random_forest')
    baseline_metrics = baseline_model.train(X_train_real, y_train_real)
    baseline_eval = baseline_model.evaluate(X_test, y_test)

    print(f"    Train accuracy: {baseline_metrics['train_accuracy']*100:.1f}%")
    print(f"    Test accuracy: {baseline_eval['exact_match_accuracy']*100:.1f}%")
    print(f"    Training time: {baseline_metrics['training_time']:.2f}s")

    # Train augmented model (real + synthetic)
    print(f"\n  Training augmented model (real + synthetic)...")
    augmented_model = ImprovedMLClassifier(model_type='random_forest')
    augmented_metrics = augmented_model.train(X_train, y_train)
    augmented_eval = augmented_model.evaluate(X_test, y_test)

    print(f"    Train accuracy: {augmented_metrics['train_accuracy']*100:.1f}%")
    print(f"    Test accuracy: {augmented_eval['exact_match_accuracy']*100:.1f}%")
    print(f"    Training time: {augmented_metrics['training_time']:.2f}s")

    # ========================================================================
    # RESULTS COMPARISON
    # ========================================================================
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Model':<30} {'Test Accuracy':>15} {'Hamming Dist':>15}")
    print("-"*60)
    print(f"{'Physics Model (template)':<30} {physics_metrics['exact_match_accuracy']*100:>14.1f}% {physics_metrics['mean_hamming_distance']:>14.2f}")
    print(f"{'ML Baseline (real only)':<30} {baseline_eval['exact_match_accuracy']*100:>14.1f}% {baseline_eval['mean_hamming_distance']:>14.2f}")
    print(f"{'ML Augmented (real+synthetic)':<30} {augmented_eval['exact_match_accuracy']*100:>14.1f}% {augmented_eval['mean_hamming_distance']:>14.2f}")

    improvement = augmented_eval['exact_match_accuracy'] - baseline_eval['exact_match_accuracy']
    print(f"\nImprovement from synthetic data: {improvement*100:+.1f}%")

    # Per-finger comparison
    print(f"\nPer-Finger Accuracy:")
    print(f"{'Finger':<10} {'Physics':>10} {'Baseline':>10} {'Augmented':>10}")
    print("-"*45)
    for i, finger in enumerate(FINGER_ORDER):
        phys_acc = physics_metrics['per_finger_accuracy'][i] * 100
        base_acc = baseline_eval['per_finger_accuracy'][finger] * 100
        aug_acc = augmented_eval['per_finger_accuracy'][finger] * 100
        print(f"{finger:<10} {phys_acc:>9.1f}% {base_acc:>9.1f}% {aug_acc:>9.1f}%")

    # Save results
    output = {
        'physics_classifier': physics_metrics,
        'ml_baseline': {
            'train_metrics': baseline_metrics,
            'test_metrics': baseline_eval
        },
        'ml_augmented': {
            'train_metrics': augmented_metrics,
            'test_metrics': augmented_eval
        },
        'dataset_info': {
            'n_real_samples': len(observations),
            'n_synthetic_samples': len(synthetic_fields),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_combos_observed': len(set(index_to_combo.values())),
            'n_combos_generated': 32
        }
    }

    # Remove numpy arrays from output (not JSON serializable)
    for key in ['predictions', 'distances']:
        if key in output['physics_classifier']:
            del output['physics_classifier'][key]
    for model_key in ['ml_baseline', 'ml_augmented']:
        if 'predictions' in output[model_key]['test_metrics']:
            del output[model_key]['test_metrics']['predictions']

    output_path = Path("ml/analysis/physics/physics_to_ml_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return output


if __name__ == '__main__':
    results = main()
