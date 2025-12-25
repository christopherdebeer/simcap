"""
Unsupervised and Semi-Supervised Learning Approaches for Finger Tracking

This module explores various approaches to leverage synthetic labeled data
for training models that can generalize to unlabeled real data.

Approaches:
1. Zero-shot transfer: Train on synthetic, evaluate on real
2. Self-supervised contrastive learning: Learn representations from temporal consistency
3. Domain adaptation: Align synthetic and real data distributions
4. Pseudo-labeling: Bootstrap labels using synthetic-trained model

Usage:
    python -m ml.unsupervised_learning --approach all
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class DataStats:
    """Statistics for a dataset."""
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    n_samples: int


def load_real_data(data_dir: str = 'data/GAMBIT', max_sessions: int = None) -> Tuple[np.ndarray, DataStats]:
    """
    Load real sensor data (unlabeled).

    Returns:
        (samples, stats) where samples is (N, 6) array of [mx, my, mz, ax, ay, az]
    """
    data_path = Path(data_dir)
    all_samples = []

    session_files = sorted(data_path.glob('*.json'))
    if max_sessions:
        session_files = session_files[:max_sessions]

    for f in session_files:
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

    samples = np.array(all_samples)

    stats = DataStats(
        mean=np.mean(samples, axis=0),
        std=np.std(samples, axis=0),
        min_val=np.min(samples, axis=0),
        max_val=np.max(samples, axis=0),
        n_samples=len(samples)
    )

    return samples, stats


def generate_synthetic_data(
    n_sessions: int = 100,
    samples_per_session: int = 500
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate labeled synthetic data.

    Returns:
        (samples, labels, finger_names) where:
        - samples is (N, 6) array
        - labels is (N, 5) array of finger states (0=extended, 1=partial, 2=flexed)
    """
    from ml.simulation import MagneticFieldSimulator, DEFAULT_MAGNET_CONFIG
    from ml.simulation.hand_model import POSE_TEMPLATES, FingerState

    all_samples = []
    all_labels = []
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    state_map = {FingerState.EXTENDED: 0, FingerState.PARTIAL: 1, FingerState.FLEXED: 2}

    poses = list(POSE_TEMPLATES.keys())
    samples_per_pose = samples_per_session // len(poses)

    for _ in range(n_sessions):
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

        for sample in session['samples']:
            features = [
                sample['mx_ut'], sample['my_ut'], sample['mz_ut'],
                sample.get('ax_g', 0), sample.get('ay_g', 0), sample.get('az_g', 0)
            ]
            all_samples.append(features)

        # Extract labels from session
        for label in session.get('labels', []):
            start_idx = label['start_sample']
            end_idx = label['end_sample']
            label_info = label.get('labels', {})
            fingers = label_info.get('fingers', {})

            label_vec = []
            for finger in finger_names:
                state_str = fingers.get(finger, 'unknown')
                if state_str == 'unknown':
                    state = 1  # Default to partial
                else:
                    state = state_map.get(FingerState(state_str), 1)
                label_vec.append(state)

            # Repeat label for all samples in range
            for _ in range(end_idx - start_idx):
                all_labels.append(label_vec)

    samples = np.array(all_samples)
    labels = np.array(all_labels[:len(samples)])  # Trim to match samples

    return samples, labels, finger_names


def normalize_data(data: np.ndarray, stats: DataStats = None) -> Tuple[np.ndarray, DataStats]:
    """Normalize data to zero mean and unit variance."""
    if stats is None:
        stats = DataStats(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0) + 1e-8,
            min_val=np.min(data, axis=0),
            max_val=np.max(data, axis=0),
            n_samples=len(data)
        )

    normalized = (data - stats.mean) / stats.std
    return normalized, stats


# ============================================================================
# Model Architectures
# ============================================================================

def build_encoder(input_dim: int = 6, latent_dim: int = 32) -> keras.Model:
    """Build a shared encoder for feature extraction."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    return keras.Model(inputs, x, name='encoder')


def build_finger_classifier(
    encoder: keras.Model,
    num_states: int = 3,
    freeze_encoder: bool = False
) -> keras.Model:
    """Build finger state classifier on top of encoder."""
    if freeze_encoder:
        encoder.trainable = False

    inputs = encoder.input
    x = encoder.output

    # Per-finger classification heads
    outputs = []
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        out = layers.Dense(16, activation='relu')(x)
        out = layers.Dropout(0.3)(out)
        out = layers.Dense(num_states, activation='softmax', name=f'{finger}_state')(out)
        outputs.append(out)

    model = keras.Model(inputs, outputs, name='finger_classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=['sparse_categorical_crossentropy'] * 5,
        metrics=[['accuracy'] for _ in range(5)]
    )

    return model


def build_contrastive_model(encoder: keras.Model, projection_dim: int = 16) -> keras.Model:
    """Build contrastive learning model (SimCLR-style)."""
    inputs = encoder.input
    features = encoder.output

    # Projection head
    x = layers.Dense(32, activation='relu')(features)
    projections = layers.Dense(projection_dim, name='projection')(x)

    return keras.Model(inputs, projections, name='contrastive_model')


def build_domain_discriminator(latent_dim: int = 32) -> keras.Model:
    """Build domain discriminator for adversarial domain adaptation."""
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    domain = layers.Dense(1, activation='sigmoid', name='domain')(x)

    model = keras.Model(inputs, domain, name='domain_discriminator')
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy')

    return model


# ============================================================================
# Training Approaches
# ============================================================================

class ZeroShotTransfer:
    """Train on synthetic data, evaluate on real (unsupervised baseline)."""

    def __init__(self, latent_dim: int = 32):
        self.encoder = build_encoder(latent_dim=latent_dim)
        self.classifier = None
        self.stats = None

    def train(
        self,
        syn_samples: np.ndarray,
        syn_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.2
    ):
        """Train classifier on synthetic data."""
        # Normalize using synthetic stats
        syn_norm, self.stats = normalize_data(syn_samples)

        # Build classifier
        self.classifier = build_finger_classifier(self.encoder)

        # Prepare labels for multi-output
        y_train = [syn_labels[:, i] for i in range(5)]

        # Train
        history = self.classifier.fit(
            syn_norm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        return history

    def evaluate_on_real(self, real_samples: np.ndarray) -> Dict:
        """
        Evaluate on real data using clustering metrics.

        Since real data is unlabeled, we use:
        - Prediction distribution analysis
        - Cluster separation metrics
        - Confidence scores
        """
        real_norm, _ = normalize_data(real_samples, self.stats)

        # Get predictions
        predictions = self.classifier.predict(real_norm, verbose=0)

        results = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for i, finger in enumerate(finger_names):
            pred_classes = np.argmax(predictions[i], axis=1)
            pred_probs = np.max(predictions[i], axis=1)

            # Class distribution
            unique, counts = np.unique(pred_classes, return_counts=True)
            dist = {int(c): int(n) for c, n in zip(unique, counts)}

            results[finger] = {
                'class_distribution': dist,
                'mean_confidence': float(np.mean(pred_probs)),
                'low_confidence_pct': float(np.mean(pred_probs < 0.5) * 100)
            }

        return results

    def get_embeddings(self, samples: np.ndarray) -> np.ndarray:
        """Get latent embeddings for visualization."""
        norm, _ = normalize_data(samples, self.stats)
        return self.encoder.predict(norm, verbose=0)


class ContrastiveLearning:
    """Self-supervised contrastive learning on real data."""

    def __init__(self, latent_dim: int = 32, projection_dim: int = 16, temperature: float = 0.1):
        self.encoder = build_encoder(latent_dim=latent_dim)
        self.contrastive_model = build_contrastive_model(self.encoder, projection_dim)
        self.temperature = temperature
        self.stats = None

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations to create positive pairs."""
        augmented = x.copy()

        # Add noise
        noise = np.random.normal(0, 0.1, x.shape)
        augmented = augmented + noise

        # Random scaling
        scale = np.random.uniform(0.9, 1.1, (x.shape[0], 1))
        augmented = augmented * scale

        return augmented

    def _contrastive_loss(self, z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
        """NT-Xent contrastive loss."""
        batch_size = tf.shape(z1)[0]

        # Normalize projections
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        # Similarity matrix
        z = tf.concat([z1, z2], axis=0)
        sim = tf.matmul(z, z, transpose_b=True) / self.temperature

        # Mask out self-similarity
        mask = tf.eye(2 * batch_size)
        sim = sim - mask * 1e9

        # Labels: positive pairs are (i, i+batch_size)
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, sim)
        return tf.reduce_mean(loss)

    def pretrain(
        self,
        real_samples: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256
    ):
        """Pre-train encoder using contrastive learning on real data."""
        real_norm, self.stats = normalize_data(real_samples)

        optimizer = keras.optimizers.Adam(1e-3)
        n_batches = len(real_norm) // batch_size

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(len(real_norm))

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                x = real_norm[batch_idx]

                # Create augmented views
                x1 = self._augment(x)
                x2 = self._augment(x)

                with tf.GradientTape() as tape:
                    z1 = self.contrastive_model(x1, training=True)
                    z2 = self.contrastive_model(x2, training=True)
                    loss = self._contrastive_loss(z1, z2)

                grads = tape.gradient(loss, self.contrastive_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.contrastive_model.trainable_variables))
                epoch_loss += loss.numpy()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def finetune_on_synthetic(
        self,
        syn_samples: np.ndarray,
        syn_labels: np.ndarray,
        epochs: int = 30,
        freeze_encoder: bool = False
    ):
        """Fine-tune classifier on synthetic data after contrastive pre-training."""
        syn_norm, _ = normalize_data(syn_samples, self.stats)

        # Build classifier with pre-trained encoder
        self.classifier = build_finger_classifier(self.encoder, freeze_encoder=freeze_encoder)

        y_train = [syn_labels[:, i] for i in range(5)]

        history = self.classifier.fit(
            syn_norm, y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        return history

    def get_embeddings(self, samples: np.ndarray) -> np.ndarray:
        """Get latent embeddings."""
        norm, _ = normalize_data(samples, self.stats)
        return self.encoder.predict(norm, verbose=0)


class DomainAdaptation:
    """Adversarial domain adaptation to align synthetic and real distributions."""

    def __init__(self, latent_dim: int = 32):
        self.encoder = build_encoder(latent_dim=latent_dim)
        self.classifier = None
        self.discriminator = build_domain_discriminator(latent_dim)
        self.stats = None

    def train(
        self,
        syn_samples: np.ndarray,
        syn_labels: np.ndarray,
        real_samples: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lambda_domain: float = 0.1
    ):
        """
        Train with adversarial domain adaptation.

        The encoder learns features that are:
        1. Discriminative for finger states (on synthetic data)
        2. Domain-invariant (confused about synthetic vs real)
        """
        # Normalize using combined stats
        combined = np.vstack([syn_samples, real_samples])
        _, self.stats = normalize_data(combined)

        syn_norm, _ = normalize_data(syn_samples, self.stats)
        real_norm, _ = normalize_data(real_samples, self.stats)

        # Build classifier
        self.classifier = build_finger_classifier(self.encoder)

        # Optimizers
        encoder_opt = keras.optimizers.Adam(1e-3)
        classifier_opt = keras.optimizers.Adam(1e-3)
        discriminator_opt = keras.optimizers.Adam(1e-4)

        n_syn = len(syn_norm)
        n_real = len(real_norm)
        n_batches = min(n_syn, n_real) // batch_size

        history = {'classifier_loss': [], 'domain_loss': [], 'accuracy': []}

        for epoch in range(epochs):
            syn_idx = np.random.permutation(n_syn)
            real_idx = np.random.permutation(n_real)

            epoch_cls_loss = 0
            epoch_dom_loss = 0

            for i in range(n_batches):
                # Get batches
                syn_batch = syn_norm[syn_idx[i*batch_size:(i+1)*batch_size]]
                real_batch = real_norm[real_idx[i*batch_size:(i+1)*batch_size]]
                syn_labels_batch = [syn_labels[syn_idx[i*batch_size:(i+1)*batch_size], j] for j in range(5)]

                # Train discriminator
                syn_features = self.encoder(syn_batch, training=False)
                real_features = self.encoder(real_batch, training=False)

                domain_features = np.vstack([syn_features, real_features])
                domain_labels = np.concatenate([np.zeros(batch_size), np.ones(batch_size)])

                d_loss = self.discriminator.train_on_batch(domain_features, domain_labels)

                # Train encoder + classifier with domain confusion
                with tf.GradientTape() as tape:
                    # Classification loss on synthetic
                    syn_features = self.encoder(syn_batch, training=True)
                    cls_preds = self.classifier(syn_batch, training=True)

                    cls_loss = sum(
                        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                        for y_true, y_pred in zip(syn_labels_batch, cls_preds)
                    )
                    cls_loss = tf.reduce_mean(cls_loss)

                    # Domain confusion loss (fool discriminator)
                    real_features = self.encoder(real_batch, training=True)
                    all_features = tf.concat([syn_features, real_features], axis=0)
                    domain_preds = self.discriminator(all_features, training=False)

                    # We want discriminator to predict 0.5 (confused)
                    target = 0.5 * tf.ones_like(domain_preds)
                    domain_loss = tf.keras.losses.binary_crossentropy(target, domain_preds)
                    domain_loss = tf.reduce_mean(domain_loss)

                    total_loss = cls_loss + lambda_domain * domain_loss

                # Update encoder and classifier
                trainable_vars = self.encoder.trainable_variables + self.classifier.trainable_variables
                grads = tape.gradient(total_loss, trainable_vars)
                encoder_opt.apply_gradients(zip(grads, trainable_vars))

                epoch_cls_loss += cls_loss.numpy()
                epoch_dom_loss += d_loss

            avg_cls_loss = epoch_cls_loss / n_batches
            avg_dom_loss = epoch_dom_loss / n_batches

            history['classifier_loss'].append(avg_cls_loss)
            history['domain_loss'].append(avg_dom_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Cls Loss: {avg_cls_loss:.4f}, Dom Loss: {avg_dom_loss:.4f}")

        return history

    def get_embeddings(self, samples: np.ndarray) -> np.ndarray:
        """Get latent embeddings."""
        norm, _ = normalize_data(samples, self.stats)
        return self.encoder.predict(norm, verbose=0)


class PseudoLabeling:
    """Use synthetic-trained model to generate pseudo-labels for real data."""

    def __init__(self, base_model: ZeroShotTransfer, confidence_threshold: float = 0.8):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.refined_classifier = None

    def generate_pseudo_labels(self, real_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate pseudo-labels for real data using confident predictions.

        Returns:
            (samples, labels, confidence) for samples above threshold
        """
        real_norm, _ = normalize_data(real_samples, self.base_model.stats)
        predictions = self.base_model.classifier.predict(real_norm, verbose=0)

        # Get max confidence per sample across all fingers
        all_confidences = np.stack([np.max(p, axis=1) for p in predictions], axis=1)
        min_confidence = np.min(all_confidences, axis=1)  # Worst finger confidence

        # Filter by confidence threshold
        confident_mask = min_confidence >= self.confidence_threshold

        confident_samples = real_samples[confident_mask]
        confident_labels = np.stack([np.argmax(p, axis=1) for p in predictions], axis=1)[confident_mask]
        confidences = min_confidence[confident_mask]

        return confident_samples, confident_labels, confidences

    def refine_on_pseudo_labels(
        self,
        syn_samples: np.ndarray,
        syn_labels: np.ndarray,
        real_samples: np.ndarray,
        epochs: int = 30,
        pseudo_weight: float = 0.5
    ):
        """
        Refine model using combination of synthetic and pseudo-labeled real data.
        """
        # Generate pseudo-labels
        pseudo_samples, pseudo_labels, _ = self.generate_pseudo_labels(real_samples)
        print(f"Generated {len(pseudo_samples)} pseudo-labeled samples ({100*len(pseudo_samples)/len(real_samples):.1f}%)")

        if len(pseudo_samples) < 100:
            print("Too few confident samples, skipping pseudo-label refinement")
            return None

        # Combine datasets with weighting
        n_syn = len(syn_samples)
        n_pseudo = len(pseudo_samples)

        # Sample synthetic data to balance
        syn_keep = int(n_pseudo / pseudo_weight)
        syn_idx = np.random.choice(n_syn, min(syn_keep, n_syn), replace=False)

        combined_samples = np.vstack([syn_samples[syn_idx], pseudo_samples])
        combined_labels = np.vstack([syn_labels[syn_idx], pseudo_labels])

        # Normalize
        combined_norm, _ = normalize_data(combined_samples, self.base_model.stats)

        # Build new classifier
        encoder = build_encoder()
        self.refined_classifier = build_finger_classifier(encoder)

        y_train = [combined_labels[:, i] for i in range(5)]

        history = self.refined_classifier.fit(
            combined_norm, y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        return history


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def evaluate_clustering_quality(
    embeddings: np.ndarray,
    predictions: List[np.ndarray]
) -> Dict:
    """
    Evaluate embedding quality using clustering metrics.

    Since we don't have ground truth labels for real data,
    we use internal clustering metrics.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # Combine predictions into single label
    combined_labels = np.stack([np.argmax(p, axis=1) for p in predictions], axis=1)
    # Create composite label (all 5 fingers)
    label_str = [''.join(map(str, row)) for row in combined_labels]
    unique_labels = {s: i for i, s in enumerate(set(label_str))}
    numeric_labels = np.array([unique_labels[s] for s in label_str])

    n_clusters = len(unique_labels)

    metrics = {
        'n_clusters': n_clusters,
    }

    if n_clusters > 1 and n_clusters < len(embeddings) - 1:
        metrics['silhouette_score'] = float(silhouette_score(embeddings, numeric_labels))
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, numeric_labels))

    return metrics


def visualize_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray] = None,
    output_path: str = 'images/embedding_comparison.png'
):
    """Create t-SNE visualization of embeddings from different approaches."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_approaches = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_approaches, figsize=(5 * n_approaches, 5))
    if n_approaches == 1:
        axes = [axes]

    for ax, (name, embeddings) in zip(axes, embeddings_dict.items()):
        # Subsample for visualization
        if len(embeddings) > 2000:
            idx = np.random.choice(len(embeddings), 2000, replace=False)
            embeddings = embeddings[idx]
            labels = labels_dict.get(name, None)
            if labels is not None:
                labels = labels[idx]
        else:
            labels = labels_dict.get(name, None) if labels_dict else None

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded_2d = tsne.fit_transform(embeddings)

        if labels is not None:
            scatter = ax.scatter(embedded_2d[:, 0], embedded_2d[:, 1],
                               c=labels, cmap='tab10', alpha=0.5, s=10)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedded_2d[:, 0], embedded_2d[:, 1], alpha=0.5, s=10)

        ax.set_title(name)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved embedding visualization to {output_path}")


def compare_approaches(
    syn_samples: np.ndarray,
    syn_labels: np.ndarray,
    real_samples: np.ndarray,
    approaches: List[str] = ['zero_shot', 'contrastive', 'domain_adaptation']
) -> Dict:
    """
    Compare different learning approaches.

    Returns metrics for each approach.
    """
    results = {}
    embeddings_dict = {}

    print("\n" + "="*70)
    print("APPROACH COMPARISON")
    print("="*70)

    # 1. Zero-shot transfer baseline
    if 'zero_shot' in approaches:
        print("\n--- Zero-Shot Transfer ---")
        zst = ZeroShotTransfer()
        zst.train(syn_samples, syn_labels, epochs=30)

        eval_results = zst.evaluate_on_real(real_samples)
        results['zero_shot'] = eval_results
        embeddings_dict['Zero-Shot'] = zst.get_embeddings(real_samples)

        print("\nPrediction distribution on real data:")
        for finger, stats in eval_results.items():
            print(f"  {finger}: {stats['class_distribution']}, conf={stats['mean_confidence']:.2f}")

    # 2. Contrastive pre-training + fine-tuning
    if 'contrastive' in approaches:
        print("\n--- Contrastive Learning ---")
        cl = ContrastiveLearning()

        print("Pre-training on real data...")
        cl.pretrain(real_samples, epochs=50)

        print("Fine-tuning on synthetic data...")
        cl.finetune_on_synthetic(syn_samples, syn_labels, epochs=20, freeze_encoder=False)

        embeddings_dict['Contrastive'] = cl.get_embeddings(real_samples)

        # Evaluate clustering quality
        real_norm, _ = normalize_data(real_samples, cl.stats)
        predictions = cl.classifier.predict(real_norm, verbose=0)
        cluster_metrics = evaluate_clustering_quality(embeddings_dict['Contrastive'], predictions)
        results['contrastive'] = cluster_metrics
        print(f"Clustering quality: {cluster_metrics}")

    # 3. Domain adaptation
    if 'domain_adaptation' in approaches:
        print("\n--- Domain Adaptation ---")
        da = DomainAdaptation()
        da.train(syn_samples, syn_labels, real_samples, epochs=30, lambda_domain=0.1)

        embeddings_dict['Domain-Adapted'] = da.get_embeddings(real_samples)

        # Evaluate
        real_norm, _ = normalize_data(real_samples, da.stats)
        predictions = da.classifier.predict(real_norm, verbose=0)
        cluster_metrics = evaluate_clustering_quality(embeddings_dict['Domain-Adapted'], predictions)
        results['domain_adaptation'] = cluster_metrics
        print(f"Clustering quality: {cluster_metrics}")

    # Visualize embeddings
    if len(embeddings_dict) > 0:
        visualize_embeddings(embeddings_dict)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Unsupervised learning approaches')
    parser.add_argument('--approach', type=str, default='all',
                       choices=['zero_shot', 'contrastive', 'domain_adaptation', 'pseudo_label', 'all'],
                       help='Which approach to run')
    parser.add_argument('--n-synthetic', type=int, default=50,
                       help='Number of synthetic sessions to generate')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')

    args = parser.parse_args()

    print("="*70)
    print("UNSUPERVISED LEARNING FOR FINGER TRACKING")
    print("="*70)

    # Load real data
    print("\nLoading real data...")
    real_samples, real_stats = load_real_data()
    print(f"Loaded {real_stats.n_samples} real samples")
    print(f"Real data stats: mean={real_stats.mean[:3]}, std={real_stats.std[:3]}")

    # Generate synthetic data
    print(f"\nGenerating {args.n_synthetic} synthetic sessions...")
    syn_samples, syn_labels, finger_names = generate_synthetic_data(n_sessions=args.n_synthetic)
    print(f"Generated {len(syn_samples)} synthetic samples with labels")

    # Run comparison
    if args.approach == 'all':
        approaches = ['zero_shot', 'contrastive', 'domain_adaptation']
    else:
        approaches = [args.approach]

    results = compare_approaches(syn_samples, syn_labels, real_samples, approaches)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for approach, metrics in results.items():
        print(f"\n{approach}:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
