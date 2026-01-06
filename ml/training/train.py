#!/usr/bin/env python3
"""
SIMCAP Training Script

Train gesture classification models on labeled GAMBIT data.

Usage:
    python -m ml.train --data-dir data/GAMBIT --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

from ml.schema import Gesture
from ml.data_loader import GambitDataset
from ml.model import (
    create_cnn_model_keras, train_model_keras,
    evaluate_model, save_model_for_inference,
    HAS_TF, HAS_TORCH
)

# Import clustering functions (optional dependency)
try:
    from ml.cluster import (
        load_unlabeled_windows, extract_features_from_windows,
        cluster_kmeans, cluster_dbscan, compute_cluster_metrics,
        analyze_clusters, reduce_dimensions, ClusterResult,
        save_cluster_results, create_label_templates,
        HAS_SKLEARN
    )
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    HAS_SKLEARN = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SIMCAP gesture classification model'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/GAMBIT',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='ml/models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--window-size', type=int, default=50,
        help='Window size in samples (50 = 1 second at 50Hz)'
    )
    parser.add_argument(
        '--stride', type=int, default=25,
        help='Window stride (25 = 50% overlap)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--framework', type=str, default='keras',
        choices=['keras', 'pytorch'],
        help='ML framework to use'
    )
    parser.add_argument(
        '--summary-only', action='store_true',
        help='Only print dataset summary, do not train'
    )
    
    # Clustering options
    parser.add_argument(
        '--cluster-only', action='store_true',
        help='Perform unsupervised clustering on unlabeled data instead of training'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=10,
        help='Number of clusters for K-means (default: 10)'
    )
    parser.add_argument(
        '--cluster-method', type=str, default='kmeans',
        choices=['kmeans', 'dbscan'],
        help='Clustering algorithm to use'
    )
    parser.add_argument(
        '--dbscan-eps', type=float, default=0.5,
        help='DBSCAN eps parameter (max distance between samples)'
    )
    parser.add_argument(
        '--dbscan-min-samples', type=int, default=5,
        help='DBSCAN min_samples parameter'
    )
    parser.add_argument(
        '--create-templates', action='store_true',
        help='Create label template files from clustering results'
    )
    parser.add_argument(
        '--visualize-clusters', action='store_true',
        help='Create 2D/3D visualizations of clusters'
    )
    
    # Finger tracking options
    parser.add_argument(
        '--model-type', type=str, default='gesture',
        choices=['gesture', 'finger_tracking'],
        help='Model type: gesture (10-class) or finger_tracking (5-finger multi-output)'
    )
    
    return parser.parse_args()


def run_clustering(args):
    """Run unsupervised clustering on unlabeled data."""
    if not HAS_CLUSTERING or not HAS_SKLEARN:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)
    
    print("=" * 60)
    print("SIMCAP Unsupervised Clustering")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading data from: {args.data_dir}")
    dataset = GambitDataset(
        args.data_dir,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Print summary
    summary = dataset.summary()
    print(f"\nDataset Summary:")
    print(f"  Total sessions: {summary['total_sessions']}")
    print(f"  Labeled sessions: {summary['labeled_sessions']}")
    print(f"  Unlabeled sessions: {summary['unlabeled_sessions']}")
    
    if summary['unlabeled_sessions'] == 0:
        print("\nNo unlabeled data found!")
        print("All sessions are already labeled.")
        return
    
    # Load unlabeled windows
    print(f"\nLoading unlabeled sessions...")
    windows, metadata = load_unlabeled_windows(dataset)
    
    if len(windows) == 0:
        print("No windows extracted from unlabeled data!")
        return
    
    print(f"  Extracted {len(windows)} windows")
    print(f"  Window shape: {windows.shape}")
    
    # Extract features
    print(f"\nExtracting statistical features...")
    features = extract_features_from_windows(windows)
    print(f"  Feature shape: {features.shape}")
    
    # Perform clustering
    print(f"\nClustering with {args.cluster_method}...")
    
    if args.cluster_method == 'kmeans':
        labels, centers = cluster_kmeans(features, n_clusters=args.n_clusters)
        n_clusters = args.n_clusters
    else:  # dbscan
        labels = cluster_dbscan(
            features,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples
        )
        centers = None
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = np.sum(labels == -1)
        print(f"  Found {n_clusters} clusters")
        if noise_count > 0:
            print(f"  Noise points: {noise_count} ({100*noise_count/len(labels):.1f}%)")
    
    # Compute metrics
    print(f"\nCluster quality metrics:")
    metrics = compute_cluster_metrics(features, labels)
    print(f"  Silhouette score: {metrics['silhouette_score']:.3f} (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
    
    # Analyze clusters
    print(f"\nCluster distribution:")
    cluster_info = analyze_clusters(windows, labels, metadata)
    for cluster_id, info in sorted(cluster_info.items()):
        print(f"  Cluster {cluster_id}: {info['size']} windows ({info['percentage']:.1f}%)")
        print(f"    Sessions: {len(info['sessions'])}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = ClusterResult(
        method=args.cluster_method,
        n_clusters=n_clusters,
        labels=labels,
        centers=centers,
        silhouette_score=metrics['silhouette_score'],
        davies_bouldin_score=metrics['davies_bouldin_score'],
        window_metadata=metadata
    )
    
    results_path = output_dir / 'clustering_results.json'
    save_cluster_results(result, results_path)
    print(f"\nSaved clustering results to: {results_path}")
    
    # Create detailed analysis
    analysis_path = output_dir / 'cluster_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump({
            'summary': summary,
            'config': {
                'method': args.cluster_method,
                'n_clusters': args.n_clusters if args.cluster_method == 'kmeans' else 'auto',
                'window_size': args.window_size,
                'stride': args.stride
            },
            'metrics': metrics,
            'clusters': cluster_info
        }, f, indent=2)
    print(f"Saved cluster analysis to: {analysis_path}")
    
    # Create label templates
    if args.create_templates:
        print(f"\nCreating label templates...")
        templates_dir = output_dir / 'label_templates'
        create_label_templates(result, templates_dir)
        print(f"\nNext steps:")
        print(f"  1. Review templates in: {templates_dir}")
        print(f"  2. Assign gesture names to CLUSTER_X placeholders")
        print(f"  3. Rename .template.json to .meta.json")
        print(f"  4. Move to data directory: {args.data_dir}")
        print(f"  5. Run training: python -m ml.train --data-dir {args.data_dir}")
    
    # Visualize clusters
    if args.visualize_clusters:
        print(f"\nCreating cluster visualizations...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # 2D PCA visualization
            print("  Reducing to 2D with PCA...")
            features_2d = reduce_dimensions(features, method='pca', n_components=2)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=labels, cmap='tab10', alpha=0.6, s=20
            )
            plt.colorbar(scatter, label='Cluster ID')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'Cluster Visualization (2D PCA) - {args.cluster_method}')
            plt.grid(True, alpha=0.3)
            
            viz_path = output_dir / 'clusters_2d_pca.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved 2D PCA plot: {viz_path}")
            
            # 3D PCA visualization
            print("  Reducing to 3D with PCA...")
            features_3d = reduce_dimensions(features, method='pca', n_components=3)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                c=labels, cmap='tab10', alpha=0.6, s=20
            )
            plt.colorbar(scatter, label='Cluster ID')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(f'Cluster Visualization (3D PCA) - {args.cluster_method}')
            
            viz_path = output_dir / 'clusters_3d_pca.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved 3D PCA plot: {viz_path}")
            
            # t-SNE visualization (slower but often better separation)
            if len(features) <= 5000:  # t-SNE is slow for large datasets
                print("  Reducing to 2D with t-SNE...")
                features_tsne = reduce_dimensions(features, method='tsne', n_components=2)
                
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    features_tsne[:, 0], features_tsne[:, 1],
                    c=labels, cmap='tab10', alpha=0.6, s=20
                )
                plt.colorbar(scatter, label='Cluster ID')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title(f'Cluster Visualization (2D t-SNE) - {args.cluster_method}')
                plt.grid(True, alpha=0.3)
                
                viz_path = output_dir / 'clusters_2d_tsne.png'
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved 2D t-SNE plot: {viz_path}")
            else:
                print("  Skipping t-SNE (too many samples, would be slow)")
            
        except ImportError:
            print("  WARNING: matplotlib not installed. Run: pip install matplotlib")
    
    print("\n" + "=" * 60)
    print("Clustering complete!")
    print("=" * 60)


def main():
    args = parse_args()
    
    # Check if clustering mode
    if args.cluster_only:
        run_clustering(args)
        return

    print("=" * 60)
    print("SIMCAP Gesture Classification Training")
    print("=" * 60)

    # Check framework availability
    if args.framework == 'keras' and not HAS_TF:
        print("ERROR: TensorFlow/Keras not installed. Run: pip install tensorflow")
        sys.exit(1)
    if args.framework == 'pytorch' and not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading data from: {args.data_dir}")
    dataset = GambitDataset(
        args.data_dir,
        window_size=args.window_size,
        stride=args.stride
    )

    # Print summary
    summary = dataset.summary()
    print(f"\nDataset Summary:")
    print(f"  Total sessions: {summary['total_sessions']}")
    print(f"  Labeled sessions: {summary['labeled_sessions']}")
    print(f"  Unlabeled sessions: {summary['unlabeled_sessions']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"\nGesture sample counts:")
    for gesture, count in summary['gesture_counts'].items():
        if count > 0:
            print(f"  {gesture}: {count}")

    if args.summary_only:
        return

    # Load train/val split based on model type
    print(f"\nLoading labeled data...")
    
    if args.model_type == 'finger_tracking':
        # Load multi-label data for finger tracking
        X, y_multi = dataset.load_finger_tracking_sessions()
        
        if len(X) == 0:
            print("\nERROR: No labeled finger tracking data found!")
            print("Please label sessions with per-finger states (V2 format).")
            sys.exit(1)
        
        # Split train/val
        n = len(X)
        indices = np.random.permutation(n)
        val_size = int(n * args.val_ratio)
        
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        X_train, y_train = X[train_idx], y_multi[train_idx]
        X_val, y_val = X[val_idx], y_multi[val_idx]
        
        print(f"  Training windows: {len(X_train)}")
        print(f"  Validation windows: {len(X_val)}")
        print(f"  Window shape: {X_train.shape[1:]}")
        print(f"  Labels per window: 5 fingers × 3 states")
        
        # Check finger state distribution
        print(f"\nFinger state distribution (training):")
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger in enumerate(fingers):
            finger_labels = y_train[:, i]
            unique, counts = np.unique(finger_labels, return_counts=True)
            print(f"  {finger}:")
            for state, count in zip(unique, counts):
                state_name = ['extended', 'partial', 'flexed'][int(state)]
                print(f"    {state_name}: {count} ({100*count/len(finger_labels):.1f}%)")
    else:
        # Standard gesture classification
        X_train, y_train, X_val, y_val = dataset.get_train_val_split(args.val_ratio)

        if len(X_train) == 0:
            print("\nERROR: No labeled data found!")
            print("Please label some sessions first using the labeling tool.")
            print("See: python -m ml.label --help")
            sys.exit(1)

        print(f"  Training windows: {len(X_train)}")
        print(f"  Validation windows: {len(X_val)}")
        print(f"  Window shape: {X_train.shape[1:]}")

        # Check class distribution
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(train_classes, train_counts):
            print(f"  {Gesture(cls).name}: {count} ({100*count/len(y_train):.1f}%)")

    # Create model
    print(f"\nCreating {args.framework} model ({args.model_type})...")
    num_classes = len(Gesture)

    if args.framework == 'keras':
        if args.model_type == 'finger_tracking':
            from .model import (create_finger_tracking_model_keras,
                              train_finger_tracking_model_keras,
                              evaluate_finger_tracking_model)
            
            model = create_finger_tracking_model_keras(
                window_size=args.window_size
            )
            model.summary()

            # Train
            print(f"\nTraining finger tracking model for up to {args.epochs} epochs...")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = str(output_dir / 'best_finger_model.keras')
            history = train_finger_tracking_model_keras(
                model, X_train, y_train, X_val, y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint_path
            )

            # Evaluate
            print("\nEvaluating finger tracking model on validation set...")
            metrics = evaluate_finger_tracking_model(model, X_val, y_val)
            print(f"  Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
            print(f"\n  Per-finger accuracy:")
            for finger, acc in metrics['per_finger_accuracy'].items():
                print(f"    {finger}: {acc*100:.1f}%")
        else:
            model = create_cnn_model_keras(
                window_size=args.window_size,
                num_classes=num_classes
            )
            model.summary()

            # Train
            print(f"\nTraining for up to {args.epochs} epochs...")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = str(output_dir / 'best_model.keras')
            history = train_model_keras(
                model, X_train, y_train, X_val, y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint_path
            )

            # Evaluate
            print("\nEvaluating on validation set...")
            metrics = evaluate_model(model, X_val, y_val)
            print(f"  Validation Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"\n  Per-class accuracy:")
            for cls, acc in metrics['per_class_accuracy'].items():
                if acc is not None:
                    print(f"    {cls}: {acc*100:.1f}%")

        # Save model
        print("\nSaving models...")
        saved = save_model_for_inference(model, args.output_dir, 'gesture_model')
        for fmt, path in saved.items():
            print(f"  {fmt}: {path}")

        # Save training results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'summary': summary,
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'metrics': metrics,
            'saved_models': saved
        }
        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results: {results_path}")

    else:
        # PyTorch training
        from .model import GestureCNN
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model = GestureCNN(window_size=args.window_size, num_classes=num_classes)
        print(model)

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining for up to {args.epochs} epochs...")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            val_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_acc={val_acc*100:.2f}%")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered")
                    break

        # Load best model and evaluate
        model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
        metrics = evaluate_model(model, X_val, y_val)
        print(f"\nFinal Validation Accuracy: {metrics['accuracy']*100:.2f}%")

        # Save model
        saved = save_model_for_inference(model, args.output_dir, 'gesture_model')
        for fmt, path in saved.items():
            print(f"  Saved {fmt}: {path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
