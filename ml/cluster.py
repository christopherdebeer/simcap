"""
SIMCAP Unsupervised Clustering

Discover gesture patterns in unlabeled data using clustering algorithms.
Useful for initial exploration and semi-automated labeling.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, asdict

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .data_loader import (
    GambitDataset, load_session_data, load_session_metadata,
    normalize_data
)
from .schema import Gesture, SessionMetadata, LabeledSegment


@dataclass
class ClusterResult:
    """Results from clustering analysis."""
    method: str  # 'kmeans', 'dbscan', etc.
    n_clusters: int
    labels: np.ndarray  # Cluster assignment for each window
    centers: Optional[np.ndarray]  # Cluster centers (if applicable)
    silhouette_score: float
    davies_bouldin_score: float
    window_metadata: List[Dict[str, Any]]  # Session info for each window


@dataclass
class WindowInfo:
    """Metadata for a single window."""
    session_file: str
    window_index: int
    start_sample: int
    end_sample: int
    cluster_id: int
    distance_to_center: Optional[float] = None


def extract_features_from_windows(windows: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from raw windows for clustering.
    
    Reduces dimensionality from (N, window_size, 9) to (N, feature_dim)
    by computing statistics over the time dimension.
    
    Args:
        windows: Shape (N, window_size, 9)
    
    Returns:
        Features: Shape (N, feature_dim) where feature_dim = 9 * 5 = 45
        (mean, std, min, max, range for each of 9 axes)
    """
    N, window_size, num_features = windows.shape
    
    features = []
    for i in range(num_features):
        axis_data = windows[:, :, i]  # (N, window_size)
        
        features.append(np.mean(axis_data, axis=1))
        features.append(np.std(axis_data, axis=1))
        features.append(np.min(axis_data, axis=1))
        features.append(np.max(axis_data, axis=1))
        features.append(np.ptp(axis_data, axis=1))  # range (max - min)
    
    return np.column_stack(features)


def load_unlabeled_windows(dataset: GambitDataset) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load all unlabeled sessions and create windows.
    
    Args:
        dataset: GambitDataset instance
    
    Returns:
        Tuple of (windows, metadata):
        - windows: Shape (N, window_size, 9)
        - metadata: List of dicts with session info for each window
    """
    all_windows = []
    all_metadata = []
    
    for json_path in sorted(dataset.data_dir.glob('*.json')):
        if json_path.name.endswith('.meta.json'):
            continue
        
        meta = load_session_metadata(json_path)
        
        # Skip labeled sessions
        if meta is not None and meta.labels:
            continue
        
        # Load and normalize data
        data = load_session_data(json_path)
        data = normalize_data(data, dataset.stats, dataset.normalize_method)
        
        # Create windows (without labels)
        num_samples = len(data)
        window_idx = 0
        
        for start in range(0, num_samples - dataset.window_size + 1, dataset.stride):
            end = start + dataset.window_size
            window_data = data[start:end]
            
            all_windows.append(window_data)
            all_metadata.append({
                'session_file': json_path.name,
                'window_index': window_idx,
                'start_sample': start,
                'end_sample': end
            })
            window_idx += 1
    
    if not all_windows:
        return np.array([]).reshape(0, dataset.window_size, 9), []
    
    return np.array(all_windows), all_metadata


def cluster_kmeans(features: np.ndarray, n_clusters: int = 10,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering.
    
    Args:
        features: Feature matrix (N, feature_dim)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Tuple of (labels, centers)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels, kmeans.cluster_centers_


def cluster_dbscan(features: np.ndarray, eps: float = 0.5,
                   min_samples: int = 5) -> np.ndarray:
    """
    Perform DBSCAN clustering (automatically determines number of clusters).
    
    Args:
        features: Feature matrix (N, feature_dim)
        eps: Maximum distance between samples in same neighborhood
        min_samples: Minimum samples in neighborhood to form core point
    
    Returns:
        Cluster labels (noise points labeled as -1)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    
    return labels


def reduce_dimensions(features: np.ndarray, method: str = 'pca',
                     n_components: int = 2) -> np.ndarray:
    """
    Reduce feature dimensionality for visualization.
    
    Args:
        features: Feature matrix (N, feature_dim)
        method: 'pca' or 'tsne'
        n_components: Target dimensions (2 or 3)
    
    Returns:
        Reduced features (N, n_components)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(features)


def compute_cluster_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        features: Feature matrix
        labels: Cluster assignments
    
    Returns:
        Dict with silhouette_score and davies_bouldin_score
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn not installed")
    
    # Filter out noise points (label -1) for metrics
    mask = labels >= 0
    if mask.sum() < 2:
        return {'silhouette_score': 0.0, 'davies_bouldin_score': 0.0}
    
    filtered_features = features[mask]
    filtered_labels = labels[mask]
    
    # Need at least 2 clusters for these metrics
    if len(np.unique(filtered_labels)) < 2:
        return {'silhouette_score': 0.0, 'davies_bouldin_score': 0.0}
    
    silhouette = silhouette_score(filtered_features, filtered_labels)
    davies_bouldin = davies_bouldin_score(filtered_features, filtered_labels)
    
    return {
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin)
    }


def analyze_clusters(windows: np.ndarray, labels: np.ndarray,
                    metadata: List[Dict]) -> Dict[str, Any]:
    """
    Analyze cluster characteristics.
    
    Args:
        windows: Raw window data (N, window_size, 9)
        labels: Cluster assignments
        metadata: Window metadata
    
    Returns:
        Dict with cluster statistics
    """
    unique_labels = np.unique(labels)
    cluster_info = {}
    
    for label in unique_labels:
        if label == -1:  # Noise in DBSCAN
            continue
        
        mask = labels == label
        cluster_windows = windows[mask]
        cluster_meta = [m for m, l in zip(metadata, labels) if l == label]
        
        # Compute statistics
        cluster_info[int(label)] = {
            'size': int(mask.sum()),
            'percentage': float(mask.sum() / len(labels) * 100),
            'sessions': list(set(m['session_file'] for m in cluster_meta)),
            'mean_values': {
                'ax': float(np.mean(cluster_windows[:, :, 0])),
                'ay': float(np.mean(cluster_windows[:, :, 1])),
                'az': float(np.mean(cluster_windows[:, :, 2])),
                'gx': float(np.mean(cluster_windows[:, :, 3])),
                'gy': float(np.mean(cluster_windows[:, :, 4])),
                'gz': float(np.mean(cluster_windows[:, :, 5])),
            }
        }
    
    return cluster_info


def save_cluster_results(result: ClusterResult, output_path: Path):
    """Save clustering results to JSON."""
    output = {
        'method': result.method,
        'n_clusters': result.n_clusters,
        'silhouette_score': result.silhouette_score,
        'davies_bouldin_score': result.davies_bouldin_score,
        'cluster_assignments': result.labels.tolist(),
        'window_metadata': result.window_metadata
    }
    
    if result.centers is not None:
        output['cluster_centers'] = result.centers.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def create_label_templates(result: ClusterResult, output_dir: Path,
                          gesture_names: Optional[List[str]] = None):
    """
    Create template metadata files for each cluster.
    
    Users can review these templates, assign gesture names, and use them
    to batch-label sessions.
    
    Args:
        result: ClusterResult from clustering
        output_dir: Directory to save templates
        gesture_names: Optional list of suggested gesture names for each cluster
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group windows by session and cluster
    session_clusters = {}
    for i, meta in enumerate(result.window_metadata):
        session = meta['session_file']
        cluster = int(result.labels[i])
        
        if cluster == -1:  # Skip noise
            continue
        
        if session not in session_clusters:
            session_clusters[session] = {}
        if cluster not in session_clusters[session]:
            session_clusters[session][cluster] = []
        
        session_clusters[session][cluster].append({
            'start_sample': meta['start_sample'],
            'end_sample': meta['end_sample']
        })
    
    # Create template for each session
    for session, clusters in session_clusters.items():
        template = {
            'timestamp': session.replace('.json', ''),
            'subject_id': 'UNKNOWN',
            'environment': 'UNKNOWN',
            'hand': 'UNKNOWN',
            'split': 'train',
            'labels': [],
            'session_notes': f'Auto-generated from clustering. Review and assign gesture names.',
            'sample_rate_hz': 50,
            'clustering_info': {
                'method': result.method,
                'n_clusters': result.n_clusters
            }
        }
        
        # Add segments for each cluster
        for cluster_id, windows in sorted(clusters.items()):
            gesture_name = (gesture_names[cluster_id] 
                          if gesture_names and cluster_id < len(gesture_names)
                          else f'CLUSTER_{cluster_id}')
            
            for window in windows:
                template['labels'].append({
                    'start_sample': window['start_sample'],
                    'end_sample': window['end_sample'],
                    'gesture': gesture_name,
                    'confidence': 'medium',
                    'cluster_id': cluster_id
                })
        
        # Save template
        template_path = output_dir / f'{session}.template.json'
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
    
    print(f"\nCreated {len(session_clusters)} label templates in {output_dir}")
    print("Review templates, assign gesture names, then rename to .meta.json")


if __name__ == '__main__':
    # Quick test
    import sys
    
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/GAMBIT'
    
    print("Loading unlabeled data...")
    dataset = GambitDataset(data_dir)
    windows, metadata = load_unlabeled_windows(dataset)
    
    if len(windows) == 0:
        print("No unlabeled data found!")
        sys.exit(1)
    
    print(f"Found {len(windows)} windows from unlabeled sessions")
    
    print("\nExtracting features...")
    features = extract_features_from_windows(windows)
    print(f"Feature shape: {features.shape}")
    
    print("\nClustering with K-means (k=10)...")
    labels, centers = cluster_kmeans(features, n_clusters=10)
    
    print(f"Found {len(np.unique(labels))} clusters")
    print(f"Cluster sizes: {np.bincount(labels)}")
    
    metrics = compute_cluster_metrics(features, labels)
    print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"Davies-Bouldin score: {metrics['davies_bouldin_score']:.3f}")
