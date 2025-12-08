#!/usr/bin/env python3
"""
SIMCAP Unified Explorer Generator

Generates a unified HTML explorer that combines:
- Session visualizations
- Clustering results
- Labeling workflow

Usage:
    python -m ml.generate_explorer --data-dir data/GAMBIT --output visualizations/explorer.html
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

from .data_loader import GambitDataset, load_session_data, load_session_metadata
from .schema import Gesture

# Try to import clustering (optional)
try:
    from .cluster import (
        load_unlabeled_windows, extract_features_from_windows,
        cluster_kmeans, reduce_dimensions, compute_cluster_metrics,
        HAS_SKLEARN
    )
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    HAS_SKLEARN = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate unified SIMCAP explorer HTML'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/GAMBIT',
        help='Path to data directory'
    )
    parser.add_argument(
        '--viz-dir', type=str, default='visualizations',
        help='Path to visualizations directory'
    )
    parser.add_argument(
        '--cluster-dir', type=str, default='ml/models',
        help='Path to clustering results directory'
    )
    parser.add_argument(
        '--output', type=str, default='visualizations/explorer.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=10,
        help='Number of clusters for K-means'
    )
    parser.add_argument(
        '--window-size', type=int, default=50,
        help='Window size in samples'
    )
    parser.add_argument(
        '--stride', type=int, default=25,
        help='Window stride'
    )
    return parser.parse_args()


def load_existing_visualizations(viz_dir: Path) -> Dict[str, Any]:
    """Load existing visualization data from index.html if available."""
    index_path = viz_dir / 'index.html'
    if not index_path.exists():
        return {}
    
    # Parse the embedded sessionsData from the HTML
    content = index_path.read_text()
    
    # Find the sessionsData array
    start_marker = 'const sessionsData = '
    end_marker = '];'
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return {}
    
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx) + 1
    
    try:
        sessions_json = content[start_idx:end_idx]
        sessions = json.loads(sessions_json)
        return {s['filename']: s for s in sessions}
    except (json.JSONDecodeError, KeyError):
        return {}


def load_clustering_results(cluster_dir: Path) -> Optional[Dict[str, Any]]:
    """Load clustering results if available."""
    results_path = cluster_dir / 'clustering_results.json'
    analysis_path = cluster_dir / 'cluster_analysis.json'
    
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        results = json.load(f)
    
    analysis = {}
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
    
    return {
        'results': results,
        'analysis': analysis
    }


def generate_cluster_data(dataset: GambitDataset, n_clusters: int = 10) -> Dict[str, Any]:
    """Generate fresh clustering data."""
    if not HAS_CLUSTERING or not HAS_SKLEARN:
        return None
    
    print("Generating clustering data...")
    windows, metadata = load_unlabeled_windows(dataset)
    
    if len(windows) == 0:
        return None
    
    # Extract features and cluster
    features = extract_features_from_windows(windows)
    labels, centers = cluster_kmeans(features, n_clusters=n_clusters)
    
    # Reduce to 2D for visualization
    features_2d = reduce_dimensions(features, method='pca', n_components=2)
    
    # Compute metrics
    metrics = compute_cluster_metrics(features, labels)
    
    # Build cluster data
    cluster_data = {
        'method': 'kmeans',
        'n_clusters': n_clusters,
        'metrics': metrics,
        'points': []
    }
    
    for i, (meta, label, coords) in enumerate(zip(metadata, labels, features_2d)):
        cluster_data['points'].append({
            'session': meta['session_file'],
            'window_index': meta['window_index'],
            'start_sample': meta['start_sample'],
            'end_sample': meta['end_sample'],
            'cluster': int(label),
            'x': float(coords[0]),
            'y': float(coords[1])
        })
    
    return cluster_data


def build_unified_data(
    data_dir: Path,
    viz_dir: Path,
    cluster_dir: Path,
    n_clusters: int = 10,
    window_size: int = 50,
    stride: int = 25
) -> Dict[str, Any]:
    """Build unified data structure for the explorer."""
    
    # Load dataset
    dataset = GambitDataset(str(data_dir), window_size=window_size, stride=stride)
    summary = dataset.summary()
    
    # Load existing visualizations
    viz_data = load_existing_visualizations(viz_dir)
    
    # Load or generate clustering
    cluster_data = load_clustering_results(cluster_dir)
    if cluster_data is None and HAS_CLUSTERING:
        cluster_data = {
            'results': None,
            'analysis': None,
            'generated': generate_cluster_data(dataset, n_clusters)
        }
    
    # Build session list with cluster info
    sessions = []
    cluster_assignments = {}
    
    # Parse cluster assignments by session/window
    if cluster_data and cluster_data.get('results'):
        results = cluster_data['results']
        for i, meta in enumerate(results.get('window_metadata', [])):
            session = meta['session_file']
            if session not in cluster_assignments:
                cluster_assignments[session] = {}
            window_key = f"{meta['start_sample']}_{meta['end_sample']}"
            cluster_assignments[session][window_key] = results['cluster_assignments'][i]
    elif cluster_data and cluster_data.get('generated'):
        for point in cluster_data['generated']['points']:
            session = point['session']
            if session not in cluster_assignments:
                cluster_assignments[session] = {}
            window_key = f"{point['start_sample']}_{point['end_sample']}"
            cluster_assignments[session][window_key] = point['cluster']
    
    # Build sessions with visualization and cluster data
    for json_path in sorted(data_dir.glob('*.json')):
        if json_path.name.endswith('.meta.json'):
            continue
        
        filename = json_path.name
        
        # Get visualization data if available
        viz = viz_data.get(filename, {})
        
        # Load raw data for duration
        try:
            raw_data = load_session_data(json_path)
            duration = len(raw_data) / 50.0  # Assuming 50Hz
        except:
            duration = viz.get('duration', 0)
        
        # Get metadata if available
        meta = load_session_metadata(json_path)
        
        # Build windows with cluster info
        windows = []
        session_clusters = cluster_assignments.get(filename, {})
        
        for i, w in enumerate(viz.get('windows', [])):
            window_key = f"{int(w.get('time_start', i) * 50)}_{int(w.get('time_end', i+1) * 50)}"
            # Try different key formats
            cluster_id = session_clusters.get(window_key)
            if cluster_id is None:
                # Try with stride-based keys
                start = i * stride
                end = start + window_size
                window_key = f"{start}_{end}"
                cluster_id = session_clusters.get(window_key, -1)
            
            windows.append({
                **w,
                'cluster_id': cluster_id if cluster_id is not None else -1
            })
        
        # If no viz windows, create from cluster data
        if not windows and filename in cluster_assignments:
            for window_key, cluster_id in cluster_assignments[filename].items():
                start, end = map(int, window_key.split('_'))
                windows.append({
                    'window_num': len(windows) + 1,
                    'time_start': start / 50.0,
                    'time_end': end / 50.0,
                    'cluster_id': cluster_id,
                    'filepath': None
                })
        
        session = {
            'filename': filename,
            'timestamp': filename.replace('.json', ''),
            'duration': duration,
            'composite_image': viz.get('composite_image'),
            'raw_images': viz.get('raw_images', []),
            'windows': windows,
            'labeled': meta is not None and bool(meta.labels),
            'split': meta.split if meta else None,
            'labels': [
                {
                    'start_sample': seg.start_sample,
                    'end_sample': seg.end_sample,
                    'gesture': seg.gesture.name,
                    'confidence': seg.confidence
                }
                for seg in (meta.labels if meta else [])
            ]
        }
        
        sessions.append(session)
    
    # Build cluster summary
    cluster_summary = []
    if cluster_data:
        analysis = cluster_data.get('analysis', {})
        clusters = analysis.get('clusters', {})
        for cluster_id in range(n_clusters):
            info = clusters.get(str(cluster_id), {})
            cluster_summary.append({
                'id': cluster_id,
                'size': info.get('size', 0),
                'percentage': info.get('percentage', 0),
                'sessions': info.get('sessions', []),
                'gesture_name': None  # To be assigned by user
            })
    
    # Build 2D points for canvas visualization
    cluster_points = []
    if cluster_data and cluster_data.get('generated'):
        cluster_points = cluster_data['generated']['points']
    elif cluster_data and cluster_data.get('results') and HAS_CLUSTERING:
        # Regenerate 2D coordinates from existing results
        print("Regenerating 2D coordinates for visualization...")
        try:
            windows, metadata = load_unlabeled_windows(dataset)
            if len(windows) > 0:
                features = extract_features_from_windows(windows)
                features_2d = reduce_dimensions(features, method='pca', n_components=2)
                results = cluster_data['results']
                assignments = results.get('cluster_assignments', [])
                
                for i, (meta, coords) in enumerate(zip(metadata, features_2d)):
                    if i < len(assignments):
                        cluster_points.append({
                            'session': meta['session_file'],
                            'window_index': meta['window_index'],
                            'start_sample': meta['start_sample'],
                            'end_sample': meta['end_sample'],
                            'cluster': int(assignments[i]),
                            'x': float(coords[0]),
                            'y': float(coords[1])
                        })
        except Exception as e:
            print(f"Warning: Could not generate 2D coordinates: {e}")
    
    return {
        'generated_at': datetime.now().isoformat(),
        'summary': summary,
        'sessions': sessions,
        'clustering': {
            'enabled': cluster_data is not None,
            'n_clusters': n_clusters,
            'metrics': cluster_data.get('analysis', {}).get('metrics', {}) if cluster_data else {},
            'clusters': cluster_summary,
            'points': cluster_points
        },
        'gestures': Gesture.names(),
        'config': {
            'window_size': window_size,
            'stride': stride,
            'sample_rate': 50
        }
    }


def generate_html(data: Dict[str, Any], output_path: Path):
    """Generate the unified explorer HTML."""
    
    # Cluster colors
    cluster_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
    ]
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIMCAP Unified Explorer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }}
        
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        h1 {{ font-size: 2em; margin-bottom: 8px; }}
        .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        
        .stats {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .stat-box {{
            background: rgba(255,255,255,0.15);
            padding: 12px 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }}
        
        .stat-box .number {{ font-size: 1.8em; font-weight: bold; display: block; }}
        .stat-box .label {{ font-size: 0.85em; opacity: 0.8; }}
        
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            background: #16213e;
            padding: 8px;
            border-radius: 10px;
        }}
        
        .tab {{
            padding: 12px 24px;
            border: none;
            background: transparent;
            color: #aaa;
            cursor: pointer;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.2s;
        }}
        
        .tab:hover {{ background: rgba(255,255,255,0.1); color: #fff; }}
        .tab.active {{ background: #667eea; color: #fff; }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .main-layout {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }}
        
        .sidebar {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            height: fit-content;
            position: sticky;
            top: 20px;
        }}
        
        .cluster-canvas-container {{
            background: #0f0f23;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        #clusterCanvas {{
            width: 100%;
            height: 280px;
            border-radius: 8px;
            cursor: crosshair;
        }}
        
        .cluster-legend {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 15px;
        }}
        
        .cluster-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9em;
        }}
        
        .cluster-item:hover {{ background: rgba(255,255,255,0.1); }}
        .cluster-item.selected {{ background: rgba(102, 126, 234, 0.3); border: 1px solid #667eea; }}
        
        .cluster-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        
        .cluster-count {{ margin-left: auto; opacity: 0.6; font-size: 0.85em; }}
        
        .content-area {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }}
        
        .filter-bar {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-bar input, .filter-bar select {{
            padding: 10px 15px;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            background: #0f0f23;
            color: #eee;
            font-size: 1em;
        }}
        
        .filter-bar input:focus, .filter-bar select:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .session-grid {{
            display: grid;
            gap: 15px;
        }}
        
        .session-card {{
            background: #0f0f23;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .session-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        
        .session-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .session-title {{ font-weight: bold; font-size: 1.1em; }}
        .session-info {{ font-size: 0.85em; opacity: 0.9; margin-top: 4px; }}
        
        .session-badges {{
            display: flex;
            gap: 6px;
        }}
        
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        
        .badge.labeled {{ background: #4daf4a; }}
        .badge.unlabeled {{ background: #ff7f00; }}
        
        .expand-icon {{
            font-size: 1.3em;
            transition: transform 0.3s;
        }}
        
        .session-card.expanded .expand-icon {{ transform: rotate(180deg); }}
        
        .session-content {{
            display: none;
            padding: 20px;
        }}
        
        .session-card.expanded .session-content {{ display: block; }}
        
        .section-title {{
            font-size: 1.2em;
            color: #667eea;
            margin: 20px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }}
        
        .section-title:first-child {{ margin-top: 0; }}
        
        .composite-image {{
            width: 100%;
            border-radius: 8px;
            cursor: pointer;
        }}
        
        .windows-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
        }}
        
        .window-card {{
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
            border: 2px solid transparent;
        }}
        
        .window-card:hover {{
            transform: scale(1.03);
        }}
        
        .window-card.highlighted {{
            border-color: #667eea;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.5);
        }}
        
        .window-cluster-bar {{
            height: 6px;
        }}
        
        .window-preview {{
            width: 100%;
            height: 100px;
            object-fit: cover;
        }}
        
        .window-info {{
            padding: 10px;
        }}
        
        .window-title {{ font-weight: bold; font-size: 0.9em; }}
        .window-time {{ font-size: 0.8em; opacity: 0.7; margin-top: 3px; }}
        .window-cluster {{ font-size: 0.75em; margin-top: 5px; }}
        
        .raw-images {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .raw-image {{
            width: 100%;
            border-radius: 8px;
            cursor: pointer;
        }}
        
        /* Labeling Panel */
        .labeling-panel {{
            background: #0f0f23;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .labeling-title {{
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #667eea;
        }}
        
        .gesture-buttons {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 8px;
            margin-bottom: 15px;
        }}
        
        .gesture-btn {{
            padding: 10px;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            background: transparent;
            color: #eee;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }}
        
        .gesture-btn:hover {{ border-color: #667eea; background: rgba(102, 126, 234, 0.1); }}
        .gesture-btn.selected {{ border-color: #667eea; background: #667eea; }}
        
        .export-btn {{
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .export-btn:hover {{ transform: scale(1.02); }}
        .export-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            padding: 20px;
        }}
        
        .modal.active {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .modal-content {{
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }}
        
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: #fff;
            font-size: 40px;
            cursor: pointer;
        }}
        
        .modal-close:hover {{ color: #667eea; }}
        
        /* Metrics display */
        .metrics-box {{
            background: #0f0f23;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2a2a4a;
        }}
        
        .metric-row:last-child {{ border-bottom: none; }}
        .metric-label {{ opacity: 0.7; }}
        .metric-value {{ font-weight: bold; }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .main-layout {{ grid-template-columns: 1fr; }}
            .sidebar {{ position: static; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 SIMCAP Unified Explorer</h1>
            <p class="subtitle">Interactive viewer for sensor data, clustering, and gesture labeling</p>
            <div class="stats">
                <div class="stat-box">
                    <span class="number" id="stat-sessions">{len(data['sessions'])}</span>
                    <span class="label">Sessions</span>
                </div>
                <div class="stat-box">
                    <span class="number" id="stat-windows">{sum(len(s['windows']) for s in data['sessions'])}</span>
                    <span class="label">Windows</span>
                </div>
                <div class="stat-box">
                    <span class="number" id="stat-clusters">{data['clustering']['n_clusters']}</span>
                    <span class="label">Clusters</span>
                </div>
                <div class="stat-box">
                    <span class="number" id="stat-labeled">{sum(1 for s in data['sessions'] if s['labeled'])}</span>
                    <span class="label">Labeled</span>
                </div>
            </div>
        </header>
        
        <div class="tabs">
            <button class="tab active" data-tab="explore">📊 Explore</button>
            <button class="tab" data-tab="clusters">🎨 Clusters</button>
            <button class="tab" data-tab="label">🏷️ Label</button>
        </div>
        
        <div class="tab-content active" id="tab-explore">
            <div class="main-layout">
                <div class="sidebar">
                    <div class="cluster-canvas-container">
                        <canvas id="clusterCanvas"></canvas>
                    </div>
                    
                    <div class="metrics-box">
                        <div class="metric-row">
                            <span class="metric-label">Silhouette Score</span>
                            <span class="metric-value">{data['clustering']['metrics'].get('silhouette_score', 0):.3f}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Davies-Bouldin</span>
                            <span class="metric-value">{data['clustering']['metrics'].get('davies_bouldin_score', 0):.3f}</span>
                        </div>
                    </div>
                    
                    <h3 style="margin-bottom: 10px; font-size: 1em;">Clusters</h3>
                    <div class="cluster-legend" id="clusterLegend"></div>
                </div>
                
                <div class="content-area">
                    <div class="filter-bar">
                        <input type="text" id="searchBox" placeholder="Search sessions..." style="flex: 1; min-width: 200px;">
                        <select id="clusterFilter">
                            <option value="all">All Clusters</option>
                        </select>
                        <select id="sortSelect">
                            <option value="timestamp-desc">Newest First</option>
                            <option value="timestamp-asc">Oldest First</option>
                            <option value="duration-desc">Longest First</option>
                            <option value="windows-desc">Most Windows</option>
                        </select>
                    </div>
                    
                    <div class="session-grid" id="sessionGrid"></div>
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="tab-clusters">
            <div class="content-area">
                <h2 style="margin-bottom: 20px;">Cluster Analysis</h2>
                <p style="margin-bottom: 20px; opacity: 0.8;">
                    Click on a cluster in the visualization or legend to filter sessions.
                    Each cluster represents a group of similar sensor patterns.
                </p>
                <div id="clusterDetails"></div>
            </div>
        </div>
        
        <div class="tab-content" id="tab-label">
            <div class="content-area">
                <h2 style="margin-bottom: 20px;">Labeling Workflow</h2>
                <p style="margin-bottom: 20px; opacity: 0.8;">
                    Assign gesture names to clusters, then export label files.
                </p>
                
                <div class="labeling-panel">
                    <h3 class="labeling-title">Assign Gestures to Clusters</h3>
                    <div id="clusterLabeling"></div>
                    
                    <button class="export-btn" id="exportBtn" style="margin-top: 20px;">
                        📥 Export Label Templates
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="modal" id="imageModal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        // Embedded data
        const explorerData = {json.dumps(data, indent=2)};
        
        // Cluster colors
        const clusterColors = {json.dumps(cluster_colors)};
        
        // State
        let selectedCluster = null;
        let clusterGestureAssignments = {{}};
        let filteredSessions = [...explorerData.sessions];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            initTabs();
            initClusterCanvas();
            initClusterLegend();
            initFilters();
            renderSessions();
            initLabelingPanel();
        }});
        
        // Tab switching
        function initTabs() {{
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
                }});
            }});
        }}
        
        // Canvas cluster visualization
        function initClusterCanvas() {{
            const canvas = document.getElementById('clusterCanvas');
            const ctx = canvas.getContext('2d');
            const points = explorerData.clustering.points || [];
            
            if (points.length === 0) {{
                ctx.fillStyle = '#666';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No clustering data available', canvas.width / 2, canvas.height / 2);
                return;
            }}
            
            // Set canvas size
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            // Find bounds
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            points.forEach(p => {{
                minX = Math.min(minX, p.x);
                maxX = Math.max(maxX, p.x);
                minY = Math.min(minY, p.y);
                maxY = Math.max(maxY, p.y);
            }});
            
            const padding = 20;
            const width = rect.width - padding * 2;
            const height = rect.height - padding * 2;
            
            function toCanvas(x, y) {{
                return {{
                    x: padding + ((x - minX) / (maxX - minX)) * width,
                    y: padding + ((y - minY) / (maxY - minY)) * height
                }};
            }}
            
            function draw() {{
                ctx.clearRect(0, 0, rect.width, rect.height);
                
                // Draw points
                points.forEach(p => {{
                    const pos = toCanvas(p.x, p.y);
                    const color = clusterColors[p.cluster % clusterColors.length];
                    const isSelected = selectedCluster === null || selectedCluster === p.cluster;
                    
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, isSelected ? 5 : 3, 0, Math.PI * 2);
                    ctx.fillStyle = isSelected ? color : color + '40';
                    ctx.fill();
                }});
            }}
            
            draw();
            
            // Click handler
            canvas.addEventListener('click', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Find closest point
                let closest = null;
                let minDist = Infinity;
                
                points.forEach(p => {{
                    const pos = toCanvas(p.x, p.y);
                    const dist = Math.sqrt((pos.x - x) ** 2 + (pos.y - y) ** 2);
                    if (dist < minDist && dist < 15) {{
                        minDist = dist;
                        closest = p;
                    }}
                }});
                
                if (closest) {{
                    selectCluster(closest.cluster);
                }} else {{
                    selectCluster(null);
                }}
                draw();
            }});
            
            window.redrawCanvas = draw;
        }}
        
        // Cluster legend
        function initClusterLegend() {{
            const legend = document.getElementById('clusterLegend');
            const clusters = explorerData.clustering.clusters || [];
            
            legend.innerHTML = clusters.map(c => `
                <div class="cluster-item" data-cluster="${{c.id}}" onclick="selectCluster(${{c.id}})">
                    <div class="cluster-dot" style="background: ${{clusterColors[c.id % clusterColors.length]}}"></div>
                    <span>Cluster ${{c.id}}</span>
                    <span class="cluster-count">${{c.size}}</span>
                </div>
            `).join('');
            
            // Populate filter dropdown
            const filter = document.getElementById('clusterFilter');
            clusters.forEach(c => {{
                const opt = document.createElement('option');
                opt.value = c.id;
                opt.textContent = `Cluster ${{c.id}} (${{c.size}})`;
                filter.appendChild(opt);
            }});
        }}
        
        function selectCluster(clusterId) {{
            selectedCluster = clusterId === selectedCluster ? null : clusterId;
            
            // Update legend
            document.querySelectorAll('.cluster-item').forEach(item => {{
                item.classList.toggle('selected', parseInt(item.dataset.cluster) === selectedCluster);
            }});
            
            // Update filter dropdown
            document.getElementById('clusterFilter').value = selectedCluster !== null ? selectedCluster : 'all';
            
            // Filter sessions
            applyFilters();
            
            // Redraw canvas
            if (window.redrawCanvas) window.redrawCanvas();
        }}
        
        // Filters
        function initFilters() {{
            document.getElementById('searchBox').addEventListener('input', applyFilters);
            document.getElementById('clusterFilter').addEventListener('change', (e) => {{
                selectedCluster = e.target.value === 'all' ? null : parseInt(e.target.value);
                applyFilters();
                if (window.redrawCanvas) window.redrawCanvas();
            }});
            document.getElementById('sortSelect').addEventListener('change', applyFilters);
        }}
        
        function applyFilters() {{
            const search = document.getElementById('searchBox').value.toLowerCase();
            const sort = document.getElementById('sortSelect').value;
            
            filteredSessions = explorerData.sessions.filter(s => {{
                // Search filter
                if (search && !s.filename.toLowerCase().includes(search)) return false;
                
                // Cluster filter
                if (selectedCluster !== null) {{
                    const hasCluster = s.windows.some(w => w.cluster_id === selectedCluster);
                    if (!hasCluster) return false;
                }}
                
                return true;
            }});
            
            // Sort
            filteredSessions.sort((a, b) => {{
                switch(sort) {{
                    case 'timestamp-desc': return b.timestamp.localeCompare(a.timestamp);
                    case 'timestamp-asc': return a.timestamp.localeCompare(b.timestamp);
                    case 'duration-desc': return b.duration - a.duration;
                    case 'windows-desc': return b.windows.length - a.windows.length;
                    default: return 0;
                }}
            }});
            
            renderSessions();
        }}
        
        // Render sessions
        function renderSessions() {{
            const grid = document.getElementById('sessionGrid');
            
            if (filteredSessions.length === 0) {{
                grid.innerHTML = '<div style="text-align: center; padding: 40px; opacity: 0.6;">No sessions match your filters</div>';
                return;
            }}
            
            grid.innerHTML = filteredSessions.map((s, idx) => `
                <div class="session-card" id="session-${{idx}}">
                    <div class="session-header" onclick="toggleSession(${{idx}})">
                        <div>
                            <div class="session-title">${{s.filename}}</div>
                            <div class="session-info">${{formatTimestamp(s.timestamp)}} | ${{s.duration.toFixed(1)}}s | ${{s.windows.length}} windows</div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div class="session-badges">
                                <span class="badge ${{s.labeled ? 'labeled' : 'unlabeled'}}">${{s.labeled ? 'Labeled' : 'Unlabeled'}}</span>
                            </div>
                            <span class="expand-icon">▼</span>
                        </div>
                    </div>
                    <div class="session-content">
                        ${{s.composite_image ? `
                            <h3 class="section-title">📊 Composite View</h3>
                            <img src="${{s.composite_image}}" class="composite-image" onclick="openModal(this.src)">
                        ` : ''}}
                        
                        <h3 class="section-title">🔍 Windows (${{s.windows.length}})</h3>
                        <div class="windows-grid">
                            ${{s.windows.map(w => `
                                <div class="window-card ${{selectedCluster !== null && w.cluster_id === selectedCluster ? 'highlighted' : ''}}">
                                    <div class="window-cluster-bar" style="background: ${{w.cluster_id >= 0 ? clusterColors[w.cluster_id % clusterColors.length] : '#444'}}"></div>
                                    ${{w.filepath ? `<img src="${{w.filepath}}" class="window-preview" onclick="openModal(this.src)">` : '<div class="window-preview" style="background: #333; display: flex; align-items: center; justify-content: center; color: #666;">No image</div>'}}
                                    <div class="window-info">
                                        <div class="window-title">Window ${{w.window_num || '?'}}</div>
                                        <div class="window-time">${{(w.time_start || 0).toFixed(2)}}s - ${{(w.time_end || 0).toFixed(2)}}s</div>
                                        <div class="window-cluster" style="color: ${{w.cluster_id >= 0 ? clusterColors[w.cluster_id % clusterColors.length] : '#666'}}">
                                            ${{w.cluster_id >= 0 ? `Cluster ${{w.cluster_id}}` : 'No cluster'}}
                                        </div>
                                    </div>
                                </div>
                            `).join('')}}
                        </div>
                        
                        ${{s.raw_images && s.raw_images.length > 0 ? `
                            <h3 class="section-title">📐 Raw Views</h3>
                            <div class="raw-images">
                                ${{s.raw_images.map(img => `<img src="${{img}}" class="raw-image" onclick="openModal(this.src)">`).join('')}}
                            </div>
                        ` : ''}}
                    </div>
                </div>
            `).join('');
        }}
        
        function toggleSession(idx) {{
            document.getElementById(`session-${{idx}}`).classList.toggle('expanded');
        }}
        
        function formatTimestamp(ts) {{
            try {{ return new Date(ts).toLocaleString(); }}
            catch {{ return ts; }}
        }}
        
        // Modal
        function openModal(src) {{
            document.getElementById('imageModal').classList.add('active');
            document.getElementById('modalImage').src = src;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
        
        document.getElementById('imageModal').addEventListener('click', (e) => {{
            if (e.target.id === 'imageModal') closeModal();
        }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});
        
        // Labeling panel
        function initLabelingPanel() {{
            const panel = document.getElementById('clusterLabeling');
            const clusters = explorerData.clustering.clusters || [];
            const gestures = explorerData.gestures || [];
            
            panel.innerHTML = clusters.map(c => `
                <div style="display: flex; align-items: center; gap: 15px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 10px;">
                    <div class="cluster-dot" style="background: ${{clusterColors[c.id % clusterColors.length]}}; width: 16px; height: 16px;"></div>
                    <span style="min-width: 80px;">Cluster ${{c.id}}</span>
                    <span style="opacity: 0.6; min-width: 60px;">(${{c.size}} windows)</span>
                    <select class="gesture-select" data-cluster="${{c.id}}" style="flex: 1; padding: 8px; background: #0f0f23; border: 2px solid #2a2a4a; border-radius: 6px; color: #eee;">
                        <option value="">-- Select Gesture --</option>
                        ${{gestures.map(g => `<option value="${{g}}">${{g}}</option>`).join('')}}
                    </select>
                </div>
            `).join('');
            
            // Track changes
            panel.querySelectorAll('.gesture-select').forEach(select => {{
                select.addEventListener('change', (e) => {{
                    const clusterId = parseInt(e.target.dataset.cluster);
                    clusterGestureAssignments[clusterId] = e.target.value || null;
                }});
            }});
            
            // Export button
            document.getElementById('exportBtn').addEventListener('click', exportLabels);
        }}
        
        function exportLabels() {{
            const assignments = clusterGestureAssignments;
            const hasAssignments = Object.values(assignments).some(v => v);
            
            if (!hasAssignments) {{
                alert('Please assign at least one gesture to a cluster before exporting.');
                return;
            }}
            
            // Build label templates for each session
            const templates = {{}};
            
            explorerData.sessions.forEach(session => {{
                const labels = [];
                
                session.windows.forEach(w => {{
                    const gesture = assignments[w.cluster_id];
                    if (gesture) {{
                        labels.push({{
                            start_sample: Math.round((w.time_start || 0) * 50),
                            end_sample: Math.round((w.time_end || 1) * 50),
                            gesture: gesture,
                            confidence: 'medium',
                            cluster_id: w.cluster_id
                        }});
                    }}
                }});
                
                if (labels.length > 0) {{
                    templates[session.filename] = {{
                        timestamp: session.timestamp,
                        labels: labels
                    }};
                }}
            }});
            
            // Download as JSON
            const blob = new Blob([JSON.stringify(templates, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'label_templates.json';
            a.click();
            URL.revokeObjectURL(url);
            
            alert(`Exported labels for ${{Object.keys(templates).length}} sessions.\\n\\nTo use:\\n1. Split the JSON into individual .meta.json files\\n2. Move to your data directory`);
        }}
    </script>
</body>
</html>'''
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Generated unified explorer: {output_path}")


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    viz_dir = Path(args.viz_dir)
    cluster_dir = Path(args.cluster_dir)
    output_path = Path(args.output)
    
    print("=" * 60)
    print("SIMCAP Unified Explorer Generator")
    print("=" * 60)
    
    # Build unified data
    print(f"\nLoading data from: {data_dir}")
    print(f"Visualizations from: {viz_dir}")
    print(f"Clustering from: {cluster_dir}")
    
    data = build_unified_data(
        data_dir, viz_dir, cluster_dir,
        n_clusters=args.n_clusters,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Generate HTML
    print(f"\nGenerating HTML...")
    generate_html(data, output_path)
    
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"Open in browser: file://{output_path.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
