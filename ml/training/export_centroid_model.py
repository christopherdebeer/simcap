#!/usr/bin/env python3
"""
Export a centroid-based classifier model for TensorFlow.js.

Instead of a neural network, we use the measured signature centroids directly.
The JS code performs nearest-centroid classification.
"""

import json
import numpy as np
from pathlib import Path
import sys


from ml.simulation.aligned_generator import AlignedGenerator


def main():
    print("=" * 70)
    print("EXPORTING CENTROID-BASED CLASSIFIER MODEL")
    print("=" * 70)
    
    # Load ground truth session
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    
    print(f"\nLoading ground truth from: {session_path.name}")
    gen = AlignedGenerator(session_path)
    
    # Extract class centroids (mean vectors for each configuration)
    centroids = {}
    
    # Baseline
    centroids['00000'] = gen.baseline.tolist()
    
    # All measured configurations
    for code, sig in gen.signatures.items():
        # Store absolute vector (baseline + delta)
        centroids[code] = (gen.baseline + sig.mean).tolist()
    
    # Generate all 32 binary configurations
    # For configs not directly measured, use interpolation
    print("\nGenerating centroids for all 32 configurations...")
    
    finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
    single_codes = ['20000', '02000', '00200', '00020', '00002']
    
    for config_num in range(32):
        code = ''
        for i in range(5):
            code += '2' if (config_num >> (4 - i)) & 1 else '0'
        
        if code in centroids:
            continue  # Already have measured value
        
        # Interpolate from single-finger signatures
        vec = gen.baseline.copy()
        for i, char in enumerate(code):
            if char == '2':
                single_code = single_codes[i]
                if single_code in gen.signatures:
                    # Apply non-additivity correction (30% reduction for multi-finger)
                    n_flexed = code.count('2')
                    correction = 1.0 - 0.2 * (n_flexed - 1) / 4
                    vec += gen.signatures[single_code].mean * correction
        
        centroids[code] = vec.tolist()
    
    # Compute normalization stats from centroids
    all_vecs = np.array(list(centroids.values()))
    stats = {
        'mean': all_vecs.mean(axis=0).tolist(),
        'std': all_vecs.std(axis=0).tolist()
    }
    
    print(f"Generated {len(centroids)} class centroids")
    print(f"Stats: mean={[f'{m:.0f}' for m in stats['mean']]}")
    print(f"       std={[f'{s:.0f}' for s in stats['std']]}")
    
    # Normalize centroids for storage
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    normalized_centroids = {}
    for code, vec in centroids.items():
        normalized_centroids[code] = ((np.array(vec) - mean) / (std + 1e-8)).tolist()
    
    # Create model output
    output_dir = Path('public/models/finger_aligned_v1')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model.json (TF.js-compatible structure but with centroids)
    model_data = {
        'format': 'centroid-classifier',
        'version': '1.0',
        'description': 'Ground truth aligned centroid-based finger classifier',
        'inputShape': [None, 3],
        'outputShape': [None, 5],  # 5 fingers, binary output
        'fingerNames': ['thumb', 'index', 'middle', 'ring', 'pinky'],
        'stateNames': ['extended', 'flexed'],
        'centroids': normalized_centroids,
        'stats': stats,
        'date': str(np.datetime64('today'))
    }
    
    model_path = output_dir / 'model.json'
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"\nSaved model: {model_path}")
    
    # Save config.json for compatibility
    config = {
        'stats': stats,
        'inputShape': [None, 3],
        'fingerNames': ['thumb', 'index', 'middle', 'ring', 'pinky'],
        'stateNames': ['extended', 'flexed'],
        'description': 'Ground truth aligned centroid classifier',
        'version': 'aligned_v1',
        'modelType': 'centroid',
        'date': str(np.datetime64('today'))
    }
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Test the model
    print("\n" + "-" * 70)
    print("Testing centroid classifier...")
    
    # Load real data for testing
    with open(session_path) as f:
        session = json.load(f)
    
    samples = session.get('samples', [])
    labels = session.get('labels', [])
    
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])
    
    correct = 0
    total = 0
    
    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})
        
        if not fingers:
            continue
        
        # Get true label
        code = ''
        for f in finger_order:
            state = fingers.get(f, 'unknown')
            if state == 'extended':
                code += '0'
            elif state == 'flexed':
                code += '2'
            else:
                continue  # Skip partial
        
        if len(code) != 5:
            continue
        
        # Classify each sample
        for i in range(start, min(end, len(mx))):
            vec = np.array([mx[i], my[i], mz[i]])
            vec_norm = (vec - mean) / (std + 1e-8)
            
            # Find nearest centroid
            min_dist = float('inf')
            pred_code = '00000'
            for c, cent in normalized_centroids.items():
                dist = np.linalg.norm(vec_norm - np.array(cent))
                if dist < min_dist:
                    min_dist = dist
                    pred_code = c
            
            if pred_code == code:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy on ground truth data: {accuracy:.1%} ({correct}/{total})")
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
