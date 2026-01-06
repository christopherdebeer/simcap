# SIMCAP Unsupervised Clustering Guide

## Overview

The clustering feature enables **unsupervised learning** on unlabeled sensor data to automatically discover gesture patterns. This is useful for:

1. **Initial exploration** - Understanding natural groupings in your data
2. **Semi-automated labeling** - Clustering similar movements, then manually assigning gesture names
3. **Validation** - Checking if labeled gestures form distinct clusters
4. **Rapid prototyping** - Quickly testing if your data contains distinguishable patterns

## Quick Start

### 1. Run Clustering

```bash
cd /Users/cdbeer/dev/simcap
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --n-clusters 10 \
    --create-templates \
    --visualize-clusters
```

This will:
- Load all unlabeled sessions from `data/GAMBIT`
- Extract statistical features from sensor windows
- Cluster windows into 10 groups using K-means
- Generate label templates for each session
- Create 2D/3D visualizations of clusters

### 2. Review Results

Check the output directory (`ml/models/` by default):

```
ml/models/
├── clustering_results.json      # Raw clustering data
├── cluster_analysis.json        # Detailed statistics
├── clusters_2d_pca.png         # 2D PCA visualization
├── clusters_3d_pca.png         # 3D PCA visualization
├── clusters_2d_tsne.png        # 2D t-SNE visualization
└── label_templates/            # Template metadata files
    ├── 2023-11-24T17:14:18.479Z.json.template.json
    ├── 2023-11-24T17:24:38.129Z.json.template.json
    └── ...
```

### 3. Assign Gesture Names

Open a template file (e.g., `ml/models/label_templates/2023-11-24T17:14:18.479Z.json.template.json`):

```json
{
  "timestamp": "2023-11-24T17:14:18.479Z",
  "labels": [
    {
      "start_sample": 0,
      "end_sample": 50,
      "gesture": "CLUSTER_3",  // ← Change this to actual gesture name
      "confidence": "medium",
      "cluster_id": 3
    },
    {
      "start_sample": 25,
      "end_sample": 75,
      "gesture": "CLUSTER_3",  // ← Change this to actual gesture name
      "confidence": "medium",
      "cluster_id": 3
    }
  ]
}
```

**Replace `CLUSTER_X` with actual gesture names:**
- `rest` - Hand at rest
- `fist` - Closed fist
- `open_palm` - Open palm
- `index_up` - Index finger pointing up
- `peace` - Peace sign (V)
- `thumbs_up` - Thumbs up
- `ok_sign` - OK sign
- `pinch` - Pinch gesture
- `grab` - Grabbing motion
- `wave` - Waving motion

### 4. Finalize Labels

Once you've assigned gesture names:

```bash
# Rename template to metadata file
cd ml/models/label_templates
mv 2023-11-24T17:14:18.479Z.json.template.json \
   2023-11-24T17:14:18.479Z.json.meta.json

# Move to data directory
mv *.meta.json ../../data/GAMBIT/
```

### 5. Train Supervised Model

Now that you have labeled data:

```bash
python -m ml.train --data-dir data/GAMBIT --epochs 50
```

## Command-Line Options

### Basic Options

```bash
--cluster-only              # Enable clustering mode (required)
--data-dir PATH            # Path to data directory (default: data/GAMBIT)
--output-dir PATH          # Output directory (default: ml/models)
--window-size N            # Window size in samples (default: 50)
--stride N                 # Window stride (default: 25)
```

### Clustering Algorithm

```bash
--cluster-method METHOD    # Algorithm: kmeans or dbscan (default: kmeans)
--n-clusters N            # Number of clusters for K-means (default: 10)
```

**K-means** (default):
- You specify the number of clusters
- Fast and deterministic
- Good for well-separated, spherical clusters

**DBSCAN** (density-based):
```bash
--cluster-method dbscan \
--dbscan-eps 0.5 \
--dbscan-min-samples 5
```
- Automatically determines number of clusters
- Can find arbitrary-shaped clusters
- Identifies noise/outliers (labeled as -1)

### Output Options

```bash
--create-templates         # Generate label template files
--visualize-clusters       # Create 2D/3D visualizations
```

## Understanding the Output

### Cluster Quality Metrics

**Silhouette Score** (range: -1 to 1)
- **> 0.5**: Strong, well-separated clusters
- **0.3 - 0.5**: Reasonable structure (typical for real-world data)
- **< 0.3**: Weak or overlapping clusters

**Davies-Bouldin Score** (lower is better)
- **< 1.0**: Good separation
- **1.0 - 2.0**: Moderate separation
- **> 2.0**: Poor separation

### Visualizations

**2D PCA Plot** (`clusters_2d_pca.png`)
- Shows clusters in 2D space using Principal Component Analysis
- Fast to compute, preserves global structure
- Good for initial overview

**3D PCA Plot** (`clusters_3d_pca.png`)
- 3D view of clusters
- Better separation visibility than 2D

**2D t-SNE Plot** (`clusters_2d_tsne.png`)
- Non-linear dimensionality reduction
- Often shows better cluster separation than PCA
- Slower to compute (skipped for >5000 samples)

## Workflow Examples

### Example 1: Quick Exploration

```bash
# Just run clustering to see what patterns exist
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --visualize-clusters
```

Review the visualizations to understand your data structure.

### Example 2: Full Semi-Automated Labeling

```bash
# 1. Cluster with template generation
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --n-clusters 10 \
    --create-templates \
    --visualize-clusters

# 2. Review visualizations
open ml/models/clusters_2d_tsne.png

# 3. Edit templates (assign gesture names)
# ... manual editing ...

# 4. Move templates to data directory
mv ml/models/label_templates/*.meta.json data/GAMBIT/

# 5. Train supervised model
python -m ml.train --data-dir data/GAMBIT --epochs 50
```

### Example 3: DBSCAN for Automatic Cluster Discovery

```bash
# Let DBSCAN find the natural number of clusters
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --cluster-method dbscan \
    --dbscan-eps 0.5 \
    --dbscan-min-samples 5 \
    --visualize-clusters
```

Check the output to see how many clusters were found.

### Example 4: Different Window Sizes

```bash
# Try larger windows (2 seconds instead of 1)
python -m ml.train \
    --data-dir data/GAMBIT \
    --cluster-only \
    --window-size 100 \
    --stride 50 \
    --n-clusters 10 \
    --visualize-clusters
```

## Tips and Best Practices

### Choosing Number of Clusters

1. **Start with gesture vocabulary size** - If you plan to recognize 10 gestures, try `--n-clusters 10`
2. **Try multiple values** - Run with 5, 10, 15 clusters and compare metrics
3. **Use DBSCAN** - Let the algorithm find the natural number of clusters
4. **Check visualizations** - Look for clear separation in t-SNE plots

### Improving Cluster Quality

1. **Collect more data** - More samples = better clustering
2. **Ensure variety** - Collect data from different sessions/subjects
3. **Clean data** - Remove sessions with sensor errors or unusual movements
4. **Adjust window size** - Try different window sizes (25, 50, 100 samples)

### Labeling Strategy

1. **Start with clear clusters** - Label the most distinct clusters first
2. **Use confidence levels** - Mark uncertain labels as "medium" or "low"
3. **Review boundaries** - Check that gesture transitions aren't included
4. **Iterate** - Train a model, review mistakes, refine labels

### Common Issues

**Low silhouette score (<0.3)**
- Data may not have clear natural groupings
- Try different number of clusters
- Consider if gestures are too similar
- May need more distinctive gestures

**Too many small clusters**
- Reduce `--n-clusters`
- Try DBSCAN with larger `--dbscan-eps`
- May indicate noisy data

**All data in one cluster**
- Increase `--n-clusters`
- Try DBSCAN with smaller `--dbscan-eps`
- Data may be too uniform

## Technical Details

### Feature Extraction

For each window (50 samples × 9 axes), we compute:
- Mean (9 features)
- Standard deviation (9 features)
- Minimum (9 features)
- Maximum (9 features)
- Range (9 features)

Total: **45 statistical features** per window

### Algorithms

**K-means**
- Iteratively assigns points to nearest cluster center
- Updates centers as mean of assigned points
- Converges to local optimum
- Requires specifying K (number of clusters)

**DBSCAN**
- Groups points that are closely packed together
- Marks points in low-density regions as outliers
- Automatically determines number of clusters
- Parameters: `eps` (neighborhood radius), `min_samples` (minimum cluster size)

### Dimensionality Reduction

**PCA (Principal Component Analysis)**
- Linear transformation to maximize variance
- Fast and deterministic
- Good for initial exploration

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Non-linear, preserves local structure
- Better cluster separation visualization
- Slower, stochastic (results vary slightly)

## Integration with Supervised Training

The clustering workflow integrates seamlessly with supervised training:

```
Unlabeled Data
    ↓
Clustering (unsupervised)
    ↓
Label Templates
    ↓
Manual Review & Gesture Assignment
    ↓
Labeled Metadata Files
    ↓
Supervised Training
    ↓
Trained Model
```

## Related Documentation

- [Main ML README](README.md) - Overview of ML pipeline
- [Labeling Tool](label.py) - Manual labeling for fine-tuning
- [Training Script](train.py) - Supervised model training
- [Data Format](README.md#data-format) - Sensor data and metadata format

## Troubleshooting

### "No unlabeled data found"
All sessions already have `.meta.json` files. Either:
- Collect new unlabeled data
- Remove existing `.meta.json` files to re-cluster

### "scikit-learn not installed"
```bash
pip install scikit-learn
```

### "matplotlib not installed" (for visualizations)
```bash
pip install matplotlib
```

### Clustering takes too long
- Reduce number of sessions
- Use K-means instead of DBSCAN
- Skip t-SNE visualization (automatic for >5000 samples)

## Future Enhancements

Potential improvements:
- [ ] Interactive HTML visualization with Plotly
- [ ] Cluster representative samples (show typical window from each cluster)
- [ ] Hierarchical clustering with dendrogram
- [ ] Automatic gesture name suggestion based on sensor patterns
- [ ] Batch labeling UI for assigning gestures to clusters
- [ ] Cross-validation of cluster stability
- [ ] Integration with active learning for optimal sample selection

---

<link rel="stylesheet" href="../src/simcap.css">
