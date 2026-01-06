# ArXiv Research Summary: IMU Orientation Invariance & Sensor-Based Recognition

## Papers Reviewed in Full

### 1. PPDA: Physically Plausible Data Augmentation (arXiv:2508.13284)

**Core Problem**: Signal Transformation-based Data Augmentation (STDA) like rotation, scaling, and time warping often produces physically implausible signals that cause data-label mismatches.

**Key Insight**: Instead of transforming recorded signals directly, modify physics simulation parameters:
- **Body (B)**: Skeletal representation
- **Dynamics (D)**: Joint orientations/movements over time
- **Placement (P)**: Sensor position/orientation relative to body segments
- **Hardware (H)**: Sensor noise and bias characteristics

**Results**:
- 3.7 percentage point average improvement in macro F1-score
- 40-80% fewer training subjects required vs no augmentation
- Particularly effective with minimal training data (1-2 subjects)

**Relevance to Our Work**: Our physics-based synthetic data generation is conceptually similar. Key improvements could include:
- Explicit modeling of hardware bias (magnetometer calibration variation)
- Sensor placement variability (±25° around each axis)
- More realistic movement dynamics rather than just position changes

---

### 2. Activity Recognition Invariant to Sensor Orientation (PMC5579846)

**Two Approaches Tested**:

1. **Heuristic OIT (9 elements)**:
   - Norms of signal and its 1st/2nd order differences
   - Angles between successive time samples
   - Angles between rotation axes from cross products
   - Result: 15.54% average accuracy drop under random rotation

2. **SVD-Based OIT**:
   - Single 3×3 rotational transformation from SVD
   - Principal axes rotate with data constellation
   - Result: **7.56% average accuracy drop** (best method)

**Baseline Comparison**:
- Random rotation without transformation: 21.21% drop
- Euclidean norm only: 13.50% drop

**Key Finding**: SVD-based transformation outperforms heuristics because it preserves more discriminative information while removing orientation dependence.

---

### 3. UniMTS: Unified Pre-training for Motion Time Series (NeurIPS 2024)

**Rotation-Invariant Augmentation**:
```
For each training iteration:
1. Sample random rotation matrix R_δ uniformly from SO(3)
2. Apply: x̂ = R_δ · x̃ (same rotation across all timesteps)
3. Model learns representations invariant to device orientation
```

**Architecture**: Spatio-temporal graph network
- Spatial edges: anatomically adjacent joints
- Temporal edges: consecutive timesteps
- Random masking of 1-5 joints during training

**Performance**:
- **340% improvement** over ImageBind in zero-shot setting
- 4.94M parameters (vs 18.69M for ImageBind)
- Tested across 18 benchmark datasets

**Key Innovation**: Contrastive learning aligns motion time series with text descriptions (GPT-3.5 paraphrases), enabling zero-shot generalization to unseen activities.

---

### 4. IMG2IMU (arXiv:2209.00945)

**Method**: Convert IMU sensor data to spectrograms (visual representations) to leverage pre-trained vision models.

**Technique**:
- Sensor-aware pre-training with custom augmentations
- Contrastive learning tailored to sensor data properties

**Results**: Outperforms sensor-pretrained baselines by **9.6%p F1-score** average across 4 IMU sensing tasks.

---

### 5. Cross-Modal Transfer Learning Survey (arXiv:2403.15444)

**Key Approaches**:
1. **Instance-Based**: Map one sensor's data to another's input space (IMUTube, CROMOSim)
2. **Feature-Based**: Project to shared representation space (RecycleML, COCOA, IMU2CLIP)

**IMU Challenges Noted**:
- Sensor drift over time
- Position/orientation variability
- Cross-device variations

**Future Direction**: Generative models for cross-modal synthesis (limited by data scarcity).

---

## Recommended Next Experiments

### Experiment 1: SO(3) Rotation Augmentation (UniMTS-Inspired)
**Rationale**: Our current synthetic data doesn't apply random rotations during training.

```python
def so3_rotation_augmentation(window, random_state=None):
    """Apply random SO(3) rotation to magnetometer window."""
    rng = np.random.default_rng(random_state)
    # Sample random rotation using scipy
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=rng)
    return (R.as_matrix() @ window.T).T
```

**Expected Outcome**: Better orientation invariance without losing discriminative power.

---

### Experiment 2: SVD-Based Orientation-Invariant Features
**Rationale**: SVD transformation showed only 7.56% accuracy drop vs 21.21% for random rotations.

```python
def svd_orientation_invariant(window):
    """Transform window to orientation-invariant representation."""
    # Center the data
    centered = window - window.mean(axis=0)
    # SVD to get principal axes
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Transform to principal axis frame
    return centered @ Vt.T
```

**Key Difference from $-family**: SVD uses data-driven axes, not arbitrary centering/scaling.

---

### Experiment 3: Heuristic OIT Features (9 elements)
**Rationale**: Explicit orientation-invariant features that preserve temporal dynamics.

```python
def heuristic_oit_features(window):
    """9 orientation-invariant elements per timestep."""
    # 1-3: Norms of signal and differences
    norms = np.linalg.norm(window, axis=1)
    diff1 = np.diff(window, axis=0)
    diff2 = np.diff(diff1, axis=0)
    norm_diff1 = np.linalg.norm(diff1, axis=1)
    norm_diff2 = np.linalg.norm(diff2, axis=1)

    # 4-6: Angles between successive samples
    angles = []
    for i in range(len(window)-1):
        cos_angle = np.dot(window[i], window[i+1]) / (norms[i] * norms[i+1] + 1e-8)
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

    # 7-9: Cross product rotation axes
    # ... (additional computations)

    return np.column_stack([norms, norm_diff1, ...])
```

---

### Experiment 4: Spectrogram + Vision Model (IMG2IMU-Inspired)
**Rationale**: Leverage pre-trained vision models on sensor spectrograms.

```python
def magnetometer_to_spectrogram(samples, fs=200, nperseg=64):
    """Convert magnetometer time series to spectrogram image."""
    from scipy import signal
    spectrograms = []
    for axis in range(3):
        f, t, Sxx = signal.spectrogram(samples[:, axis], fs, nperseg=nperseg)
        spectrograms.append(Sxx)
    return np.stack(spectrograms, axis=-1)  # H x W x 3 (RGB-like)
```

---

### Experiment 5: Physics-Based Augmentation (PPDA-Inspired)
**Rationale**: Our synthetic data could model more physical variations.

**New Parameters to Model**:
1. **Hardware bias**: Random per-axis magnetometer bias [-1.0, 1.0] μT
2. **Placement variation**: ±25° rotation around each axis
3. **Calibration error**: Scale factors [0.95, 1.05] per axis

```python
def physics_augmentation(samples, seed=None):
    rng = np.random.default_rng(seed)

    # Hardware bias
    bias = rng.uniform(-1.0, 1.0, size=3)

    # Placement rotation (small)
    angles = rng.uniform(-np.pi/7.2, np.pi/7.2, size=3)  # ±25°
    R = Rotation.from_euler('xyz', angles).as_matrix()

    # Calibration scale
    scale = rng.uniform(0.95, 1.05, size=3)

    return (samples @ R.T) * scale + bias
```

---

### Experiment 6: Contrastive Learning (UniMTS-Inspired)
**Rationale**: Learn representations that are invariant to sensor orientation through contrastive loss.

**Approach**:
1. Create positive pairs: same gesture, different orientations (via SO(3) augmentation)
2. Create negative pairs: different gestures
3. Train encoder to minimize contrastive loss

```python
def contrastive_loss(z_i, z_j, temperature=0.5):
    """NT-Xent contrastive loss."""
    batch_size = z_i.shape[0]
    z = tf.concat([z_i, z_j], axis=0)
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature

    # Mask out self-similarity
    mask = tf.eye(2 * batch_size)
    sim_matrix = sim_matrix - mask * 1e9

    # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, sim_matrix)
```

---

## Priority Ranking

| Priority | Experiment | Effort | Expected Impact |
|----------|------------|--------|-----------------|
| 1 | SO(3) Rotation Augmentation | Low | High - directly addresses orientation |
| 2 | SVD-Based Features | Low | Medium - principled alternative to $-family |
| 3 | Physics-Based Augmentation | Medium | High - more realistic synthetic data |
| 4 | Contrastive Learning | High | High - state-of-the-art approach |
| 5 | Heuristic OIT Features | Medium | Medium - explicit invariant features |
| 6 | Spectrogram + Vision | High | Medium - requires pre-trained model |

---

## Key Takeaways

1. **$-family centering hurts magnetometer data** because it destroys absolute field information. This is consistent with our ablation findings.

2. **SVD-based transformation** is more principled than arbitrary normalization - it finds data-driven principal axes.

3. **SO(3) rotation augmentation** during training is the state-of-the-art approach for orientation invariance (UniMTS).

4. **Physics-based augmentation** outperforms signal transformation because it produces physically plausible variations.

5. **Contrastive learning** enables zero-shot generalization by learning semantic representations that transfer across orientations.

---

## Sources

- [PPDA: Physics-Based IMU Data Augmentation](https://arxiv.org/abs/2508.13284)
- [Activity Recognition Invariant to Sensor Orientation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5579846/)
- [UniMTS: Unified Pre-training for Motion Time Series (NeurIPS 2024)](https://arxiv.org/abs/2410.19818)
- [IMG2IMU: Cross-Modal Transfer](https://arxiv.org/abs/2209.00945)
- [Cross-Modal Transfer Learning Survey](https://arxiv.org/abs/2403.15444)
- [Differential Rotational Transformations with Quaternions](https://www.mdpi.com/1424-8220/18/8/2725)
- [UniMTS GitHub Repository](https://github.com/xiyuanzh/UniMTS)
