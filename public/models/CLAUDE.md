# Models Directory

This directory contains TensorFlow.js models for client-side inference in the Gambit app.

## Directory Structure

```
public/models/
├── CLAUDE.md              # This file
├── gesture_v1/            # 10-class gesture classification
│   ├── model.json
│   └── group1-shard1of1.bin
├── finger_contrastive_v1/ # 6-feature contrastive finger model
│   ├── model.json
│   ├── config.json
│   └── group1-shard1of1.bin
├── finger_aligned_v1/     # V1: 3-feature binary finger model
│   ├── model.json
│   ├── config.json
│   └── group1-shard1of1.bin
├── finger_aligned_v3/     # V3: 3-feature, w=10, 50% synthetic (91% cross-orient)
│   ├── model.json
│   ├── config.json
│   └── group1-shard1of1.bin
└── finger_aligned_v4/     # V4-Regularized: 70.1% cross-orient, best generalization (CURRENT)
    ├── model.json
    ├── config.json
    ├── group1-shard1of1.bin
    ├── model.keras         # Original Keras model
    └── saved_model/        # TensorFlow SavedModel format
```

## Model Lifecycle

### 1. Training (ml/ directory)

Models are trained using Python/TensorFlow in the `ml/` directory:

```bash
# Train a new model
python -m ml.train_finger_model --output models/finger_v2

# The training script should output:
# - model.keras (Keras format for further training)
# - Normalization stats (mean, std) for the model
```

### 2. Conversion to TensorFlow.js

After training, convert to TensorFlow.js format:

```bash
# Install tensorflowjs if needed
pip install tensorflowjs

# Convert Keras model to TensorFlow.js
tensorflowjs_converter --input_format=keras \
    ml/models/your_model.keras \
    public/models/your_model_v1/
```

### 3. Register in Code

Add the model to the unified registry in `apps/gambit/gesture-inference.ts`:

```typescript
// In ALL_MODELS array:
{
  id: 'your_model_v1',
  name: 'Your Model (v1)',
  type: 'finger_magnetic',  // or 'gesture'
  path: '/models/your_model_v1/model.json',
  stats: {
    mean: [/* from training */],
    std: [/* from training */]
  },
  description: 'Description of what this model does',
  date: '2025-01-01',
  active: true,  // Set to true if this should be the default
  inputFeatures: 3,  // For finger models
  numStates: 2       // 2=binary, 3=extended/partial/flexed
}
```

### 4. Update UI (if needed)

The Gambit app (`apps/gambit/gambit-app.ts`) automatically shows the appropriate UI based on model type:
- `gesture` models → 10-class probability bars
- `finger_magnetic` models → 5-finger state display

If adding a new model type, update `loadModel()` and `wrappedUpdateData()` in `gambit-app.ts`.

## Keeping Things in Sync

### When Adding a New Model

1. Train model in `ml/`
2. Convert to TensorFlow.js format
3. Copy to `public/models/<model_name>/`
4. Add entry to `ALL_MODELS` in `gesture-inference.ts`
5. Set `active: true` if it should be the default
6. Set previous model's `active: false`
7. Build and test: `npm run build && npm run dev`

### When Updating an Existing Model

1. Retrain with same architecture
2. Re-convert to TensorFlow.js
3. Replace files in `public/models/<model_name>/`
4. Update `stats` in the model registry if normalization changed
5. Update `date` field
6. Build and test

### When Removing a Model

1. Remove directory from `public/models/`
2. Remove entry from `ALL_MODELS` in `gesture-inference.ts`
3. If it was the active model, set another model's `active: true`
4. Build and test

## Model Types

| Type | Input | Output | Inference Class |
|------|-------|--------|-----------------|
| `gesture` | 9 features × 50 samples | 10 class probabilities | `GestureInference` |
| `finger_magnetic` | 3-6 features × 1 sample | 5 fingers × N states | `MagneticFingerInference` |
| `finger_window` | 9 features × 50 samples | 5 fingers × N states | `FingerTrackingInference` |

## Normalization Stats

Every model requires normalization statistics from training:

```python
# During training, compute and save:
mean = X_train.mean(axis=0)  # Per-feature mean
std = X_train.std(axis=0)    # Per-feature std

# Save to config.json alongside model
{
  "mean": [5188, 6179, 17152],
  "std": [5732, 7181, 16189]
}
```

These stats MUST match between training and inference, or predictions will be wrong.

## Testing a Model

1. Start dev server: `npm run dev`
2. Open Gambit app: http://localhost:5173/gambit/
3. Select model from dropdown in Inference card
4. Connect device or play back a session
5. Verify predictions appear and make sense

## Troubleshooting

### Model fails to load
- Check browser console for 404 errors
- Verify `model.json` path is correct (should start with `/models/`)
- Check that all shard files (`group1-shard1of1.bin`) are present

### Predictions are wrong/random
- Verify normalization stats match training
- Check input feature order matches training
- Verify model was converted correctly

### "TensorFlow.js not loaded"
- Ensure `<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs">` is in HTML
- Check for JavaScript errors before model loading
