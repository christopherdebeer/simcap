#!/usr/bin/env python3
"""Convert Keras model to TensorFlow.js format."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path
import subprocess
import json

# Load the model
model_path = Path('public/models/finger_aligned_v1/model.keras')
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# Save as SavedModel format
saved_model_path = Path('public/models/finger_aligned_v1/saved_model')
print(f"Exporting as SavedModel to {saved_model_path}...")
model.export(saved_model_path)

# Convert to TF.js graph model
output_path = Path('public/models/finger_aligned_v1')
print(f"Converting to TensorFlow.js...")

result = subprocess.run([
    'tensorflowjs_converter',
    '--input_format=tf_saved_model',
    '--output_format=tfjs_graph_model',
    '--signature_name=serving_default',
    str(saved_model_path),
    str(output_path)
], capture_output=True, text=True)

if result.returncode == 0:
    print("Conversion successful!")
    print(f"Model files: {list(output_path.glob('*.json'))}")
else:
    print(f"Conversion failed: {result.stderr}")
    
# Update config with model type
config_path = output_path / 'config.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    config['modelType'] = 'graph'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("Updated config.json with modelType=graph")
