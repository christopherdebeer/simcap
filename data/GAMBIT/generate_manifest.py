#!/usr/bin/env python3
"""
Generate manifest.json for GAMBIT session files.

Scans the data/GAMBIT directory for session JSON files and creates
a manifest with metadata for the web UI to load.

Usage:
    python generate_manifest.py

Or run from project root:
    python data/GAMBIT/generate_manifest.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

def parse_timestamp_from_filename(filename):
    """Parse ISO timestamp from filename like '2025-12-11T13_26_33.209Z.json'"""
    # Remove .json extension
    name = filename.replace('.json', '')
    # Replace underscores back to colons for parsing
    name = name.replace('_', ':')
    try:
        return datetime.fromisoformat(name.replace('Z', '+00:00'))
    except ValueError:
        return None

def get_session_info(filepath):
    """Extract metadata from a session file."""
    filename = os.path.basename(filepath)
    stat = os.stat(filepath)

    info = {
        'filename': filename,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat() + 'Z',
    }

    # Parse timestamp from filename
    ts = parse_timestamp_from_filename(filename)
    if ts:
        info['timestamp'] = ts.isoformat()

    # Try to read file metadata without loading all samples
    try:
        with open(filepath, 'r') as f:
            # Read first chunk to detect format
            chunk = f.read(1000)

            if chunk.strip().startswith('{'):
                # New v2.0 format with metadata
                f.seek(0)
                data = json.load(f)
                info['version'] = data.get('version', '1.0')
                info['timestamp'] = data.get('timestamp', info.get('timestamp'))
                info['sampleCount'] = len(data.get('samples', []))

                # Extract additional metadata if present
                if 'label' in data:
                    info['label'] = data['label']
                if 'gesture' in data:
                    info['gesture'] = data['gesture']

            elif chunk.strip().startswith('['):
                # Old format - just array of samples
                f.seek(0)
                data = json.load(f)
                info['version'] = '1.0'
                info['sampleCount'] = len(data) if isinstance(data, list) else 0
            else:
                info['version'] = 'unknown'
                info['sampleCount'] = 0

    except (json.JSONDecodeError, IOError) as e:
        info['version'] = 'error'
        info['sampleCount'] = 0
        info['error'] = str(e)

    # Calculate duration estimate (assuming ~20Hz sample rate)
    if info.get('sampleCount', 0) > 0:
        info['durationSec'] = round(info['sampleCount'] / 20, 1)

    return info

def generate_manifest(data_dir=None):
    """Generate manifest.json for all session files in the directory."""

    # Default to script's directory
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)

    # Find all session JSON files (exclude manifest, calibration, etc.)
    exclude_files = {'manifest.json', 'gambit_calibration.json', 'dataset_stats.npz'}

    sessions = []
    for filepath in sorted(data_dir.glob('*.json')):
        if filepath.name in exclude_files:
            continue
        if filepath.name.startswith('.'):
            continue

        print(f"Processing: {filepath.name}")
        info = get_session_info(filepath)
        sessions.append(info)

    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    manifest = {
        'generated': datetime.utcnow().isoformat() + 'Z',
        'directory': str(data_dir.name),
        'sessionCount': len(sessions),
        'sessions': sessions
    }

    # Write manifest
    manifest_path = data_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated manifest with {len(sessions)} sessions")
    print(f"Output: {manifest_path}")

    return manifest

if __name__ == '__main__':
    generate_manifest()
