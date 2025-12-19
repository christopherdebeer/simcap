#!/usr/bin/env python3
"""
Generate Manifests for GitHub Branching Storage

Generates manifest files for session data and visualizations:
- data/GAMBIT/manifest.json - Index of all session files (to be committed to main)
- visualizations/manifests/index.json - Index of all visualization manifests

These manifests enable efficient listing without API calls to GitHub.

Usage:
    # Generate session manifest from local data
    python scripts/generate-manifests.py --sessions --data-dir data/GAMBIT

    # Generate visualization manifest index from local manifests
    python scripts/generate-manifests.py --visualizations --manifest-dir visualizations/manifests

    # Generate both
    python scripts/generate-manifests.py --all

    # Upload manifests to GitHub after generation
    python scripts/generate-manifests.py --all --upload
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# GitHub configuration
GITHUB_OWNER = "christopherdebeer"
GITHUB_REPO = "simcap"
DATA_BRANCH = "data"
IMAGES_BRANCH = "images"
MAIN_BRANCH = "main"


def get_raw_url(branch: str, path: str) -> str:
    """Construct raw.githubusercontent.com URL."""
    return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{branch}/{path}"


def normalize_timestamp(timestamp: str) -> str:
    """Convert underscore-based timestamp to colon-based (ISO format)."""
    return re.sub(r'T(\d{2})_(\d{2})_(\d{2})', r'T\1:\2:\3', timestamp)


def denormalize_timestamp(timestamp: str) -> str:
    """Convert colon-based timestamp to underscore-based (filename safe)."""
    return re.sub(r'T(\d{2}):(\d{2}):(\d{2})', r'T\1_\2_\3', timestamp)


def parse_timestamp_from_filename(filename: str) -> Optional[str]:
    """Parse ISO timestamp from filename like '2025-12-11T13_26_33.209Z.json'"""
    name = filename.replace('.json', '')
    # Convert underscores to colons
    name = normalize_timestamp(name)
    try:
        # Validate it's a valid timestamp
        datetime.fromisoformat(name.replace('Z', '+00:00'))
        return name
    except ValueError:
        return None


def get_session_info(filepath: Path, base_url: str) -> Dict[str, Any]:
    """Extract metadata from a session file."""
    filename = filepath.name
    stat = filepath.stat()

    info = {
        'filename': filename,
        'size': stat.st_size,
        'url': f"{base_url}/{filename}",
    }

    # Parse timestamp from filename
    ts = parse_timestamp_from_filename(filename)
    if ts:
        info['timestamp'] = ts

    # Try to read file metadata
    try:
        with open(filepath, 'r') as f:
            chunk = f.read(2000)

            if chunk.strip().startswith('{'):
                f.seek(0)
                data = json.load(f)
                info['version'] = data.get('version', '2.0')
                info['timestamp'] = data.get('timestamp', info.get('timestamp'))
                info['sampleCount'] = len(data.get('samples', []))

                # Calculate duration
                metadata = data.get('metadata', {})
                sample_rate = metadata.get('sample_rate', 26)
                if info.get('sampleCount', 0) > 0:
                    info['durationSec'] = round(info['sampleCount'] / sample_rate, 1)

            elif chunk.strip().startswith('['):
                f.seek(0)
                data = json.load(f)
                info['version'] = '1.0'
                info['sampleCount'] = len(data) if isinstance(data, list) else 0
                if info.get('sampleCount', 0) > 0:
                    info['durationSec'] = round(info['sampleCount'] / 20, 1)

    except (json.JSONDecodeError, IOError) as e:
        info['version'] = 'error'
        info['error'] = str(e)

    return info


def generate_sessions_manifest(
    data_dir: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate manifest.json for all session files.

    Args:
        data_dir: Directory containing session JSON files
        output_path: Where to write manifest (default: data_dir/manifest.json)
        verbose: Print progress

    Returns:
        Generated manifest dict
    """
    if output_path is None:
        output_path = data_dir / 'manifest.json'

    base_url = get_raw_url(DATA_BRANCH, f"GAMBIT")

    # Find all session JSON files
    exclude_files = {'manifest.json', 'gambit_calibration.json', 'dataset_stats.npz', 'index.json'}

    sessions = []
    for filepath in sorted(data_dir.glob('*.json')):
        if filepath.name in exclude_files:
            continue
        if filepath.name.startswith('.'):
            continue

        if verbose:
            print(f"  Processing: {filepath.name}")

        info = get_session_info(filepath, base_url)
        sessions.append(info)

    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    manifest = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'directory': 'GAMBIT',
        'branch': DATA_BRANCH,
        'baseUrl': base_url,
        'sessionCount': len(sessions),
        'sessions': sessions,
    }

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\nGenerated session manifest: {output_path}")
        print(f"  Sessions: {len(sessions)}")

    return manifest


def parse_visualization_manifest(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parse a visualization manifest file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def generate_visualization_index(
    manifest_dir: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate index.json for visualization manifests.

    Args:
        manifest_dir: Directory containing visualization manifest JSON files
        output_path: Where to write index (default: manifest_dir/index.json)
        verbose: Print progress

    Returns:
        Generated index dict
    """
    if output_path is None:
        output_path = manifest_dir / 'index.json'

    base_image_url = get_raw_url(IMAGES_BRANCH, "")

    # Group manifests by session, keeping latest
    sessions: Dict[str, Dict[str, Any]] = {}

    for filepath in sorted(manifest_dir.glob('*.json')):
        if filepath.name == 'index.json':
            continue

        manifest = parse_visualization_manifest(filepath)
        if not manifest:
            continue

        session_ts = manifest.get('sessionTimestamp', '')
        generated_at = manifest.get('generatedAt', '')

        if verbose:
            print(f"  Processing: {filepath.name}")

        # Keep only the latest manifest per session
        if session_ts not in sessions or generated_at > sessions[session_ts].get('generatedAt', ''):
            sessions[session_ts] = {
                'sessionTimestamp': session_ts,
                'generatedAt': generated_at,
                'hasComposite': bool(manifest.get('images', {}).get('composite')),
                'windowCount': len(manifest.get('windows', [])),
                'manifestPath': f"visualizations/manifests/{filepath.name}",
            }

    # Convert to sorted list
    session_list = sorted(
        sessions.values(),
        key=lambda x: x.get('sessionTimestamp', ''),
        reverse=True
    )

    index = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'imageBranch': IMAGES_BRANCH,
        'baseImageUrl': base_image_url,
        'sessions': session_list,
    }

    # Write index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"\nGenerated visualization index: {output_path}")
        print(f"  Sessions: {len(session_list)}")

    return index


def upload_manifest(
    filepath: Path,
    dest_path: str,
    branch: str = MAIN_BRANCH,
    verbose: bool = True
) -> bool:
    """
    Upload manifest to GitHub.

    Requires GITHUB_TOKEN environment variable.
    """
    try:
        from ml.github_upload import upload_content_to_github
    except ImportError:
        print("Error: Could not import github_upload module")
        return False

    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    try:
        result = upload_content_to_github(
            content=content,
            path=dest_path,
            branch=branch,
            token=token,
            message=f"Update {dest_path}",
        )

        if verbose:
            print(f"Uploaded: {dest_path} -> {result.get('url', 'unknown')}")

        return True

    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifests for GitHub branching storage"
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Generate session manifest (data/GAMBIT/manifest.json)",
    )
    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Generate visualization index (visualizations/manifests/index.json)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all manifests",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "GAMBIT",
        help="Directory containing session data (default: data/GAMBIT)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "manifests",
        help="Directory containing visualization manifests (default: visualizations/manifests)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for manifests (default: same as input)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated manifests to GitHub (requires GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not args.sessions and not args.visualizations and not args.all:
        args.all = True

    verbose = not args.quiet

    if args.all or args.sessions:
        if args.data_dir.exists():
            if verbose:
                print(f"Generating session manifest from {args.data_dir}")

            output_path = args.output_dir / "data" / "GAMBIT" / "manifest.json" if args.output_dir else None
            manifest = generate_sessions_manifest(args.data_dir, output_path, verbose)

            if args.upload:
                manifest_path = output_path or args.data_dir / "manifest.json"
                upload_manifest(manifest_path, "data/GAMBIT/manifest.json", MAIN_BRANCH, verbose)
        else:
            print(f"Warning: Data directory not found: {args.data_dir}")

    if args.all or args.visualizations:
        if args.manifest_dir.exists():
            if verbose:
                print(f"\nGenerating visualization index from {args.manifest_dir}")

            output_path = args.output_dir / "visualizations" / "manifests" / "index.json" if args.output_dir else None
            index = generate_visualization_index(args.manifest_dir, output_path, verbose)

            if args.upload:
                index_path = output_path or args.manifest_dir / "index.json"
                upload_manifest(index_path, "visualizations/manifests/index.json", MAIN_BRANCH, verbose)
        else:
            print(f"Warning: Manifest directory not found: {args.manifest_dir}")


if __name__ == "__main__":
    main()
