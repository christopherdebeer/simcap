#!/usr/bin/env python3
"""
GitHub Upload Utility for Visualizations

Uploads generated visualization files to GitHub using the Contents API.
Images are committed to the 'images' branch, manifests to 'main' branch.

Environment Variables:
    GITHUB_TOKEN: GitHub PAT with repo write access (required)

Usage:
    # Upload all visualizations to images branch
    python -m ml.github_upload --input-dir visualizations

    # Upload specific session with manifest
    python -m ml.github_upload --input-dir visualizations --session 2025-12-15T22_35_15.567Z --manifest

    # Dry run (list files without uploading)
    python -m ml.github_upload --input-dir visualizations --dry-run

    # Upload and generate manifest index
    python -m ml.github_upload --input-dir visualizations --manifest --update-index
"""

import os
import sys
import json
import base64
import argparse
import mimetypes
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# GitHub configuration
GITHUB_API_URL = "https://api.github.com"
DEFAULT_OWNER = "christopherdebeer"
DEFAULT_REPO = "simcap"
DATA_BRANCH = "data"
IMAGES_BRANCH = "images"
MAIN_BRANCH = "main"

# Manifest version
MANIFEST_VERSION = "1.0"


def get_content_type(filepath: Path) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type or "application/octet-stream"


def get_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    """Construct raw.githubusercontent.com URL."""
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def get_file_sha(
    owner: str,
    repo: str,
    path: str,
    branch: str,
    token: str
) -> Optional[str]:
    """Get SHA of existing file (needed for updates)."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {"ref": branch}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json().get("sha")
        return None
    except requests.RequestException:
        return None


def upload_file_to_github(
    filepath: Path,
    path: str,
    branch: str,
    token: str,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    message: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Upload a single file to GitHub.

    Args:
        filepath: Local file path
        path: Destination path in repository
        branch: Target branch
        token: GitHub PAT
        owner: Repository owner
        repo: Repository name
        message: Commit message (auto-generated if not provided)
        dry_run: If True, skip actual upload

    Returns:
        Dict with upload result
    """
    file_size = filepath.stat().st_size

    if dry_run:
        return {
            "path": path,
            "branch": branch,
            "size": file_size,
            "url": f"(dry-run) {get_raw_url(owner, repo, branch, path)}",
            "uploaded": False,
        }

    # Read and encode file
    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    # Check if file exists
    existing_sha = get_file_sha(owner, repo, path, branch, token)

    # Prepare request
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    payload = {
        "message": message or f"Upload {path}",
        "content": content,
        "branch": branch,
    }

    if existing_sha:
        payload["sha"] = existing_sha

    try:
        response = requests.put(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        raw_url = get_raw_url(owner, repo, branch, path)

        return {
            "path": path,
            "branch": branch,
            "size": file_size,
            "url": raw_url,
            "commit_sha": result.get("commit", {}).get("sha"),
            "uploaded": True,
        }

    except requests.HTTPError as e:
        error_body = e.response.text if e.response else str(e)
        raise Exception(f"Upload failed for {path}: {e.response.status_code} - {error_body}")
    except requests.RequestException as e:
        raise Exception(f"Network error uploading {path}: {e}")


def upload_content_to_github(
    content: str,
    path: str,
    branch: str,
    token: str,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    message: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Upload string content to GitHub (for JSON files).

    Args:
        content: String content to upload
        path: Destination path in repository
        branch: Target branch
        token: GitHub PAT
        owner: Repository owner
        repo: Repository name
        message: Commit message
        dry_run: If True, skip actual upload

    Returns:
        Dict with upload result
    """
    content_bytes = content.encode("utf-8")
    content_size = len(content_bytes)

    if dry_run:
        return {
            "path": path,
            "branch": branch,
            "size": content_size,
            "url": f"(dry-run) {get_raw_url(owner, repo, branch, path)}",
            "uploaded": False,
        }

    # Encode content
    encoded_content = base64.b64encode(content_bytes).decode()

    # Check if file exists
    existing_sha = get_file_sha(owner, repo, path, branch, token)

    # Prepare request
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    payload = {
        "message": message or f"Upload {path}",
        "content": encoded_content,
        "branch": branch,
    }

    if existing_sha:
        payload["sha"] = existing_sha

    try:
        response = requests.put(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        raw_url = get_raw_url(owner, repo, branch, path)

        return {
            "path": path,
            "branch": branch,
            "size": content_size,
            "url": raw_url,
            "commit_sha": result.get("commit", {}).get("sha"),
            "uploaded": True,
        }

    except requests.HTTPError as e:
        error_body = e.response.text if e.response else str(e)
        raise Exception(f"Upload failed for {path}: {e.response.status_code} - {error_body}")
    except requests.RequestException as e:
        raise Exception(f"Network error uploading {path}: {e}")


def collect_visualization_files(
    input_dir: Path,
    session_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Collect all visualization files to upload.

    Args:
        input_dir: Directory containing visualizations
        session_filter: Optional session timestamp to filter

    Returns:
        List of dicts with 'filepath' and 'path' keys
    """
    files = []

    # Image extensions to upload
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            filepath = Path(root) / filename

            # Skip non-image files
            ext = filepath.suffix.lower()
            if ext not in image_extensions:
                continue

            # Get relative path from input_dir
            rel_path = filepath.relative_to(input_dir)
            path = str(rel_path)

            # Apply session filter if specified
            if session_filter:
                if session_filter not in path:
                    continue

            files.append({
                "filepath": filepath,
                "path": path,
            })

    return files


def normalize_timestamp(timestamp: str) -> str:
    """Convert underscore-based timestamp to colon-based (ISO format)."""
    return re.sub(r'T(\d{2})_(\d{2})_(\d{2})', r'T\1:\2:\3', timestamp)


def denormalize_timestamp(timestamp: str) -> str:
    """Convert colon-based timestamp to underscore-based (filename safe)."""
    return re.sub(r'T(\d{2}):(\d{2}):(\d{2})', r'T\1_\2_\3', timestamp)


def extract_session_timestamp(filepath: str) -> Optional[str]:
    """Extract session timestamp from a visualization filepath."""
    patterns = [
        r'(composite|calibration_stages|orientation_3d|orientation_track|raw_axes)_(.+?)\.png$',
        r'windows_(.+?)/window_\d+',
        r'trajectory_comparison_(.+?)/',
    ]

    for pattern in patterns:
        match = re.search(pattern, filepath)
        if match:
            groups = match.groups()
            ts = groups[-1] if len(groups) > 0 else None
            if ts:
                return ts
    return None


def group_files_by_session(files: List[Dict]) -> Dict[str, List[Dict]]:
    """Group uploaded files by session timestamp."""
    sessions: Dict[str, List[Dict]] = {}

    for f in files:
        path = f.get("path", "")
        ts = extract_session_timestamp(path)
        if ts:
            if ts not in sessions:
                sessions[ts] = []
            sessions[ts].append(f)

    return sessions


def build_manifest(
    session_timestamp: str,
    files: List[Dict],
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    session_metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Build a visualization manifest from uploaded files.

    Args:
        session_timestamp: Session timestamp (underscore format)
        files: List of uploaded file dicts with 'path' and 'url'
        owner: Repository owner
        repo: Repository name
        session_metadata: Optional session metadata dict

    Returns:
        Manifest dict ready for JSON serialization
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    normalized_ts = normalize_timestamp(session_timestamp)
    generated_ts_safe = denormalize_timestamp(generated_at.replace('+00:00', 'Z'))
    manifest_id = f"{session_timestamp}_{generated_ts_safe}"

    # Build URL lookup
    url_map: Dict[str, str] = {}
    for f in files:
        path = f.get("path", "")
        url = f.get("url", "")
        if path and url:
            url_map[path] = url

    # Session-level images
    images: Dict[str, str] = {}
    for img_type in ["composite", "calibration_stages", "orientation_3d", "orientation_track", "raw_axes"]:
        key = f"{img_type}_{session_timestamp}.png"
        if key in url_map:
            images[img_type] = url_map[key]

    # Trajectory comparison images
    trajectory_comparison: Dict[str, str] = {}
    traj_prefix = f"trajectory_comparison_{session_timestamp}/"
    for path, url in url_map.items():
        if path.startswith(traj_prefix):
            filename = path.replace(traj_prefix, "").replace(".png", "")
            traj_type = filename.replace("_3d", "").replace("_overlay", "")
            trajectory_comparison[traj_type] = url

    # Window images
    windows: Dict[int, Dict[str, Any]] = {}
    window_prefix = f"windows_{session_timestamp}/"

    for path, url in url_map.items():
        if not path.startswith(window_prefix):
            continue

        window_match = re.match(
            rf'{re.escape(window_prefix)}window_(\d+)(?:/(.+)|\.png)$',
            path
        )
        if not window_match:
            continue

        window_num = int(window_match.group(1))
        sub_path = window_match.group(2)

        if window_num not in windows:
            windows[window_num] = {
                "window_num": window_num,
                "images": {},
                "trajectory_images": {},
            }

        window = windows[window_num]

        if sub_path is None:
            window["composite"] = url
        else:
            image_name = sub_path.replace(".png", "")
            if image_name.startswith("trajectory_") and not any(
                x in image_name for x in ["accel", "gyro", "mag", "combined"]
            ):
                traj_key = image_name.replace("trajectory_", "")
                window["trajectory_images"][traj_key] = url
            else:
                window["images"][image_name] = url

    windows_list = [windows[k] for k in sorted(windows.keys())]

    # Build session metadata
    session_info = {
        "filename": f"{session_timestamp}.json",
        "duration": 0,
        "sample_count": 0,
        "sample_rate": 50,
    }

    if session_metadata:
        session_info.update({
            "duration": session_metadata.get("duration", 0),
            "sample_count": session_metadata.get("sample_count", 0),
            "sample_rate": session_metadata.get("sample_rate", 50),
            "device": session_metadata.get("device"),
            "firmware_version": session_metadata.get("firmware_version"),
            "session_type": session_metadata.get("session_type"),
            "hand": session_metadata.get("hand"),
            "magnet_type": session_metadata.get("magnet_type"),
            "notes": session_metadata.get("notes"),
            "custom_labels": session_metadata.get("custom_labels", []),
        })

    manifest = {
        "version": MANIFEST_VERSION,
        "sessionTimestamp": normalized_ts,
        "generatedAt": generated_at,
        "manifestId": manifest_id,
        "session": session_info,
        "images": images,
        "trajectory_comparison": trajectory_comparison,
        "windows": windows_list,
    }

    return manifest


def upload_visualizations(
    input_dir: Path,
    token: str,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    session_filter: Optional[str] = None,
    dry_run: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Upload all visualization files to GitHub images branch.

    Args:
        input_dir: Directory containing visualizations
        token: GitHub PAT
        owner: Repository owner
        repo: Repository name
        session_filter: Optional session timestamp to filter
        dry_run: If True, list files without uploading
        max_workers: Number of parallel upload threads
        verbose: Print progress information

    Returns:
        Dict with upload summary
    """
    files = collect_visualization_files(input_dir, session_filter)

    if not files:
        return {
            "uploaded": 0,
            "failed": 0,
            "total_size": 0,
            "files": [],
        }

    if verbose:
        print(f"Found {len(files)} files to upload to '{IMAGES_BRANCH}' branch")
        if dry_run:
            print("DRY RUN - no files will be uploaded")

    results = []
    failed = []
    total_size = 0

    # Upload files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                upload_file_to_github,
                f["filepath"],
                f["path"],
                IMAGES_BRANCH,
                token,
                owner,
                repo,
                f"Upload visualization: {f['path']}",
                dry_run,
            ): f
            for f in files
        }

        for i, future in enumerate(as_completed(future_to_file)):
            file_info = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                total_size += result["size"]

                if verbose:
                    status = "OK" if result["uploaded"] else "SKIP"
                    print(f"  [{i+1}/{len(files)}] {status}: {result['path']} ({result['size']:,} bytes)")

            except Exception as e:
                failed.append({
                    "path": file_info["path"],
                    "error": str(e),
                })
                if verbose:
                    print(f"  [{i+1}/{len(files)}] FAIL: {file_info['path']} - {e}")

    summary = {
        "uploaded": len(results),
        "failed": len(failed),
        "total_size": total_size,
        "files": results,
        "errors": failed if failed else None,
    }

    if verbose:
        print(f"\nSummary: {len(results)} uploaded, {len(failed)} failed, {total_size:,} bytes total")

    return summary


def upload_manifest(
    manifest: Dict[str, Any],
    token: str,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Upload a visualization manifest to main branch.

    Args:
        manifest: Manifest dict to upload
        token: GitHub PAT
        owner: Repository owner
        repo: Repository name
        dry_run: If True, skip actual upload
        verbose: Print progress

    Returns:
        Upload result dict
    """
    manifest_id = manifest.get("manifestId", "unknown")
    path = f"visualizations/manifests/{manifest_id}.json"

    if verbose:
        print(f"Uploading manifest: {path}")

    manifest_json = json.dumps(manifest, indent=2)

    result = upload_content_to_github(
        content=manifest_json,
        path=path,
        branch=MAIN_BRANCH,
        token=token,
        owner=owner,
        repo=repo,
        message=f"Add visualization manifest for {manifest.get('sessionTimestamp', 'unknown')}",
        dry_run=dry_run,
    )

    result["manifest"] = manifest
    return result


def upload_with_manifest(
    input_dir: Path,
    token: str,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    session_filter: Optional[str] = None,
    session_metadata: Optional[Dict] = None,
    dry_run: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Upload visualizations and generate manifests.

    Args:
        input_dir: Directory containing visualizations
        token: GitHub PAT
        owner: Repository owner
        repo: Repository name
        session_filter: Optional session timestamp filter
        session_metadata: Optional metadata for the session
        dry_run: If True, skip actual uploads
        max_workers: Parallel upload threads
        verbose: Print progress

    Returns:
        Dict with upload summary including manifests
    """
    # First upload all image files
    upload_result = upload_visualizations(
        input_dir=input_dir,
        token=token,
        owner=owner,
        repo=repo,
        session_filter=session_filter,
        dry_run=dry_run,
        max_workers=max_workers,
        verbose=verbose,
    )

    # Group files by session
    files = upload_result.get("files", [])
    sessions = group_files_by_session(files)

    if verbose:
        print(f"\nGenerating manifests for {len(sessions)} session(s)...")

    # Generate and upload manifests
    manifests = []
    manifest_errors = []

    for session_ts, session_files in sessions.items():
        try:
            # Build manifest
            manifest = build_manifest(
                session_timestamp=session_ts,
                files=session_files,
                owner=owner,
                repo=repo,
                session_metadata=session_metadata if session_filter else None,
            )

            # Upload manifest
            manifest_result = upload_manifest(
                manifest=manifest,
                token=token,
                owner=owner,
                repo=repo,
                dry_run=dry_run,
                verbose=verbose,
            )

            manifests.append(manifest_result)

            if verbose:
                status = "OK" if manifest_result["uploaded"] else "SKIP"
                print(f"  {status}: {manifest_result['path']}")

        except Exception as e:
            manifest_errors.append({
                "session": session_ts,
                "error": str(e),
            })
            if verbose:
                print(f"  FAIL: Manifest for {session_ts} - {e}")

    # Combine results
    upload_result["manifests"] = manifests
    upload_result["manifest_errors"] = manifest_errors if manifest_errors else None

    if verbose:
        print(f"\nManifests: {len(manifests)} generated, {len(manifest_errors)} failed")

    return upload_result


def main():
    parser = argparse.ArgumentParser(
        description="Upload visualizations to GitHub"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory containing visualizations (default: visualizations)",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Filter to specific session timestamp",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel upload threads (default: 4)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Generate and upload manifests after uploading images",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=DEFAULT_OWNER,
        help=f"GitHub repository owner (default: {DEFAULT_OWNER})",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"GitHub repository name (default: {DEFAULT_REPO})",
    )

    args = parser.parse_args()

    # Get token from environment
    token = os.environ.get("GITHUB_TOKEN")
    if not token and not args.dry_run:
        print("Error: GITHUB_TOKEN environment variable not set", file=sys.stderr)
        print("Create a PAT at: https://github.com/settings/tokens", file=sys.stderr)
        sys.exit(1)

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Run upload
    if args.manifest:
        result = upload_with_manifest(
            input_dir=args.input_dir,
            token=token or "",
            owner=args.owner,
            repo=args.repo,
            session_filter=args.session,
            dry_run=args.dry_run,
            max_workers=args.workers,
            verbose=not args.quiet and not args.json,
        )
    else:
        result = upload_visualizations(
            input_dir=args.input_dir,
            token=token or "",
            owner=args.owner,
            repo=args.repo,
            session_filter=args.session,
            dry_run=args.dry_run,
            max_workers=args.workers,
            verbose=not args.quiet and not args.json,
        )

    # Output results
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.quiet:
        if result["failed"] > 0:
            print(f"\nFailed uploads:")
            for err in result.get("errors", []):
                print(f"  - {err['path']}: {err['error']}")

    sys.exit(1 if result["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
