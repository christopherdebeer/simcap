#!/usr/bin/env python3
"""
Vercel Blob Upload Utility for Visualizations

Uploads generated visualization files to Vercel Blob storage.
Uses the Vercel Blob REST API for server-side uploads.

Supports manifest-based visualization system:
- Generates JSON manifests with all image URLs
- Stores manifests with versioned naming: {session_ts}_{generated_ts}.json
- Enables efficient listing via manifest prefix queries

Environment Variables:
    BLOB_READ_WRITE_TOKEN: Vercel Blob read/write token (required)

Usage:
    # Upload all visualizations with manifest generation
    python -m ml.blob_upload --input-dir visualizations --manifest

    # Upload specific session with manifest
    python -m ml.blob_upload --input-dir visualizations --session 2025-12-15T22_35_15.567Z --manifest

    # Dry run (list files without uploading)
    python -m ml.blob_upload --input-dir visualizations --dry-run

    # Upload without manifest (legacy mode)
    python -m ml.blob_upload --input-dir visualizations
"""

import os
import sys
import json
import argparse
import mimetypes
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Vercel Blob API endpoint
BLOB_API_URL = "https://blob.vercel-storage.com"

# Manifest version
MANIFEST_VERSION = "1.0"


def get_content_type(filepath: Path) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type or "application/octet-stream"


def upload_file(
    filepath: Path,
    pathname: str,
    token: str,
    dry_run: bool = False
) -> Dict:
    """
    Upload a single file to Vercel Blob.

    Args:
        filepath: Local file path
        pathname: Destination path in blob storage (e.g., "visualizations/composite.png")
        token: Vercel Blob read/write token
        dry_run: If True, skip actual upload

    Returns:
        Dict with upload result including url, pathname, size
    """
    if dry_run:
        return {
            "pathname": pathname,
            "size": filepath.stat().st_size,
            "url": f"(dry-run) {pathname}",
            "uploaded": False
        }

    content_type = get_content_type(filepath)

    with open(filepath, "rb") as f:
        file_data = f.read()

    # Vercel Blob PUT request
    url = f"{BLOB_API_URL}/{pathname}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "x-api-version": "7",
        "x-content-type": content_type,
    }

    request = Request(url, data=file_data, headers=headers, method="PUT")

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode())
            return {
                "pathname": pathname,
                "url": result.get("url", url),
                "size": len(file_data),
                "uploaded": True
            }
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise Exception(f"Upload failed for {pathname}: {e.code} - {error_body}")
    except URLError as e:
        raise Exception(f"Network error uploading {pathname}: {e.reason}")


def collect_visualization_files(
    input_dir: Path,
    session_filter: Optional[str] = None
) -> List[Dict]:
    """
    Collect all visualization files to upload.

    Args:
        input_dir: Directory containing visualizations
        session_filter: Optional session timestamp to filter

    Returns:
        List of dicts with 'filepath' and 'pathname' keys
    """
    files = []

    # Image extensions to upload
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            filepath = Path(root) / filename

            # Skip non-image files (except session-data.js)
            ext = filepath.suffix.lower()
            if ext not in image_extensions and filename != "session-data.js":
                continue

            # Get relative path from input_dir
            rel_path = filepath.relative_to(input_dir)
            pathname = f"visualizations/{rel_path}"

            # Apply session filter if specified
            if session_filter:
                if session_filter not in str(rel_path):
                    continue

            files.append({
                "filepath": filepath,
                "pathname": pathname
            })

    return files


def upload_visualizations(
    input_dir: Path,
    token: str,
    session_filter: Optional[str] = None,
    dry_run: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict:
    """
    Upload all visualization files to Vercel Blob.

    Args:
        input_dir: Directory containing visualizations
        token: Vercel Blob read/write token
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
            "files": []
        }

    if verbose:
        print(f"Found {len(files)} files to upload")
        if dry_run:
            print("DRY RUN - no files will be uploaded")

    results = []
    failed = []
    total_size = 0

    # Upload files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                upload_file,
                f["filepath"],
                f["pathname"],
                token,
                dry_run
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
                    print(f"  [{i+1}/{len(files)}] {status}: {result['pathname']} ({result['size']:,} bytes)")

            except Exception as e:
                failed.append({
                    "pathname": file_info["pathname"],
                    "error": str(e)
                })
                if verbose:
                    print(f"  [{i+1}/{len(files)}] FAIL: {file_info['pathname']} - {e}")

    summary = {
        "uploaded": len(results),
        "failed": len(failed),
        "total_size": total_size,
        "files": results,
        "errors": failed if failed else None
    }

    if verbose:
        print(f"\nSummary: {len(results)} uploaded, {len(failed)} failed, {total_size:,} bytes total")

    return summary


def normalize_timestamp(timestamp: str) -> str:
    """Convert underscore-based timestamp to colon-based (ISO format)."""
    # T22_35_15 -> T22:35:15
    return re.sub(r'T(\d{2})_(\d{2})_(\d{2})', r'T\1:\2:\3', timestamp)


def denormalize_timestamp(timestamp: str) -> str:
    """Convert colon-based timestamp to underscore-based (filename safe)."""
    # T22:35:15 -> T22_35_15
    return re.sub(r'T(\d{2}):(\d{2}):(\d{2})', r'T\1_\2_\3', timestamp)


def extract_session_timestamp(filepath: str) -> Optional[str]:
    """Extract session timestamp from a visualization filepath."""
    # Match patterns like:
    # composite_2025-12-15T22_35_15.567Z.png
    # windows_2025-12-15T22_35_15.567Z/window_001/...
    # trajectory_comparison_2025-12-15T22_35_15.567Z/...
    patterns = [
        r'(composite|calibration_stages|orientation_3d|orientation_track|raw_axes)_(.+?)\.png$',
        r'windows_(.+?)/window_\d+/',
        r'trajectory_comparison_(.+?)/',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filepath)
        if match:
            # Get the timestamp group (last group in match)
            groups = match.groups()
            ts = groups[-1] if len(groups) > 0 else None
            if ts:
                return ts
    return None


def group_files_by_session(files: List[Dict]) -> Dict[str, List[Dict]]:
    """Group uploaded files by session timestamp."""
    sessions: Dict[str, List[Dict]] = {}
    
    for f in files:
        pathname = f.get("pathname", "")
        ts = extract_session_timestamp(pathname)
        if ts:
            if ts not in sessions:
                sessions[ts] = []
            sessions[ts].append(f)
    
    return sessions


def build_manifest(
    session_timestamp: str,
    files: List[Dict],
    session_metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Build a visualization manifest from uploaded files.
    
    Args:
        session_timestamp: Session timestamp (underscore format)
        files: List of uploaded file dicts with 'pathname' and 'url'
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
        pathname = f.get("pathname", "")
        url = f.get("url", "")
        if pathname and url:
            # Store relative path from visualizations/
            rel_path = pathname.replace("visualizations/", "")
            url_map[rel_path] = url
    
    # Session-level images
    images: Dict[str, str] = {}
    for img_type in ["composite", "calibration_stages", "orientation_3d", "orientation_track", "raw_axes"]:
        key = f"{img_type}_{session_timestamp}.png"
        if key in url_map:
            images[img_type] = url_map[key]
    
    # Trajectory comparison images
    trajectory_comparison: Dict[str, str] = {}
    traj_prefix = f"trajectory_comparison_{session_timestamp}/"
    for rel_path, url in url_map.items():
        if rel_path.startswith(traj_prefix):
            # Extract type from filename (e.g., raw_3d.png -> raw)
            filename = rel_path.replace(traj_prefix, "").replace(".png", "")
            traj_type = filename.replace("_3d", "").replace("_overlay", "")
            trajectory_comparison[traj_type] = url
    
    # Window images
    windows: List[Dict[str, Any]] = {}
    window_prefix = f"windows_{session_timestamp}/"
    
    for rel_path, url in url_map.items():
        if not rel_path.startswith(window_prefix):
            continue
            
        # Parse window path: windows_TS/window_001/image.png or windows_TS/window_001.png
        window_match = re.match(
            rf'{re.escape(window_prefix)}window_(\d+)(?:/(.+)|\.png)$',
            rel_path
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
            # This is the composite window image (window_001.png)
            window["composite"] = url
        else:
            # This is a sub-image
            image_name = sub_path.replace(".png", "")
            
            # Categorize trajectory vs regular images
            if image_name.startswith("trajectory_") and not any(x in image_name for x in ["accel", "gyro", "mag", "combined"]):
                # trajectory_raw, trajectory_iron, trajectory_fused, trajectory_filtered
                traj_key = image_name.replace("trajectory_", "")
                window["trajectory_images"][traj_key] = url
            else:
                window["images"][image_name] = url
    
    # Convert windows dict to sorted list
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


def upload_manifest(
    manifest: Dict[str, Any],
    token: str,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Upload a manifest JSON to Vercel Blob.
    
    Args:
        manifest: Manifest dict to upload
        token: Vercel Blob token
        dry_run: If True, skip actual upload
        verbose: Print progress
        
    Returns:
        Upload result dict
    """
    manifest_id = manifest.get("manifestId", "unknown")
    pathname = f"visualizations/manifests/{manifest_id}.json"
    
    if verbose:
        print(f"Uploading manifest: {pathname}")
    
    if dry_run:
        return {
            "pathname": pathname,
            "url": f"(dry-run) {pathname}",
            "size": len(json.dumps(manifest)),
            "uploaded": False,
            "manifest": manifest,
        }
    
    # Serialize manifest
    manifest_json = json.dumps(manifest, indent=2)
    manifest_bytes = manifest_json.encode("utf-8")
    
    # Upload
    url = f"{BLOB_API_URL}/{pathname}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-api-version": "7",
        "x-content-type": "application/json",
    }
    
    request = Request(url, data=manifest_bytes, headers=headers, method="PUT")
    
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode())
            return {
                "pathname": pathname,
                "url": result.get("url", url),
                "size": len(manifest_bytes),
                "uploaded": True,
                "manifest": manifest,
            }
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise Exception(f"Manifest upload failed: {e.code} - {error_body}")
    except URLError as e:
        raise Exception(f"Network error uploading manifest: {e.reason}")


def upload_with_manifest(
    input_dir: Path,
    token: str,
    session_filter: Optional[str] = None,
    session_metadata: Optional[Dict] = None,
    dry_run: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict:
    """
    Upload visualizations and generate manifests.
    
    Args:
        input_dir: Directory containing visualizations
        token: Vercel Blob token
        session_filter: Optional session timestamp filter
        session_metadata: Optional metadata for the session
        dry_run: If True, skip actual uploads
        max_workers: Parallel upload threads
        verbose: Print progress
        
    Returns:
        Dict with upload summary including manifests
    """
    # First upload all files
    upload_result = upload_visualizations(
        input_dir=input_dir,
        token=token,
        session_filter=session_filter,
        dry_run=dry_run,
        max_workers=max_workers,
        verbose=verbose
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
                session_metadata=session_metadata if session_filter else None
            )
            
            # Upload manifest
            manifest_result = upload_manifest(
                manifest=manifest,
                token=token,
                dry_run=dry_run,
                verbose=verbose
            )
            
            manifests.append(manifest_result)
            
            if verbose:
                status = "OK" if manifest_result["uploaded"] else "SKIP"
                print(f"  {status}: {manifest_result['pathname']}")
                
        except Exception as e:
            manifest_errors.append({
                "session": session_ts,
                "error": str(e)
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
        description="Upload visualizations to Vercel Blob storage"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory containing visualizations (default: visualizations)"
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Filter to specific session timestamp"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel upload threads (default: 4)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Generate and upload manifests after uploading images"
    )

    args = parser.parse_args()

    # Get token from environment
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token and not args.dry_run:
        print("Error: BLOB_READ_WRITE_TOKEN environment variable not set", file=sys.stderr)
        print("Get a token from: https://vercel.com/dashboard/stores", file=sys.stderr)
        sys.exit(1)

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Run upload (with or without manifest generation)
    if args.manifest:
        result = upload_with_manifest(
            input_dir=args.input_dir,
            token=token or "",
            session_filter=args.session,
            dry_run=args.dry_run,
            max_workers=args.workers,
            verbose=not args.quiet and not args.json
        )
    else:
        result = upload_visualizations(
            input_dir=args.input_dir,
            token=token or "",
            session_filter=args.session,
            dry_run=args.dry_run,
            max_workers=args.workers,
            verbose=not args.quiet and not args.json
        )

    # Output results
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.quiet:
        if result["failed"] > 0:
            print(f"\nFailed uploads:")
            for err in result.get("errors", []):
                print(f"  - {err['pathname']}: {err['error']}")

    sys.exit(1 if result["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
