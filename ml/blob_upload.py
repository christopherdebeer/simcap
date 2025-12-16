#!/usr/bin/env python3
"""
Vercel Blob Upload Utility for Visualizations

Uploads generated visualization files to Vercel Blob storage.
Uses the Vercel Blob REST API for server-side uploads.

Environment Variables:
    BLOB_READ_WRITE_TOKEN: Vercel Blob read/write token (required)

Usage:
    # Upload all visualizations
    python -m ml.blob_upload --input-dir visualizations

    # Upload specific session
    python -m ml.blob_upload --input-dir visualizations --session 2025-12-15T22_35_15.567Z

    # Dry run (list files without uploading)
    python -m ml.blob_upload --input-dir visualizations --dry-run
"""

import os
import sys
import json
import argparse
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Vercel Blob API endpoint
BLOB_API_URL = "https://blob.vercel-storage.com"


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

    # Run upload
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
