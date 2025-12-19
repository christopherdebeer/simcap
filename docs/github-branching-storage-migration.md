# GitHub Branching Storage Migration Analysis

## Overview

This document outlines the required changes to migrate from Vercel Blob storage back to a GitHub-based storage model with branch separation:

- **`data` branch**: Session data (JSON files from collector)
- **`images` branch**: Visualization images (PNG files)
- **`main`/feature branches**: Manifests only (JSON)

## Current Architecture (Vercel Blob)

### Frontend Upload Flow (`apps/gambit/shared/blob-upload.ts`)
1. Client stores secret in localStorage
2. Client requests upload token from `/api/upload` with secret
3. Server validates secret in `onBeforeGenerateToken` callback
4. Client uploads directly to Vercel Blob using `@vercel/blob/client`
5. Files stored at `sessions/{timestamp}.json`

### API Endpoints
- `POST /api/upload` - Token generation for blob uploads
- `GET /api/sessions` - Lists blobs with `sessions/` prefix
- `GET /api/visualizations` - Lists manifests from `visualizations/manifests/` prefix

### Python Visualization Pipeline (`ml/blob_upload.py`)
1. Upload images to `visualizations/` prefix
2. Generate manifest JSON with image URLs
3. Upload manifest to `visualizations/manifests/{session_ts}_{generated_ts}.json`

---

## Proposed Architecture (GitHub Branches)

### Branch Structure
```
data (branch)
├── GAMBIT/
│   ├── 2025-12-15T22_35_15.567Z.json
│   └── 2025-12-16T10_20_30.123Z.json

images (branch)
├── composite_2025-12-15T22_35_15.567Z.png
├── calibration_stages_2025-12-15T22_35_15.567Z.png
├── windows_2025-12-15T22_35_15.567Z/
│   ├── window_001.png
│   └── window_001/
│       ├── accel.png
│       └── gyro.png
└── trajectory_comparison_2025-12-15T22_35_15.567Z/
    └── raw_3d.png

main (branch) - also feature branches
├── data/
│   └── GAMBIT/
│       └── manifest.json  (list of sessions)
└── visualizations/
    └── manifests/
        └── manifest.json  (list of all visualization manifests)
```

### URL Construction
- **Session data**: `https://raw.githubusercontent.com/{owner}/{repo}/data/GAMBIT/{filename}`
- **Images**: `https://raw.githubusercontent.com/{owner}/{repo}/images/{path}`
- **Manifests**: Checked into source, bundled with deployment or fetched from raw.githubusercontent

---

## Required Changes

### 1. Frontend Upload (`apps/gambit/shared/blob-upload.ts`)

**Current**: Uses `@vercel/blob/client` for two-phase upload

**New**: Create `github-upload.ts` with GitHub Contents API

```typescript
// New module: apps/gambit/shared/github-upload.ts

interface GitHubUploadOptions {
  token: string;           // GitHub PAT with repo write access
  owner: string;           // Repository owner (e.g., "christopherdebeer")
  repo: string;            // Repository name (e.g., "simcap")
  branch: string;          // Target branch (e.g., "data")
  path: string;            // File path (e.g., "GAMBIT/2025-12-15T22_35_15.567Z.json")
  content: string;         // File content
  message: string;         // Commit message
  onProgress?: (progress: UploadProgress) => void;
}

export async function uploadToGitHub(options: GitHubUploadOptions): Promise<UploadResult> {
  // 1. Base64 encode content
  // 2. Check if file exists (GET /repos/{owner}/{repo}/contents/{path}?ref={branch})
  // 3. Create/update file (PUT /repos/{owner}/{repo}/contents/{path})
  //    - If exists: include sha for update
  //    - Set branch parameter
  // 4. Return commit URL
}
```

**Changes to `collector-app.ts`**:
- Replace `uploadToBlob()` calls with `uploadToGitHub()`
- Store GitHub PAT instead of upload secret (or use API proxy)
- Target `data` branch with path `GAMBIT/{timestamp}.json`

**Security Consideration**:
- Option A: Direct PAT (stored in localStorage - less secure)
- Option B: API proxy that validates secret and uses server-side PAT

### 2. API Endpoints

#### `/api/upload` → `/api/github-upload`

**Current**: Vercel Blob token generation

**New**: GitHub Contents API proxy

```typescript
// api/github-upload.ts
export default async function handler(request: Request): Promise<Response> {
  // 1. Validate client secret (from clientPayload)
  // 2. Parse request: { branch, path, content, message }
  // 3. Use server-side GITHUB_TOKEN to commit
  // 4. Return commit result
}
```

**Environment Variables**:
- `GITHUB_TOKEN` - Server-side PAT with repo write access
- `SIMCAP_UPLOAD_SECRET` - Client authentication (keep existing)

#### `/api/sessions`

**Current**: Lists blobs with `list({ prefix: 'sessions/' })`

**New**: Fetch manifest from GitHub or list directory

```typescript
// Option A: Read manifest.json from main branch
const manifestUrl = `https://raw.githubusercontent.com/${owner}/${repo}/main/data/GAMBIT/manifest.json`;
const manifest = await fetch(manifestUrl).then(r => r.json());

// Option B: Use GitHub Contents API to list directory
const listUrl = `https://api.github.com/repos/${owner}/${repo}/contents/GAMBIT?ref=data`;
const files = await fetch(listUrl, { headers: { Authorization: `token ${token}` } }).then(r => r.json());
```

**URL Construction for sessions**:
```typescript
const sessionUrl = `https://raw.githubusercontent.com/${owner}/${repo}/data/GAMBIT/${filename}`;
```

#### `/api/visualizations`

**Current**: Lists manifest blobs from `visualizations/manifests/`

**New**: Read manifest from main branch

```typescript
const manifestUrl = `https://raw.githubusercontent.com/${owner}/${repo}/main/visualizations/manifests/manifest.json`;
```

**Image URL Construction**:
```typescript
// Images stored in images branch
const imageUrl = `https://raw.githubusercontent.com/${owner}/${repo}/images/${imagePath}`;
```

### 3. Python Visualization Upload (`ml/blob_upload.py`)

**Current**: Uses Vercel Blob REST API

**New**: Create `ml/github_upload.py` with GitHub Contents API

```python
# ml/github_upload.py

import base64
import requests
from pathlib import Path

def upload_to_github(
    filepath: Path,
    path: str,
    branch: str,
    token: str,
    owner: str = "christopherdebeer",
    repo: str = "simcap",
    message: str = "Upload visualization"
) -> dict:
    """Upload file to GitHub using Contents API."""

    # Read and encode file
    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    # Check if file exists
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    existing_sha = None
    resp = requests.get(url, headers=headers, params={"ref": branch})
    if resp.status_code == 200:
        existing_sha = resp.json().get("sha")

    # Create/update file
    payload = {
        "message": message,
        "content": content,
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()

    return resp.json()


def upload_visualization_images(
    input_dir: Path,
    token: str,
    session_filter: Optional[str] = None,
    dry_run: bool = False
) -> dict:
    """Upload all visualization images to 'images' branch."""
    # Similar to current upload_visualizations() but targeting GitHub
    pass


def upload_manifest_to_main(
    manifest: dict,
    token: str,
    session_timestamp: str
) -> dict:
    """Upload/update manifest in main branch."""
    # Manifests go to main branch, merged into visualization manifest index
    pass
```

**Workflow Changes**:
1. Generate visualizations locally (unchanged)
2. Upload images to `images` branch
3. Generate manifest with `raw.githubusercontent.com/...` URLs
4. Commit manifest to `main` branch (or create PR)

### 4. Manifest Structure Changes

#### Session Manifest (`data/GAMBIT/manifest.json`)

```json
{
  "generated": "2025-12-19T...",
  "directory": "GAMBIT",
  "branch": "data",
  "baseUrl": "https://raw.githubusercontent.com/christopherdebeer/simcap/data/GAMBIT",
  "sessionCount": 42,
  "sessions": [
    {
      "filename": "2025-12-15T22_35_15.567Z.json",
      "timestamp": "2025-12-15T22:35:15.567Z",
      "size": 123456,
      "version": "2.1",
      "sampleCount": 2500,
      "durationSec": 96.2,
      "url": "https://raw.githubusercontent.com/christopherdebeer/simcap/data/GAMBIT/2025-12-15T22_35_15.567Z.json"
    }
  ]
}
```

#### Visualization Manifest (`visualizations/manifests/manifest.json`)

```json
{
  "generated": "2025-12-19T...",
  "imageBranch": "images",
  "baseImageUrl": "https://raw.githubusercontent.com/christopherdebeer/simcap/images",
  "sessions": [
    {
      "sessionTimestamp": "2025-12-15T22:35:15.567Z",
      "generatedAt": "2025-12-18T14:00:00.000Z",
      "images": {
        "composite": "composite_2025-12-15T22_35_15.567Z.png",
        "calibration_stages": "calibration_stages_2025-12-15T22_35_15.567Z.png"
      },
      "windowCount": 10
    }
  ]
}
```

**Note**: Individual session visualization manifests can still be stored as separate files if needed for versioning.

---

## Implementation Steps

### Phase 1: GitHub Upload Infrastructure
1. Create `apps/gambit/shared/github-upload.ts` with Contents API
2. Create `api/github-upload.ts` proxy endpoint
3. Add `GITHUB_TOKEN` to Vercel environment
4. Update TypeScript types in `packages/api/src/types.ts`

### Phase 2: Session Data Migration
1. Create `data` branch from empty state (or `data` orphan branch)
2. Update `collector-app.ts` to use GitHub upload
3. Update `/api/sessions` to read from GitHub
4. Generate session manifest after uploads

### Phase 3: Visualization Migration
1. Create `images` branch (orphan, no source history)
2. Create `ml/github_upload.py` for Python uploads
3. Update visualization manifest generation
4. Update `/api/visualizations` to read from GitHub

### Phase 4: Cleanup
1. Remove Vercel Blob dependencies (`@vercel/blob`)
2. Remove LFS configuration (`.gitattributes`)
3. Update CLAUDE.md documentation
4. Archive old blob data if needed

---

## Considerations

### Advantages of GitHub Branches
- No external storage dependencies
- Works in all development environments (Claude Code, local, Vercel)
- Version controlled (git history)
- Free storage (within GitHub limits)

### Disadvantages / Risks
- GitHub API rate limits (5000 req/hr authenticated)
- Raw content CDN caching (may delay visibility)
- File size limits (100 MB per file via API)
- Branch pollution (many commits to data/images branches)

### Mitigations
- Use conditional requests (If-None-Match) to reduce API calls
- Cache manifests client-side
- Batch commits where possible (multiple files per commit)
- Use GitHub Actions for manifest regeneration

---

## Environment Variables

### Current (Vercel Blob)
```
BLOB_READ_WRITE_TOKEN    # Vercel Blob token
SIMCAP_UPLOAD_SECRET     # Client auth secret
```

### New (GitHub)
```
GITHUB_TOKEN             # GitHub PAT with repo write access
SIMCAP_UPLOAD_SECRET     # Client auth secret (keep for API proxy)
GITHUB_OWNER             # Repository owner (optional, can hardcode)
GITHUB_REPO              # Repository name (optional, can hardcode)
```

---

## Files to Modify

### Frontend
- `apps/gambit/shared/blob-upload.ts` → `github-upload.ts` (new or replace)
- `apps/gambit/collector-app.ts` - Update upload calls
- `apps/gambit/gambit-app.ts` - Update if needed

### API
- `api/upload.ts` → `api/github-upload.ts` (new or replace)
- `api/sessions.ts` - Change from blob list to GitHub
- `api/visualizations.ts` - Change from blob list to GitHub

### Python
- `ml/blob_upload.py` → `ml/github_upload.py` (new or replace)
- `data/GAMBIT/generate_manifest.py` - Update for new structure

### Types
- `packages/api/src/types.ts` - Add GitHub-specific types

### Configuration
- `CLAUDE.md` - Update documentation
- `.gitattributes` - Remove LFS rules
- `package.json` - Remove `@vercel/blob` dependency
