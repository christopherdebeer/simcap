# GitHub Branching Storage Workflow

Session data and visualization assets are stored in GitHub branches:
- **`data` branch**: Session data JSON files
- **`images` branch**: Visualization images (PNG)
- **`main` branch**: Manifests and source code

## Environment Variables

```bash
GITHUB_TOKEN             # GitHub PAT with repo write access (for uploads)
SIMCAP_UPLOAD_SECRET     # Secret for browser-based uploads via API proxy
```

## Branch Structure

```
data (branch)
└── GAMBIT/
    └── {timestamp}.json      # Session data files

images (branch)
├── composite_{timestamp}.png
├── windows_{timestamp}/
│   └── window_*.png
└── trajectory_comparison_{timestamp}/
    └── *.png

main (branch)
├── data/GAMBIT/manifest.json           # Session index
└── visualizations/manifests/
    ├── index.json                       # Visualization index
    └── {session}_{generated}.json       # Individual manifests
```

## Fetching Session Data

```bash
# Fetch all sessions from GitHub data branch
npm run fetch:sessions

# Fetch specific session
npm run fetch:sessions -- --session 2025-12-15T22_35_15.567Z

# List available sessions without downloading
npm run fetch:sessions -- --list
```

## Fetching Visualizations

```bash
# Fetch all visualizations
npm run fetch:visualizations

# Fetch for specific session
npm run fetch:visualizations -- --session 2025-12-15T22_35_15.567Z
```

## Generating New Visualizations

```bash
# Generate visualizations for all local session data
python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations

# Generate for specific session
python -m ml.visualize --data-dir data/GAMBIT --session 2025-12-15T22_35_15.567Z
```

## Uploading to GitHub

### Session Data (via Web Collector)
Session data is uploaded automatically via the web collector interface.
Uses API proxy at `/api/github-upload` which commits to the `data` branch.

### Visualizations (via Python)
```bash
# Upload all visualizations to images branch
python -m ml.github_upload --input-dir visualizations

# Upload with manifest generation
python -m ml.github_upload --input-dir visualizations --manifest

# Upload specific session
python -m ml.github_upload --input-dir visualizations --session 2025-12-15T22_35_15.567Z --manifest

# Dry run (preview without uploading)
python -m ml.github_upload --input-dir visualizations --dry-run
```

## Generating Manifests

Manifests enable efficient listing without GitHub API calls.

```bash
# Generate session manifest
python scripts/generate-manifests.py --sessions

# Generate visualization index
python scripts/generate-manifests.py --visualizations

# Generate all manifests
python scripts/generate-manifests.py --all

# Generate and upload to main branch
python scripts/generate-manifests.py --all --upload
```

## API Endpoints

- `GET /api/sessions` - List all sessions with URLs and metadata
- `GET /api/visualizations` - List all visualizations grouped by session
- `GET /api/visualizations?session={ts}` - Get manifest for specific session
- `POST /api/github-upload` - Proxy for browser-based uploads (uses server GITHUB_TOKEN)

## Full Workflow: Processing New Data

1. **Collect data** via web interface (uploads to `data` branch)
2. **Fetch locally** for processing:
   ```bash
   npm run fetch:sessions
   ```
3. **Generate visualizations**:
   ```bash
   python -m ml.visualize --data-dir data/GAMBIT --output-dir visualizations
   ```
4. **Upload visualizations** to `images` branch:
   ```bash
   python -m ml.github_upload --input-dir visualizations --manifest
   ```
5. **Update manifests** in `main` branch:
   ```bash
   python scripts/generate-manifests.py --all --upload
   ```

## Notes

- Session data goes to `data` branch, images to `images` branch
- Manifests are committed to `main` branch for source control
- Browser uploads use API proxy (SIMCAP_UPLOAD_SECRET for auth)
- Python uploads use GITHUB_TOKEN directly
- Raw content served via `raw.githubusercontent.com` (cached by GitHub CDN)
