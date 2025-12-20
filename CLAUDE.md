# GitHub Branching Storage Workflow

Session data and visualization assets are stored in GitHub branches:
- **`data` branch**: Session data JSON files
- **`images` branch**: Visualization images (PNG)
- **`main` branch**: Manifests and source code

## Local Development Setup (Worktrees)

Git worktrees provide local access to the `data` and `images` branches without switching branches.
A **Claude Code session start hook** automatically runs setup when starting a new session.

### Automatic Setup (Session Start Hook)

The `.claude/settings.json` configures a SessionStart hook that runs `scripts/setup-worktrees.sh`.
When you start a Claude Code session, the worktrees are automatically created if they don't exist.

### Manual Setup

```bash
# Run the setup script manually
npm run setup:worktrees

# Or directly
bash scripts/setup-worktrees.sh
```

### Directory Layout

```
repo/                          # main branch (source code)
├── .worktrees/
│   ├── data/                  # worktree → data branch
│   │   └── GAMBIT/
│   │       └── *.json
│   └── images/                # worktree → images branch
│       └── *.png
├── data -> .worktrees/data    # symlink (Unix only)
└── images -> .worktrees/images # symlink (Unix only)
```

### Working with Worktrees

```bash
# Access data branch content
ls data/GAMBIT/

# Commit changes to data branch
cd .worktrees/data
git add .
git commit -m "Add new session data"
git push origin data

# Access images branch content
ls images/

# Commit changes to images branch
cd .worktrees/images
git add .
git commit -m "Add new visualizations"
git push origin images
```

### Important Notes

- **Never switch branches** in a worktree directory - each worktree is tied to its branch
- **Worktrees are ignored** by git (via `.gitignore`) - they won't be committed
- **Symlinks are local** - they provide convenient `./data` and `./images` paths
- **One branch per worktree** - Git only allows a branch to be checked out in one place

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
└── visualizations/manifests/
    ├── sessions.json                    # Session index (lists data branch files)
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
# Generate visualizations to images worktree (default output)
python -m ml.visualize --data-dir data/GAMBIT

# Generate for specific session
python -m ml.visualize --data-dir data/GAMBIT --session 2025-12-15T22_35_15.567Z

# Custom output directory (not recommended)
python -m ml.visualize --data-dir data/GAMBIT --output-dir /path/to/output
```

## Uploading to GitHub

### Session Data (via Web Collector)
Session data is uploaded automatically via the web collector interface.
Uses API proxy at `/api/github-upload` which commits to the `data` branch.

### Visualizations (via Python or Git)
```bash
# Option 1: Commit directly via worktree (recommended)
cd .worktrees/images
git add .
git commit -m "Add visualizations for session X"
git push origin images

# Option 2: Upload via GitHub API
python -m ml.github_upload --input-dir images --manifest

# Dry run (preview without uploading)
python -m ml.github_upload --input-dir images --dry-run
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
2. **Generate visualizations** (outputs to images worktree):
   ```bash
   python -m ml.visualize --data-dir data/GAMBIT
   ```
3. **Commit visualizations** to `images` branch:
   ```bash
   cd .worktrees/images && git add . && git commit -m "Add visualizations" && git push origin images
   ```
4. **Update manifests** in `main` branch:
   ```bash
   python scripts/generate-manifests.py --all --upload
   ```

## Notes

- Session data goes to `data` branch, images to `images` branch
- Manifests are committed to `main` branch for source control
- Browser uploads use API proxy (SIMCAP_UPLOAD_SECRET for auth)
- Python uploads use GITHUB_TOKEN directly
- Raw content served via `raw.githubusercontent.com` (cached by GitHub CDN)
