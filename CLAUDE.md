# Git LFS Workflow


## LFS-tracked files (via .gitattributes)
- `data/GAMBIT/*.json` - session data
- `visualizations/**/*.png` - generated plots
- `visualizations/session-data.js`

## After clone/checkout

```bash
git lfs pull          # Download actual file content
```

## Adding new LFS files

```bash
# Add file normally - LFS handles it via .gitattributes
git add data/GAMBIT/new-session.json
git commit -m "Add session"
git push              # Pushes commit + LFS objects
```

## Verify LFS status

```bash
git lfs ls-files      # List tracked files (* = has content, - = missing)
git lfs status        # Show pending LFS operations
```

## Fix missing LFS content

```bash
git lfs fetch --all   # Download all LFS objects
git lfs checkout      # Replace pointers with content
```

## Push LFS objects manually

```bash
git lfs push --all origin main
```

## Troubleshooting

If files show as "modified" unexpectedly, check that `.gitattributes` exists and contains the LFS tracking patterns.

