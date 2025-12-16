# Git LFS Workflow

## LFS-tracked files
- `data/GAMBIT/*.json` - session data
- `visualizations/*.png` - generated plots

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

## ⚠️ Don't commit pointer→content changes

If `git diff --staged` shows LFS pointer changing to binary content:
```bash
git restore --staged <file>   # Unstage it
```

This happens after `git lfs checkout` - it's normal. The pointer in git is correct.

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
