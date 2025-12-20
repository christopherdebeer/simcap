#!/bin/bash
#
# Setup git worktrees for data and images branches
#
# This script creates worktrees under .worktrees/ for the data and images
# branches, allowing local access to branch content without switching branches.
#
# Usage: ./scripts/setup-worktrees.sh
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREES_DIR="$REPO_ROOT/.worktrees"

echo "Setting up worktrees in $WORKTREES_DIR..."

# Create .worktrees directory
mkdir -p "$WORKTREES_DIR"

# Function to setup a worktree
setup_worktree() {
    local branch="$1"
    local worktree_path="$WORKTREES_DIR/$branch"

    if [ -d "$worktree_path" ]; then
        echo "✓ Worktree for '$branch' already exists at $worktree_path"
        return 0
    fi

    # Check if branch exists locally or on remote
    if git show-ref --verify --quiet "refs/heads/$branch" 2>/dev/null; then
        echo "Creating worktree for local branch '$branch'..."
        git worktree add "$worktree_path" "$branch"
    elif git show-ref --verify --quiet "refs/remotes/origin/$branch" 2>/dev/null; then
        echo "Creating worktree for remote branch 'origin/$branch'..."
        git worktree add "$worktree_path" "origin/$branch" -b "$branch"
    else
        echo "⚠ Branch '$branch' not found locally or on remote."
        echo "  Creating orphan branch..."
        git worktree add --detach "$worktree_path"
        (
            cd "$worktree_path"
            git switch --orphan "$branch"
            git commit --allow-empty -m "Initialize $branch branch"
        )
    fi

    echo "✓ Created worktree for '$branch' at $worktree_path"
}

# Fetch remote branches first
echo "Fetching remote branches..."
git fetch origin data images 2>/dev/null || git fetch origin 2>/dev/null || true

# Setup worktrees for data and images branches
setup_worktree "data"
setup_worktree "images"

# Create symlinks (Unix only)
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
    # Create symlinks for convenient access
    for branch in data images; do
        link_path="$REPO_ROOT/$branch"
        target=".worktrees/$branch"

        if [ -L "$link_path" ]; then
            echo "✓ Symlink '$branch' already exists"
        elif [ -e "$link_path" ]; then
            echo "⚠ '$link_path' exists but is not a symlink - skipping"
        else
            ln -s "$target" "$link_path"
            echo "✓ Created symlink: $branch -> $target"
        fi
    done
fi

echo ""
echo "Worktree setup complete!"
echo ""
echo "Directory structure:"
echo "  .worktrees/data/   - Session data files (data branch)"
echo "  .worktrees/images/ - Visualization images (images branch)"
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
    echo "  data -> .worktrees/data   (symlink)"
    echo "  images -> .worktrees/images (symlink)"
fi
echo ""
echo "To commit changes to data branch:"
echo "  cd .worktrees/data && git add . && git commit -m 'message' && git push"
