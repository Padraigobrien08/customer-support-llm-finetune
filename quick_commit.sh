#!/bin/bash
# Quick commit script for model training updates

set -e

cd "$(dirname "$0")"

echo "=" 
echo "Quick Commit"
echo "="
echo

# Check if git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not a git repository. Initializing..."
    git init
    echo "✓ Git repository initialized"
    echo
fi

# Show status
echo "Changes to commit:"
git status --short
echo

# Ask for commit message
if [ -z "$1" ]; then
    COMMIT_MSG="Update training notebook for Llama-3-8B-Instruct and add quality assessment system"
else
    COMMIT_MSG="$1"
fi

# Add all changes
echo "Staging changes..."
git add -A

# Commit
echo "Committing with message: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo
echo "✓ Commit complete!"
echo
echo "To push: git push"
echo "To see commit: git log -1"
