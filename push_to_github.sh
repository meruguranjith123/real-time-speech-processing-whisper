#!/bin/bash

# Automated GitHub Push Script
# Run this AFTER creating the repository on GitHub

set -e

GITHUB_USER="Ranjith"
REPO_NAME="real-time-speech-processing-whisper"

echo "=========================================="
echo "Pushing to GitHub Repository"
echo "=========================================="
echo ""
echo "Repository: $GITHUB_USER/$REPO_NAME"
echo ""

# Check if remote already exists
if git remote get-url origin &>/dev/null; then
    echo "⚠️  Remote 'origin' already exists."
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
    else
        echo "Aborted."
        exit 1
    fi
fi

# Add remote
echo "📡 Adding remote repository..."
git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git

# Ensure we're on main branch
echo "🌿 Setting branch to main..."
git branch -M main

# Push to GitHub
echo "🚀 Pushing to GitHub..."
echo ""
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "You can now view your repository at:"
echo "https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""

