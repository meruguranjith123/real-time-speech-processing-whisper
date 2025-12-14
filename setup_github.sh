#!/bin/bash

# GitHub Repository Setup Script
# Project: Real-Time Speech Processing with Whisper

REPO_NAME="real-time-speech-processing-whisper"
GITHUB_USER=$(git config user.name 2>/dev/null || echo "YOUR_GITHUB_USERNAME")

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""
echo "Repository Name: $REPO_NAME"
echo "GitHub Username: $GITHUB_USER"
echo ""
echo "=========================================="
echo "STEP 1: Create Repository on GitHub"
echo "=========================================="
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Description: Real-Time Speech Processing with Whisper - Stuttering Detection and Next Sentence Prediction"
echo "4. Choose: Public or Private"
echo "5. DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "6. Click 'Create repository'"
echo ""
echo "=========================================="
echo "STEP 2: After creating the repo, run:"
echo "=========================================="
echo ""
echo "git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "Or if you prefer SSH:"
echo "git remote add origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "=========================================="
echo "Quick Setup (copy and paste):"
echo "=========================================="
echo ""
echo "# Set your GitHub username:"
echo "export GITHUB_USER='$GITHUB_USER'"
echo ""
echo "# Add remote (HTTPS):"
echo "git remote add origin https://github.com/\$GITHUB_USER/$REPO_NAME.git"
echo ""
echo "# Or add remote (SSH):"
echo "git remote add origin git@github.com:\$GITHUB_USER/$REPO_NAME.git"
echo ""
echo "# Push to GitHub:"
echo "git branch -M main"
echo "git push -u origin main"
echo ""





