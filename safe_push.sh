#!/bin/bash

# Safe push script - follows git best practices:
# 1. Pull latest changes before pushing (prevents conflicts and overwriting remote changes)
# 2. Backup visitors.db to prevent accidental data loss
# Note: visitors.db is ignored by .gitignore, so it won't be pushed

REPO_DIR="/Users/pd3rvr/Documents/pubs/THESIS/thetex/huggingface/spad_for_vision_space"
DB_FILE="visitors.db"

cd "$REPO_DIR" || exit 1

echo "=========================================="
echo "Safe Push Script - Git Best Practices"
echo "=========================================="
echo "This script will:"
echo "  1. Pull latest changes from remote (good practice)"
echo "  2. Backup visitors.db (data preservation)"
echo "  3. Push your changes to remote"
echo ""

# Check if we have uncommitted changes
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "⚠️  You have uncommitted changes. Please commit or stash them first."
    echo ""
    git status --short
    echo ""
    read -p "Continue anyway? (y/n): " continue_anyway
    if [[ "$continue_anyway" != "y" && "$continue_anyway" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Backup local visitors.db if it exists
if [ -f "$DB_FILE" ]; then
    echo "📦 Backing up local $DB_FILE..."
    backup_file="${DB_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$DB_FILE" "$backup_file"
    echo "✅ Backup created: $backup_file"
    local_db_exists=true
else
    local_db_exists=false
    echo "ℹ️  Local $DB_FILE does not exist"
fi
echo ""

# Stash visitors.db temporarily if it exists (since it's ignored, this won't affect it)
# But we'll back it up anyway

# Pull latest changes (best practice: always pull before pushing)
echo "⬇️  Pulling latest changes from remote (best practice)..."
echo "   This ensures you have the latest code and prevents conflicts"
git pull origin main 2>&1
pull_exit_code=$?

if [ $pull_exit_code -ne 0 ]; then
    echo ""
    echo "⚠️  Pull had conflicts or errors. This means:"
    echo "   - Remote has changes you don't have locally"
    echo "   - You may need to resolve conflicts before pushing"
    echo ""
    git status
    echo ""
    
    # Restore local backup if pull failed
    if [ "$local_db_exists" = true ] && [ -f "$backup_file" ]; then
        echo "Restoring local $DB_FILE from backup..."
        cp "$backup_file" "$DB_FILE"
    fi
    
    read -p "Resolve conflicts and continue with push? (y/n): " continue_push
    if [[ "$continue_push" != "y" && "$continue_push" != "Y" ]]; then
        echo ""
        echo "Aborted. Please resolve conflicts manually and try again."
        echo "You may need to:"
        echo "  1. Review conflicts: git status"
        echo "  2. Resolve conflicts in affected files"
        echo "  3. Stage resolved files: git add <files>"
        echo "  4. Commit: git commit"
        exit 1
    fi
fi

echo "✅ Pull completed - you're now synced with remote"
echo ""

# Check visitors.db status
if [ -f "$DB_FILE" ]; then
    db_size=$(du -h "$DB_FILE" | cut -f1)
    echo "📊 Current $DB_FILE size: $db_size"
    echo "ℹ️  Note: $DB_FILE is ignored by .gitignore and won't be pushed"
    echo "   (This is intentional to preserve visitor data on the server)"
else
    echo "ℹ️  $DB_FILE does not exist locally"
fi
echo ""

# Show what will be pushed
echo "📤 Files to be pushed:"
git status --short
echo ""

# Ask for confirmation
read -p "Push changes to remote? (y/n): " confirm_push

if [[ "$confirm_push" != "y" && "$confirm_push" != "Y" ]]; then
    echo "Push cancelled."
    exit 0
fi

# Push changes
echo ""
echo "🚀 Pushing to remote..."
git push -v origin main 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to remote"
    echo "✅ $DB_FILE preserved locally (not pushed, as it's in .gitignore)"
    if [ "$local_db_exists" = true ] && [ -f "$backup_file" ]; then
        echo "✅ Backup available at: $backup_file"
    fi
    echo ""
    echo "✅ Followed git best practices:"
    echo "   - Pulled latest changes before pushing"
    echo "   - Preserved local data"
else
    echo ""
    echo "❌ Push failed"
    echo "   Check the error message above and resolve any issues"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Safe push completed!"
echo "=========================================="

