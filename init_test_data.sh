#!/bin/bash
#
# Script to initialize git submodules in CI/CD environments
# This script is designed to work in GitHub Actions and other CI environments
#

set -e  # Exit on any error

echo "Initializing submodules in CI environment..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check if .gitmodules exists
if [ -f ".gitmodules" ]; then
    echo "Found .gitmodules, initializing submodules..."
    
    # Initialize and update all submodules
    git submodule update --init --recursive
    
    # Verify the test data submodule if it exists
    if [ -d "tests/data" ]; then
        if [ "$(ls -A tests/data 2>/dev/null)" ]; then
            echo "✅ Test data directory contains files:"
            ls -la tests/data/ | head -5
        else
            echo "⚠️  Test data directory exists but is empty"
        fi
    else
        echo "ℹ️  No test data directory found"
    fi
else
    echo "ℹ️  No .gitmodules file found - no submodules to initialize"
fi

echo "Submodule initialization complete!"
