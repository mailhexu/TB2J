#!/bin/bash
#
# Script to ADD the test data repository as a git submodule for the first time
# This script should be run ONCE locally to add the submodule to your repository
#

set -e  # Exit on any error

SUBMODULE_URL="https://github.com/mailhexu/TB2J_test_data.git"
SUBMODULE_PATH="tests/data"

echo "Adding test data submodule to repository..."

# Check if submodule already exists
if [ -f ".gitmodules" ] && grep -q "$SUBMODULE_PATH" .gitmodules; then
    echo "Error: Submodule already exists in .gitmodules"
    echo "If you need to update it, use: git submodule update --remote"
    echo "If you need to reinitialize it, use: ./init_test_data.sh"
    exit 1
fi

# Check if directory already exists
if [ -d "$SUBMODULE_PATH" ]; then
    echo "Error: Directory $SUBMODULE_PATH already exists"
    echo "Please remove it first or choose a different path"
    exit 1
fi

# Create tests directory if it doesn't exist
if [ ! -d "tests" ]; then
    echo "Creating tests directory..."
    mkdir -p tests
fi

# Add the submodule
echo "Adding submodule: $SUBMODULE_URL -> $SUBMODULE_PATH"
git submodule add "$SUBMODULE_URL" "$SUBMODULE_PATH"

echo ""
echo "✅ Test data submodule added successfully!"
echo ""
echo "Next steps:"
echo "1. Commit the changes: git add .gitmodules tests/data && git commit -m 'Add test data submodule'"
echo "2. Push the changes: git push"
echo ""
echo "The submodule will now be automatically initialized in CI/CD environments."
