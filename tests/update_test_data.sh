#!/usr/bin/env bash
#
# Update the TB2J_test_data submodule to the latest remote commit.
#
# Usage (from anywhere):
#   ./tests/update_test_data.sh
#
# After running this, you should usually commit the updated
# submodule pointer in the TB2J repository:
#   git add tests/data
#   git commit -m "Update tests/data submodule to latest TB2J_test_data"

set -euo pipefail

# Move to repository root (one level above this script)
ROOT_DIR="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Updating tests/data submodule to latest remote..."
git submodule update --remote tests/data

echo "Done. Run 'git status' to review changes."
