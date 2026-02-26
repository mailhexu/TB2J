#!/usr/bin/env bash
#
# Convenience script to initialize test data (if configured)
# and run pytest from the repository root.

set -euo pipefail

# Ensure we are at the repository root
ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Initialize test-data submodule if it is configured
if [ -f ".gitmodules" ] && grep -q "tests/data" .gitmodules; then
    if [ -x "./tests/init_test_data.sh" ]; then
        echo "Initializing test data submodule using ./tests/init_test_data.sh ..."
        ./tests/init_test_data.sh
    else
        echo "./tests/init_test_data.sh not found or not executable; skipping submodule initialization." >&2
    fi
fi

echo "Running pytest $*"
pytest "$@"
