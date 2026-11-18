#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

repository="${TWINE_REPOSITORY:-pypi}"
upload=false

usage() {
    cat <<'EOF'
Usage: ./upload_to_pip.sh [--upload] [--repository NAME]

Build TB2J distributions and validate them with twine.

Options:
  --upload            Upload checked artifacts. Without this, only build/check.
  --repository NAME   Twine repository name from ~/.pypirc (default: pypi).
  -h, --help          Show this help.

Environment:
  TWINE_REPOSITORY    Default repository when --repository is not given.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)
            upload=true
            shift
            ;;
        --repository)
            repository="${2:?--repository requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

rm -rf -- dist build
uv build --no-sources

artifacts=(dist/*)
if [[ ! -e "${artifacts[0]}" ]]; then
    echo "No distribution artifacts were built in dist/." >&2
    exit 1
fi

python -m twine check --strict "${artifacts[@]}"

if [[ "$upload" == true ]]; then
    python -m twine upload --repository "$repository" --verbose "${artifacts[@]}"
else
    echo "Build and twine check passed. Re-run with --upload to publish to '$repository'."
fi
