#!/usr/bin/env bash
# Run FLAIR's CPU-only tooling in Docker with ./outputs mounted, so saved
# records land on the host instead of vanishing with the container.
#
#   ./run-local.sh test                         # the test suite
#   ./run-local.sh explain "a very red car"     # routing decisions, saved
#   ./run-local.sh summary                      # what has been saved so far
#   ./run-local.sh shell                        # poke around
#
# Images are NOT produced here -- generation needs a GPU. Use
# notebooks/flair_kaggle.ipynb for that.
set -euo pipefail

IMAGE=flair-test
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/outputs"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "building $IMAGE ..."
    docker build -q -t "$IMAGE" "$HERE" >/dev/null
fi

# MSYS/Git-Bash rewrites /app to a Windows path without the leading slash.
run() { MSYS_NO_PATHCONV=1 docker run --rm -v "$HERE/outputs:/app/outputs" "$@"; }

case "${1:-test}" in
    test)    run "$IMAGE" python -m pytest -q ;;
    explain) shift; run "$IMAGE" python scripts/explain.py "$@" --save outputs/ ;;
    summary) run "$IMAGE" python -c \
                 "from flair_t2i.artifacts import summarise; print(summarise('outputs'))" ;;
    shell)   run -it "$IMAGE" /bin/bash ;;
    *)       echo "usage: $0 {test|explain <prompt> [flags]|summary|shell}" >&2; exit 1 ;;
esac
