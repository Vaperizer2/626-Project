#!/usr/bin/env bash
# Copy this checkout's quant tools and compclass to carlpc2 and record the commit they came from.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
R=carlpc2:/home/carl/626-runs/quant/repo
rsync -a --delete --exclude __pycache__ "$REPO/tools/quant/" "$R/tools/quant/"
rsync -a --exclude __pycache__ "$REPO/tools/speed/" "$R/tools/speed/"
rsync -a --exclude __pycache__ "$REPO/src/compclass/" "$R/src/compclass/"
sha=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- tools/quant || echo -dirty)
ssh carlpc2 "sed -i '/QUANT_GIT_SHA/d' /home/carl/626-runs/quant/env.sh; echo export QUANT_GIT_SHA=$sha >> /home/carl/626-runs/quant/env.sh"
echo "synced $sha"
