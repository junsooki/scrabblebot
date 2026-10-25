#!/bin/sh
# Build the adaptive-shading version in its own target dir (the live binary
# is untouched) and compare early_hot thresholds on held-out matches.
# early_hot=0 is today's live behaviour.
set -e
cd "$(dirname "$0")"
CARGO_TARGET_DIR=target-v3 cargo build --release --bins 2>&1 | tail -2
for h in 0 0.9 1.0 1.1 1.25 1.5; do
  printf 'early_hot=%s  ' "$h"
  target-v3/release/replay --theta theta-tournament.txt --split held-out --set early_hot=$h | sed -n 2p
done
