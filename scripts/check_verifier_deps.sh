#!/usr/bin/env bash
# Verifier dependency hygiene (CI job "Verifier portability").
#
# The `aerie-falcon` graph without default features is what a verifier-only
# consumer links. It must never pull the honest signer, Rayon parallelism, or
# host diagnostics.
set -euo pipefail
cd "$(dirname "$0")/.."

forbidden=(fn-dsa rayon sysinfo aerie-profiling criterion)

graph="$(cargo tree --locked -p aerie-falcon --no-default-features -e normal --prefix none \
  | awk '{print $1}' | sort -u)"

status=0
for crate in "${forbidden[@]}"; do
  if grep -qx "$crate" <<<"$graph"; then
    echo "error: forbidden crate in the verifier dependency graph: $crate" >&2
    status=1
  fi
done

count="$(wc -l <<<"$graph" | tr -d ' ')"
echo "verifier graph: $count crates, forbidden set absent"
exit "$status"
