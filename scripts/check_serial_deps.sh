#!/usr/bin/env bash
# Serial normal/build dependency graph hygiene for proving and verification.
set -euo pipefail
cd "$(dirname "$0")/.."

# Proving may link witness preparation and producer Akita crates, but never
# the honest signer, Rayon, diagnostics, or benchmark instrumentation.
serial_forbidden=(rayon rayon-core sysinfo aerie-profiling criterion dhat)
verifier_forbidden=(aerie-prover aerie-falcon-witness aerie-bench aerie-invariants
  akita-pcs akita-prover akita-cpu-backend akita-setup akita-planner)

check_graph() {
  local package="$1"
  local features="$2"
  local role="$3"
  local graph crate count
  local args=(--locked -p "$package" --no-default-features -e normal,build --prefix none)
  if [ -n "$features" ]; then
    args+=(--features "$features")
  fi
  graph="$(cargo tree "${args[@]}" | awk '{print $1}' | sort -u)"
  for crate in "${serial_forbidden[@]}"; do
    if grep -qx "$crate" <<<"$graph"; then
      echo "error: forbidden crate in $package serial $role normal/build graph: $crate" >&2
      return 1
    fi
  done
  if grep -Eq '^fn-dsa($|-)' <<<"$graph"; then
    echo "error: signer in $package serial $role normal/build graph" >&2
    return 1
  fi
  if [ "$role" = verifier ]; then
    for crate in "${verifier_forbidden[@]}"; do
      if grep -qx "$crate" <<<"$graph"; then
        echo "error: forbidden crate in $package serial verifier normal/build graph: $crate" >&2
        return 1
      fi
    done
  fi
  count="$(wc -l <<<"$graph" | tr -d ' ')"
  echo "$package serial $role normal/build graph: $count crates, forbidden set absent"
}

check_graph aerie prover prover
check_graph aerie verifier verifier
check_graph aerie-falcon "" verifier
check_graph aerie-verifier "" verifier
