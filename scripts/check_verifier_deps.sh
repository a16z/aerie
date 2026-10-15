#!/usr/bin/env bash
# Verifier normal/build dependency graph hygiene, with and without parallelism.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

check_graph() {
  local package="$1"
  local features="$2"
  local mode="$3"
  local graph root crate rest
  local args=(--locked -p "$package" --no-default-features -e normal,build
    --prefix none --format '{p}')
  if [ -n "$features" ]; then
    args+=(--features "$features")
  fi
  if ! graph="$(cargo tree "${args[@]}")"; then
    echo "error: cargo tree failed for $package $mode" >&2
    return 1
  fi
  if [ -z "$graph" ]; then
    echo "error: empty dependency graph for $package $mode" >&2
    return 1
  fi
  read -r root rest <<<"$graph"
  if [ "$root" != "$package" ]; then
    echo "error: expected graph root $package, got $root ($mode)" >&2
    return 1
  fi
  while read -r crate rest; do
    case "$crate" in
      aerie-prover|aerie-falcon-witness|aerie-bench|aerie-invariants|aerie-profiling|\
      akita-pcs|akita-prover|akita-cpu-backend|akita-setup|akita-planner|\
      fn-dsa|fn-dsa-*|sysinfo|criterion|dhat)
        echo "error: forbidden crate in $package $mode normal/build graph: $crate" >&2
        return 1
        ;;
      rayon|rayon-core)
        if [ "$mode" != parallel ]; then
          echo "error: forbidden crate in $package $mode normal/build graph: $crate" >&2
          return 1
        fi
        ;;
    esac
  done <<<"$graph"
  echo "ok $package $mode"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  check_graph aerie-verifier "" scalar
  check_graph aerie-falcon "" scalar
  check_graph aerie verifier scalar
  check_graph aerie-verifier parallel parallel
  check_graph aerie-falcon parallel parallel
  check_graph aerie verifier,parallel parallel
fi
