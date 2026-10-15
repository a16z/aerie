#!/usr/bin/env bash
# Build a standalone verifier consumer and check its normal/build dependencies.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo/scripts/check_verifier_deps.sh"
consumer="$repo/target/verifier-consumer"
mkdir -p "$consumer/src"

cat > "$consumer/Cargo.toml" <<EOF
[package]
name = "aerie-verifier-consumer"
version = "0.0.0"
edition = "2021"

[workspace]

[dependencies]
aerie = { path = "$repo/crates/aerie", default-features = false, features = ["verifier"] }
EOF
cat > "$consumer/src/main.rs" <<'EOF'
use aerie::falcon::{verify_aggregate, Falcon512, Falcon1024, FalconVerifierSetup};

fn main() {
    let _ = FalconVerifierSetup::<Falcon512>::new;
    let _ = verify_aggregate::<Falcon512>;
    let _ = FalconVerifierSetup::<Falcon1024>::new;
    let _ = verify_aggregate::<Falcon1024>;
}
EOF
cp "$repo/Cargo.lock" "$consumer/Cargo.lock"
cd "$consumer"
export CARGO_TARGET_DIR="$consumer/target"
# Adapt the copied lock to the standalone root while preserving package identities.
cargo tree --offline --no-default-features -e normal,build > /dev/null
awk '
  BEGIN { RS = ""; FS = "\n" }
  /^\[\[package\]\]/ {
    identity = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^(name|version|source|checksum) = /) identity = identity $i "\n"
    }
    if (FILENAME == ARGV[1]) {
      locked[identity] = 1
    } else if ($2 != "name = \"aerie-verifier-consumer\"" && !(identity in locked)) {
      print "error: consumer package differs from workspace lock:\n" identity > "/dev/stderr"
      failed = 1
    }
  }
  END { exit failed }
' "$repo/Cargo.lock" "$consumer/Cargo.lock"
cargo build --locked
check_graph aerie-verifier-consumer "" scalar
