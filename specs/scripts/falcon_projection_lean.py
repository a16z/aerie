#!/usr/bin/env python3
"""Link between the projection constants and their Lean formalization.

The norm check and the ring-equation check of `specs/falcon.md` Section 6 are formalized in
Lean in the `certified-projections` library. This script ties the three places that hold
the constants of those checks:

1. `falcon_projection_certificate.json`, produced by `falcon_projection_certificate.py`;
2. the Rust verifier (`crates/aerie-falcon/src/protocol/projection`);
3. the Lean definitions, through the Lean file `CertifiedProjections/Tests/AerieSpecConstants.lean`.

The Lean file is generated here from the certificate. Each of its lines states that a Lean
definition equals the value in the certificate, and Lean checks every line when
`certified-projections` is built. `falcon_projection_lean.json` records the SHA-256 of the
file. It does not record a commit of `certified-projections`.

    falcon_projection_lean.py --check     compare the Rust constants with the certificate, and
                                          the generated Lean file with the recorded SHA-256
    falcon_projection_lean.py --print     write the generated Lean file to standard output
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CERTIFICATE = HERE / "falcon_projection_certificate.json"
RECORD = HERE / "falcon_projection_lean.json"
LEAN_FILE = "CertifiedProjections/Tests/AerieSpecConstants.lean"

PROFILE_RS = ROOT / "crates/aerie-falcon/src/protocol/projection/profile.rs"
NATIVE_PARAMS_RS = ROOT / "crates/aerie-falcon-native/src/params.rs"
SUBSTRATE_RS = ROOT / "crates/aerie-falcon/src/substrate.rs"

HEADER = """\
/-
Copyright (c) 2026 Quang Dao. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Quang Dao
-/

import CertifiedProjections.Aerie.Geometry

/-!
# Constants of the Aerie specification

Every example states that a definition of the Aerie instance equals the value of the same constant
in the Aerie specification. The file is generated in the Aerie repository by
`specs/scripts/falcon_projection_lean.py` from `specs/scripts/falcon_projection_certificate.json`,
and that repository records the SHA-256 of this file. Do not edit it by hand.
-/

namespace CertifiedProjections.Tests.AerieSpecConstants

open CertifiedProjections.MultiStage CertifiedProjections.Aerie
"""

FOOTER = """
end CertifiedProjections.Tests.AerieSpecConstants
"""


def vector(values):
    return "![" + ", ".join(str(v) for v in values) + "]"


def chosen(certificate, n):
    found = [c for c in certificate["configurations"] if c["role"] == "chosen" and c["n"] == n]
    if len(found) != 1:
        raise SystemExit(f"expected one chosen configuration for n = {n}")
    return found[0]


def high_bits(config):
    """The number R of record-index bits above the first radix, at the largest batch."""
    return sum(config["norm_geometry"]["radix_exponents"][1:])


def lean_file(certificate):
    k = certificate["constants"]
    widths = k["norm_widths"]
    if len(set(widths)) != 1:
        raise SystemExit("the branch stages do not have a common number of rows")
    lines = [HEADER]
    lines += [
        "/-! ## Constants shared by both parameter sets -/",
        "",
        f"example : branchRows = {widths[0]} := rfl",
        f"example : numBranchStages = {len(widths)} := rfl",
        f"example : numBranches = {k['norm_groups']} := rfl",
        f"example : finalRows = {k['norm_outputs']} := rfl",
        f"example : normCheckError = 2⁻¹ ^ {k['eps_bits']} := rfl",
    ]
    for n in (512, 1024):
        c = chosen(certificate, n)
        P = f"falcon{n}"
        G = f"({P}Geometry {high_bits(c)})"
        m = c["exactness"]["m"]
        numerator, denominator = c["Q_S"].split("/")
        residual = c["residual"]
        lines += [
            "",
            f"/-! ## Falcon-{n} -/",
            "",
            f"example : {P}.p = {k['p']} := by decide",
            f"example : {P}.q = {k['q']} := rfl",
            f"example : {P}.n = {n} := rfl",
            f"example : {P}.betaSq = {c['B']} := rfl",
            f"example : {P}.maxSignatures = {k['records']} := by decide",
            f"example : {P}.acceptSqNorm = {c['C_S']} := rfl",
            f"example : {P}.fanInMax = {vector(c['exactness']['fan_in'])} := rfl",
            f"example : {P}.s2Bound = {k['s2_max']} := rfl",
            f"example : {P}.publicBound = {(k['q'] - 1) // 2} := rfl",
            f"example : {P}.residualEntryBound = {c['L_W']} := rfl",
            f"example : {P}.ringEntryBound = {residual['b']} := rfl",
            f"example : {P}.ringRows = {vector(residual['widths'])} := rfl",
            f"example : {P}.overflowOrder = {residual['wrap_order']} := rfl",
            f"example : {P}.honestOrder = {c['cap_order']} := rfl",
            f"example : eta = {c['eta_proj']} := by norm_num [eta]",
            f"example : {P}.soundSqNorm = {numerator} / {denominator} := by",
            f"  norm_num [Params.soundSqNorm, {P}, eta]",
            f"example : {P}.honestSqNorm = {c['H_S']} := by decide",
            f"example : {P}.residualSqNorm = {c['H_W']} := by decide",
            f"example : Nat.sqrt {P}.betaSq = {c['cmax']} := by",
            "  symm; rw [Nat.eq_sqrt]; decide",
            f"/-- The integer square root of the acceptance bound of Falcon-{n}. -/",
            f"theorem {P}_sqrtAcceptSqNorm : {P}.sqrtAcceptSqNorm = {m} := by",
            "  unfold Params.sqrtAcceptSqNorm; symm; rw [Nat.eq_sqrt]; decide",
        ]
        for i, tau in enumerate(c["exactness"]["tau"]):
            lines += [
                f"example : {P}.tau {i} = {tau} := by",
                f"  rw [Params.tau, {P}_sqrtAcceptSqNorm]; decide",
            ]
        lines += [
            f"example : {G}.numRecords = {k['records']} := by decide",
            f"example : {G}.normRadices {P}.n = {vector(c['norm_geometry']['radices'])} := by",
            "  decide",
            f"example : {G}.ringRadices {P}.n = {vector(c['residual_geometry']['radices'])} := by",
            "  decide",
            f"example : ({G}.normShape {P}.n).f = {vector(c['exactness']['fan_in'])} := by",
            "  decide",
        ]
    lines.append(FOOTER)
    return "\n".join(lines)


def rust_constant(text, name):
    match = re.search(rf"\bconst {name}: \w+ = ([0-9_]+);", text)
    if match is None:
        raise SystemExit(f"Rust constant {name} not found")
    return int(match.group(1).replace("_", ""))


def check_rust(certificate):
    """Compare the constants of the Rust verifier with the certificate."""
    k = certificate["constants"]
    c512, c1024 = chosen(certificate, 512), chosen(certificate, 1024)
    profile = PROFILE_RS.read_text()
    native_params = NATIVE_PARAMS_RS.read_text()
    substrate = SUBSTRATE_RS.read_text()
    expected = {
        "NORM_GROUPS": k["norm_groups"],
        "NORM_PREFIX_LAYERS": len(k["norm_widths"]),
        "NORM_PREFIX_WIDTH": k["norm_widths"][0],
        "NORM_TERMINAL_ROWS": k["norm_outputs"],
        "RESIDUAL_LAYERS": len(c512["residual"]["widths"]),
        "RESIDUAL_WIDTH": c512["residual"]["widths"][0],
        "RESIDUAL_BOUND": c512["residual"]["b"],
        "FALCON_Q": k["q"],
        "NORM_FIRST_BLOCK_BITS": c512["norm_geometry"]["radix_exponents"][0],
        "RESIDUAL_FIRST_BLOCK_BITS": c512["residual_geometry"]["radix_exponents"][0],
        "NORM_LATER_BLOCK_BITS": c512["norm_geometry"]["radix_exponents"][1],
        "NORM_CAP_FALCON_512": c512["C_S"],
        "NORM_CAP_FALCON_1024": c1024["C_S"],
    }
    failures = []
    for name, value in expected.items():
        found = rust_constant(native_params if name == "FALCON_Q" else profile, name)
        if found != value:
            failures.append(f"{name}: Rust has {found}, the certificate has {value}")
    match = re.search(r"assert!\(FIELD_MODULUS == u64::MAX - ([0-9_]+)\);", substrate)
    if match is None:
        raise SystemExit("Rust constant FIELD_MODULUS not found")
    modulus = 2**64 - 1 - int(match.group(1).replace("_", ""))
    if modulus != k["p"]:
        failures.append(f"FIELD_MODULUS: Rust has {modulus}, the certificate has {k['p']}")
    for c in (c512, c1024):
        if (c["residual"]["b"], c["residual"]["widths"]) != (
            c512["residual"]["b"],
            c512["residual"]["widths"],
        ):
            failures.append("the two parameter sets use different ring-equation checks")
        exponents = c["norm_geometry"]["radix_exponents"]
        if exponents[:2] != c512["norm_geometry"]["radix_exponents"][:2]:
            failures.append("the two parameter sets use different first radices")
        if c["residual_geometry"]["radix_exponents"][0] != expected["RESIDUAL_FIRST_BLOCK_BITS"]:
            failures.append("the two parameter sets use different first residual radices")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--print", action="store_true")
    args = parser.parse_args()
    certificate = json.loads(CERTIFICATE.read_text())
    text = lean_file(certificate)
    if args.print:
        sys.stdout.write(text)
        return
    failures = check_rust(certificate)
    record = json.loads(RECORD.read_text())
    digest = hashlib.sha256(text.encode()).hexdigest()
    if record["file"] != LEAN_FILE:
        failures.append(f"the record names {record['file']}, expected {LEAN_FILE}")
    if record["sha256"] != digest:
        failures.append(
            f"the Lean file generated from the certificate has SHA-256 {digest}, "
            f"the record has {record['sha256']}: regenerate the file in certified-projections "
            "and update falcon_projection_lean.json"
        )
    if failures:
        for failure in failures:
            print("FAIL", failure)
        raise SystemExit(1)
    print(f"ok: Rust constants match the certificate; {LEAN_FILE} has the recorded "
          f"SHA-256 ({digest[:16]})")


if __name__ == "__main__":
    main()
