#!/usr/bin/env python3
"""Check indexed-coordinate field grinding and nonce sizes against the catalog.

The selected schedule geometry comes from the checked-in native Akita catalog
audit (SHA-256 2bb3b0c78a3d69daab47ed5fd2d8e34d4aff831ff280b44928e79f480a1ed8de).
PCS field-event enumeration and tree debits are compared with the Aerie
paper's whole-protocol tree evidence, computed independently from a frozen
catalog input (SHA-256
178b00e63a596d1ba3b3c5d81d6f656fc1346908377107adcf4f43536df761e7) at Akita
3720078d. The evidence is vendored as a gzip file at
`specs/evidence/whole-protocol-tree-checks.json.gz`; its SHA-256
59c2c978af41a677e4b25b89e9eb5a4e968753b8cbaeefe6ffb6aa15442b6ad8
is checked on the decompressed bytes. `--paper-json` selects another copy.
A missing or changed file fails.
The admitted shapes are derived from the padding rule of falcon.md Section 2.1
for every record count in [1, 2^16]: 14 Falcon-512 and 15 Falcon-1024 padded
batches. For each shape, this script independently expands the selected
checked-in Akita catalog schedule, prices every PIOP and PCS field event over
the exact field cardinality, and compares event IDs, debits, minimal grinding
bits, sparse-coordinate groups, and completion counts exactly, including the
root EvaluationBatch site. All allocation and support calculations use exact
integers and fractions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from fractions import Fraction
from math import comb, prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = ROOT / "specs" / "falcon.md"
BEGIN = "<!-- BEGIN TRANSCRIPT GRINDING TABLE -->"
END = "<!-- END TRANSCRIPT GRINDING TABLE -->"
# Default when --paper-json is not given.
VENDORED_PAPER_JSON = ROOT / "specs" / "evidence" / "whole-protocol-tree-checks.json.gz"
# SHA-256 of the uncompressed paper evidence.
PAPER_JSON_SHA256 = "59c2c978af41a677e4b25b89e9eb5a4e968753b8cbaeefe6ffb6aa15442b6ad8"

P = 2**64 - 59
K = P * P
T = 2**128
D64_SUPPORT = comb(64, 41) * comb(41, 31) * 2**41
D128_RAW_SUPPORT = comb(128, 31) * 2**31
D256_RAW_SUPPORT = comb(256, 23) * 2**23
# ceil(|support| * the pinned D128 operator-norm acceptance probability).
D128_ACCEPTED_SUPPORT = 502822610611069237129947393810277364817
D128_NORM_THRESHOLD = 13
D128_COLLISION_FACTOR = 64
NATIVE_CATALOG_AUDIT_SHA256 = "2bb3b0c78a3d69daab47ed5fd2d8e34d4aff831ff280b44928e79f480a1ed8de"
THETA = Fraction(1, 2**135)
assert D64_SUPPORT == 361715808335458406596740595030299770880
assert D128_RAW_SUPPORT == 1046949673839956231392050469602223718400
assert D256_RAW_SUPPORT == 2873108161471518200301057969538203648000

# falcon.md Section 2.1 and `FalconDegree::padded_records`.
MAX_RECORDS = 1 << 16
MIN_PADDED_COEFFICIENTS = 1 << 12


def padded_records(degree: int, records: int) -> int:
    if not 1 <= records <= MAX_RECORDS:
        raise ValueError("record count must be in [1, 2^16]")
    return max(1 << (records - 1).bit_length(), MIN_PADDED_COEFFICIENTS // degree)


def shape_name(degree: int, records: int) -> str:
    return (f"Falcon-{degree}-Nstar-{records}" if records < 1024
            else f"Falcon-{degree}-{records // 1024}K")


EXPECTED_RECORDS = {
    degree: tuple(sorted({padded_records(degree, records)
                          for records in range(1, MAX_RECORDS + 1)}))
    for degree in (512, 1024)
}
EXPECTED_SHAPES = {shape_name(degree, records)
                   for degree, padded in EXPECTED_RECORDS.items()
                   for records in padded}
EXPECTED_SHAPE_COUNT = 29
if (EXPECTED_RECORDS[512] != tuple(1 << e for e in range(3, 17))
        or EXPECTED_RECORDS[1024] != tuple(1 << e for e in range(2, 17))
        or len(EXPECTED_SHAPES) != EXPECTED_SHAPE_COUNT):
    raise SystemExit("FAIL: admitted padded shapes differ from falcon.md Section 2.1")

sys.path.insert(0, str(HERE))
import falcon_msis_audit as audit  # noqa: E402

# The native catalog audit must cover exactly the admitted padded shapes.
if (audit.FALCON_512_BATCHES != EXPECTED_RECORDS[512]
        or audit.FALCON_1024_BATCHES != EXPECTED_RECORDS[1024]):
    raise SystemExit("FAIL: native catalog audit batches differ from the admitted padded shapes")


def ceil_log2(n: int) -> int:
    if n <= 0:
        raise ValueError("ceil_log2 input must be positive")
    return (n - 1).bit_length()


def minimum_grind_bits(error: Fraction) -> int:
    """Least g such that 2^g >= 2^128 * the exact site debit."""
    if error <= 0:
        raise ValueError("grinding error must be positive")
    numerator = T * error.numerator
    denominator = error.denominator
    bits = 0
    while denominator * (1 << bits) < numerator:
        bits += 1
    if bits and denominator * (1 << (bits - 1)) >= numerator:
        raise AssertionError("grinding allocation is not minimal")
    return bits


def nonce_leb128_max_bytes(bits: int) -> int:
    """Maximum inline unsigned-LEB128 size for a nonzero PoW site."""
    if bits <= 0:
        return 0
    width = bits + 7
    return (width + 6) // 7


def expected_nonce_leb128_bytes(bits: int) -> float:
    """Expected first-success nonce size, conditioned on success in its range."""
    if bits <= 0:
        return 0.0
    limit = 1 << (bits + 7)
    log_failure = math.log1p(-(2.0 ** -bits))
    denominator = -math.expm1(limit * log_failure)
    # For a nonnegative integer N, its LEB128 length is one plus the number
    # of powers 128^j (j >= 1) that N reaches. Sum the conditioned geometric
    # tails P[N >= threshold | N < limit].
    extra = 0.0
    threshold = 128
    while threshold < limit:
        tail = (math.exp(threshold * log_failure)
                - math.exp(limit * log_failure)) / denominator
        extra += tail
        threshold *= 128
    return 1.0 + extra


def route_from_profile(profile: dict) -> dict:
    matrix = profile["inner"]["matrix"]
    routes = matrix["security_route"]
    if "Linf" in routes:
        route = routes["Linf"]
        return {"kind": "Linf", "ring_dimension": int(route["ring_dimension"])}
    if "L2" in routes:
        route = routes["L2"]
        key = route["table_key"]
        return {
            "kind": "L2",
            "ring_dimension": int(key["ring_dimension"]),
            "collision_l2_sq": int(key["collision_l2_sq"]),
            "response_l2_sq_cap": int(route["response_l2_sq_cap"]),
            "norm_proof_shape": route["norm_proof_shape"],
        }
    raise ValueError("unknown Akita inner security route")


def normalize_group(entry: dict) -> dict:
    profile = entry["profile"]
    opening = entry["opening"]
    method = opening["opening_method"]
    if isinstance(method, dict):
        method_name = "SubringCoefficientPacking"
        subring_dimension = int(method[method_name]["challenge_subring_dimension"])
    else:
        method_name = str(method)
        subring_dimension = None
    inner = profile["inner"]["matrix"]
    outer = profile["outer"]["matrix"]
    return {
        "num_vars": int(profile["group"]["num_vars"]),
        "num_polynomials": int(profile["group"]["num_polynomials"]),
        "live_blocks": int(profile["blocks"]["live_blocks"]),
        "positions_per_block": int(profile["blocks"]["positions_per_block"]),
        "inner_output_rank": int(inner["output_rank"]),
        "inner_ring_dimension": route_from_profile(profile)["ring_dimension"],
        "outer_slice_count": int(profile["outer_slice_count"]),
        "outer_output_rank": int(outer["output_rank"]),
        "inner_route": route_from_profile(profile),
        "opening_method": method_name,
        "opening_log_basis": int(opening["log_basis_open"]),
        "fold_count_pm1": int(opening["fold_challenge_config"]["count_pm1"]),
        "fold_count_pm2": int(opening["fold_challenge_config"]["count_pm2"]),
        "subring_dimension": subring_dimension,
        "setup_natural_len": entry["setup_natural_len"],
    }


def normalize_step(step: dict) -> dict:
    params = step["params"]
    return {
        "groups": [normalize_group(e) for e in params["groups"]["entries"]],
        "open_output_rank": int(params["open_matrix"]["output_rank"]),
        "output_witness_len": int(step["output_witness_len"]),
        "payload_mode": str(params["payload_mode"]),
    }


def normalize_terminal(terminal: dict) -> dict:
    return {
        "inner_ring_dimension": route_from_profile({"inner": terminal["inner"]})["ring_dimension"],
        "positions_per_block": int(terminal["blocks"]["positions_per_block"]),
        "live_blocks": int(terminal["blocks"]["live_blocks"]),
        "fold_count_pm1": int(terminal["fold_challenge_config"]["count_pm1"]),
        "fold_count_pm2": int(terminal["fold_challenge_config"]["count_pm2"]),
        "inner_route": route_from_profile({"inner": terminal["inner"]}),
    }


def schedule_profile(degree: int, records: int, row: dict) -> dict:
    schedule = audit.schedule_from_row(row)
    root = normalize_step(schedule["root"])
    recursive = [normalize_step(step) for step in schedule["recursive_folds"]]
    terminal = normalize_terminal(schedule["terminal"])
    return {
        "shape": shape_name(degree, records),
        "degree": degree,
        "n_sig": records,
        "d_coef": degree.bit_length() - 1,
        "d_sig": records.bit_length() - 1,
        "d": (degree * records).bit_length() - 1,
        "root": root,
        "recursive_folds": recursive,
        "terminal": terminal,
    }


def relation_rounds(output_witness_len: int, successor_ring_dimension: int) -> int:
    live_rings = (output_witness_len + successor_ring_dimension - 1) // successor_ring_dimension
    padded_rings = 1 << (live_rings - 1).bit_length()
    return (padded_rings * successor_ring_dimension).bit_length() - 1


def recursive_opening_vars(group: dict) -> int:
    return (group["inner_ring_dimension"].bit_length() - 1
            + group["positions_per_block"].bit_length() - 1
            + ceil_log2(group["live_blocks"]))


def row_index_width(step: dict) -> int:
    rows = sum(1 + g["inner_output_rank"] + g["outer_slice_count"] * g["outer_output_rank"]
               for g in step["groups"])
    rows += step["open_output_rank"]
    if step["payload_mode"] == "Compressed":
        rows += (len(step["groups"]) + 1) * 2
    elif step["payload_mode"] != "Raw":
        raise ValueError(f"unknown payload mode {step['payload_mode']}")
    return ceil_log2(rows + 1)


def field_event(events: list[dict], level: str, family: str, loss: int,
                factor: int | None = None, ordinal: int = 1) -> None:
    if loss <= 0:
        raise ValueError(f"nonpositive loss for PCS field event {family}")
    events.append({"site_id": f"PCS/{level}/{family}/{ordinal}", "family": family,
                   "error": Fraction(loss, K),
                   "factor": factor if factor is not None else loss + 1})


def sumcheck_events(events: list[dict], level: str, family: str,
                    stage: int, rounds: int, degree: int) -> None:
    for round_no in range(1, rounds + 1):
        field_event(events, level, family, degree, degree + 1,
                    ordinal=stage * 1000 + round_no)


def add_eor(events: list[dict], level: str, max_vars: int, row_batch: bool = False) -> None:
    if max_vars < 1:
        raise ValueError("extension-opening reduction requires a positive width")
    field_event(events, level, "EORSplitPoint", 1, 2)
    if row_batch:
        field_event(events, level, "EORClaimBatch", 1, 2)
    sumcheck_events(events, level, "EORSumcheck", 0, max_vars - 1, 2)


def stage1_shapes(log_basis: int, route_kind: str) -> tuple[list[tuple[int, int]], bool, int]:
    arities = {2: [], 3: [], 4: [2], 5: [4], 6: [2, 4]}.get(log_basis)
    if arities is None:
        raise ValueError(f"unsupported stage-1 opening log basis {log_basis}")
    stages: list[tuple[int, int]] = []
    children = 1
    for arity in arities:
        children *= arity
        stages.append((arity + 1, children))
    leaf_degree = 2 if log_basis == 2 else 4
    is_l2 = route_kind == "L2"
    if not is_l2:
        if route_kind != "Linf":
            raise ValueError(f"unsupported inner route {route_kind}")
        stages.append((leaf_degree + 1, 0))
    return stages, is_l2, leaf_degree


def l2_admission(route: dict) -> None:
    if route["kind"] != "L2" or route["ring_dimension"] != 128:
        raise ValueError("unsupported L2 route in catalog")
    if list(route["norm_proof_shape"]) != ["Direct"]:
        raise ValueError("unsupported L2 norm proof shape")
    required = (D128_COLLISION_FACTOR * D128_NORM_THRESHOLD**2
                * route["response_l2_sq_cap"])
    if route["collision_l2_sq"] < required:
        raise ValueError("catalog L2 collision filter is not admitted")
    if int(route["norm_proof_shape"]["Direct"]["physical_response_len"]) <= 0:
        raise ValueError("empty catalog L2 physical response")


def sparse_support(group: dict) -> int:
    pm1, pm2 = group["fold_count_pm1"], group["fold_count_pm2"]
    if group["opening_method"] == "SubringCoefficientPacking":
        if group["inner_route"]["kind"] != "Linf":
            raise ValueError("coefficient-packing sparse support requires Linf")
        # Akita's production ladder: D64 uses 31 +-1 and 10 +-2 entries,
        # D128 uses 31 +-1 entries, D256 uses 23 +-1 entries; all are raw
        # (unfiltered) Linf supports.
        if (group["subring_dimension"], pm1, pm2) == (64, 31, 10):
            return D64_SUPPORT
        if (group["subring_dimension"], pm1, pm2) == (128, 31, 0):
            return D128_RAW_SUPPORT
        if (group["subring_dimension"], pm1, pm2) == (256, 23, 0):
            return D256_RAW_SUPPORT
        raise ValueError("unregistered coefficient-packing sparse support")
    if group["opening_method"] != "EvaluationTrace" or pm1 != 31 or pm2 != 0:
        raise ValueError("unregistered evaluation-trace sparse support")
    if group["inner_ring_dimension"] != 128:
        raise ValueError("evaluation-trace sparse support requires D128")
    if group["inner_route"]["kind"] == "L2":
        l2_admission(group["inner_route"])
        return D128_ACCEPTED_SUPPORT
    if group["inner_route"]["kind"] != "Linf":
        raise ValueError("unknown evaluation-trace route")
    return D128_RAW_SUPPORT


def expand_pcs(profile: dict) -> tuple[list[dict], list[dict]]:
    field: list[dict] = []
    coordinate_groups: list[dict] = []
    root, recursive, terminal = profile["root"], profile["recursive_folds"], profile["terminal"]
    predecessors = [("L0", root, True)]
    predecessors.extend((f"L{i + 1}", step, False) for i, step in enumerate(recursive))
    previous_rounds = None

    for index, (level, step, is_root) in enumerate(predecessors):
        groups = step["groups"]
        if not groups:
            raise ValueError(f"{profile['shape']} {level}: empty group set")
        if is_root:
            layout_vars = max(g["num_vars"] for g in groups)
            layout_polys = sum(g["num_polynomials"] for g in groups)
        else:
            if previous_rounds is None:
                raise ValueError("missing prior relation round count")
            layout_vars, layout_polys = previous_rounds, 1
            if len(groups) != 1 or groups[0]["num_polynomials"] != 1:
                raise ValueError(f"{profile['shape']} {level}: expected scalar successor")
            if groups[0]["num_vars"] != previous_rounds:
                raise ValueError(f"{profile['shape']} {level}: opening width mismatch")

        methods = {g["opening_method"] for g in groups}
        if len(methods) != 1:
            raise ValueError(f"{profile['shape']} {level}: mixed opening methods")
        method = next(iter(methods))
        if method == "EvaluationTrace":
            if layout_polys != 1:
                raise ValueError("v1 registry does not admit multi-claim EOR")
            add_eor(field, level, layout_vars, layout_polys > 1)
        elif method != "SubringCoefficientPacking":
            raise ValueError(f"{profile['shape']} {level}: unsupported opening method")

        successor = recursive[index] if index < len(recursive) else {"groups": [terminal]}
        successor_group = successor["groups"][0]
        successor_d = successor_group["inner_ring_dimension"]
        rounds = relation_rounds(step["output_witness_len"], successor_d)
        successor_vars = recursive_opening_vars(successor_group)
        if rounds != successor_vars:
            raise ValueError(f"{profile['shape']} {level}: relation rounds/opening width mismatch")
        tau1 = row_index_width(step)
        for group_index, group in enumerate(groups):
            coordinate_groups.append({
                "event_id": f"PCS/fold/L{index}",
                "group": group_index,
                "coordinate_count": group["num_polynomials"] * group["live_blocks"],
                "support": sparse_support(group),
            })

        if layout_polys > 1:
            if is_root and len(groups) > 1:
                if layout_polys != 9:
                    raise ValueError("catalog grouped root no longer has nine polynomials")
                # Coordinate-wise tree charge: the extractor branches on base
                # eta plus nine one-coordinate alternatives, so the site has
                # debit 9/|K| and ten children (falcon.md Section 10.1).
                field_event(field, level, "EvaluationBatch", layout_polys, layout_polys + 1)
            else:
                field_event(field, level, "EvaluationBatch", max(1, layout_polys - 1), layout_polys)

        # Akita prices the ring-switch challenge by the native inner ring for
        # both opening methods: the quotient-lift residual of the
        # inner-consistency rows has degree below 2 * d_max.
        alpha_loss = max(2 * g["inner_ring_dimension"] - 1 for g in groups)
        field_event(field, level, "RingSwitchAlpha", alpha_loss, alpha_loss + 1)
        field_event(field, level, "Tau0Point", rounds, 2**rounds)
        field_event(field, level, "Tau1Point", tau1, 2**tau1)

        stage_group = groups[-1]
        stages, has_l2, leaf_degree = stage1_shapes(
            stage_group["opening_log_basis"], stage_group["inner_route"]["kind"])
        for stage_no, (degree, child_claims) in enumerate(stages):
            sumcheck_events(field, level, "Stage1RangeSumcheck", stage_no, rounds, degree)
            if child_claims > 0:
                loss = max(child_claims - 1, 1)
                field_event(field, level, "Stage1InterstageBatch", loss,
                            child_claims, ordinal=stage_no + 1)
        if has_l2:
            l2_admission(stage_group["inner_route"])
            field_event(field, level, "L2NormMerge", 1, 2)
            sumcheck_events(field, level, "PhysicalL2Sumcheck", 0, rounds, leaf_degree + 1)
            field_event(field, level, "L2VirtualBatch", 1, 2)
        if step["payload_mode"] == "Compressed":
            field_event(field, level, "CompressionBinary", 1, 2)
        field_event(field, level, "Stage2Batch", 1, 2)
        sumcheck_events(field, level, "Stage2Sumcheck", 0, rounds, 3)
        if any(g["setup_natural_len"] is not None for g in groups):
            raise ValueError(f"{profile['shape']} {level}: setup prefix is unsupported")
        previous_rounds = rounds

    terminal_width = recursive_opening_vars(terminal)
    if previous_rounds is None or previous_rounds > terminal_width:
        raise ValueError("terminal extension opening width is invalid")
    add_eor(field, f"L{len(recursive) + 1}-terminal", previous_rounds)
    terminal_group = {
        **terminal,
        "opening_method": "EvaluationTrace",
        "subring_dimension": None,
        "num_polynomials": 1,
    }
    if terminal["inner_route"]["kind"] != "L2":
        raise ValueError("terminal schedule must use the certified L2 support")
    terminal_support = sparse_support(terminal_group)
    coordinate_groups.append({
        "event_id": f"PCS/fold/L{len(predecessors)}",
        "group": 0,
        "coordinate_count": terminal["live_blocks"],
        "support": terminal_support,
    })
    return field, coordinate_groups


def expand_piop(profile: dict) -> list[dict]:
    d = int(profile["d"])
    degree = int(profile["degree"])
    gammas = (3, 8, 12) if degree == 512 else (4, 9, 12)
    out = [
        {"site_id": "PIOP/mu", "family": "PIOP projection-output batching vector",
         "error": Fraction(8, K), "factor": 2**8},
        {"site_id": "PIOP/tau", "family": "PIOP factor-product vector",
         "error": Fraction(d, K), "factor": 2**d},
    ]
    for stage, width in enumerate(gammas, 1):
        out.append({"site_id": f"PIOP/gamma{stage}",
                    "family": f"PIOP stage {stage} independent batching coefficients",
                    "error": Fraction(width, K), "factor": width + 1})
    for stage, degree_bound in enumerate((3, 5, 3), 1):
        for round_no in range(1, d + 1):
            out.append({"site_id": f"PIOP/stage{stage}/sumcheck-round{round_no}",
                        "family": f"PIOP stage {stage} degree-{degree_bound} sumcheck round",
                        "error": Fraction(degree_bound, K), "factor": degree_bound + 1})
    return out


def artifact_support_id(support: int) -> str:
    try:
        return {
            D64_SUPPORT: "D64_signed_sparse_31_plus_10",
            D128_RAW_SUPPORT: "D128_signed_sparse_31",
            D256_RAW_SUPPORT: "D256_signed_sparse_23",
            D128_ACCEPTED_SUPPORT: "D128_operator_norm_accepted_lower_bound",
        }[support]
    except KeyError as exc:
        raise ValueError(f"unregistered sparse support {support}") from exc


def compare_artifact(profile: dict, piop: list[dict], pcs_field: list[dict],
                     coordinate_groups: list[dict], paper_row: dict) -> dict:
    indexed = paper_row["indexed_allocation"]
    if indexed["convention"] != "indexed-coordinate accounting (deployed)":
        raise SystemExit(f"FAIL: {profile['shape']} paper convention is not indexed-coordinate")
    if indexed["field_allocation_rule"] != "least g >= 0 with 2^g >= 2^128 * e_r":
        raise SystemExit(f"FAIL: {profile['shape']} paper field allocation rule changed")

    fields = piop + pcs_field
    if len(indexed["field_events"]) != len(fields):
        raise SystemExit(f"FAIL: {profile['shape']} field-event count differs from paper artifact")
    for local, saved in zip(fields, indexed["field_events"]):
        error = local["error"]
        bits = local["grinding_bits"]
        if saved["event_id"] != local["site_id"] or saved["family"] != local["family"]:
            raise SystemExit(
                f"FAIL: {profile['shape']} field event order/identity differs at {local['site_id']}"
            )
        if int(saved["tree_error_numerator"]) != error.numerator \
                or int(saved["tree_factor"]) != int(local["factor"]):
            raise SystemExit(f"FAIL: {profile['shape']} debit/tree factor differs at {local['site_id']}")
        if error != Fraction(int(saved["tree_error_numerator"]), K):
            raise SystemExit(f"FAIL: {profile['shape']} exact debit differs at {local['site_id']}")
        if bits != int(saved["grinding_bits"]) or not saved["minimality_checked"]:
            raise SystemExit(f"FAIL: {profile['shape']} field grinding bits differ at {local['site_id']}")
        priced_error = Fraction(int(saved["priced_error"]["numerator"]),
                                 int(saved["priced_error"]["denominator"]))
        if priced_error != error / (1 << bits):
            raise SystemExit(f"FAIL: {profile['shape']} priced debit differs at {local['site_id']}")

    saved_groups = indexed["sparse_coordinate_groups"]
    if len(saved_groups) != len(coordinate_groups):
        raise SystemExit(f"FAIL: {profile['shape']} sparse-coordinate group count differs")
    for local, saved in zip(coordinate_groups, saved_groups):
        support = local["support"]
        if (saved["event_id"] != local["event_id"]
                or int(saved["group"]) != local["group"]
                or int(saved["coordinate_count"]) != local["coordinate_count"]
                or saved["support_id"] != artifact_support_id(support)
                or int(saved["support_lower_bound"]) != support
                or int(saved["grinding_bits"]) != 0):
            raise SystemExit(
                f"FAIL: {profile['shape']} sparse-coordinate group differs at "
                f"{local['event_id']} group {local['group']}"
            )
        if Fraction(int(saved["per_touch_error"]["numerator"]),
                    int(saved["per_touch_error"]["denominator"])) != Fraction(1, support):
            raise SystemExit(f"FAIL: {profile['shape']} coordinate support debit differs")

    fold_ids = list(dict.fromkeys(group["event_id"] for group in coordinate_groups))
    if indexed["fold_grinding_bits"] != [0] * len(fold_ids):
        raise SystemExit(f"FAIL: {profile['shape']} paper artifact assigns sparse-fold work")
    field_completions = len(fields) + 1  # The projection-seed predicate is also a field completion.
    coordinate_completions = sum(group["coordinate_count"] for group in coordinate_groups)
    completion = indexed["completion"]
    if (int(indexed["coordinate_count_total"]) != coordinate_completions
            or int(completion["R_field_upper_bound"]) != field_completions
            or int(completion["R_coord_upper_bound"]) != coordinate_completions
            or int(completion["R_total_upper_bound"]) != field_completions + coordinate_completions):
        raise SystemExit(f"FAIL: {profile['shape']} verifier completion counts differ")

    seed_error = max(Fraction(3, 2**256), Fraction(4, 3) * THETA)
    saved_seed_error = indexed["seed_predicate_error"]
    if seed_error != Fraction(int(saved_seed_error["numerator"]),
                              int(saved_seed_error["denominator"])):
        raise SystemExit(f"FAIL: {profile['shape']} projection-seed debit differs")
    field_priced_errors = [event["error"] / (1 << event["grinding_bits"])
                           for event in fields]
    coordinate_errors = [Fraction(1, group["support"]) for group in coordinate_groups]
    rho = max([seed_error, *field_priced_errors, *coordinate_errors])
    if rho > Fraction(1, T):
        raise SystemExit(f"FAIL: {profile['shape']} indexed address weight exceeds 2^-128")
    saved_rho = indexed["maximum_error_per_charged_address"]
    if (rho != Fraction(int(saved_rho["numerator"]), int(saved_rho["denominator"]))
            or not indexed["maximum_le_2^-128_exact"]):
        raise SystemExit(f"FAIL: {profile['shape']} maximum indexed address weight differs")
    field_missing_error = seed_error + sum(field_priced_errors, Fraction())
    coordinate_missing_error = sum(
        (group["coordinate_count"] * Fraction(1, group["support"])
         for group in coordinate_groups), Fraction())
    for name, actual in (("all_field_streams_missing_error_upper_bound", field_missing_error),
                         ("all_coordinate_streams_missing_error_upper_bound", coordinate_missing_error)):
        saved = completion[name]
        if actual != Fraction(int(saved["numerator"]), int(saved["denominator"])):
            raise SystemExit(f"FAIL: {profile['shape']} {name} differs")

    fold_counts: dict[str, int] = {}
    for group in coordinate_groups:
        fold_counts[group["event_id"]] = (
            fold_counts.get(group["event_id"], 0) + group["coordinate_count"])
    complete_tree_B = 2 * prod(int(event["factor"]) for event in fields)
    complete_tree_B *= prod(count + 1 for count in fold_counts.values())
    d = int(profile["d"])
    explicit_length = int(profile["degree"]) * int(profile["n_sig"])
    if explicit_length != 1 << d:
        raise SystemExit(f"FAIL: {profile['shape']} explicit length is not 2^d")
    complete_tree_exponent = 0
    while complete_tree_B > 1 << (d * complete_tree_exponent):
        complete_tree_exponent += 1
    saved_tree = paper_row["complete_tree"]
    if (not indexed["tree_unchanged"]
            or complete_tree_B != int(saved_tree["including_seed_factor_B"])
            or complete_tree_exponent != int(saved_tree["minimal_common_power_exponent"])
            or not saved_tree["exact_exponent_check"]["B_le_2^(d_times_exponent)"]):
        raise SystemExit(f"FAIL: {profile['shape']} complete-tree admission differs")

    return {
        "field_events": fields,
        "field_sites": len(fields),
        "R_field": field_completions,
        "pcs_field_sites": len(pcs_field),
        "pio_field_sites": len(piop),
        "sparse_coordinate_groups": coordinate_groups,
        "sparse_coordinates": coordinate_completions,
        "fold_count": len(fold_ids),
        "maximum_address_error": rho,
        "complete_tree_B": complete_tree_B,
        "complete_tree_exponent": complete_tree_exponent,
    }


def native_allocation_summary(profile: dict, piop: list[dict], pcs_field: list[dict],
                              coordinate_groups: list[dict]) -> dict:
    """Compute the address bound and tree admission from Akita's native plan."""
    fields = piop + pcs_field
    seed_error = max(Fraction(3, 2**256), Fraction(4, 3) * THETA)
    field_priced_errors = [event["error"] / (1 << event["grinding_bits"])
                           for event in fields]
    coordinate_errors = [Fraction(1, group["support"]) for group in coordinate_groups]
    rho = max([seed_error, *field_priced_errors, *coordinate_errors])
    if rho > Fraction(1, T):
        raise SystemExit(f"FAIL: {profile['shape']} native indexed address weight exceeds 2^-128")

    fold_counts: dict[str, int] = {}
    for group in coordinate_groups:
        fold_counts[group["event_id"]] = (
            fold_counts.get(group["event_id"], 0) + group["coordinate_count"])
    complete_tree_B = 2 * prod(int(event["factor"]) for event in fields)
    complete_tree_B *= prod(count + 1 for count in fold_counts.values())
    d = int(profile["d"])
    complete_tree_exponent = 0
    while complete_tree_B > 1 << (d * complete_tree_exponent):
        complete_tree_exponent += 1

    return {
        "field_events": fields,
        "field_sites": len(fields),
        "R_field": len(fields) + 1,
        "pcs_field_sites": len(pcs_field),
        "pio_field_sites": len(piop),
        "sparse_coordinate_groups": coordinate_groups,
        "sparse_coordinates": sum(group["coordinate_count"]
                                   for group in coordinate_groups),
        "fold_count": len(fold_counts),
        "maximum_address_error": rho,
        "field_missing_error": seed_error + sum(field_priced_errors, Fraction()),
        "coordinate_missing_error": sum(
            (group["coordinate_count"] * Fraction(1, group["support"])
             for group in coordinate_groups), Fraction()),
        "complete_tree_B": complete_tree_B,
        "complete_tree_exponent": complete_tree_exponent,
    }


ROOT_EVALUATION_BATCH = "PCS/L0/EvaluationBatch/1"
# The complete tree branches on base eta and nine one-coordinate alternatives
# (ten children); the checkpoint extractor does not branch at eta, where a
# fixed source with a false claim has one bad answer per nonzero linear form.
ROOT_EXACT_P2_RULE = {"L": 9, "checkpoint_extractor_branches": False,
                      "checkpoint_fixed_source_numerator": 1, "child_count": 10,
                      "complete_tree_branches": True, "g": 4}


def check_root_evaluation_batch(native_pcs_field: list[dict], paper_row: dict,
                                shape: str) -> None:
    """Require the paper's root allocation to equal the native ten-child rule."""
    saved_roots = [event for event in paper_row["indexed_allocation"]["field_events"]
                   if event["event_id"] == ROOT_EVALUATION_BATCH]
    native_roots = [event for event in native_pcs_field
                    if event["site_id"] == ROOT_EVALUATION_BATCH]
    if len(saved_roots) != 1 or len(native_roots) != 1:
        raise SystemExit(f"FAIL: {shape} must contain one root EvaluationBatch event")

    saved, native = saved_roots[0], native_roots[0]
    rule = ROOT_EXACT_P2_RULE
    if (native["error"] != Fraction(rule["L"], K)
            or native["grinding_bits"] != rule["g"]
            or int(native["factor"]) != rule["child_count"]):
        raise SystemExit(f"FAIL: {shape} native EvaluationBatch exact-p2 rule differs")
    if (int(saved["tree_error_numerator"]) != rule["L"]
            or int(saved["tree_factor"]) != rule["child_count"]
            or int(saved["grinding_bits"]) != rule["g"]
            or saved.get("exact_p2_rule") != rule):
        raise SystemExit(f"FAIL: {shape} paper EvaluationBatch allocation differs from native L=9, g=4")


def derive_row(profile: dict, paper_row: dict) -> dict:
    piop = expand_piop(profile)
    pcs_field, coordinate_groups = expand_pcs(profile)
    for event in piop + pcs_field:
        bits = minimum_grind_bits(event["error"])
        if bits > 25:
            raise SystemExit(f"FAIL: {profile['shape']} work target exceeds native u32 policy")
        if Fraction(1 << bits) < T * event["error"]:
            raise AssertionError("per-site grinding target not met")
        if bits and Fraction(1 << (bits - 1)) >= T * event["error"]:
            raise AssertionError("per-site grinding difficulty is not minimal")
        event["grinding_bits"] = bits

    check_root_evaluation_batch(pcs_field, paper_row, profile["shape"])
    # Every site, including the root EvaluationBatch, must match exactly.
    compare_artifact(profile, piop, pcs_field, coordinate_groups, paper_row)
    if (int(paper_row["degree"]) != int(profile["degree"])
            or int(paper_row["n_sig"]) != int(profile["n_sig"])
            or paper_row["schedule_shape"] != profile["shape"]):
        raise SystemExit(f"FAIL: {profile['shape']} shape identity differs from paper artifact")

    native_counts = native_allocation_summary(profile, piop, pcs_field, coordinate_groups)
    field_count = native_counts["field_sites"]
    piop_nonce_bits = [event["grinding_bits"] for event in piop]
    piop_nonce_fields = sum(bits > 0 for bits in piop_nonce_bits)
    piop_nonce_worst_bytes = sum(nonce_leb128_max_bytes(bits)
                                 for bits in piop_nonce_bits)
    piop_nonce_expected_bytes = math.fsum(
        expected_nonce_leb128_bytes(bits) for bits in piop_nonce_bits
    )
    row = {
        "shape": profile["shape"],
        "degree": profile["degree"],
        "n_sig": profile["n_sig"],
        "d": profile["d"],
        "piop_bits": pio_grinding_summary(piop),
        "pio_field_sites": native_counts["pio_field_sites"],
        "pcs_field_sites": native_counts["pcs_field_sites"],
        "field_sites": field_count,
        "R_field": native_counts["R_field"],
        "sparse_coordinates": native_counts["sparse_coordinates"],
        "fold_count": native_counts["fold_count"],
        "maximum_address_error": native_counts["maximum_address_error"],
        "field_missing_error": native_counts["field_missing_error"],
        "coordinate_missing_error": native_counts["coordinate_missing_error"],
        "complete_tree_B": native_counts["complete_tree_B"],
        "complete_tree_exponent": native_counts["complete_tree_exponent"],
        "pio_nonce_fields": piop_nonce_fields,
        "pio_nonce_worst_bytes": piop_nonce_worst_bytes,
        "pio_nonce_expected_bytes": piop_nonce_expected_bytes,
        "pcs_field_bit_histogram": histogram(e["grinding_bits"] for e in pcs_field),
        "pcs_family_bit_histograms": family_histograms(pcs_field),
        "events": piop + pcs_field,
        "coordinate_groups": coordinate_groups,
    }
    return row


def histogram(values) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def family_histograms(events: list[dict]) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for event in events:
        values = result.setdefault(event["family"], {})
        bits = event["grinding_bits"]
        values[bits] = values.get(bits, 0) + 1
    return result


def pio_grinding_summary(events: list[dict]) -> str:
    bits = {event["site_id"]: event["grinding_bits"] for event in events}
    stages = []
    for stage in (1, 2, 3):
        stage_events = [event for event in events
                        if event["site_id"].startswith(f"PIOP/stage{stage}/")]
        stage_bits = {event["grinding_bits"] for event in stage_events}
        if len(stage_bits) != 1:
            raise ValueError(f"PIOP stage {stage} site difficulties are not uniform")
        stages.append(f"S{stage} {next(iter(stage_bits))}x{len(stage_events)}")
    gamma = ",".join(str(bits[f"PIOP/gamma{i}"]) for i in (1, 2, 3))
    return (f"mu {bits['PIOP/mu']}; tau {bits['PIOP/tau']}; "
            f"gamma {gamma}; sumcheck {'; '.join(stages)}")


def piop_plan_rows() -> list[dict]:
    """Emit the normative PIOP allocation without requiring paper evidence."""
    rows = []
    for degree in (512, 1024):
        for records in EXPECTED_RECORDS[degree]:
            d = (degree * records).bit_length() - 1
            profile = {"degree": degree, "d": d}
            events = expand_piop(profile)
            for event in events:
                event["grinding_bits"] = minimum_grind_bits(event["error"])
            bits = {event["site_id"]: event["grinding_bits"] for event in events}
            rows.append({
                "degree": degree,
                "n_sig": records,
                "rounds": d,
                "mu": bits["PIOP/mu"],
                "tau": bits["PIOP/tau"],
                "gamma": [bits[f"PIOP/gamma{stage}"] for stage in (1, 2, 3)],
                "sumcheck": [
                    bits[f"PIOP/stage{stage}/sumcheck-round1"]
                    for stage in (1, 2, 3)
                ],
            })
    if len(rows) != EXPECTED_SHAPE_COUNT:
        raise AssertionError("PIOP plan must cover exactly the admitted padded shapes")
    return rows


def paper_json_path(explicit: Path | None = None) -> Path:
    """Select --paper-json, else the vendored copy."""
    if explicit is not None:
        return explicit
    return VENDORED_PAPER_JSON


def load_paper_artifact(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"FAIL: missing paper indexed-allocation evidence at {path}")
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PAPER_JSON_SHA256:
        raise SystemExit(f"FAIL: paper evidence hash changed: {digest}")
    artifact = json.loads(raw)
    if artifact.get("deployed_convention") != "indexed-coordinate accounting":
        raise SystemExit("FAIL: paper evidence does not deploy indexed-coordinate accounting")
    if set(artifact.get("shapes", {})) != EXPECTED_SHAPES:
        raise SystemExit("FAIL: paper evidence does not contain exactly the admitted padded shapes")
    return artifact


def checked_rows() -> list[dict]:
    """Return rows checked against the selected paper evidence; fail if absent."""
    return all_rows(load_paper_artifact(paper_json_path()))


def validate_native_catalog() -> dict:
    """Load and validate the pinned native catalog report."""
    report = audit.load_catalog_report()
    catalog_sha256 = hashlib.sha256(audit.REPORT.read_bytes()).hexdigest()
    if catalog_sha256 != NATIVE_CATALOG_AUDIT_SHA256:
        raise SystemExit(f"FAIL: native catalog audit hash changed: {catalog_sha256}")
    if audit.validate_report(report, quiet=True) != 0:
        raise SystemExit("FAIL: current native PCS catalog audit is invalid")
    return report


def all_rows(paper: dict, report_native_allocation: bool = False) -> list[dict]:
    report = validate_native_catalog()
    rows = []
    for degree in (512, 1024):
        native_rows = audit.production_rows(report, degree)
        for records in EXPECTED_RECORDS[degree]:
            profile = schedule_profile(degree, records, native_rows[records])
            try:
                paper_row = paper["shapes"][profile["shape"]]
            except KeyError as exc:
                raise SystemExit(f"FAIL: no paper evidence for {profile['shape']}") from exc
            rows.append(derive_row(profile, paper_row))
    names = {row["shape"] for row in rows}
    if names != EXPECTED_SHAPES or len(rows) != EXPECTED_SHAPE_COUNT:
        raise SystemExit("FAIL: catalog must cover exactly the admitted padded shapes")
    if report_native_allocation:
        print("INFO: root PCS EvaluationBatch equals the paper allocation at every "
              "shape: L=9, g=4, ten children (flat coordinate branching).")
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        BEGIN,
        "| Shape | PIOP field-site work bits | PIOP field sites | PCS field sites | PCS site-family work bits (g:events) | $`R_{\\mathrm{field}}`$ | Indexed sparse-coordinate completions (zero work) | PIOP nonce fields | Worst-case PIOP nonce bytes | Expected PIOP nonce bytes |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pcs_families = []
        for family, counts in sorted(row["pcs_family_bit_histograms"].items()):
            work = ",".join(f"{bits}:{count}" for bits, count in sorted(counts.items()))
            pcs_families.append(f"{family} {work}")
        pcs_summary = "; ".join(pcs_families)
        lines.append(
            f"| {row['shape']} | {row['piop_bits']} | {row['pio_field_sites']} | "
            f"{row['pcs_field_sites']} | {pcs_summary} | {row['R_field']} | "
            f"{row['sparse_coordinates']} | {row['pio_nonce_fields']} | "
            f"{row['pio_nonce_worst_bytes']} | {row['pio_nonce_expected_bytes']:.2f} |"
        )
    lines.extend((END, ""))
    return "\n".join(lines)


def check_spec(rendered: str) -> None:
    if not SPEC.is_file():
        raise SystemExit(f"FAIL: missing {SPEC}")
    text = SPEC.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise SystemExit(f"FAIL: {SPEC.relative_to(ROOT)} must contain one table marker pair")
    start = text.index(BEGIN)
    end = text.index(END, start) + len(END)
    actual = text[start:end] + "\n"
    if actual != rendered:
        raise SystemExit("FAIL: normative transcript grinding table differs; use --print")


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--print", action="store_true")
    modes.add_argument("--json", action="store_true")
    modes.add_argument("--piop-plan-json", action="store_true")
    parser.add_argument(
        "--paper-json",
        type=Path,
        default=None,
        help="the paper's whole-protocol-tree-checks.json (default: the vendored "
        + str(VENDORED_PAPER_JSON.relative_to(ROOT)) + ")",
    )
    args = parser.parse_args()
    if args.piop_plan_json:
        print(json.dumps(piop_plan_rows(), sort_keys=True, separators=(",", ":")))
        return
    paper_path = paper_json_path(args.paper_json)
    paper = load_paper_artifact(paper_path)
    rows = all_rows(paper, report_native_allocation=args.check)
    rendered = render_table(rows)
    if args.print:
        print(rendered, end="")
        return
    if args.json:
        def encode(obj):
            if isinstance(obj, Fraction):
                return {"numerator": str(obj.numerator), "denominator": str(obj.denominator)}
            if isinstance(obj, dict):
                return {str(key): encode(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [encode(value) for value in obj]
            return obj
        print(json.dumps(encode(rows), indent=2, sort_keys=True))
        return
    check_spec(rendered)
    print(f"PASS: catalog-derived indexed grinding table matches the {EXPECTED_SHAPE_COUNT}-shape "
          f"paper evidence ({paper_path}) and specs/falcon.md")


if __name__ == "__main__":
    main()
