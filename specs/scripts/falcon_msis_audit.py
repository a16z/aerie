#!/usr/bin/env python3
"""Audit the checked-in Falcon four-group PCS catalog report.

The report is emitted by the aerie-pcs-config regeneration harness from schedules
planned and validated by Akita e023b6a46. This script checks report provenance and structure,
then counts the exact SIS matrix occurrence multiset Akita derives from each
schedule, including compression and setup-prefix matrices. It does not turn
Akita's 128-bit estimator gate into a probability or a concrete MSIS advantage.
"""

import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "falcon_pcs_catalog_audit.json"
EXPECTED_SCHEMA = "aerie-pcs-catalog-audit-v1"
EXPECTED_AKITA = "e023b6a46143f7b23199ccecb2684192baf7b640"
EXPECTED_FAMILIES = (
    "aerie_fp64_signed_width14",
    "aerie_fp64_signed_width4",
)
# Exact claimed.p0 from Akita 3720078d763bd87df3434f50dfe69aac071b1dab,
# scripts/operator_norm/d128/cert_d128_w31_gamma13.json. The certificate is
# byte-for-byte identical at db5efa205 and 515a21668 (SHA-256 e9f255359549bf3b7726eb5b71e80b6d8aae29160832ed06f920194fa34d41f7). Retain the rational
# here so CI needs no external checkout; derive ceil(C128 * p0) below. This
# certificate covers the strict fixed-point predicate at threshold 13.
D128_ACCEPTANCE_P0 = Fraction(
    "4865195740716121784397629768877885937089252993943257404592882856209937833980"
    "1329542064977413374121562895733964894651201466162662551320768817527014937711"
    "6416998325073961060611659330297/10130043849301306876940913187098365613989529"
    "5329036323069862958730806862999531371420756868273685201183654268697205939601"
    "2386050206992870987463622987078066966989874839179755520000000000"
)
# Exact claimed.p0 from Akita 3720078d763bd87df3434f50dfe69aac071b1dab,
# scripts/operator_norm/d64/cert_d64_a31_b11_gamma18.json (SHA-256
# cd967cd643a21f89543b5340d54830837ce73beb4c91bb154baf6e5942e9e230, identical
# at 515a21668). It covers the D=64 (31, 11) shell under the strict
# fixed-point predicate at threshold 18 (OperatorNormRejection::D64_SELECTIVE_L2).
D64_ACCEPTANCE_P0 = Fraction(
    "3950345228831770106447954194274496298213348879331438320951824378373604150916"
    "09701043860418006919606612147275605263845077835591257/1681637318810762621397"
    "2390587628634421542687605226240322594159295650278187624333344394095431847989"
    "85374212724335307550778982400000"
)
# Certified selective-L2 families: (dimension, count_pm1, count_pm2) -> p0.
L2_ACCEPTANCE_P0 = {
    (64, 31, 11): D64_ACCEPTANCE_P0,
    (128, 31, 0): D128_ACCEPTANCE_P0,
}
CURRENT_SIS_TABLE_DIGESTS = {
    bytes.fromhex(
        "ddc82aa1d362bc174169291768516d3f"
        "5d019bef4b0453049bfac755ca3be44b"
    ),
    bytes.fromhex(
        "a1cc0a06089744145b61919cf001ea26"
        "0c95a4bba161ffdaff55391d0dfa102a"
    ),
}
FAILURES = []
QUIET = False

FALCON_512_BATCHES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                      8192, 16384, 32768, 65536)
FALCON_1024_BATCHES = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                       4096, 8192, 16384, 32768, 65536)
GROUP_ROLES = ("S1", "e", "a0,a1,a2,sigma", "u0,u1,u2")
GROUP_POLYNOMIALS = (1, 1, 4, 3)


def require(label, ok, detail=""):
    if not QUIET:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)


def check_positive_int(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    return isinstance(value, str) and value.isdecimal() and int(value) > 0


def table_digests(value):
    """Collect serialized table identities from the complete native schedule."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "table_digest" and isinstance(child, list):
                try:
                    found.append(bytes(child))
                except (TypeError, ValueError):
                    found.append(b"")
            else:
                found.extend(table_digests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(table_digests(child))
    return found


def profiles_from_row(row):
    profiles = row.get("profiles")
    if isinstance(profiles, dict) and isinstance(profiles.get("exact_json"), str):
        return json.loads(profiles["exact_json"], parse_int=int)
    if isinstance(profiles, dict):
        return profiles
    raise ValueError("catalog audit row has no structured profiles")


def group_profiles_from_row(row):
    """Restore exact fallback serialization on any oversized profile value."""
    groups = row.get("group_profiles")
    if not isinstance(groups, list):
        raise ValueError("catalog audit row has no group profile list")
    restored = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("catalog group profile is not an object")
        item = dict(group)
        profile = item.get("profile")
        if isinstance(profile, dict) and isinstance(profile.get("exact_json"), str):
            item["profile"] = json.loads(profile["exact_json"], parse_int=int)
        if not isinstance(item.get("profile"), dict):
            raise ValueError("catalog group profile has no structured committed profile")
        restored.append(item)
    return restored


def matrix_ring_dimension(matrix):
    route = matrix["security_route"]
    if "Linf" in route:
        return route["Linf"]["ring_dimension"]
    if "L2" in route:
        return route["L2"]["table_key"]["ring_dimension"]
    raise ValueError("unknown inner matrix security route")


def support_size(config, dimension):
    """Raw support arithmetic; L2 rejection requires the certified bound below."""
    pm1 = config["count_pm1"]
    pm2 = config["count_pm2"]
    weight = pm1 + pm2
    if weight > dimension:
        raise ValueError("sparse challenge weight exceeds its ring dimension")
    return math.comb(dimension, weight) * math.comb(weight, pm1) * (1 << weight)


def challenge_support_lower_bound(config, dimension, matrix):
    """Certified support after selective rejection, or exact unfiltered support."""
    raw = support_size(config, dimension)
    route = matrix["security_route"]
    if "Linf" in route:
        return raw
    if "L2" not in route:
        raise ValueError("unknown inner matrix security route")
    p0 = L2_ACCEPTANCE_P0.get(
        (dimension, config.get("count_pm1"), config.get("count_pm2")))
    if (p0 is None or matrix_ring_dimension(matrix) != dimension
            or set(config) != {"count_pm1", "count_pm2"}):
        raise ValueError("L2 challenge family has no accepted-support certificate here")
    accepted = raw * p0
    return (accepted.numerator + accepted.denominator - 1) // accepted.denominator


def sufficient_margin_bits(term):
    """Least b with term * 2^128 <= 2^b; no floating-point boundary rounding."""
    bits = 0
    while term.numerator * (1 << 128) > term.denominator * (1 << bits):
        bits += 1
    return bits


def profile_support_size(profile, opening):
    method = opening["opening_method"]
    if isinstance(method, dict) and "SubringCoefficientPacking" in method:
        dimension = method["SubringCoefficientPacking"]["challenge_subring_dimension"]
    elif method == "EvaluationTrace":
        dimension = matrix_ring_dimension(profile["inner"]["matrix"])
    else:
        raise ValueError(f"unknown opening method {method!r}")
    return challenge_support_lower_bound(
        opening["fold_challenge_config"], dimension, profile["inner"]["matrix"])


def conditional_cwss_terms(schedule):
    """Return rational conditional CWSS upper bounds from native schedule geometry.

    For every nonterminal group, claims are its polynomial count and each claim
    contributes its exact live-block count. The terminal schedule has one
    terminal witness source, so it contributes one claim. L2 denominators are
    certified accepted-support lower bounds, not exact accepted cardinalities.
    """
    terms = []
    folds = [schedule["root"], *schedule["recursive_folds"]]
    for fold_index, fold in enumerate(folds):
        entries = fold["params"]["groups"]["entries"]
        for group_index, entry in enumerate(entries):
            profile = entry["profile"]
            claims = profile["group"]["num_polynomials"]
            blocks = profile["blocks"]["live_blocks"]
            support = profile_support_size(profile, entry["opening"])
            terms.append((f"fold{fold_index}.group{group_index}",
                          Fraction(claims * blocks, support)))

    terminal = schedule["terminal"]
    dimension = matrix_ring_dimension(terminal["inner"]["matrix"])
    support = challenge_support_lower_bound(
        terminal["fold_challenge_config"], dimension, terminal["inner"]["matrix"])
    blocks = terminal["blocks"]["live_blocks"]
    terms.append(("terminal", Fraction(blocks, support)))
    return terms


def conditional_cwss_summary(schedule):
    """Return per-fold upper bounds, terminal bound, and their rational sum."""
    terms = conditional_cwss_terms(schedule)
    fold_count = len(schedule["recursive_folds"]) + 1
    fold_sums = [
        sum((term for label, term in terms if label.startswith(f"fold{index}.")),
            Fraction())
        for index in range(fold_count)
    ]
    terminal = next(term for label, term in terms if label == "terminal")
    return fold_sums, terminal, sum(fold_sums, terminal)


def load_catalog_report():
    """Load the checked-in native audit report without changing JSON integers."""
    if not REPORT.is_file():
        raise FileNotFoundError(f"missing native catalog audit report: {REPORT}")
    return json.loads(REPORT.read_text(), parse_int=int)


def schedule_from_row(row):
    """Prefer the direct exact serialization exported by the Rust harness."""
    schedule_json = row.get("schedule_json")
    if isinstance(schedule_json, str):
        return json.loads(schedule_json, parse_int=int)
    schedule = row.get("schedule")
    if isinstance(schedule, dict) and isinstance(schedule.get("exact_json"), str):
        return json.loads(schedule["exact_json"], parse_int=int)
    if isinstance(schedule, dict):
        return schedule
    raise ValueError("catalog audit row has neither exact nor structured schedule")


def expected_group_vars(degree, records):
    if degree not in (512, 1024) or records <= 0 or records & (records - 1):
        raise ValueError(f"unsupported Falcon shape ({degree}, {records})")
    d_coef = degree.bit_length() - 1
    d_sig = records.bit_length() - 1
    return tuple(max(13, value) for value in
                 (d_coef + d_sig, 2 + d_sig, d_coef + d_sig, d_sig))


def production_rows(report, degree):
    """Resolve all admitted grouped rows by their four native group profiles."""
    expected_batches = FALCON_512_BATCHES if degree == 512 else FALCON_1024_BATCHES
    width4 = next((family for family in report["families"]
                   if family["name"] == EXPECTED_FAMILIES[1]), None)
    if width4 is None:
        raise ValueError("width-4 catalog family is missing")
    rows = {}
    for row in width4["rows"]:
        try:
            groups = group_profiles_from_row(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(groups) != 4:
            continue
        if tuple(group.get("role") for group in groups) != GROUP_ROLES:
            continue
        shapes = tuple(group["profile"]["group"]["num_vars"] for group in groups)
        matching_batches = [records for records in expected_batches
                            if shapes == expected_group_vars(degree, records)]
        for records in matching_batches:
            if records in rows:
                raise ValueError(f"duplicate grouped audit row for degree={degree}, records={records}")
            rows[records] = row

    missing = sorted(set(expected_batches) - set(rows))
    extra = sorted(set(rows) - set(expected_batches))
    if missing or extra:
        raise ValueError(f"degree {degree} grouped rows missing={missing}, extra={extra}")
    for records, row in rows.items():
        check_production_group_row(row, degree, records)
    return rows


def check_production_group_row(row, degree, records):
    """Check canonical group shapes, widths, and decomposition bases."""
    groups = group_profiles_from_row(row)
    expected_vars = expected_group_vars(degree, records)
    if len(groups) != 4 or tuple(g.get("role") for g in groups) != GROUP_ROLES:
        raise ValueError(f"degree={degree}, records={records}: group order mismatch")
    if len(row.get("key", {}).get("precommitteds", [])) != 3:
        raise ValueError(f"degree={degree}, records={records}: expected three precommits")
    row_profiles = profiles_from_row(row)
    expected_profiles = [*row_profiles["precommitteds"], row_profiles["final_group"]]
    if len(expected_profiles) != len(groups):
        raise ValueError(f"degree={degree}, records={records}: committed profile count mismatch")
    for index, (group, role, polys, nvars) in enumerate(
            zip(groups, GROUP_ROLES, GROUP_POLYNOMIALS, expected_vars)):
        profile = group["profile"]
        if profile != expected_profiles[index]:
            raise ValueError(f"degree={degree}, records={records}: {role} profile mismatch")
        if profile["group"] != {"num_vars": nvars, "num_polynomials": polys}:
            raise ValueError(f"degree={degree}, records={records}: {role} shape mismatch")
        source_width = 14 if index < 2 else 4
        inner = group["inner_decomposition"]
        outer = group["outer_decomposition"]
        if inner["log_basis"] != profile["inner"]["digits"]["log_basis"] or \
                inner["num_digits"] != profile["inner"]["digits"]["num_digits"]:
            raise ValueError(f"degree={degree}, records={records}: {role} inner metadata mismatch")
        if outer["log_basis"] != profile["outer"]["digits"]["log_basis"] or \
                outer["num_digits"] != profile["outer"]["digits"]["num_digits"]:
            raise ValueError(f"degree={degree}, records={records}: {role} outer metadata mismatch")
        if source_width == 4:
            basis_ok = inner["log_basis"] == 4 and inner["num_digits"] == 1
        else:
            basis_ok = 3 <= inner["log_basis"] <= 14
            representable = ((1 << (inner["log_basis"] - 1))
                             * (((1 << inner["log_basis"]) ** inner["num_digits"] - 1)
                                // ((1 << inner["log_basis"]) - 1)))
            basis_ok = basis_ok and representable >= (1 << (source_width - 1))
        if not basis_ok:
            raise ValueError(f"degree={degree}, records={records}: {role} width/basis mismatch")
    schedule = schedule_from_row(row)
    root_entries = schedule["root"]["params"]["groups"]["entries"]
    if len(root_entries) != 4:
        raise ValueError(f"degree={degree}, records={records}: root group count mismatch")
    recursive_folds = schedule["recursive_folds"]
    for fold_index, fold in enumerate(recursive_folds):
        entries = fold["params"]["groups"]["entries"]
        if len(entries) != 1:
            raise ValueError(f"degree={degree}, records={records}: recursive fold {fold_index} group count mismatch")

    all_entries = [("root", index, entry)
                   for index, entry in enumerate(root_entries)]
    for fold_index, fold in enumerate(recursive_folds):
        all_entries.extend((f"recursive fold {fold_index}", 0, entry)
                           for entry in fold["params"]["groups"]["entries"])
    for fold_name, group_index, entry in all_entries:
        opening = entry["opening"]
        opening_basis = opening["log_basis_open"]
        if not 3 <= opening_basis <= 6:
            role = GROUP_ROLES[group_index] if fold_name == "root" else fold_name
            raise ValueError(f"degree={degree}, records={records}: {role} opening basis outside 3..6")
        expected_open_digits = (64 + opening_basis - 1) // opening_basis
        if opening["num_digits_open"] != expected_open_digits:
            role = GROUP_ROLES[group_index] if fold_name == "root" else fold_name
            raise ValueError(f"degree={degree}, records={records}: {role} opening digit count mismatch")


def production_ledger_rows(report, degree, batch_logs):
    """Return exact selected catalog metrics for the rows charged by a ledger."""
    rows = production_rows(report, degree)
    selected = []
    for log_records in batch_logs:
        records = 1 << log_records
        try:
            row = rows[records]
        except KeyError as error:
            raise ValueError(f"degree={degree} has no production row for N=2^{log_records}") from error
        schedule = schedule_from_row(row)
        fold_cwss, terminal_cwss, total_cwss = conditional_cwss_summary(schedule)
        occurrence_counts = Counter(
            (occurrence["role"], occurrence["bound"]["kind"])
            for occurrence in row["sis_occurrences"]
        )
        selected.append({
            "records": records,
            "row": row,
            "cwss": total_cwss,
            "fold_cwss": fold_cwss,
            "terminal_cwss": terminal_cwss,
            "cwss_term_count": sum(
                len(fold["params"]["groups"]["entries"])
                for fold in [schedule["root"], *schedule["recursive_folds"]]
            ) + 1,
            "occurrence_counts": occurrence_counts,
            "occurrence_total": len(row["sis_occurrences"]),
        })
    return selected


def validate_occurrence(occurrence, row_label, index):
    label = f"{row_label} SIS occurrence {index}"
    required = {
        "location", "role", "output_rank", "input_width", "ring_dimension",
        "modulus_profile", "bound",
    }
    if not isinstance(occurrence, dict) or not required <= occurrence.keys():
        return None

    valid = (
        isinstance(occurrence["location"], str) and bool(occurrence["location"])
        and occurrence["role"] in {"Inner", "Outer", "Open", "Compression"}
        and all(check_positive_int(occurrence[name]) for name in
                ("output_rank", "input_width", "ring_dimension"))
        and occurrence["modulus_profile"] == "Q64Offset59"
        and isinstance(occurrence["bound"], dict)
    )
    bound = occurrence["bound"]
    valid = valid and (
        set(bound) == {"kind", "value"}
        and bound["kind"] in {"linf", "l2_squared"}
        and check_positive_int(bound["value"])
    )
    if not valid:
        return None
    return occurrence["role"], bound["kind"]


def validate_report(report=None, quiet=False):
    global QUIET
    QUIET = quiet
    FAILURES.clear()
    if report is None:
        try:
            report = load_catalog_report()
        except (OSError, json.JSONDecodeError, FileNotFoundError) as error:
            if not quiet:
                print(f"STOP: cannot read native catalog audit report: {error}")
            return 2

    if not isinstance(report, dict):
        if not quiet:
            print("STOP: native catalog audit report root must be an object")
        return 2

    if not quiet:
        print("Falcon four-group PCS/MSIS catalog audit")
    require("native audit schema is current", report.get("schema") == EXPECTED_SCHEMA,
            str(report.get("schema")))
    require("catalog audit provenance matches its recorded Akita revision",
            report.get("akita_revision") == EXPECTED_AKITA,
            str(report.get("akita_revision")))
    families = report.get("families")
    require("both width families are present in canonical order",
            isinstance(families, list)
            and all(isinstance(family, dict) for family in families)
            and tuple(family.get("name") for family in families) == EXPECTED_FAMILIES)
    if (not isinstance(families, list)
            or len(families) != len(EXPECTED_FAMILIES)
            or any(not isinstance(family, dict) for family in families)):
        return 1

    any_rows = False
    for family, expected_name in zip(families, EXPECTED_FAMILIES):
        name = family.get("name")
        rows = family.get("rows")
        require(f"{expected_name} family name", name == expected_name, str(name))
        require(f"{expected_name} has nonempty catalog rows",
                isinstance(rows, list) and bool(rows))
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, dict) for row in rows)):
            continue
        any_rows = True
        keys = [json.dumps(row.get("key"), sort_keys=True) for row in rows]
        require(f"{expected_name} row keys are unique", len(keys) == len(set(keys)))
        role_counts = Counter()
        bound_counts = Counter()
        occurrence_total = 0
        schedule_folds = 0
        for row_index, row in enumerate(rows):
            row_label = f"{name} row {row_index}"
            require(f"{row_label} has key, profiles, groups, exact schedule and audit",
                    all(field in row for field in
                        ("key", "profiles", "group_profiles", "fold_count", "planner_estimate",
                         "sis_occurrences"))
                    and ("schedule_json" in row or "schedule" in row))
            try:
                schedule = schedule_from_row(row)
                profiles = profiles_from_row(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                require(f"{row_label} has exact parseable schedule and profiles", False,
                        str(error))
                continue
            estimate = row.get("planner_estimate")
            occurrences = row.get("sis_occurrences")
            valid_shape = (
                isinstance(schedule, dict)
                and isinstance(schedule.get("root"), dict)
                and isinstance(schedule.get("recursive_folds"), list)
                and isinstance(schedule.get("terminal"), dict)
                and isinstance(profiles, dict)
                and isinstance(profiles.get("final_group"), dict)
                and isinstance(profiles.get("precommitteds"), list)
                and isinstance(row.get("group_profiles"), list)
                and isinstance(estimate, dict)
                and check_positive_int(estimate.get("estimated_proof_payload_bytes"))
                and check_positive_int(estimate.get("estimated_num_setup_field_elements"))
                and isinstance(occurrences, list)
                and bool(occurrences)
            )
            require(f"{row_label} has coherent native schedule fields", valid_shape)
            if not valid_shape:
                continue
            expected_folds = len(schedule["recursive_folds"]) + 2
            require(f"{row_label} fold count matches schedule",
                    row["fold_count"] == expected_folds,
                    f"{row['fold_count']} folds")
            digests = table_digests(schedule)
            require(f"{row_label} embeds current Akita SIS table identities",
                    bool(digests) and set(digests) <= CURRENT_SIS_TABLE_DIGESTS,
                    f"{len(digests)} matrix table references")
            schedule_folds += expected_folds
            occurrence_total += len(occurrences)
            valid_occurrence_count = 0
            try:
                cwss = conditional_cwss_terms(schedule)
                expected_cwss_terms = sum(
                    len(fold["params"]["groups"]["entries"])
                    for fold in [schedule["root"], *schedule["recursive_folds"]]
                ) + 1
            except (KeyError, TypeError, ValueError) as error:
                require(f"{row_label} conditional CWSS geometry is derivable", False,
                        str(error))
            else:
                require(f"{row_label} has one CWSS term per fold group plus terminal",
                        len(cwss) == expected_cwss_terms,
                        f"{len(cwss)} terms")
                _, _, row_cwss = conditional_cwss_summary(schedule)
                if not quiet:
                    print(f"    {row_label}: schedule sum of per-fold conditional CWSS "
                          f"upper bounds plus terminal = {row_cwss.numerator}/"
                          f"{row_cwss.denominator} ({len(cwss)} terms)")
            for index, occurrence in enumerate(occurrences):
                result = validate_occurrence(occurrence, row_label, index)
                if result is not None:
                    role_counts[result[0]] += 1
                    bound_counts[result[1]] += 1
                    valid_occurrence_count += 1
            require(f"{row_label} all SIS occurrence descriptors are valid",
                    valid_occurrence_count == len(occurrences),
                    f"{valid_occurrence_count}/{len(occurrences)} valid")

        if not quiet:
            print(f"\n  {name}: {len(rows)} catalog rows, {schedule_folds} total fold levels")
            print(f"    exact SIS occurrence multiset: {occurrence_total}")
            print("    occurrences by role: " + ", ".join(
                f"{role}={role_counts[role]}" for role in
                ("Inner", "Outer", "Open", "Compression")))
            print("    occurrence bounds: " + ", ".join(
                f"{kind}={bound_counts[kind]}" for kind in ("linf", "l2_squared")))
    require("audit contains rows", any_rows)
    accepted = challenge_support_lower_bound(
        {"count_pm1": 31, "count_pm2": 0}, 128,
        {"security_route": {"L2": {"table_key": {"ring_dimension": 128}}}})
    require("D128 certified accepted support", 0 < D128_ACCEPTANCE_P0 <= 1
            and accepted == 502822610611069237129947393810277364817,
            f"ceil(C128 * p0) = {accepted}; log2 = {math.log2(accepted):.13f}")
    accepted = challenge_support_lower_bound(
        {"count_pm1": 31, "count_pm2": 11}, 64,
        {"security_route": {"L2": {"table_key": {"ring_dimension": 64}}}})
    require("D64 certified accepted support", 0 < D64_ACCEPTANCE_P0 <= 1
            and accepted == 355332842887957753779026572075354372186,
            f"ceil(N64 * p0) = {accepted}; log2 = {math.log2(accepted):.13f}")
    for degree, batches in ((512, FALCON_512_BATCHES),
                            (1024, FALCON_1024_BATCHES)):
        try:
            matched = production_rows(report, degree)
        except (KeyError, TypeError, ValueError) as error:
            require(f"Falcon-{degree} production group rows have canonical shapes and bases",
                    False, str(error))
        else:
            require(f"Falcon-{degree} production group rows have canonical shapes and bases",
                    tuple(sorted(matched)) == tuple(sorted(batches)),
                    f"{len(matched)} rows")
            reference_records = 1 << (15 if degree == 512 else 14)
            folds, terminal, _ = conditional_cwss_summary(
                schedule_from_row(matched[reference_records]))
            margins = [sufficient_margin_bits(term) for term in [*folds, terminal]]
            require(f"Falcon-{degree} sufficient whole-fold margins at reference shape",
                    margins == [8, 5, 4, 2, 3, 3, 2], str(margins))
    if not quiet:
        print("\nMSIS ledger form: sum over the reported occurrence multiset I of Adv_MSIS(I).")
        print("The ADPS16 table gate is a modeled attack-cost threshold, not an advantage or probability.")
        print("No numerical MSIS probability is inferred from that threshold.")
        print("CWSS uses certified accepted support for L2 routes; bounds are conditional.")
        print("Whole-fold margins are hypothetical; indexed sparse atoms need zero extra bits.")
        print("Indexed-coordinate accounting is the deployed query convention (paper Lemma C.3);")
        print("the assembled classical-ROM bound is the paper's Theorem 5.4.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(validate_report())
