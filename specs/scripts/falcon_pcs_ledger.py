#!/usr/bin/env python3
"""Shared current-catalog reporting for the Falcon composition ledgers."""

from collections import Counter
from fractions import Fraction

import falcon_msis_audit as audit
from falcon_transcript_grinding import checked_rows


class CatalogAuditError(RuntimeError):
    pass


def terminal_digest_loss(q_h):
    """Exact two-digest loss for a terminal-oracle query budget."""
    if type(q_h) is not int or q_h < 0:
        raise ValueError("terminal digest query budget must be a nonnegative integer")
    return Fraction(q_h * (q_h + 3), 2**257)


def commitment_coefficient_cost(group):
    """Mirror Aerie's exact profile-to-CompressionChainPlan source geometry.

    Akita's pinned Q64 ladder is [32, 16] with a 128-byte terminal. The source
    length uses the complete sliced B image, not polynomial_count times rank.
    """
    profile = group["profile"]
    outer = profile["outer"]["matrix"]
    modulus = outer["sis_table_key"]["modulus_profile"]
    if modulus != "Q64Offset59":
        raise ValueError(f"unexpected commitment modulus profile {modulus!r}")
    slices = profile["outer_slice_count"]
    rank = outer["output_rank"]
    ring_dimension = outer["sis_table_key"]["ring_dimension"]
    source_coefficients = slices * rank * ring_dimension
    source_bytes = source_coefficients * 8
    if source_coefficients <= 0 or source_bytes > 8 * 1024:
        raise ValueError("commitment source image is outside Akita's Q64 compression domain")

    input_coefficients = source_coefficients
    terminal_coefficients = None
    for compression_dimension in (32, 16):
        input_width = (input_coefficients * 64 + compression_dimension - 1) \
            // compression_dimension
        if input_width <= 0:
            raise ValueError("empty input to Akita compression map")
        input_coefficients = compression_dimension
        if input_coefficients * 8 == 128:
            terminal_coefficients = input_coefficients
    if terminal_coefficients is None:
        raise ValueError("Akita Q64 compression ladder did not reach 128 bytes")
    return {
        "slices": slices,
        "outer_rank": rank,
        "outer_ring_dimension": ring_dimension,
        "source_coefficients": source_coefficients,
        "source_bytes": source_bytes,
        "terminal_coefficients": terminal_coefficients,
        "terminal_bytes": terminal_coefficients * 8,
    }


def current_metrics(degree, batch_logs):
    """Load, audit, and select the production shapes charged by a ledger."""
    try:
        report = audit.load_catalog_report()
    except (OSError, ValueError) as error:
        raise CatalogAuditError(str(error)) from error
    if audit.validate_report(report, quiet=True) != 0:
        raise CatalogAuditError("native four-group catalog audit did not pass")
    try:
        return audit.production_ledger_rows(report, degree, batch_logs)
    except (KeyError, TypeError, ValueError) as error:
        raise CatalogAuditError(str(error)) from error


def occurrence_summary(counts):
    roles = ("Inner", "Outer", "Open", "Compression")
    kinds = ("linf", "l2_squared")
    return ", ".join(f"{role}/{kind}={counts[(role, kind)]}"
                      for role in roles for kind in kinds)


def print_current_metrics(degree, batch_logs):
    """Check terminal examples and print current symbolic and conditional data."""
    rows = current_metrics(degree, batch_logs)
    plans = checked_rows()
    examples = (
        (0, Fraction(0), "0"),
        (1, Fraction(1, 2**255), "2^-255"),
        (2, Fraction(5, 2**256), "5*2^-256"),
        (2**32, Fraction(1, 2**193) + Fraction(3, 2**225),
         "2^-193 + 3*2^-225"),
        (2**64, Fraction(1, 2**129) + Fraction(3, 2**193),
         "2^-129 + 3*2^-193"),
    )
    print("  Terminal digest loss: epsilon_terminal(Q_H) = "
          "(binom(Q_H,2)+2*Q_H)/2^256 = Q_H*(Q_H+3)/2^257")
    print("  Q_H counts terminal-digest oracle queries in the reduction experiment, "
          "including final verification; Q counts transcript/sampler queries separately.")
    for q_h, expected, expression in examples:
        if terminal_digest_loss(q_h) != expected:
            raise CatalogAuditError(f"terminal digest example failed at Q_H={q_h}")
        print(f"    [ok] exact terminal example Q_H={q_h}: {expression}")
    for bad in (-1, 1.5, "2", True, None):
        try:
            terminal_digest_loss(bad)
        except ValueError:
            continue
        raise CatalogAuditError(f"invalid terminal digest budget accepted: {bad!r}")
    for plan in plans:
        c = plan["R_field"] + plan["sparse_coordinates"]
        known = Fraction(2**64 + c, 2**128) + terminal_digest_loss(2**64)
        if known != (Fraction(1, 2**64) + Fraction(2 * c + 1, 2**129)
                     + Fraction(3, 2**193)):
            raise CatalogAuditError("assigned-budget partial example failed")
    print(f"    [ok] exact assigned-budget partial examples for {len(plans)} shapes; "
          f"maximum C={max(p['R_field'] + p['sparse_coordinates'] for p in plans)}")
    print("  Full symbolic classical-ROM knowledge bound: "
          "(Q+R_field+R_coord)*2^-128 + epsilon_terminal(Q_H) + delta_X + "
          "Delta_setup + epsilon_bind + delta_samp + sum_I Adv_MSIS,I^exp(T_I)")
    print("  Known partial numerical examples assign Q=Q_H=2^64; "
          "all other terms remain to be added. This assignment does not identify the budgets.")
    work = {row["n_sig"]: row for row in plans if row["degree"] == degree}
    print(f"Falcon-{degree} native four-group PCS rows (Akita planner estimates)")
    for entry in rows:
        row = entry["row"]
        estimate = row["planner_estimate"]
        print(f"  N_sig={entry['records']}: payload="
              f"{estimate['estimated_proof_payload_bytes']} B, setup="
              f"{estimate['estimated_num_setup_field_elements']} field elements, "
              f"{row['fold_count']} folds")
        plan = work.get(entry["records"])
        if plan is None:
            raise CatalogAuditError(
                f"no Section 10.1 work plan for Falcon-{degree} N_sig={entry['records']}")
        c = plan["R_field"] + plan["sparse_coordinates"]
        print(f"    verifier completions: R_field={plan['R_field']}, "
              f"sparse_coordinates (R_coord)={plan['sparse_coordinates']}, C={c}")
        print(f"    known partial at assigned Q=Q_H=2^64: "
              f"2^-64 + {2 * c + 1}*2^-129 + 3*2^-193")
        print(f"    PIOP native work: {plan['pio_nonce_fields']} nonce fields, "
              f"at most {plan['pio_nonce_worst_bytes']} B, "
              f"expected {plan['pio_nonce_expected_bytes']:.2f} B "
              "conditional on successful geometric searches")
        print("    PCS native nonce size delta: measured from built proofs; fold work=0")
        print(f"    PIOP work: {plan['piop_bits']}")
        value = entry["cwss"]
        largest_fold = max(entry["fold_cwss"])
        terminal = entry["terminal_cwss"]
        print(f"    largest per-fold conditional CWSS: "
              f"{largest_fold.numerator}/{largest_fold.denominator}; "
              f"terminal: {terminal.numerator}/{terminal.denominator}")
        print(f"    schedule sum of all per-fold CWSS terms plus terminal: "
              f"{value.numerator}/{value.denominator} "
              f"({entry['cwss_term_count']} exact terms)")
        print(f"    exact SIS matrix occurrences: {entry['occurrence_total']} "
              f"({occurrence_summary(entry['occurrence_counts'])})")
        commitments = [
            (group["role"], commitment_coefficient_cost(group))
            for group in audit.group_profiles_from_row(row)
        ]
        print("    compressed group commitment coefficients: " + ", ".join(
            f"{role}: {cost['slices']} slices × rank {cost['outer_rank']} × "
            f"D={cost['outer_ring_dimension']} = {cost['source_coefficients']} source coeffs/"
            f"{cost['source_bytes']} B → {cost['terminal_coefficients']} terminal coeffs/"
            f"{cost['terminal_bytes']} B" for role, cost in commitments))
    print("  MSIS ledger: sum Adv_MSIS(I) over each row's exact SIS occurrence multiset.")
    print("  V1 work uses the complete-tree debits in falcon.md Section 10.1; the verifier "
          "enforces the PIOP plan and Akita enforces its native PCS plan.")
