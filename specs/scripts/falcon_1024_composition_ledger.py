#!/usr/bin/env python3
"""Falcon-1024 composition ledger for N = 2^14 through 2^16.

This script checks the three-stage PIOP numerator, the Falcon-1024 reference
profile bound, the pinned clear-projection certificate, and the current native
four-group PCS catalog report. It reports exact conditional CWSS and MSIS
occurrence counts, plus the normative indexed-coordinate field work and nonce-size
allocation. MSIS remains an advantage term at the theorem's reduction time.
Exact terminal-digest and assigned-budget partial examples are checked; the
full knowledge bound retains separate query budgets and symbolic terms.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

from falcon_pcs_ledger import CatalogAuditError, print_current_metrics

N_DEG = 1024
LOG_N = 10
P = 2**64 - 59
K_SIZE = P * P
BATCH_LOGS = (14, 15, 16)
EXPECTED = {14: 441, 15: 461, 16: 481}
REFERENCE_LOG = 14
FAILURES = []


def lg(x):
    if isinstance(x, Fraction):
        return math.log2(x.numerator) - math.log2(x.denominator)
    return math.log2(x)


def require(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def pow2_neg_upper(bits, margin=Fraction(1, 10**9)):
    """Return a rational upper bound for the certificate's reported error."""
    b = Fraction(bits) - margin
    whole = math.floor(b)
    frac = b - whole
    mantissa = math.ceil(2.0 ** (-float(frac)) * 2**40) + 1
    return Fraction(mantissa, 2**40 * 2**whole)


def piop_terms(ell):
    """Return the three-stage PIOP numerator components over |K|."""
    rounds_per_stage = LOG_N + ell
    round_checks = 11 * rounds_per_stage
    basis16_product = 2 * rounds_per_stage + ell
    range_leaf = rounds_per_stage + 2 * ell
    stage3_zero_checks = rounds_per_stage + 2 * ell
    projection_weights = 8
    stage_combiners = 3
    total = (round_checks + basis16_product + range_leaf + stage3_zero_checks
             + projection_weights + stage_combiners)
    return {
        "rounds_per_stage": rounds_per_stage,
        "round_checks": round_checks,
        "basis16_product": basis16_product,
        "range_leaf": range_leaf,
        "stage3_zero_checks": stage3_zero_checks,
        "projection_weights": projection_weights,
        "stage_combiners": stage_combiners,
        "total": total,
    }


def projection_row(n):
    cert_path = Path(__file__).resolve().parent / "falcon_projection_certificate.json"
    cert = json.loads(cert_path.read_text())
    rows = [row for row in cert["configurations"]
            if row["n"] == n and row["role"] == "chosen"]
    if len(rows) != 1:
        raise SystemExit(f"certificate has {len(rows)} chosen rows for n = {n}")
    return cert, rows[0]


print("Falcon-1024 PIOP algebraic terms (three stages)")
for ell in BATCH_LOGS:
    terms = piop_terms(ell)
    print(f"\n  N = 2^{ell}, {terms['rounds_per_stage']} rounds per stage")
    print(f"    sumcheck rounds: {terms['round_checks']}/|K|")
    print(f"    basis16_product zero checks at tau: {terms['basis16_product']}/|K|")
    print(f"    range_leaf zero checks at r^(1): {terms['range_leaf']}/|K|")
    print(f"    stage 3 zero checks at r^(2): {terms['stage3_zero_checks']}/|K|")
    print("    monomial output batching and stage combiners: 11/|K|")
    print(f"    total: {terms['total']}/|K| = 2^{lg(Fraction(terms['total'], K_SIZE)):.2f}")
    require(f"PIOP numerator at N = 2^{ell} is {EXPECTED[ell]}",
            terms["total"] == EXPECTED[ell], str(terms["total"]))
    if ell == REFERENCE_LOG:
        require("Falcon-1024 reference profile is at most 460/|K|",
                terms["total"] <= 460, f"{terms['total']}/|K|")

print("\nFalcon-1024 clear-projection certificate")
CERT, CROW = projection_row(N_DEG)
print(f"  schema {CERT['schema']}")
require("certificate schema is aerie/falcon/projection-certificate/*",
        CERT["schema"].startswith("aerie/falcon/projection-certificate/"))
require("chosen n = 1024 row is certified and every check passes",
        CROW["certified"] and all(CROW["checks"].values()),
        f"{len(CROW['checks'])} checks")
kill = Fraction(CROW["residual"]["kill"])
wrap_bits = CROW["residual"]["wrap_bits"]
norm_bits = CROW["norm_err_bits"]
total_bits = CROW["total_err_bits"]
leg_sum = 2.0 ** -norm_bits + float(kill) + 2.0 ** -wrap_bits
print(f"  norm leg      2^-{norm_bits:.4f}")
print(f"  residual kill 2^-{lg(1 / kill):.4f} (exact {kill})")
print(f"  residual wrap 2^-{wrap_bits:.4f} (order {CROW['residual']['wrap_order']})")
print(f"  total         2^-{total_bits:.4f}")
require("residual kill is exactly 3/25^31", kill == Fraction(3, 25**31))
require("total equals norm + kill + wrap to 1e-6 bits",
        abs(-math.log2(leg_sum) - total_bits) < 1e-6,
        f"{(-math.log2(leg_sum)):.6f}")
EPS_PROJ = pow2_neg_upper(total_bits)
require("charged rational upper-bounds 2^-total",
        EPS_PROJ >= Fraction(2.0 ** -total_bits)
        and lg(EPS_PROJ) + total_bits < 1e-6,
        f"2^{lg(EPS_PROJ):.6f}")
print(f"  charged rational upper bound: 2^{lg(EPS_PROJ):.4f}")
print("  charged once per transcript; certificate covers N* <= 2^16")
print("\nFour-group PCS/catalog and conditional extraction terms")
try:
    print_current_metrics(1024, BATCH_LOGS)
except CatalogAuditError as error:
    require("current native four-group catalog report passes shape and audit checks",
            False, str(error))

if FAILURES:
    raise SystemExit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
print("all PIOP, projection, catalog, and terminal accounting checks passed")
