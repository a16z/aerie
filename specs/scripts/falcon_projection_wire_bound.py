#!/usr/bin/env python3
"""Size bounds of the projection output Rice section (specs/falcon.md Section 10.4).

Standard library only; exits non-zero on any failed check.

The section is two parameter bytes and 187 zigzag Rice codes (156 norm codes
of ctr_p(y_j), 31 residual codes of ctr_p(z_j)/q), each leg at its smallest
length-minimizing parameter k in [0, 63], zero-padded to a byte boundary.
"""

import sys
from math import isqrt

P = 2**64 - 59
Q = 12_289
NORM_ROWS, RESIDUAL_WIDTH, MAX_K = 156, 31, 63
CAPS = {
    "Falcon-512": 5_092_300_183_991_442_303_232,
    "Falcon-1024": 10_513_136_047_130_310_946_696,
}
HALF = (P - 1) // 2
U = HALF // Q  # largest |ctr_p(z)/q| whose multiple of q is centered

failures = 0


def check(label, ok):
    global failures
    print(("PASS " if ok else "FAIL ") + label)
    failures += not ok


def section_bytes(bits):
    return 2 + -(-bits // 8)


def per_code_worst(max_code, rows):
    """min_k rows * (k + 1 + floor(max_code / 2^k)): every code at max_code."""
    return min(rows * (k + 1 + (max_code >> k)) for k in range(MAX_K + 1))


# Any decodable section: norm codes up to m = p - 1, residual codes up to 2U.
NORM_ANY = per_code_worst(P - 1, NORM_ROWS)
RESIDUAL_ANY = per_code_worst(2 * U, RESIDUAL_WIDTH)
check("U = floor((p-1)/(2q)) = 750,538,858,886,384 < 2^49.42",
      U == 750_538_858_886_384 and U < 2**49.42)
check("norm codes cost at most 65 bits, residual codes 52 bits",
      (NORM_ANY, RESIDUAL_ANY) == (NORM_ROWS * 65, RESIDUAL_WIDTH * 52))
check("any decodable section is at most 1,471 bytes (MAX_ENCODED_BYTES) < 1,496",
      section_bytes(NORM_ANY + RESIDUAL_ANY) == 1_471 < 8 * 187 == 1_496)

# Accepted proofs: sum ctr(y)^2 <= C_S and Cauchy-Schwarz give
# sum m_j <= 2 isqrt(156 C_S); floor sums are at most the floor of the sum.
for (name, cap), (norm_bits, bound) in zip(
        CAPS.items(), [(5_511, 893), (5_602, 904)]):
    total = 2 * isqrt(NORM_ROWS * cap)
    best = min((NORM_ROWS * (k + 1) + (total >> k), k) for k in range(MAX_K + 1))
    check(f"{name}: accepted norm leg at most {norm_bits:,} bits at k_y = 33",
          best == (norm_bits, 33))
    check(f"{name}: accepted section at most {bound} bytes",
          section_bytes(best[0] + RESIDUAL_ANY) == bound)

# A static per-leg width at the accepted tails (38 and 51 bits signed).
check("static accepted widths 38 and 51 bits cost 939 bytes",
      -(-(NORM_ROWS * 38 + RESIDUAL_WIDTH * 51) // 8) == 939
      and 2 * isqrt(CAPS["Falcon-1024"]) < 2**38 and 2 * U < 2**51)

sys.exit(1 if failures else 0)
