#!/usr/bin/env python3
"""Numerical checks for the Falcon-1024 profile in ``specs/falcon.md``.

The script checks Falcon byte and norm constants, the exact encoding budget,
projection geometry and certificate, integer-lifting bounds, four Akita group
shapes, and the three PIOP stages. Arithmetic is exact unless a check is
labeled FLOAT. The projection certificate is read-only here; its separate
check script recomputes it.

Citation keys:
  [spec §N]   = specs/falcon.md, section N
  [cert]      = projection certificate, role "chosen" at the stated n
  [Falcon]    = Falcon specification, Table 3.3

Run from the repository root with any Python 3 (stdlib only):
  python3 specs/scripts/falcon_1024_parameter_checks.py
"""

import json
import math
from fractions import Fraction
from pathlib import Path

FAILURES = []


def check(name, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def isqrt_ceil(x):
    r = math.isqrt(x)
    return r if r * r == x else r + 1


# ---------------------------------------------------------------- constants
P = 2**64 - 59                      # base field modulus [spec §3]
Q = 12289                           # Falcon modulus, both degrees
HT = (Q - 1) // 2                   # centered Falcon coefficient bound [spec §2]
S2 = 2047                           # compressed coefficient limit, both degrees
MAX_NSTAR = 2**16                   # supported batch maximum [spec §5]

# Falcon-512 profile values [Falcon], [spec §2]
N512 = 512
B512 = 34_034_726
SIG512, PK512, SALT = 666, 897, 40

# Falcon-1024 profile [Falcon]
N = 1024
B = 70_265_242
SIG, PK = 1280, 897 * 2 - 1
check("Falcon-1024 public key is 1,793 bytes", PK == 1793,
      "1 header byte + 1024 * 14 bits")
check("Falcon-1024 public-key body is 14 bits per coefficient",
      (PK - 1) * 8 == N * 14)

# pinned projection certificate [cert]
CERT_PATH = Path(__file__).resolve().parent / "falcon_projection_certificate.json"
CERT = json.loads(CERT_PATH.read_text())


def cert_config(n, role="chosen"):
    rows = [c for c in CERT["configurations"] if c["n"] == n and c["role"] == role]
    if len(rows) != 1:
        raise SystemExit(f"certificate has {len(rows)} {role} rows for n = {n}")
    return rows[0]


CERT_1024 = cert_config(N)
CERT_512 = cert_config(N512)
# Only keys shared by every pinned schema version are read; the check names
# inside "checks" differ between versions, so they are required to be all true
# without being named.
check("certificate schema is aerie/falcon/projection-certificate/*",
      CERT["schema"].startswith("aerie/falcon/projection-certificate/"),
      CERT["schema"])
check("certificate: chosen n = 1024 and n = 512 rows certified, every check true",
      all(c["certified"] and all(c["checks"].values()) for c in (CERT_1024, CERT_512)),
      f"{len(CERT_1024['checks'])} checks per row")

# --------------------------------------------- 1. exact norm bound, spec §5
S1_NORM_MAX = math.isqrt(B)
check("floor sqrt(B_1024) = 8,382", S1_NORM_MAX == 8_382,
      f"{S1_NORM_MAX}^2 = {S1_NORM_MAX**2} <= {B} < {(S1_NORM_MAX + 1)**2}")
check("Falcon-512 reference: floor sqrt(B_512) = 5,833 [spec §5]",
      math.isqrt(B512) == 5_833)
S1_CENTERED_MAX = min(S1_NORM_MAX, HT)
SIGNED14_MAX = 2**13 - 1
check("centered S1 bound 6,144 fits signed width 14 [spec §5]",
      S1_CENTERED_MAX <= SIGNED14_MAX)
check("Falcon-512 S1 norm bound is 5,833; Falcon-1024 is capped at 6,144",
      min(math.isqrt(B512), HT) == 5_833 and S1_CENTERED_MAX == 6_144)

# ------------------------------------------- 2. encodability budget, spec §4
BODY_BITS = (SIG - 1 - SALT) * 8
FIXED_BITS = N * (1 + 7 + 1)
BUDGET = BODY_BITS - FIXED_BITS
check("padded body has 9,912 bits", BODY_BITS == 9_912)
check("fixed cost is 9,216 bits", FIXED_BITS == 9_216)
check("unary high-part budget is 696 bits", BUDGET == 696)
check("Falcon-512 reference budget is 392 bits [spec §4]",
      (SIG512 - 1 - SALT) * 8 - N512 * 9 == 392)
check("a2 in [0,15] still covers |s2| <= 2,047", 15 * 128 + 127 == S2)
# three balanced budget digits (8, 8, 16): u0, u1 in [-4,3], u2 in [-8,7]
BUDGET_CONST = 4 + 8 * 4 + 64 * 8
check("budget recomposition constant is 548", BUDGET_CONST == 548)
lo = -4 - 8 * 4 - 64 * 8 + BUDGET_CONST
hi = 3 + 8 * 3 + 64 * 7 + BUDGET_CONST
check("budget digits (8,8,16) cover [0,1023] and hence [0,696]",
      lo == 0 and hi == 1023 and hi >= BUDGET)
check("three radix-8 digits would NOT cover the 1024 budget",
      511 < BUDGET, "Falcon-512 layout [spec §4] tops out at 511")
check("budget identity lifts: 1024*15 + 1023 < p", N * 15 + hi < P,
      f"lhs = {N * 15 + hi}")
S2_NORM2 = N * S2**2
check("||s2||_2^2 <= 1024 * 2047^2 = 4,290,774,016", S2_NORM2 == 4_290_774_016)

# ----------------------------- 3. domains and projection geometry, spec §6.2
NU, NU512 = 10, 9                   # log2 n


def k_split(nu):
    """Low record bits in the first projection block [spec §6.2]."""
    return 12 - (nu + 1)


K1024, K512 = k_split(NU), k_split(NU512)
check("tensor split k = 12 - (10 + 1) = 1 at degree 1024 [spec §6.2]",
      K1024 == 1)
check("Falcon-512 reference: k = 2 [spec §6.2]", K512 == 2)
check("geometry (ell >= k) excludes only N* = 1 at degree 1024 [spec §6.2]",
      [ell for ell in range(17) if ell < K1024] == [0])


def domains(nu, ell):
    """(coefficient, slack, row, norm, projection) arities [spec §6.2]."""
    return (nu + ell, 2 + ell, ell, nu + 1 + ell, 2 + nu + ell)


check("domain arities at ell are (10+ell, 2+ell, ell, 11+ell, 12+ell)",
      all(domains(NU, ell) == (10 + ell, 2 + ell, ell, 11 + ell, 12 + ell)
          for ell in range(17)))
check("projection-link arity d = 12 + ell <= 28 for N* <= 2^16",
      max(domains(NU, ell)[4] for ell in range(17)) == 28)
check("norm source storage bound 2,048 N* <= 2^27 [spec §6.2]",
      2 * N * MAX_NSTAR == 2**27)

# Flat indices in the projection geometry cover their ranges at a small batch.
NSTAR_SMALL = 4
U = sorted(j + N * a + 2 * N * i for i in range(NSTAR_SMALL)
           for a in range(2) for j in range(N))
X = sorted(j + N * (a + 2 * b) + 4 * N * i for i in range(NSTAR_SMALL)
           for b in range(2) for a in range(2) for j in range(N))
check("flat indices u and x of (6.1) are bijections at N* = 4 [spec §6.2]",
      U == list(range(2 * N * NSTAR_SMALL)) and X == list(range(4 * N * NSTAR_SMALL)))

NORM_WIDTH, NORM_GROUPS, NORM_OUT, RES_WIDTH = 43, 4, 156, 31


def norm_radix_bits(nu, ell):
    """Norm-leg block bits [spec §6.2]."""
    rest = ell - k_split(nu)
    r2 = min(4, rest)
    r3 = min(4, rest - r2)
    r4 = min(4, rest - r2 - r3)
    return [nu + 1 + k_split(nu), r2, r3, r4, rest - r2 - r3 - r4]


def residual_radix_bits(nu, ell):
    """Residual-leg block bits [spec §6.2]."""
    rest = ell - k_split(nu)
    return [nu + k_split(nu), rest // 2, rest - rest // 2]


def core_coefficients(norm_bits, res_bits):
    rn = [2**b for b in norm_bits]
    rr = [2**b for b in res_bits]
    norm = NORM_GROUPS * (NORM_WIDTH * rn[0]
                          + NORM_WIDTH**2 * (rn[1] + rn[2] + rn[3])) \
        + NORM_OUT * NORM_GROUPS * NORM_WIDTH * rn[4]
    res = RES_WIDTH * rr[0] + RES_WIDTH**2 * (rr[1] + rr[2])
    return norm, res


check("first blocks are 12 bits (j,a,i_0) and 11 bits (j,i_0) at degree 1024",
      norm_radix_bits(NU, 16)[0] == 12 and residual_radix_bits(NU, 16)[0] == 11)
check("norm and residual blocks tile 11+ell and 10+ell bits for every ell",
      all(sum(norm_radix_bits(NU, ell)) == NU + 1 + ell
          and sum(residual_radix_bits(NU, ell)) == NU + ell
          and min(norm_radix_bits(NU, ell)) >= 0
          for ell in range(K1024, 17)))
RN = [2**b for b in norm_radix_bits(NU, 16)]
RR = [2**b for b in residual_radix_bits(NU, 16)]
check("radices at N* = 2^16: (4096,16,16,16,8) and (2048,128,256) [spec §6.3]",
      RN == [4096, 16, 16, 16, 8] and RR == [2048, 128, 256])
check("radices at N* = 2^16 match the certificate [cert]",
      RN == CERT_1024["norm_geometry"]["radices"]
      and RR == CERT_1024["residual_geometry"]["radices"])
check("Falcon-512 reference radices (4096,16,16,16,4), (2048,128,128) [spec §6.3]",
      [2**b for b in norm_radix_bits(NU512, 16)] == [4096, 16, 16, 16, 4]
      == CERT_512["norm_geometry"]["radices"]
      and [2**b for b in residual_radix_bits(NU512, 16)] == [2048, 128, 128]
      == CERT_512["residual_geometry"]["radices"])
CN, CR = core_coefficients(norm_radix_bits(NU, 16), residual_radix_bits(NU, 16))
check("core coefficients at N* = 2^16: 1,274,176 + 432,512 = 1,706,688 [spec §6.3]",
      (CN, CR, CN + CR) == (1_274_176, 432_512, 1_706_688))
check("core coefficients match the certificate [cert]",
      CN == CERT_1024["norm_geometry"]["coefficients"]
      and CR == CERT_1024["residual_geometry"]["coefficients"]
      and CN + CR == CERT_1024["total_coefficients"])
CN5, CR5 = core_coefficients(norm_radix_bits(NU512, 16), residual_radix_bits(NU512, 16))
check("Falcon-512 reference projection coefficients 1,166,848 + 309,504 [spec §6.3]",
      (CN5, CR5) == (1_166_848, 309_504) == (
          CERT_512["norm_geometry"]["coefficients"],
          CERT_512["residual_geometry"]["coefficients"]))
check("outputs 156 + 31 = 187, 1,496 LE64 limb bytes at both degrees [spec §6.3]",
      NORM_OUT + RES_WIDTH == 187 == CERT_1024["outputs"] == CERT_512["outputs"]
      and 8 * 187 == 1_496 == CERT_1024["clear_bytes"])
FIRST_CHANNELS = NORM_GROUPS * NORM_WIDTH + RES_WIDTH
check("203 first-core channels (4 * 43 norm + 31 residual) [spec §6.3]",
      FIRST_CHANNELS == 203)
PHASE1 = FIRST_CHANNELS * 2**(16 - K1024)
check("rest-first phase-1 tables have 203 * 2^15 = 6,651,904 entries [spec §6.2]",
      PHASE1 == 6_651_904)
EXT_N, EXT_R = CERT_1024["norm_geometry"]["ext"], CERT_1024["residual_geometry"]["ext"]
MERGED = EXT_N + EXT_R - (2**11 - 1) + 187
check("merged projection evaluation uses 15,530 products [spec §6.3]",
      (EXT_N, EXT_R, MERGED) == (13_039, 4_351, 15_530)
      and MERGED == CERT_1024["merged_ext"])

# --------------------------------- 4. projection predicate and exact lift, spec §9
C_S = 10_513_136_047_130_310_946_696
check("C_S equals the certificate's order-46 cap [spec §6.3, cert]",
      C_S == CERT_1024["C_S"] and CERT_1024["cap_order"] == 46)
check("2^73 < C_S < 2^74, so the u128 accumulator rule applies [spec §6.3]",
      2**73 < C_S < 2**74, f"log2 C_S = {math.log2(C_S):.2f}")
check("honest energy ceiling H_S = 2^16 B [spec §6.3, cert]",
      2**16 * B == CERT_1024["H_S"])
check("same rho at both degrees [spec §6.3, cert]",
      CERT_1024["rho"] == CERT_512["rho"])
QS_LOG2 = CERT_1024["Q_S_log2"]
LIFT_LHS = 2**QS_LOG2 + N * S2**2
check("norm lift: Q_S + 1024 * 2047^2 < p (FLOAT) [spec §9]",
      LIFT_LHS < P,
      f"log2 Q_S = {QS_LOG2:.4f}, ratio to p = {LIFT_LHS / P:.3f}")
SLACK = math.log2(P - N * S2**2) - QS_LOG2
check("lift slack log2(p - 1024 * 2047^2) - log2 Q_S agrees with the "
      "certificate (FLOAT) [spec §9]",
      SLACK > 0 and abs(SLACK - CERT_1024["lift_slack_bits"]) < 1e-9,
      f"slack = {SLACK:.2f} bits")

# ------------------------------------------- 5. residual leg, spec §9
DELTA_POST = S1_NORM_MAX + isqrt_ceil(HT**2 * N * B) + HT
check("projection residual bound matches the pinned certificate [spec §9]",
      DELTA_POST == 1_648_068_606 == CERT_1024["L_W"], f"Δ = {DELTA_POST}")
check("residual bound is below p/2, so its centered lift is unique [spec §9]",
      2 * DELTA_POST < P)
check("extracted residual quotient is at most 134,109 [spec §9]",
      DELTA_POST // Q == 134_109)
H_DELTA = 2**16 * N * DELTA_POST**2
check("certificate residual energy cap matches the lifted bound [spec §9, cert]",
      H_DELTA == CERT_1024["H_W"], f"log2 cap = {math.log2(H_DELTA):.2f}")
check("projection wrap carrier 6,144 sqrt(nB) is in [2^30, 2^31) [spec §9]",
      2**30 <= isqrt_ceil(HT**2 * N * B) < 2**31)
RES = CERT_1024["residual"]
check("shared residual profile b = 12, widths (31,31,31) [spec §6.3, cert]",
      RES["b"] == 12 and RES["widths"] == [31, 31, 31]
      and CERT_512["residual"]["b"] == 12)
check("annihilation term is exactly 3 * 25^-31 [spec §9, cert]",
      Fraction(RES["kill"]) == Fraction(3, 25**31),
      f"2^-{RES['kill_bits']:.2f}")
check("wrap term <= 2^-153.75 at best order 60 (FLOAT) [spec §9, cert]",
      RES["wrap_bits"] >= 153.74 and RES["wrap_order"] == 60,
      f"2^-{RES['wrap_bits']:.2f}")
REJ = cert_config(N, "rejected")
check("residual b = 17, widths (28,28,28) fails at degree 1024 (FLOAT) [spec §6.3]",
      not REJ["certified"] and RES["b"] != REJ["residual"]["b"]
      and REJ["residual"]["wrap_bits"] < 46,
      f"wrap 2^-{REJ['residual']['wrap_bits']:.1f}")
TOTAL_BITS = CERT_1024["total_err_bits"]
LEGS = 2**-CERT_1024["norm_err_bits"] + 2**-RES["kill_bits"] + 2**-RES["wrap_bits"]
check("total projection term is the norm + annihilation + wrap sum, "
      "at least 2^-136 (FLOAT) [spec §9.5, cert]",
      TOTAL_BITS >= 136 and abs(-math.log2(LEGS) - TOTAL_BITS) < 1e-6,
      f"2^-{TOTAL_BITS:.3f}")

# ------------------------------------------------ 6. Akita groups and PIOP stages
GROUPS = {
    "S1": (1, 14, 6144),
    "e": (1, 14, 8191),
    "digits": (4, 4, 8),
    "budget": (3, 4, 8),
}
check("four same-shape Akita groups carry 1, 1, 4, and 3 polynomials [spec §5]",
      len(GROUPS) == 4 and [g[0] for g in GROUPS.values()] == [1, 1, 4, 3])
check("signed width 14 contains every S1 and slack value [spec §5]",
      GROUPS["S1"][2] <= 2**13 - 1 and GROUPS["e"][2] <= 2**13 - 1)
check("signed width 4 contains the digit alphabets [spec §5]",
      all(g[2] <= 2**3 for k, g in GROUPS.items() if k in {"digits", "budget"}))
check("Falcon-512 and Falcon-1024 share the group widths [spec §5]",
      (14, 14, 4, 4) == (GROUPS["S1"][1], GROUPS["e"][1],
                         GROUPS["digits"][1], GROUPS["budget"][1]))

DIGIT_MAGNITUDES = {a0 + 8 * a1 + 128 * a2 + 1092
                    for a0 in range(-4, 4) for a1 in range(-8, 8)
                    for a2 in range(-8, 8)}
check("3,4,4-bit digits reconstruct exactly [0,2047] [spec §3]",
      DIGIT_MAGNITUDES == set(range(2048)))
IMG8 = sorted({z * (z + 1) for z in range(-4, 4)})
IMG16 = sorted({z * (z + 1) for z in range(-8, 8)})
check("base-8 range image is {0,2,6,12} [spec §7]", IMG8 == [0, 2, 6, 12])
check("base-16 range image is {0,2,6,12,20,30,42,56} [spec §7]",
      IMG16 == [0, 2, 6, 12, 20, 30, 42, 56])
PREIMG = [z for z in range(-100, 101) if z * (z + 1) in IMG16]
check("z(z+1) in the base-16 image characterizes [-8,7] over the integers",
      PREIMG == list(range(-8, 8)))
STAGES = {
    "stage 1": {"basis16_product": 3, "projection_link": 2},
    "stage 2": {"range_leaf": 5},
    "stage 3": {"range_source_link": 3, "sign_check": 3,
                "norm_equality": 3, "encoding_budget": 2,
                "ring_functional": 3, "s1_reduction": 2,
                "slack_reduction": 2},
}
STAGE_NAMES = {name: tuple(instances) for name, instances in STAGES.items()}
STAGE_DEGREES = tuple(max(instances.values()) for instances in STAGES.values())
check("three stages contain the required batched checks [spec §7]",
      STAGE_NAMES["stage 1"] == ("basis16_product", "projection_link")
      and STAGE_NAMES["stage 2"] == ("range_leaf",)
      and len(STAGE_NAMES["stage 3"]) == 7)
check("maximum stage degrees are 3, 5, and 3 [spec §7]",
      STAGE_DEGREES == (3, 5, 3))
TERMINAL_OPENINGS = sum(group[0] for group in GROUPS.values())
check("terminal PCS evaluations equal the opened group polynomial count [spec §7]",
      TERMINAL_OPENINGS == 9)
check("wire header 16 + 2 + 1 + 8 = 27 bytes [spec §10]", 16 + 2 + 1 + 8 == 27)

# ------------------------------------------------ 8. HashToPoint output size
# rejection sampling on 16-bit chunks with threshold 5q = 61,445
exp_bytes_512 = Fraction(2 * N512 * 65536, 5 * Q)
exp_bytes_1024 = Fraction(2 * N * 65536, 5 * Q)
check("expected SHAKE256 output roughly doubles (1,092 -> 2,185 bytes) (FLOAT)",
      1091 < float(exp_bytes_512) < 1093 and 2183 < float(exp_bytes_1024) < 2186,
      f"{float(exp_bytes_512):.1f} -> {float(exp_bytes_1024):.1f}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all checks passed")
