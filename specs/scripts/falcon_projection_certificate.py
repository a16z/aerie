#!/usr/bin/env python3
"""Clear-projection soundness certificate for Aerie Falcon-512 and Falcon-1024.

Certifies the clear structured-projection profile that replaces the grouped
JL layer (specs/falcon.md Section 6; specs/falcon-projection-theorem.md):

  norm leg      4 groups x 4 prefix cores of width 43, then one fresh
                156-row ternary terminal; one centered integer energy cap
                C_S from the order-46 honest moment bound.
  residual leg  b = 12, core widths (31, 31, 31), entries uniform on
                {-b..b}; 31 divisibility-by-q tests.

The norm leg is certified along the elementary chain of the companion
theorem (Section 3 there), with no imported lemma:

  Lemma A   half-propagation of a large coordinate through one ternary row
            (guard: 2 (tau_{i+1} - 1) < tau_i at every stage).
  Lemma B   a stage whose input entries are below tau_i reduces exactly
            modulo p (check: fan_in_i * (tau_i - 1) <= (p - 1)/2).
  Lemma C   E exp(-t |a^T X|^2/|X|^2) <= sup_{0<w<=1} h_t(w), with
            h_t(w) = E|cos(sqrt(t w/2) G)|^(2/w).  This script certifies
            sup_w h_t(w) <= U(t) in exact rational arithmetic and asserts
            K >= U(t) for the pair (t, K) it is given.
  Lemma D   the comparison coupling on the unreduced integer chain.
  Assembly  Pr <= 2^-156 + Pr[Z_156 * sum_g W_g < rho], every prefix zero
            atom boosted by eps_F = 1 - prod_i (1 - 2^-c_i).  Here rho is
            the largest grid threshold that fits; the theorem states the
            bound for the threshold eta = 2^15 <= rho.

For each degree n in {512, 1024} it checks, in exact rational arithmetic:

  threshold   eta <= rho, with eta = 2^15 and rho the certified grid point
  lift        Q_S + n*2047^2 < p, with Q_S = (C_S + 1)/eta
  accept      floor(sqrt(C_S)) < tau_5           (accepted y is "small")
  lemma_a_i   2 (tau_{i+1} - 1) < tau_i, i = 0..4 (Lemma A guards)
  lemma_b_i   fan_in_i (tau_i - 1) <= (p-1)/2, i = 0..4 (Lemma B)
  2b < q
  total       norm-leg error + residual wrap + residual kill <= 2^-136
  honest      2^-129 + residual wrap <= 2^-128.9

with tau_5 = 2 floor(sqrt(C_S)) + 1 and tau_i = 2^(5-i) tau_5 (so the
terminal input threshold is tau_4 = 2 tau_5), and fan_in_i the largest
fan-in of stage i over every supported N* (attained at N* = 2^16).

It also evaluates the rejected alternative b = 17, widths (28, 28, 28),
and asserts, as a documented negative check, that it passes at n = 512
and fails the 2^-136 total at n = 1024 (the residual wrap term).

Floating point appears only in *_bits / *_log2 / *_display fields. Every
accept/reject decision is an exact integer or Fraction comparison; real
constants (exp, pi, square roots) enter only through rational enclosures
with proven remainder bounds, rounded outward to dyadic rationals.

The script also certifies the Laplace supremum (laplace_sup_certificate),
with a LaplacePair type that binds t and K together, and the spec radix rule.

Usage (from the repository root, stdlib only):
  python3 specs/scripts/falcon_projection_certificate.py           report
  python3 specs/scripts/falcon_projection_certificate.py --check   CI gate
  python3 specs/scripts/falcon_projection_certificate.py --write   re-pin
"""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

PIN_PATH = Path(__file__).resolve().with_suffix(".json")
SCHEMA = "aerie/falcon/projection-certificate/v3"
FLOAT_TOLERANCE = 1e-9


class CertificateError(Exception):
    """An internal invariant of the certificate failed."""


def require(ok, message):
    # Explicit raise, not assert: the checks must survive `python3 -O`.
    if not ok:
        raise CertificateError(message)


# ----------------------------------------------------------------- constants
P = 2**64 - 59                      # base field modulus
HALF_P = (P - 1) // 2               # largest centered magnitude
Q = 12289                           # Falcon modulus
N_RECORDS = 2**16                   # batch maximum (records)
LOG_N_RECORDS = 16
EPS_BITS = 136                      # target projection soundness
ETA_PROJ = 2**15                    # norm comparison threshold (Theorem 3.1)
HONEST_BITS = 129                   # honest-abort budget for the cap search
HONEST_FLOOR = Fraction(1289, 10)   # asserted honest-rejection bits
HT = (Q - 1) // 2                   # centered |h|, |t| bound = 6144
S2_MAX = 2047                       # |S2*| bound from R1/R2/F alphabets
FALCON = {                          # squared norm bound B, floor(sqrt(B))
    512: {"B": 34_034_726, "cmax": 5_833},
    1024: {"B": 70_265_242, "cmax": 8_382},
}
# Residual coefficient bounds: falcon_1024_parameter_checks.py W_POST
# and specs/falcon.md Section 9 (integer lifting).
L_W_SPEC = {512: 811_061_395, 1024: 1_648_068_606}

# Norm leg shape (fixed for both degrees).
S_WIDTHS = (43, 43, 43, 43)
S_GROUPS = 4
S_OUT = 156
NORM_FIRST_BITS = 12                # first radix 2^12 (specs/falcon.md 6.2)
PREFIX_MAX_BITS = 4                 # cores 2..4 take at most 4 bits
CAP_MAX_ORDER = 200                 # source moment orders searched
WRAP_MAX_ORDER = 300                # residual wrap orders searched
MIN_LOG_PADDED_COEFFS = 12          # n N* >= 2^12 (N* >= 4096/n)

CHOSEN = (12, (31, 31, 31))
REJECTED = (17, (28, 28, 28))
# The four certified configurations.
CONFIGS = (
    (512, REJECTED, "rejected"),
    (512, CHOSEN, "chosen"),
    (1024, CHOSEN, "chosen"),
    (1024, REJECTED, "rejected"),
)

# The scalar Laplace pair of Lemma C. They are only ever used together,
# through a LaplacePair, whose construction certifies K >= sup_w h_t(w).
LAPLACE_T = Fraction(3)
LAPLACE_K = Fraction(21, 40)

# ---- grid parameters
A = Fraction(1, 1024)
R = Fraction(33, 32)
MASS_BITS = 192
D = 1 << MASS_BITS
WORK_BITS = 160
FP = 1 << WORK_BITS
DEGREE = 128

# ---- rigorous real enclosures
DYADIC_BITS = 320                   # outward rounding grid 2^-320
EXP_TAYLOR_TERMS = 40               # Taylor terms after halving to [0,1]
WALLIS_SHIFT = 2048                 # a_0(q) = a_0(q+M) * prod(...)
CELL_BITS = 40                      # q-cell endpoints on the 2^-40 grid
TAIL_EXPONENT = 64                  # Fourier sum truncated once t k^2/q >= 64
SUP_DECIMALS = 6                    # U(t) = h(1) rounded up at 1e-6
MAX_CELLS = 10_000
MAX_SMALL_W_Q0 = 1024


# ============================================= rigorous real enclosures
def round_down(x):
    x = Fraction(x)
    return Fraction(math.floor(x * 2**DYADIC_BITS), 2**DYADIC_BITS)


def round_up(x):
    x = Fraction(x)
    return Fraction(math.ceil(x * 2**DYADIC_BITS), 2**DYADIC_BITS)


@lru_cache(None)
def exp_bounds(y):
    """(lo, hi) with lo <= exp(y) <= hi, for rational y >= 0.

    Halve y into [0, 1], take the Taylor sum to EXP_TAYLOR_TERMS terms
    (a lower bound, all terms positive) and add the Lagrange-type tail
    z^(N+1)/(N+1)! * (N+2)/(N+2-z), then square back up with outward
    rounding.
    """
    y = Fraction(y)
    require(y >= 0, "exp_bounds needs y >= 0")
    halvings = 0
    z = y
    while z > 1:
        z /= 2
        halvings += 1
    term = total = Fraction(1)
    for j in range(1, EXP_TAYLOR_TERMS + 1):
        term = term * z / j
        total += term
    n = EXP_TAYLOR_TERMS
    tail = term * z / (n + 1) * Fraction(n + 2) / (n + 2 - z)
    lo, hi = round_down(total), round_up(total + tail)
    for _ in range(halvings):
        lo, hi = round_down(lo * lo), round_up(hi * hi)
    require(0 < lo <= hi, "exp enclosure inverted")
    return lo, hi


def exp_neg_upper(y):
    """Upper bound for exp(-y), y >= 0."""
    return round_up(1 / exp_bounds(y)[0])


def exp_neg_lower(y):
    """Lower bound for exp(-y), y >= 0."""
    return round_down(1 / exp_bounds(y)[1])


def atan_bounds(x, terms=40):
    """Alternating Taylor series of atan on 0 < x < 1: consecutive partial
    sums bracket the limit because the terms decrease in magnitude."""
    x = Fraction(x)
    require(0 < x < 1, "atan_bounds needs 0 < x < 1")
    partial = Fraction(0)
    sums = []
    for k in range(terms + 1):
        partial += (-1)**k * x**(2 * k + 1) / (2 * k + 1)
        sums.append(partial)
    return min(sums[-2:]), max(sums[-2:])


@lru_cache(None)
def pi_bounds():
    """Machin: pi = 16 atan(1/5) - 4 atan(1/239)."""
    lo5, hi5 = atan_bounds(Fraction(1, 5))
    lo239, hi239 = atan_bounds(Fraction(1, 239))
    lo, hi = 16 * lo5 - 4 * hi239, 16 * hi5 - 4 * lo239
    require(lo < hi and hi - lo < Fraction(1, 2**100), "pi enclosure")
    return lo, hi


def is_square(v):
    return v >= 0 and math.isqrt(v)**2 == v


def sqrt_upper(x):
    """Upper bound for sqrt(x), exact when x is a rational square."""
    x = Fraction(x)
    require(x >= 0, "sqrt_upper needs x >= 0")
    a, b = x.numerator, x.denominator
    if is_square(a) and is_square(b):
        return Fraction(math.isqrt(a), math.isqrt(b))
    # sqrt(a/b) = sqrt(a b)/b
    scale = 2**DYADIC_BITS
    return Fraction(math.isqrt(a * b * scale * scale) + 1, b * scale)


def wallis_upper(q):
    """Upper bound for a_0(q) = (1/pi) int_0^pi |cos u|^(2q) du, q >= 0.

    a_0(n) = C(2n, n)/4^n at integers and a_0(x) = a_0(x+1)(2x+2)/(2x+1)
    (integration by parts), so a_0(q) = a_0(q+M) prod_{j<M} (2q+2j+2) /
    (2q+2j+1). a_0 is log-convex (Hoelder), so on [n, n+1] it lies below
    the geometric, hence below the arithmetic, interpolation of its values
    at n and n+1.
    """
    q = Fraction(q)
    require(q >= 0, "wallis_upper needs q >= 0")
    x = q + WALLIS_SHIFT
    n = math.floor(x)
    lam = x - n
    a_n = Fraction(math.comb(2 * n, n), 4**n)
    a_n1 = a_n * (2 * n + 1) / (2 * n + 2)
    interp = (1 - lam) * a_n + lam * a_n1
    u, v = q.numerator, q.denominator
    num = math.prod(2 * u + (2 * j + 2) * v for j in range(WALLIS_SHIFT))
    den = math.prod(2 * u + (2 * j + 1) * v for j in range(WALLIS_SHIFT))
    scaled = -(-(interp.numerator * num * 2**DYADIC_BITS)
               // (interp.denominator * den))
    return Fraction(scaled, 2**DYADIC_BITS)


@lru_cache(None)
def fourier_h_upper(q, t):
    """Upper bound for h_t(1/q) = E|cos(sqrt(t/(2q)) G)|^(2q), q >= 1.

    Fourier/Parseval (companion theorem, Lemma C.2):
      h = a_0(q) [1 + 2 sum_{k>=1} r_k(q) exp(-t k^2/q)],
      r_k(q) = prod_{j=1..k} (q - j + 1)/(q + j),  |r_k| <= 1.
    Positive terms use an upper bound for exp(-y), negative terms a lower
    bound; the tail beyond the last term k0 - 1 is at most
    sum_{k>=k0} exp(-t k0 k/q) = x^k0/(1 - x), x = exp(-t k0/q).
    """
    q, t = Fraction(q), Fraction(t)
    require(q >= 1 and t > 0, "fourier_h_upper needs q >= 1, t > 0")
    s = Fraction(1)
    r = Fraction(1)
    k = 1
    while t * k * k / q < TAIL_EXPONENT or t * k / q < 1:
        r = r * (q - k + 1) / (q + k)
        y = t * k * k / q
        if r > 0:
            s += 2 * r * exp_neg_upper(y)
        elif r < 0:
            s += 2 * r * exp_neg_lower(y)
        k += 1
    x = exp_neg_upper(t * k / q)
    require(x < 1, "Fourier tail ratio")
    tail = round_up(x**k / (1 - x))
    s += 2 * tail
    require(s > 0, "Fourier sum not positive")
    return round_up(wallis_upper(q) * s)


def small_w_upper(t, q0):
    """Upper bound, valid for every 0 < w <= 1/q0, of
    (1+t)^(-1/2) + P(|G| > a),  a = pi sqrt(q0/(2t)),
    with the Mills bound P(|G| > a) <= sqrt(2/pi) exp(-a^2/2)/a
    = sqrt(4t/(pi^3 q0)) exp(-pi^2 q0/(4t)) (companion, Lemma C.3)."""
    pi_lo, _ = pi_bounds()
    limit = sqrt_upper(1 / (1 + t))
    prefactor = sqrt_upper(4 * t / (pi_lo**3 * q0))
    gauss = exp_neg_upper(pi_lo**2 * q0 / (4 * t))
    return round_up(limit + prefactor * gauss)


@lru_cache(None)
def laplace_sup_certificate(t):
    """Certify sup_{0<w<=1} h_t(w) <= U(t) in exact rational arithmetic.

    U(t) is max(h_t(1), (1+t)^(-1/2)) (upper enclosures) rounded up at
    SUP_DECIMALS decimals. Writing q = 1/w:
      * q >= q0: small_w_upper(t, q0) <= U, which is monotone in q;
      * 1 <= q <= q0: consecutive cells [qa, qb] with
        (qb/qa) * fourier_h_upper(qa)^2 <= U^2, using
        h(q) <= sqrt(qb/qa) h(qa) on the cell (companion, Lemma C.4).
    """
    t = Fraction(t)
    require(t > 0, "t must be positive")
    h1_up = fourier_h_upper(Fraction(1), t)
    limit_up = sqrt_upper(1 / (1 + t))
    top = max(h1_up, limit_up)
    scale = 10**SUP_DECIMALS
    sup_bound = Fraction(math.floor(top * scale) + 1, scale)

    q0 = None
    for cand in range(2, MAX_SMALL_W_Q0 + 1):
        if small_w_upper(t, cand) <= sup_bound:
            q0 = cand
            break
    require(q0 is not None, "no small-w cut-off q0 reaches the bound")
    small = small_w_upper(t, q0)

    cells = []
    worst = Fraction(0)                 # max (qb/qa) h_up(qa)^2
    qa = Fraction(1)
    while qa < q0:
        require(len(cells) < MAX_CELLS, "Laplace sup: too many cells")
        hu = fourier_h_upper(qa, t)
        require(hu < sup_bound, f"Laplace sup: h_t(1/{qa}) not below bound")
        ratio = qa * sup_bound**2 / hu**2
        qb = Fraction(math.floor(ratio * 2**CELL_BITS), 2**CELL_BITS)
        require(qb > qa, "Laplace sup: cell does not advance")
        cell_sq = qb / qa * hu**2
        require(cell_sq <= sup_bound**2, "Laplace sup: cell bound")
        worst = max(worst, cell_sq)
        cells.append((qa, qb))
        qa = qb
    fingerprint = hashlib.sha256(
        ";".join(f"{a.numerator:x}/{a.denominator:x},"
                 f"{b.numerator:x}/{b.denominator:x}"
                 for a, b in cells).encode()).hexdigest()
    h1_lo = (1 + exp_neg_lower(t)) / 2
    return {
        "t": t,
        "sup_bound": sup_bound,
        "h1_lower": h1_lo,
        "h1_upper": h1_up,
        "limit_upper": limit_up,
        "small_w_q0": q0,
        "small_w_upper": small,
        "cells": len(cells),
        "cells_sha256": fingerprint,
        "max_cell_bound_sq": worst,
    }


@dataclass(frozen=True)
class LaplacePair:
    """The (t, K) pair of Lemma C, certified together.

    Construction recomputes (cached) the supremum certificate at this t and
    refuses any K below it, so K cannot be changed without t or vice versa.
    """
    t: Fraction
    K: Fraction

    def __post_init__(self):
        require(isinstance(self.t, Fraction) and isinstance(self.K, Fraction),
                "Laplace pair must be exact rationals")
        require(0 < self.K < 1, "K must lie in (0, 1)")
        bound = laplace_sup_certificate(self.t)["sup_bound"]
        require(self.K >= bound,
                f"K = {self.K} is below the certified supremum "
                f"{bound} (~{float(bound):.7f}) of h_t at t = {self.t}")


def e3_above_20():
    """e^3 > 20 by the Taylor partial sum sum_{j<=10} 3^j/j!."""
    return sum(Fraction(3**j, math.factorial(j)) for j in range(11)) > 20


# --------------------------------------------------- grid distributions
def ceil_div(a, b):
    return (a + b - 1) // b


def _laplace(lap):
    require(isinstance(lap, LaplacePair), "engine needs a LaplacePair")
    return lap


@lru_cache(None)
def exp_grid_upper(j, lap):
    """FP times exp(t*A*R**j), rounded upward with an exact Taylor tail."""
    x = _laplace(lap).t * A * R**j
    require(0 <= x < DEGREE + 2, "exp_grid_upper argument out of range")
    a, b = x.numerator, x.denominator
    term = total = FP
    for k in range(1, DEGREE + 1):
        term = ceil_div(term * a, b * k)
        total += term
    omitted = ceil_div(term * a, b * (DEGREE + 1))
    remainder = ceil_div(omitted * b * (DEGREE + 2), b * (DEGREE + 2) - a)
    return total + remainder


@lru_cache(None)
def distribution(width, lap):
    """Grid law (zero mass, masses) of Z_width, rounded down to the grid.

    CDF upper bound min(1, K^width exp(t u)) at u = A R^j (Corollary C').
    """
    p = _laplace(lap).K**width

    def cdf(j):
        return min(D, ceil_div(D * p.numerator * exp_grid_upper(j, lap),
                               p.denominator * FP))

    zero = previous = cdf(0)
    masses = []
    for j in range(4000):
        current = cdf(j + 1)
        require(current >= previous, "grid CDF not monotone")
        masses.append(current - previous)
        if current == D:
            require(zero + sum(masses) == D, "grid law mass mismatch")
            return zero, tuple(masses)
        previous = current
    raise CertificateError("grid unexpectedly large")


def polynomial_product(polynomials, mass_bits=MASS_BITS):
    """Exact convolution by coefficient packing, with no inter-slot carries."""
    layers = len(polynomials)
    slot_bytes = (layers * mass_bits + 8) // 8
    slot_bits = 8 * slot_bytes
    bound = 1 << slot_bits
    require(math.prod(sum(p) for p in polynomials) < bound,
            "packed convolution slot overflow")
    encoded = 1
    for poly in polynomials:
        packed = int.from_bytes(
            b"".join(v.to_bytes(slot_bytes, "little") for v in poly),
            "little")
        encoded *= packed
    length = 1 + sum(len(p) - 1 for p in polynomials)
    raw = encoded.to_bytes(length * slot_bytes, "little")
    return tuple(
        int.from_bytes(raw[i * slot_bytes:(i + 1) * slot_bytes], "little")
        for i in range(length))


# --------------------------------------------------------- shifted sums
@lru_cache(None)
def sum_shift(delta):
    """Largest k with R**k <= 1 + R**(-delta): grid offset of R**a + R**b."""
    lhs = 1 + R**(-delta)
    k = 0
    while R**(k + 1) <= lhs:
        k += 1
    return k


SHIFT_CUTOFF = 200
require(R**(-SHIFT_CUTOFF) < R - 1, "shift cutoff too small")
SHIFTS = [sum_shift(d) for d in range(SHIFT_CUTOFF + 1)]


def grid_sum(x, y):
    """Law of X+Y for independent grid variables, rounded DOWN to the grid."""
    zx, mx = x
    zy, my = y
    out = [0] * (max(len(mx), len(my)) + SHIFTS[0] + 1)
    for a, ma in enumerate(mx):
        if ma:
            out[a] += ma * zy
    for b, mb in enumerate(my):
        if mb:
            out[b] += zx * mb
    for a, ma in enumerate(mx):
        if not ma:
            continue
        for b, mb in enumerate(my):
            if not mb:
                continue
            d = a - b if a >= b else b - a
            shift = SHIFTS[d] if d <= SHIFT_CUTOFF else 0
            out[(a if a >= b else b) + shift] += ma * mb
    while out and out[-1] == 0:
        out.pop()
    return zx * zy, tuple(out)


@lru_cache(None)
def prefix_law(widths, lap):
    polys = [distribution(c, lap)[1] for c in widths]
    masses = polynomial_product(polys)
    positive = math.prod(sum(p) for p in polys)
    return D**len(widths) - positive, masses


def escape_probability(widths):
    """eps_F = 1 - prod_i (1 - 2^-c_i): some prefix core of one group lets
    a large input become small (Lemma A, chained)."""
    keep = math.prod(1 - Fraction(1, 2**c) for c in widths)
    return 1 - keep


@lru_cache(None)
def boosted_prefix_law(widths, lap):
    """Prefix product law with eps_F extra mass moved from the top of the
    grid law to the zero atom (the law of W_g 1[not F_g] is dominated by
    it)."""
    zero, masses = prefix_law(widths, lap)
    total = D**len(widths)
    eps = escape_probability(widths)
    delta = ceil_div(eps.numerator * total, eps.denominator)
    masses = list(masses)
    remaining = delta
    i = len(masses) - 1
    while remaining:
        require(i >= 0, "atom boost exceeds the prefix mass")
        take = min(remaining, masses[i])
        masses[i] -= take
        remaining -= take
        i -= 1
    while masses and masses[-1] == 0:
        masses.pop()
    require(zero + delta + sum(masses) == total, "boosted law mass")
    return zero + delta, tuple(masses)


@lru_cache(None)
def summed_boosted_law(widths, groups, lap):
    """Lower bound for the sum of `groups` iid boosted prefix products."""
    result = None
    power = boosted_prefix_law(widths, lap)
    g = groups
    while g:
        if g & 1:
            result = power if result is None else grid_sum(result, power)
        g >>= 1
        if g:
            power = grid_sum(power, power)
    return result


# ------------------------------------------------------ rational roots
def ceil_rational_root(numerator, denominator, order):
    """Smallest integer x with denominator*x**order >= numerator."""
    if numerator < 0 or denominator <= 0 or order < 1:
        raise ValueError("expected nonnegative numerator, positive den/order")
    if numerator == 0:
        return 0
    bit_bound = max(0, numerator.bit_length() - denominator.bit_length() + 1)
    hi = 1 << ((bit_bound + order - 1) // order)
    while denominator * hi**order < numerator:
        hi *= 2
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if denominator * mid**order >= numerator:
            hi = mid
        else:
            lo = mid
    return hi


# ----------------------------------------- mixed tail and source moments
@lru_cache(None)
def mixed_tail(bits, widths, groups, out, lap):
    """Norm-leg failure bound and its grid rho.

    Returns (num, den, escape, exponent, rho) with
      num/den = Pr[Z_out * sum_g W~_g < rho]   (boosted grid laws),
      escape  = 2^-out                         (terminal Lemma A),
    and num/den + escape <= (7/8) 2^-bits, with the grid exponent maximal.
    """
    zs, ms = summed_boosted_law(widths, groups, lap)
    zf, mf = distribution(out, lap)
    layers = len(widths) * groups + 1
    den = D**layers
    coeff = polynomial_product([ms, mf], mass_bits=MASS_BITS * layers)
    require(zs + sum(ms) == D**(len(widths) * groups), "summed law mass")
    require(zf + sum(mf) == D, "terminal law mass")
    require(sum(coeff) == sum(ms) * sum(mf), "mixed law mass")
    escape = Fraction(1, 2**out)

    def fits(num):
        # num/den + 2^-out <= 7/(8 * 2^bits), in integers.
        return 8 * 2**bits * (num * 2**out + den) <= 7 * den * 2**out

    num = den - sum(ms) * sum(mf)
    require(fits(num), "zero atom exceeds budget")
    exponent = 0
    for mass in coeff:
        if not fits(num + mass):
            break
        num += mass
        exponent += 1
    else:
        raise CertificateError("full probability cannot pass")
    require(fits(num) and not fits(num + mass),
            "grid exponent is not maximal")
    return num, den, escape, exponent, A**(len(widths) + 1) * R**exponent


@lru_cache(None)
def source_moments(max_order, widths, groups, out):
    mu = [1]
    for k in range(max_order):
        mu.append(mu[-1] * math.prod(c + 2 * k for c in widths))
    cur = [1] + [0] * max_order
    for _ in range(groups):
        cur = [sum(math.comb(p, k) * cur[p - k] * mu[k] for k in range(p + 1))
               for p in range(max_order + 1)]
    terminal = 1
    moments = [Fraction(1)]
    for p in range(1, max_order + 1):
        terminal *= out + 2 * (p - 1)
        moments.append(
            Fraction(cur[p] * terminal, 2**((len(widths) + 1) * p)))
    return moments


def best_cap(energy, moments, bits):
    """Exhaustive exact roots over the finite moment-order search."""
    best = None
    for order in range(1, len(moments)):
        rhs = 2**bits * energy**order * moments[order]
        cap = ceil_rational_root(rhs.numerator, rhs.denominator, order)
        require((cap - 1)**order < rhs <= cap**order, "cap root not tight")
        if best is None or cap < best["cap"]:
            best = {"cap": cap, "order": order, "moment": moments[order]}
    return best


def fixed_first_geometry(first_bits, widths, groups, out, total_bits):
    layers = len(widths) + 1

    @lru_cache(None)
    def walk(i, used):
        if i == layers:
            return (0, ()) if used == total_bits else None
        best = None
        choices = ([first_bits] if i == 0 else
                   range(1, total_bits - used - (layers - i - 1) + 1))
        for e in choices:
            if used + e > total_bits:
                continue
            tail = walk(i + 1, used + e)
            if tail is None:
                continue
            a = (groups * (1 if i == 0 else widths[i - 1])
                 * (widths[i] if i < len(widths) else out))
            cand = (a * 2**e + tail[0], (e,) + tail[1])
            if best is None or cand < best:
                best = cand
        return best

    coefficients, exps = walk(0, 0)
    stored = used = 0
    for c, e in zip(widths, exps):
        used += e
        stored += groups * c * 2**(total_bits - used)
    extension = 2**exps[0] - 1
    for i, e in enumerate(exps[1:], 1):
        inputs = groups * widths[i - 1]
        outputs = widths[i] if i < len(widths) else out
        extension += min(inputs * (2**e - 1), (2**e - 1) + inputs * outputs)
    return {"radix_exponents": list(exps), "radices": [2**e for e in exps],
            "coefficients": coefficients, "ext": extension,
            "prefix_entries": stored}


# -------------------------------------------------------- residual leg
def residual_moments(b, widths, max_order):
    # Uniform{-b..b} has variance b(b+1)/3; a c-row core contributes
    # (2 sigma^2)^r (c/2)_r = sigma^(2r) prod_{k<r} (c + 2k).
    variance = Fraction(b * (b + 1), 3)
    out = [Fraction(1)]
    for r in range(1, max_order + 1):
        out.append(out[-1] * variance**len(widths)
                   * math.prod(c + 2 * (r - 1) for c in widths))
    return out


def lw_bound(n, falcon_b, cmax):
    """Least L with (L - cmax - 6144)^2 >= 6144^2 n B."""
    target = HT**2 * n * falcon_b
    lw = cmax + HT + math.isqrt(target)
    while (lw - cmax - HT)**2 < target:
        lw += 1
    require((lw - 1 - cmax - HT)**2 < target, "L_W is not least")
    return lw


def log2_exact_pow2(x):
    require(x > 0 and x & (x - 1) == 0, f"{x} is not a power of two")
    return x.bit_length() - 1


def bits(f):
    """Display only: -log2 of a positive Fraction."""
    return math.log2(f.denominator) - math.log2(f.numerator)


def log2f(x):
    """Display only: log2 of a positive int or Fraction."""
    x = Fraction(x)
    return math.log2(x.numerator) - math.log2(x.denominator)


def digest(f):
    """Platform-independent fingerprint of an exact rational."""
    f = Fraction(f)
    text = f"{f.numerator:x}/{f.denominator:x}"
    return hashlib.sha256(text.encode()).hexdigest()


def exact(f, inline_limit=256):
    """JSON form of an exact rational: inline if small, else a digest."""
    f = Fraction(f)
    size = max(f.numerator.bit_length(), f.denominator.bit_length())
    if size <= inline_limit:
        return str(f)
    return {"sha256": digest(f), "numerator_bits": f.numerator.bit_length(),
            "denominator_bits": f.denominator.bit_length()}


# ----------------------------------------------- norm-leg geometry (6.2)
def norm_radices(n, log_n_star):
    """Radices (r_1..r_5) of specs/falcon.md Section 6.2 at N* = 2^ell."""
    nu = log2_exact_pow2(n)
    k = NORM_FIRST_BITS - (nu + 1)
    rest = log_n_star - k
    require(rest >= 0, "N* below the first block")
    rho = []
    left = rest
    for _ in range(len(S_WIDTHS) - 1):
        take = min(PREFIX_MAX_BITS, left)
        rho.append(take)
        left -= take
    rho.append(left)
    return (2**NORM_FIRST_BITS,) + tuple(2**e for e in rho)


def stage_fan_ins(radices):
    """Fan-in of the core that consumes stage-i table T_i, i = 0..4."""
    channels = (1,) + S_WIDTHS[:-1] + (S_GROUPS * S_WIDTHS[-1],)
    return tuple(c * r for c, r in zip(channels, radices))


def max_fan_ins(n):
    """Largest fan-in per stage over every supported N*."""
    nu = log2_exact_pow2(n)
    lowest = MIN_LOG_PADDED_COEFFS - nu
    rows = [stage_fan_ins(norm_radices(n, ell))
            for ell in range(lowest, LOG_N_RECORDS + 1)]
    return tuple(max(col) for col in zip(*rows)), rows[-1]


def thresholds(cap):
    """tau_5 = 2 floor(sqrt(C_S)) + 1, tau_i = 2^(5-i) tau_5 (tau_4 = 2 tau_5)."""
    m = math.isqrt(cap)
    tau5 = 2 * m + 1
    stages = len(S_WIDTHS) + 1
    return m, tuple(tau5 * 2**(stages - i) for i in range(stages + 1))


# ------------------------------------------------------------ norm leg
@lru_cache(None)
def norm_leg(n, lap):
    """Everything that depends on the degree but not the residual profile."""
    falcon_b, cmax = FALCON[n]["B"], FALCON[n]["cmax"]
    fail_num, fail_den, escape, exponent, rho = mixed_tail(
        EPS_BITS, S_WIDTHS, S_GROUPS, S_OUT, lap)
    moments = source_moments(CAP_MAX_ORDER, S_WIDTHS, S_GROUPS, S_OUT)
    h_s = N_RECORDS * falcon_b
    cap = best_cap(h_s, moments, HONEST_BITS)
    # The tail below rho bounds the tail below any smaller threshold.
    q_s = Fraction(cap["cap"] + 1, ETA_PROJ)
    m, tau = thresholds(cap["cap"])
    fan_in, fan_in_top = max_fan_ins(n)
    checks = {"eta_proj_at_most_grid_threshold": ETA_PROJ <= rho,
              "lift": q_s + n * S2_MAX**2 < P,
              "accept_below_tau5": m < tau[-1]}
    for i in range(len(fan_in)):
        checks[f"lemma_a_stage_{i}"] = 2 * (tau[i + 1] - 1) < tau[i]
    for i in range(len(fan_in)):
        checks[f"lemma_b_stage_{i}"] = fan_in[i] * (tau[i] - 1) <= HALF_P
    return {"falcon_b": falcon_b, "cmax": cmax,
            "fail_num": fail_num, "fail_den": fail_den, "escape": escape,
            "rho_exponent": exponent, "rho": rho, "h_s": h_s, "cap": cap,
            "q_s": q_s, "m": m, "tau": tau, "fan_in": fan_in,
            "fan_in_top": fan_in_top, "checks": checks}


def certify(n, b, widths, lap):
    """Certify (n, b, widths) along the elementary norm chain."""
    norm = norm_leg(n, lap)
    falcon_b, cmax = norm["falcon_b"], norm["cmax"]
    require(cmax**2 <= falcon_b < (cmax + 1)**2, "cmax != floor(sqrt(B))")
    cap, q_s = norm["cap"], norm["q_s"]
    lw = lw_bound(n, falcon_b, cmax)
    h_w = N_RECORDS * n * lw**2
    moments = residual_moments(b, widths, WRAP_MAX_ORDER)
    x = Fraction(4 * h_w, P * P)
    # Exact minimization; ties go to the smallest order, as in min().
    wrap_order = min(range(1, WRAP_MAX_ORDER + 1),
                     key=lambda r: x**r * moments[r])
    wrap = x**wrap_order * moments[wrap_order]
    kill = sum((Fraction(1, (2 * b + 1)**c) for c in widths), Fraction(0))
    norm_grid = Fraction(norm["fail_num"], norm["fail_den"])
    norm_err = norm_grid + norm["escape"]
    err = norm_err + wrap + kill
    honest = Fraction(1, 2**HONEST_BITS) + wrap
    norm_bits = log2_exact_pow2(N_RECORDS) + log2_exact_pow2(2 * n)
    residual_bits = log2_exact_pow2(N_RECORDS) + log2_exact_pow2(n)
    sg = fixed_first_geometry(NORM_FIRST_BITS, S_WIDTHS, S_GROUPS, S_OUT,
                              norm_bits)
    vg = fixed_first_geometry(NORM_FIRST_BITS - 1, widths[:-1], 1,
                              widths[-1], residual_bits)
    merged = (sg["ext"] + vg["ext"] - (2**(NORM_FIRST_BITS - 1) - 1)
              + S_OUT + widths[-1])
    checks = dict(norm["checks"])
    checks["spec_radices_match_geometry"] = (
        list(norm_radices(n, LOG_N_RECORDS)) == sg["radices"])
    checks["two_b_below_q"] = 2 * b < Q
    checks["total_at_least_136_bits"] = err <= Fraction(1, 2**EPS_BITS)
    # honest <= 2^-128.9  <=>  honest^10 * 2^1289 <= 1, exactly.
    checks["honest_at_least_128_9_bits"] = (
        honest**HONEST_FLOOR.denominator * 2**HONEST_FLOOR.numerator <= 1)
    tau = norm["tau"]
    return {
        "n": n,
        "B": falcon_b,
        "cmax": cmax,
        "H_S": norm["h_s"],
        "H_S_log2": math.log2(norm["h_s"]),
        "C_S": cap["cap"],
        "C_S_log2": math.log2(cap["cap"]),
        "cap_order": cap["order"],
        "cap_moment": exact(cap["moment"]),
        "rho": f"(1/1024)^{len(S_WIDTHS) + 1}*(33/32)^{norm['rho_exponent']}",
        "rho_grid_exponent": norm["rho_exponent"],
        "rho_log2": log2f(norm["rho"]),
        "eta_proj": ETA_PROJ,
        "Q_S": exact(q_s),
        "Q_S_log2": log2f(q_s),
        "lift_slack_bits": (math.log2(P - n * S2_MAX**2) - log2f(q_s)),
        "exactness": {
            "m": norm["m"],
            "tau": list(tau),
            "tau_log2": [math.log2(v) for v in tau],
            "fan_in": list(norm["fan_in"]),
            "margin_bits": [math.log2(HALF_P) - math.log2(f * (t - 1))
                            for f, t in zip(norm["fan_in"], tau)],
        },
        "norm_grid_err": exact(norm_grid),
        "norm_grid_err_bits": bits(norm_grid),
        "norm_escape": str(norm["escape"]),
        "norm_err": exact(norm_err),
        "norm_err_bits": bits(norm_err),
        "L_W": lw,
        "H_W": h_w,
        "H_W_log2": math.log2(h_w),
        "residual": {
            "b": b,
            "widths": list(widths),
            "kill": exact(kill),
            "kill_bits": bits(kill),
            "wrap": exact(wrap),
            "wrap_bits": bits(wrap),
            "wrap_order": wrap_order,
        },
        "total_err": exact(err),
        "total_err_bits": bits(err),
        "honest": exact(honest),
        "honest_bits": bits(honest),
        "checks": checks,
        "certified": all(checks.values()),
        "outputs": S_OUT + widths[-1],
        "clear_bytes": 8 * (S_OUT + widths[-1]),
        "norm_geometry": sg,
        "residual_geometry": vg,
        "total_coefficients": sg["coefficients"] + vg["coefficients"],
        "merged_ext": merged,
    }


# -------------------------------------------------- global self-checks
def exp_upper(x, terms=128):
    """tail_bounds.py:94-105. Exact rational upper bound on exp(x), x>=0."""
    require(0 <= x < terms + 2, "exp_upper argument out of range")
    term = total = Fraction(1)
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    first_omitted = term * x / (terms + 1)
    return total + first_omitted / (1 - x / (terms + 2))


def is_prime(v):
    return v >= 2 and all(v % d for d in range(2, math.isqrt(v) + 1))


def global_checks(lap):
    """Profile-independent checks; (name, ok) pairs."""
    out = []
    # Self-test: packed convolution vs schoolbook.
    polys = [(1, 2, 3), (4, 1), (0, 2)]
    expected = [1]
    for poly in polys:
        acc = [0] * (len(expected) + len(poly) - 1)
        for i, a in enumerate(expected):
            for j, c in enumerate(poly):
                acc[i + j] += a * c
        expected = acc
    out.append(("packed convolution matches schoolbook",
                polynomial_product(polys, mass_bits=4) == tuple(expected)))
    # Self-test: the fixed-point grid exp is an upper bound.
    out.append(("exp_grid_upper(j) >= exp_upper(t A R^j) at j in {0,100,250}",
                all(Fraction(exp_grid_upper(j, lap), FP)
                    >= exp_upper(lap.t * A * R**j) for j in (0, 100, 250))))
    # Enclosure self-tests against independent rational facts.
    pi_lo, pi_hi = pi_bounds()
    out.append(("pi enclosure lies inside (333/106, 355/113)",
                Fraction(333, 106) < pi_lo < pi_hi < Fraction(355, 113)))
    lo, hi = exp_bounds(Fraction(3))
    taylor_lo = sum(Fraction(3**j, math.factorial(j)) for j in range(80))
    out.append(("exp(3) enclosure is consistent with independent Taylor "
                "bounds and narrower than 2^-150",
                lo <= exp_upper(Fraction(3)) and taylor_lo <= hi
                and hi - lo < Fraction(1, 2**150)))
    out.append(("e^3 > 20 (so (1 + e^-3)/2 < 21/40 at p = 1)",
                e3_above_20()))
    sup = laplace_sup_certificate(lap.t)
    out.append((f"Lemma C: sup_(0<w<=1) h_t(w) <= U = {sup['sup_bound']} "
                f"<= K = {lap.K} at t = {lap.t}",
                sup["sup_bound"] <= lap.K
                and sup["max_cell_bound_sq"] <= sup["sup_bound"]**2
                and sup["small_w_upper"] <= sup["sup_bound"]))
    # Generic consistency asserts.
    out.append(("q = 12289 is prime", is_prime(Q)))
    out.append(("centered h, t bound (q-1)/2 = 6144", HT == 6144))
    out.append((f"3^(p-1) <= (2p-1)!! for p <= {WRAP_MAX_ORDER}",
                all(3**(p - 1) <= math.prod(range(1, 2 * p, 2))
                    for p in range(1, WRAP_MAX_ORDER + 1))))
    out.append((f"Lemma A terminal escape 2^-{S_OUT} fits 2^-{EPS_BITS}",
                Fraction(1, 2**S_OUT) < Fraction(1, 2**EPS_BITS)))
    for n, spec in L_W_SPEC.items():
        d = FALCON[n]
        out.append((f"n={n}: cmax = isqrt(B) = {d['cmax']}",
                    math.isqrt(d["B"]) == d["cmax"]))
        out.append((f"n={n}: L_W equals the existing spec value {spec:,}",
                    lw_bound(n, d["B"], d["cmax"]) == spec))
    return out


def laplace_report(lap):
    sup = laplace_sup_certificate(lap.t)
    return {
        "t": str(lap.t),
        "K": str(lap.K),
        "sup_bound": str(sup["sup_bound"]),
        "sup_bound_display": float(sup["sup_bound"]),
        "K_minus_sup_bound": str(lap.K - sup["sup_bound"]),
        "K_minus_sup_bound_display": float(lap.K - sup["sup_bound"]),
        "h_at_w1_lower_display": float(sup["h1_lower"]),
        "h_at_w1_upper_display": float(sup["h1_upper"]),
        "h_at_w1_upper": exact(sup["h1_upper"]),
        "gaussian_limit_upper": str(sup["limit_upper"]),
        "small_w_q0": sup["small_w_q0"],
        "small_w_upper": exact(sup["small_w_upper"]),
        "small_w_upper_display": float(sup["small_w_upper"]),
        "cells": sup["cells"],
        "cells_sha256": sup["cells_sha256"],
        "max_cell_bound": exact(sup["max_cell_bound_sq"]),
        "max_cell_bound_display": math.sqrt(float(sup["max_cell_bound_sq"])),
        "escape_eps_F": exact(escape_probability(S_WIDTHS)),
        "escape_eps_F_log2": log2f(escape_probability(S_WIDTHS)),
    }


# ------------------------------------------------------------- report
def build_report(t=LAPLACE_T, k=LAPLACE_K):
    started = time.monotonic()
    lap = LaplacePair(t=Fraction(t), K=Fraction(k))
    configs = []
    for n, (b, widths), role in CONFIGS:
        row = certify(n, b, widths, lap)
        configs.append({"role": role, **row})
    report = {
        "schema": SCHEMA,
        "provenance": {
            "norm_leg": ("elementary: Lemmas A, B, C, C', D and the assembly "
                         "of specs/falcon-projection-theorem.md Section 3; "
                         "sup_w h_t(w) certified here in exact arithmetic"),
        },
        "constants": {
            "p": P, "q": Q, "records": N_RECORDS, "eps_bits": EPS_BITS,
            "honest_cap_bits": HONEST_BITS,
            "honest_floor_bits": str(HONEST_FLOOR), "s2_max": S2_MAX,
            "A": str(A), "R": str(R), "mass_bits": MASS_BITS,
            "norm_widths": list(S_WIDTHS), "norm_groups": S_GROUPS,
            "norm_outputs": S_OUT, "cap_max_order": CAP_MAX_ORDER,
            "wrap_max_order": WRAP_MAX_ORDER,
            "dyadic_bits": DYADIC_BITS, "wallis_shift": WALLIS_SHIFT,
            "cell_bits": CELL_BITS,
        },
        "laplace": laplace_report(lap),
        "global_checks": {name: ok for name, ok in global_checks(lap)},
        "configurations": configs,
    }
    return report, time.monotonic() - started


def verdicts(report):
    """(name, ok, detail) for every decision the certificate makes."""
    out = [(name, ok, "") for name, ok in report["global_checks"].items()]
    for row in report["configurations"]:
        tag = f"n={row['n']} b={row['residual']['b']} " \
              f"widths={tuple(row['residual']['widths'])}"
        if row["role"] == "chosen":
            for name, ok in row["checks"].items():
                out.append((f"{tag}: {name}", ok, ""))
            out.append((f"{tag}: cap order is 46", row["cap_order"] == 46,
                        f"order {row['cap_order']}"))
        elif row["n"] == 512:
            out.append((f"{tag} (rejected profile): passes at n=512",
                        row["certified"],
                        f"total 2^-{row['total_err_bits']:.2f}"))
        else:
            wrap_fails = not row["checks"]["total_at_least_136_bits"] and all(
                ok for name, ok in row["checks"].items()
                if name not in ("total_at_least_136_bits",
                                "honest_at_least_128_9_bits"))
            out.append((f"{tag} (rejected profile): fails at n=1024 "
                        f"(NEGATIVE CHECK: only the residual wrap fails)",
                        wrap_fails,
                        f"wrap 2^-{row['residual']['wrap_bits']:.2f}"))
    return out


def print_report(report):
    lap = report["laplace"]
    print(f"--- Lemma C at t={lap['t']}, K={lap['K']}")
    print(f"  h_t(1) in [{lap['h_at_w1_lower_display']:.10f}, "
          f"{lap['h_at_w1_upper_display']:.10f}]  "
          f"sup_w h_t(w) <= U={lap['sup_bound']} "
          f"(~{lap['sup_bound_display']:.6f})")
    print(f"  K - U = {lap['K_minus_sup_bound']} "
          f"(~{lap['K_minus_sup_bound_display']:.3e})  "
          f"{lap['cells']} q-cells on [1, {lap['small_w_q0']}], "
          f"max cell bound {lap['max_cell_bound_display']:.9f}; "
          f"w <= 1/{lap['small_w_q0']}: {lap['small_w_upper_display']:.6f}")
    print(f"  eps_F = 2^{lap['escape_eps_F_log2']:.4f}")
    for row in report["configurations"]:
        res = row["residual"]
        print(f"--- n={row['n']}  b={res['b']}  widths={tuple(res['widths'])}"
              f"  [{row['role']}]")
        print(f"  B={row['B']:,}  floor(sqrt B)={row['cmax']:,}  "
              f"H_S=2^{row['H_S_log2']:.2f}")
        print(f"  C_S={row['C_S']:,} (2^{row['C_S_log2']:.2f}, "
              f"order {row['cap_order']})  rho={row['rho']}"
              f" = 2^{row['rho_log2']:.4f}  eta_proj={row['eta_proj']}")
        print(f"  log2 Q_S={row['Q_S_log2']:.4f}  "
              f"lift slack={row['lift_slack_bits']:.2f} bits  "
              f"norm err=2^-{row['norm_err_bits']:.4f}")
        ex = row["exactness"]
        print(f"  tau_0..5 = 2^" + ", 2^".join(f"{v:.3f}"
                                              for v in ex["tau_log2"])
              + f"  fan-in {tuple(ex['fan_in'])}")
        print("  exactness margins " + ", ".join(
            f"{v:.3f}" for v in ex["margin_bits"]) + " bits")
        print(f"  L_W={row['L_W']:,}  H_W=2^{row['H_W_log2']:.2f}  "
              f"kill=2^-{res['kill_bits']:.2f}  "
              f"wrap=2^-{res['wrap_bits']:.2f} (r={res['wrap_order']})")
        print(f"  total=2^-{row['total_err_bits']:.3f}  "
              f"honest=2^-{row['honest_bits']:.8f}  "
              f"outputs={row['outputs']} ({row['clear_bytes']:,} bytes)")
        ng, rg = row["norm_geometry"], row["residual_geometry"]
        print(f"  norm radices {tuple(ng['radices'])}; "
              f"{ng['coefficients']:,} coeffs; {ng['ext']:,} ext")
        print(f"  residual radices {tuple(rg['radices'])}; "
              f"{rg['coefficients']:,} coeffs; {rg['ext']:,} ext")
        print(f"  verifier total {row['total_coefficients']:,} coeffs; "
              f"{row['merged_ext']:,} merged ext")
    print()
    failures = []
    for name, ok, detail in verdicts(report):
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)
    return failures


# ------------------------------------------------------- pin comparison
def diff(pinned, fresh, path="$"):
    """Exact comparison, except floats (display fields) within tolerance."""
    if isinstance(pinned, float) or isinstance(fresh, float):
        ok = (isinstance(pinned, (int, float)) and
              isinstance(fresh, (int, float)) and
              not isinstance(pinned, bool) and not isinstance(fresh, bool) and
              abs(pinned - fresh) <= FLOAT_TOLERANCE)
        return [] if ok else [f"{path}: pinned {pinned!r}, got {fresh!r}"]
    if isinstance(pinned, dict) and isinstance(fresh, dict):
        out = []
        for key in sorted(set(pinned) | set(fresh)):
            if key not in pinned or key not in fresh:
                out.append(f"{path}.{key}: present only in "
                           f"{'pin' if key in pinned else 'recomputation'}")
            else:
                out += diff(pinned[key], fresh[key], f"{path}.{key}")
        return out
    if isinstance(pinned, list) and isinstance(fresh, list):
        if len(pinned) != len(fresh):
            return [f"{path}: length {len(pinned)} != {len(fresh)}"]
        out = []
        for i, (a, b) in enumerate(zip(pinned, fresh)):
            out += diff(a, b, f"{path}[{i}]")
        return out
    if type(pinned) is not type(fresh) or pinned != fresh:
        return [f"{path}: pinned {pinned!r}, got {fresh!r}"]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help=f"fail if the recomputed report differs from "
                           f"{PIN_PATH.name}")
    mode.add_argument("--write", action="store_true",
                      help=f"rewrite {PIN_PATH.name} (only if all checks pass)")
    args = parser.parse_args()

    try:
        report, seconds = build_report()
    except CertificateError as error:
        print(f"[FAIL] certificate invariant: {error}")
        return 1
    # Round-trip so the comparison sees exactly what the pin file holds.
    fresh = json.loads(json.dumps(report))
    failures = print_report(fresh)
    print(f"\ncomputed in {seconds:.1f} s")
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1

    if args.write:
        PIN_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {PIN_PATH}")
    elif args.check:
        try:
            pinned = json.loads(PIN_PATH.read_text())
        except (OSError, ValueError) as error:
            print(f"[FAIL] cannot read pin {PIN_PATH}: {error}")
            return 1
        differences = diff(pinned, fresh)
        for line in differences:
            print(f"[FAIL] pin mismatch {line}")
        if differences:
            print(f"{len(differences)} fields differ from {PIN_PATH.name}")
            return 1
        print(f"[ok  ] recomputed report matches {PIN_PATH.name}")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
