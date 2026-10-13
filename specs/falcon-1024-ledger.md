# Falcon-1024 composition ledger

This ledger records the three-stage PIOP terms and the clear-projection term
for Falcon-1024, together with the audited four-group production catalog
estimates, conditional extraction upper bounds, and exact MSIS occurrence
data. Section 10.1 of [falcon.md](falcon.md#101-transcript-registry) supplies
the normative complete-tree work allocation. The paper proves the target
construction's 128-bit classical-ROM knowledge rate for every admitted padded
batch. The verifier enforces the
Section 10.1 PIOP work plan and Akita enforces its native PCS plan
(falcon.md Section 12.3).

The unground interactive terms below are retained as algebraic checks. They
are not the complete-tree debits used to allocate work. The
challenge field is $`K=\mathbb F_{p^2}`$, where $`p=2^{64}-59`$. Falcon-512 terms
and the ledger convention are in [`falcon-ledger.md`](falcon-ledger.md).

## 1. PIOP terms

Let $`R=d_{\rm coef}+d_{\rm sig}`$. Each of the three stages has $`R`$ rounds at degrees
3, 5, and 3, respectively. Stage 1 checks $`a_1`$, $`a_2`$, and the base-16
digit $`u_2`$; stage 2 checks $`a_0`$, $`u_0`$, and $`u_1`$; stage 3 checks
`sign_check`, `norm_equality`, and `encoding_budget`.

| Source | Falcon-1024 numerator over $`\vert K\vert `$ |
| --- | ---: |
| Sumcheck rounds | $`11R`$ |
| `basis16_product` at $`\tau`$: $`a_1`$, $`a_2`$, $`u_2`$ | $`2R+d_{\rm sig}`$ |
| `range_leaf` at $`r^{(1)}`$: $`a_0`$, $`u_0`$, $`u_1`$ | $`R+2d_{\rm sig}`$ |
| `sign_check`, `norm_equality`, `encoding_budget` at $`r^{(2)}`$ | $`R+2d_{\rm sig}`$ |
| Monomial projection weights from $`\mu\in K^8`$ | 8 |
| Three fresh stage combiners | 3 |
| **Total** | **$`15R+5d_{\rm sig}+11`$** |

The design reference profile is Falcon-1024 at $`N_{\rm sig}=2^{14}`$. Its numerator is
441, below 460, so its unground union bound is at most $`460/|K|\lt 2^{-119}`$.
This is an algebraic check, not the deployed PIOP term; Section 3 charges each
ground PIOP site at weight at most $`2^{-128}`$.
The full $`N_{\rm sig}^*=2^{16}`$ cap has a numerator of 481; the reference-profile
bound does not extend to that cap.

| $`N_{\rm sig}`$ | $`d_{\rm sig}`$ | Rounds per stage | PIOP numerator | $`\leq460`$? |
| ---: | ---: | ---: | ---: | --- |
| $`2^{14}`$ | 14 | 24 | 441 | yes; Falcon-1024 reference profile |
| $`2^{15}`$ | 15 | 25 | 461 | no |
| $`2^{16}`$ | 16 | 26 | 481 | no |

The script checks all three rows and the reference-profile bound. The shared
point reuse and sumcheck argument are specified in [`falcon.md`](falcon.md),
Sections 7 and 9.

## 2. Clear-projection term

The projection theorem's chosen Falcon-1024 certificate row is charged once
per transcript. The source tables are fixed before the projection seed. The
current certificate gives:

| Leg | Term |
| --- | ---: |
| Norm leg | $`2^{-136.4416}`$ |
| Residual kill | $`3/25^{31}=2^{-142.3746}`$ |
| Residual wrap, order 60 | $`2^{-153.7467}`$ |
| Total | $`2^{-136.4182}`$ |

The script reads the row from
[`scripts/falcon_projection_certificate.json`](scripts/falcon_projection_certificate.json),
checks the leg sum, and charges a rational upper bound for the total. The
projection theorem and integer-lifting argument are in
[`falcon-projection-theorem.md`](falcon-projection-theorem.md) and
[`falcon.md`](falcon.md), Section 9.

## 3. PCS, MSIS, and assembled error

The selected layout is $`[S_1]`$ width 14, $`[e]`$ width 14,
$`[a_0,a_1,a_2,\sigma]`$ width 4, and final group
$`[u_0,u_1,u_2]`$ width 4. The full catalog has 14 width-14 rows and 33
width-4 rows: 14 standalone producer rows and 19 distinct grouped production
rows (13 Falcon-512 shapes and 6 Falcon-1024-specific shapes). The supported
Falcon-1024 live batch range is $`3`$ through $`65{,}536`$. Rounding up to a power
of two gives padded counts $`4,8,\ldots,65{,}536`$; multiple padded counts alias
the same group-shape row, and the audit validates all 15 padded-count aliases.
The exact report and validator are
[`scripts/falcon_pcs_catalog_audit.json`](scripts/falcon_pcs_catalog_audit.json)
and [`scripts/falcon_msis_audit.py`](scripts/falcon_msis_audit.py).

The figures below are Akita planner estimates for the shared four-group
opening, not full proof-frame sizes. Each group commitment contributes 128
terminal Q64 coefficient bytes; the source geometries and pinned compression
plan are in [`falcon-pcs-extraction.md`](falcon-pcs-extraction.md), Section 4.

| $`N_{\rm sig}`$ | Grouped payload estimate | Setup field-element estimate | Fold levels | SIS occurrences: Inner $`L_\infty/L_2^2`$; Outer; Open; Compression |
| ---: | ---: | ---: | ---: | --- |
| $`2^{14}`$ | 68,723 B | 749,056 | 7 | 51: 7/3; 9; 6; 26 |
| $`2^{15}`$ | 68,872 B | 1,048,576 | 7 | 51: 7/3; 9; 6; 26 |
| $`2^{16}`$ | 68,988 B | 1,469,952 | 7 | 51: 7/3; 9; 6; 26 |

For a fixed fold prefix $`\ell`$, the conditional CWSS upper bound is
$`\kappa_\ell\leq\sum_g P_gB_g/L_g`$, using each group's polynomial count,
exact live-block count, and a certified challenge-support lower bound.
With $`C_{64}`$, $`C_{128}`$, $`C_{256}`$, and $`A_{128}=\lceil C_{128}p_0\rceil`$ defined in
[`falcon-pcs-extraction.md`](falcon-pcs-extraction.md), Section 2, the
$`L_\infty`$ routes use raw support and the selective $`L_2`$ routes use
$`A_{128}=502822610611069237129947393810277364817`$. The latter is derived
from the exact `claimed.p0` rational in Akita `515a21668`'s D128 certificate,
unchanged at the current pin `e023b6a46`.
The audit computes rational upper bounds from the serialized schedule.
Their sums, including the terminal witness term, are:

| $`N_{\rm sig}`$ | Conditional schedule upper bound |
| ---: | ---: |
| $`2^{14}`$ | $`266/C_{64}+131/C_{128}+21/A_{128}`$ |
| $`2^{15}`$ | $`200/C_{64}+162/C_{128}+21/A_{128}`$ |
| $`2^{16}`$ | $`604/C_{64}+41/C_{128}+21/A_{128}`$ |

The deployed convention is indexed-coordinate accounting, the paper's
Lemma C.3 ("Sparse Folds with an Indexed Query Budget"). Sparse-fold
work is zero. Each coordinate uses Akita's independent indexed XOF stream
on its fold seed and coordinate index. Response-nonce trials count as
ordinary oracle queries in $`Q`$, with no additional search penalty. The
conditional schedule sums above remain useful CWSS diagnostics; they are
not multiplied into the deployed per-query rate.

The normative field sites are `mu`, `tau`, the three `stage{s}/gamma`
vectors, every `stage{s}/round/t/challenge`, and the PCS field families in
Section 10.1. Their tree debits are respectively $`8/|K|`$, $`d/|K|`$,
$`m_s/|K|`$, and $`(3,5,3)/|K|`$ for the PIOP; PCS uses its declared degree or
coordinate-tree debit divided by exact $`|K|=(2^{64}-59)^2`$; the root
`EvaluationBatch` is a ten-child coordinate tree with debit $`9/|K|`$ and
$`g=4`$ (Section 10.1). Each field event
gets the least $`g\geq0`$ with $`2^{128}e\leq2^g`$, without a global site-count
multiplier. Nonzero field work uses Akita's native candidate range
$`[0,2^{g+7})`$, separate predicate squeeze, and inline canonical unsigned
LEB128 nonce; a zero-bit site has no nonce field. Section 10.2 gives the
PIOP's worst-case and geometric expected byte counts. PCS nonce encoding
is unchanged, and its realized size delta is measured from built proofs.
The projection seed has no work. The table covers all 29 admitted padded
shapes, Falcon-512 $`N^*=8,\ldots,2^{16}`$ and Falcon-1024 $`N^*=4,\ldots,2^{16}`$.
Its expanded inventory is
recomputed from the selected catalogs by
[`falcon_transcript_grinding.py`](scripts/falcon_transcript_grinding.py),
whose `--check` compares every site with the paper's vendored
`shapes.<shape>.indexed_allocation` evidence by path.

For each production shape, the MSIS term is recorded symbolically as
$`\sum_{I\in\mathcal I}\mathrm{Adv}_{\mathrm{MSIS}}(I)`$ over the exact
native occurrence multiset, including setup-prefix and compression matrices.
Each row has 51 occurrences: 7 inner $`L_\infty`$, 3 inner $`L_2^2`$, 9 outer,
6 open, and 26 compression matrices. The JSON report carries the complete
occurrence descriptors and table identities. Passing Akita's 128-bit ADPS16
table gate is a modeled attack-cost threshold, not probability $`2^{-128}`$.

For the specified transcript at the 29 admitted padded shapes, the maximum
address weight is
```math
\rho=\max\{\eta,\max_r2^{-g_r}e_r,\max_{j,g}C_{jg}^{-1}\}\leq2^{-128},
\qquad \eta=\max\{3\cdot2^{-256},(4/3)\vartheta\},\quad
\vartheta\leq2^{-135}.
```
Here $`C_{jg}`$ is the exact raw sparse support or its certified accepted
support lower bound, as appropriate for that route. The classical-ROM
knowledge bound is
```math
(Q+R_{\rm field}+R_{\rm coord})2^{-128}+\delta_X+\Delta_{\rm setup}
+\sum_I\mathrm{Adv}^{\rm exp}_{{\rm MSIS},I}(T_I).
```
$`R_{\rm field}`$ bounds untouched field-family addresses, including the
projection seed, completed by the final verifier; $`R_{\rm coord}`$ bounds
its untouched indexed sparse-coordinate streams. The table records full
verification upper bounds for both. The paper's Theorem 5.4
states this bound for every admitted $`N^*=4,\ldots,2^{16}`$. Here $`Q`$ includes failed grinding
attempts, response-nonce trials, and sampler queries, $`\delta_X`$ is the
expansion-oracle bad-event term, and $`\Delta_{\rm setup}`$ is the setup
exceptional-event term. The scheduled Module-SIS assumptions apply at the
theorem's expected reduction times $`T_I`$. Compressed public records and
transcript-state digests, including sparse-coordinate seeds, additionally
incur their binding collision advantage at the extractor's expected time;
a nonideal joint sampler incurs its error over the complete query experiment.
This is a per-counted-hash-evaluation knowledge rate, not an unconditional
numerical probability for a deployed hash function. Verifier completions
cannot be replaced by one. It does not establish QROM security.

The proof source is the Aerie paper: Section 4 (the protocol), Theorem 5.4 ("128-Bit Classical-ROM Knowledge Soundness")
and Appendix C (indexed-coordinate address accounting and tree allocation),
with the whole-protocol tree evidence vendored in
[`evidence/`](evidence/). The local table script reads and checks the
paper's event inventories against current catalog geometry.
No source-fixed or completed-grind conditioning is assumed in this accounting.

## 4. Reproduction

Run the stdlib-only script from the repository root:

```sh
python3 specs/scripts/falcon_1024_composition_ledger.py
python3 specs/scripts/falcon_transcript_grinding.py --check
```

It checks the three-stage numerators for $`N_{\rm sig}=2^{14},2^{15},2^{16}`$, verifies
the Falcon-1024 reference-profile bound and pinned projection certificate,
then audits the native PCS report and prints grouped planner estimates, rational
conditional CWSS upper bounds, and SIS occurrence counts. The separate grinding
check verifies the normative work allocation. The `aerie-falcon` tests in
`protocol/grinding.rs` compare the implementation's PIOP plan with the
script's `--piop-plan-json` output and require the planned shapes to equal
those `FalconVerifierSetup::supports` admits. PCS work is Akita-native.
