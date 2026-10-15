# Falcon composition ledger

This ledger records the three-stage PIOP terms and the clear-projection term
for Falcon-512, together with the audited four-group production catalog
estimates, conditional extraction upper bounds, and exact MSIS occurrence
data. Section 10.1 of [falcon.md](falcon.md#101-transcript-registry) supplies
the normative complete-tree work allocation. The paper proves the target
construction's 128-bit classical-ROM knowledge rate for every admitted padded
batch. The verifier enforces the
Section 10.1 PIOP work plan and Akita enforces its native PCS plan
(falcon.md Section 12.3).

The unground interactive terms below are retained as algebraic checks. They
are not the complete-tree debits used to allocate work. The
challenge field is $`K=\mathbb F_{p^2}`$, where $`p=2^{64}-59`$. See
[`falcon-1024-ledger.md`](falcon-1024-ledger.md) for Falcon-1024.

## 1. PIOP terms

Let $`R=d_{\rm coef}+d_{\rm sig}`$. Stage 1 has degree 3, stage 2 has degree 5, and
stage 3 has degree 3. Each stage has $`R`$ rounds, so the sumcheck-round term
is $`11R/|K|`$.

The zero-check and batching terms are:

| Source | Falcon-512 numerator over $`\vert K\vert `$ |
| --- | ---: |
| `basis16_product` at $`\tau`$: $`a_1`$, $`a_2`$ | $`2R`$ |
| `range_leaf` at $`r^{(1)}`$: $`a_0`$, $`u_0`$, $`u_1`$, $`u_2`$ | $`R+3d_{\rm sig}`$ |
| `sign_check`, `norm_equality`, `encoding_budget` at $`r^{(2)}`$ | $`R+2d_{\rm sig}`$ |
| Monomial projection weights from $`\mu\in K^8`$ | 8 |
| Three fresh stage combiners | 3 |
| **Total** | **$`15R+5d_{\rm sig}+11`$** |

The three stage combiners contribute one each. The projection batching
polynomial has degree at most 8, so its error is $`8/|K|`$. Zero-checks reuse
points only after all tested tables have been committed, as specified in
[`falcon.md`](falcon.md) Sections 7 and 9.

| $`N_{\rm sig}`$ | $`d_{\rm sig}`$ | Rounds per stage | PIOP numerator | $`\leq460`$? |
| ---: | ---: | ---: | ---: | --- |
| $`2^{14}`$ | 14 | 23 | 426 | yes |
| $`2^{15}`$ | 15 | 24 | 446 | yes; Falcon-512 reference profile |
| $`2^{16}`$ | 16 | 25 | 466 | no |

Thus the unground reference-profile bound $`\leq460/|K|`$ holds at Falcon-512's
specified $`N_{\rm sig}=2^{15}`$. It is not a bound for the full $`N_{\rm sig}^*=2^{16}`$ cap, and it is
not the deployed PIOP term; Section 3 charges each ground PIOP site at weight at
most $`2^{-128}`$. The script checks all three rows and the reference-profile bound.

## 2. Clear-projection term

The projection theorem's chosen Falcon-512 certificate row is charged once
per transcript. The source tables are fixed before the projection seed, so
no union bound over records, groups, or coordinates is required. The current
certificate gives:

| Leg | Term |
| --- | ---: |
| Norm leg | $`2^{-136.4416}`$ |
| Residual kill | $`3/25^{31}=2^{-142.3746}`$ |
| Residual wrap, order 136 | $`2^{-437.7379}`$ |
| Total | $`2^{-136.4182}`$ |

The script reads these values from
[`scripts/falcon_projection_certificate.json`](scripts/falcon_projection_certificate.json),
checks that the three legs sum to the recorded total, and charges a rational
upper bound for the total. The proof and integer-lifting argument are in
[`falcon-projection-theorem.md`](falcon-projection-theorem.md) and
[`falcon.md`](falcon.md), Section 9.

## 3. PCS, MSIS, and assembled error

The selected layout is $`[S_1]`$ width 14, $`[e]`$ width 14,
$`[a_0,a_1,a_2,\sigma]`$ width 4, and the final group
$`[u_0,u_1,u_2]`$ width 4. The production audit report has 14 width-14 rows and
33 width-4 rows, including 13 Falcon-512 grouped rows; the exact schedules
cover every admitted Falcon-512 batch size. Full report and validator:
[`scripts/falcon_pcs_catalog_audit.json`](scripts/falcon_pcs_catalog_audit.json)
and [`scripts/falcon_msis_audit.py`](scripts/falcon_msis_audit.py).

The grouped-opening payload and setup figures below are Akita planner
estimates for the shared four-group opening, not full proof-frame sizes.
Each of the four group commitments carries 16 terminal Q64 coefficients
(128 bytes); source profile geometry and the pinned compression plan are
recorded in [`falcon-pcs-extraction.md`](falcon-pcs-extraction.md), Section 4.

| $`N_{\rm sig}`$ | Grouped payload estimate | Setup field-element estimate | Fold levels | SIS occurrences: Inner $`L_\infty/L_2^2`$; Outer; Open; Compression |
| ---: | ---: | ---: | ---: | --- |
| $`2^{14}`$ | 63,273 B | 524,288 | 6 | 40: 6/3; 8; 5; 18 |
| $`2^{15}`$ | 64,692 B | 749,056 | 7 | 51: 7/3; 9; 6; 26 |
| $`2^{16}`$ | 64,842 B | 1,143,296 | 7 | 51: 7/3; 9; 6; 26 |

For a fixed fold prefix $`\ell`$, the conditional CWSS upper bound is
$`\kappa_\ell\leq\sum_g P_gB_g/L_g`$, using each group's polynomial count,
exact live-block count, and a certified challenge-support lower bound.
With $`C_{64}`$, $`C_{128}`$, $`C_{256}`$, and $`A_{128}=\lceil C_{128}p_0\rceil`$ defined in
[`falcon-pcs-extraction.md`](falcon-pcs-extraction.md), Section 2, the
$`L_\infty`$ routes use raw support and the selective $`L_2`$ routes use
$`A_{128}=502822610611069237129947393810277364817`$. The latter is derived
from the exact `claimed.p0` rational in LayerZero-Labs/akita at
`e023b6a46`, `scripts/operator_norm/d128/cert_d128_w31_gamma13.json`.
The audit computes rational upper bounds from the serialized schedule.
Their sums, including the terminal witness term, are:

| $`N_{\rm sig}`$ | Conditional schedule upper bound |
| ---: | ---: |
| $`2^{14}`$ | $`138/C_{64}+87/C_{128}+24/A_{128}`$ |
| $`2^{15}`$ | $`266/C_{64}+132/C_{128}+21/A_{128}`$ |
| $`2^{16}`$ | $`134/C_{64}+169/C_{128}+62/C_{256}+21/A_{128}`$ |

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

For each proof row, the MSIS term is recorded symbolically as
$`\sum_{I\in\mathcal I}\mathrm{Adv}_{\mathrm{MSIS}}(I)`$ over its exact
native occurrence multiset, including setup-prefix and compression matrices.
The table gives the count by matrix role and bound type; the JSON report carries
the complete descriptors and table identities. Passing Akita's 128-bit ADPS16
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
(Q+R_{\rm field}+R_{\rm coord})2^{-128}+\epsilon_{\rm terminal}(Q_H)
+\delta_X+\Delta_{\rm setup}+\epsilon_{\rm bind}+\delta_{\rm samp}
+\sum_I\mathrm{Adv}^{\rm exp}_{{\rm MSIS},I}(T_I).
```
$`R_{\rm field}`$ bounds untouched field-family addresses, including the
projection seed, completed by the final verifier; $`R_{\rm coord}`$ bounds
its untouched indexed sparse-coordinate streams. The table records full
verification upper bounds for both. Here $`Q`$ counts the original prover's transcript/sampler queries, including
failed grinding attempts and response-nonce trials, $`\delta_X`$ is the
expansion-oracle bad-event term, and $`\Delta_{\rm setup}`$ is the setup
exceptional-event term. The scheduled Module-SIS assumptions apply at the
theorem's expected reduction times $`T_I`$. The term $`\epsilon_{\rm bind}`$
is the binding collision advantage for compressed public records and
transcript-state digests, including sparse-coordinate seeds, at the
extractor's expected time. It is zero in an ideal collision-free binding
model; ideal random-oracle hashing alone does not make it zero.
The term $`\delta_{\rm samp}`$ is the error of a nonideal joint sampler over
the complete query experiment. It is zero when the joint sampler exactly
implements the stipulated ideal-oracle distribution. Neither specialization
is silently assumed for deployed hashes.
The $`2^{-128}`$ coefficient is the transcript/sampler per-counted-query
knowledge rate. The combined bound also includes the separate terminal
budget and symbolic losses; it is not an unconditional numerical probability
for a deployed hash function.
Verifier completions cannot be replaced by one. It does not establish QROM
security.

Let $`Q_H`$ count terminal-digest oracle queries in the reduction experiment,
including the final verifier, separately from the transcript/sampler budget
$`Q`$. Aerie enables Akita's `dev-protocol` last-block recovery: the verifier
checks 32-byte digests of the recovered $`E`$ and $`\mathbf t`$ blocks.
[LayerZero-Labs/akita at `e023b6a46143f7b23199ccecb2684192baf7b640`,
`book/src/how/verifying/terminal.md`, equation (11)](https://github.com/LayerZero-Labs/akita/blob/e023b6a46143f7b23199ccecb2684192baf7b640/book/src/how/verifying/terminal.md)
gives the classical-ROM collision and unopened-digest loss; the two checks
are in [`crates/akita-verifier/src/terminal/direct.rs`](https://github.com/LayerZero-Labs/akita/blob/e023b6a46143f7b23199ccecb2684192baf7b640/crates/akita-verifier/src/terminal/direct.rs#L368-L369).
The additive charge is
```math
\epsilon_{\rm terminal}(Q_H)
=\frac{\binom{Q_H}{2}+2Q_H}{2^{256}}
=\frac{Q_H(Q_H+3)}{2^{257}}.
```
This bound already covers both terminal digests in the shared four-group
opening. It is charged once, without a factor for digests or groups. It does
not change $`\rho`$ or prescribe terminal grinding. Substituting $`Q`$ or
$`Q+R_{\rm field}+R_{\rm coord}`$ for $`Q_H`$ requires explicit accounting of
the terminal oracle; a common-budget statement is not assumed here.

Exact terminal examples are
$`\epsilon_{\rm terminal}(0)=0`$,
$`\epsilon_{\rm terminal}(1)=2^{-255}`$,
$`\epsilon_{\rm terminal}(2)=5\cdot2^{-256}`$,
$`\epsilon_{\rm terminal}(2^{32})=2^{-193}+3\cdot2^{-225}`$, and
$`\epsilon_{\rm terminal}(2^{64})=2^{-129}+3\cdot2^{-193}`$.
For the explicitly assigned diagnostic budgets $`Q=Q_H=2^{64}`$, let
$`C=R_{\rm field}+R_{\rm coord}`$. The known partial contribution is
```math
(2^{64}+C)2^{-128}+\epsilon_{\rm terminal}(2^{64})
=2^{-64}+(2C+1)2^{-129}+3\cdot2^{-193}.
```
This assignment does not identify the two budgets in every attack. The
expansion, setup, binding, sampler, and MSIS terms remain to be added.
The current completion counts and coefficients are:

| $`N_{\rm sig}`$ | $`R_{\rm field}`$ | $`R_{\rm coord}`$ | $`C`$ | $`2C+1`$ |
| ---: | ---: | ---: | ---: | ---: |
| $`2^{14}`$ | 463 | 249 | 712 | 1425 |
| $`2^{15}`$ | 530 | 419 | 949 | 1899 |
| $`2^{16}`$ | 536 | 386 | 922 | 1845 |

Across all 29 admitted shapes, the maximum $`C`$ is 1207. The composition
script checks the exact terminal examples and the assigned-budget partial
identity for every shape, using `checked_rows()` from the grinding script.

The proof source is the Aerie paper: Section 4 (the protocol), Theorem 5.4 ("128-Bit Classical-ROM Knowledge Soundness")
and Appendix C (indexed-coordinate address accounting and tree allocation),
with the whole-protocol tree evidence vendored in
[`evidence/`](evidence/). The local table script reads and checks the
paper's event inventories against current catalog geometry.
No source-fixed or completed-grind conditioning is assumed in this accounting.

## 4. Reproduction

Run the stdlib-only script from the repository root:

```sh
python3 specs/scripts/falcon_composition_ledger.py
python3 specs/scripts/falcon_transcript_grinding.py --check
```

It checks the three-stage numerators for $`N_{\rm sig}=2^{14},2^{15},2^{16}`$, verifies
the Falcon-512 reference-profile bound and pinned projection certificate, then
validates the current native catalog report and prints the grouped-opening
estimates, rational conditional CWSS upper bounds, and SIS occurrence counts.
It also checks the exact terminal-digest examples and known partial
contributions at assigned budgets $`Q=Q_H=2^{64}`$, and prints the full
symbolic bound and completion counts. The separate grinding check verifies
the normative work allocation. The `aerie-falcon` tests in
`protocol/grinding.rs` compare the implementation's PIOP plan with the
script's `--piop-plan-json` output and require the planned shapes to equal
those `FalconVerifierSetup::supports` admits. PCS work is Akita-native.
