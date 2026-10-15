# Falcon four-group PCS extraction

Companion to [`falcon.md`](falcon.md), Sections 5 and 7. This document states
what the Akita opening must bind for the four-group layout and records the
audited production catalog costs and conditional extraction terms.

## 1. Commitment layout

The prover commits to four multi-polynomial Akita groups. A group contains any
number of polynomials with the same number of variables. Each group has one
commitment, one frozen `CommittedGroupProfile`, and one opening point.

| Group | Polynomials | Variables | Signed width | Honest value range | Opening point |
| --- | --- | ---: | ---: | --- | --- |
| $`[S_1]`$ | $`S_1`$ | $`d_{\rm coef} + d_{\rm sig}`$ | 14 | $`[-6144,6144]`$ | $`r^{(3)}`$ |
| $`[e]`$ | $`e`$ | $`2 + d_{\rm sig}`$ | 14 | $`[0,8191]`$ | $`r^{(3)}_e`$ |
| $`[a_0,a_1,a_2,\sigma]`$ | $`a_0`$, $`a_1`$, $`a_2`$, $`\sigma`$ | $`d_{\rm coef} + d_{\rm sig}`$ | 4 | $`[-8,7]`$ | $`r^{(3)}`$ |
| $`[u_0,u_1,u_2]`$ | $`u_0`$, $`u_1`$, $`u_2`$ | $`d_{\rm sig}`$ | 4 | $`[-8,7]`$ | $`r^{(3)}_{\mathrm{sig}}`$ |

A signed width $`w`$ admits honest integer values in
$`[-2^{w-1},2^{w-1}-1]`$. There is no power-of-two polynomial-count rule. One
opening proof covers the four groups at their respective points.

The final group is the width-4 budget group $`[u_0,u_1,u_2]`$, preserving
the canonical group order. Group parameters and planner estimates are
recorded in [`falcon.md`](falcon.md), Section 5.

The three points are suffixes of $`r^{(3)}`$. The $`e`$ point omits the first
$`d_{\rm coef} - 2`$ coefficient coordinates. The budget point contains only the
signature coordinates. The verifier derives the point arities and the group
profiles from the statement shape and the catalog selected for that shape.

## 2. Extraction statement and theorem boundary

**Conditional statement (per-group polynomial preservation).** If the pinned
Akita extraction chain applies to this heterogeneous four-group opening, then
the extractor obtains, for each group, committed polynomials that open to the
claimed values at that group's opening point and are bound to that group's
frozen `CommittedGroupProfile`.

The claim is per group. It does not assert that an extracted field
coefficient has an integer lift in the group's honest-value range. The signed
widths size honest witness production and PCS parameters; they do not prove
ranges against a malicious prover. Falcon soundness obtains every required
integer range from the PIOP predicates and the clear-projection integer-lift
argument in [`falcon.md`](falcon.md), Sections 4, 6, and 9.

The catalogs and `FoldSchedule::sis_occurrences` expose the exact native
schedule and SIS matrix occurrences, including setup-prefix and compression
matrices. They do not prove the heterogeneous extraction theorem. In
particular, the table's 128-bit ADPS16 gate is a modeled attack-cost threshold
for its SIS matrices; it is not a probability bound of $`2^{-128}`$.

For a fixed fold prefix, the conditional coordinate-wise special-soundness
debit is bounded by

```math
\kappa_{\mathrm{CWSS}}^{\mathrm{cond}}
\leq \sum_g\frac{P_g B_g}{L_g},
\qquad
C_g^{\rm raw}
=\binom{D_g}{m_{1,g}+m_{2,g}}
  \binom{m_{1,g}+m_{2,g}}{m_{1,g}}
  2^{m_{1,g}+m_{2,g}}.
```

Here $`P_g`$ is the number of polynomial claims in group $`g`$, $`B_g`$ is its exact
live-block count per claim, and $`L_g`$ is a lower bound on the support of the
sampled challenge. The raw family has $`m_{1,g}`$ entries of magnitude one and
$`m_{2,g}`$ entries of magnitude two in dimension $`D_g`$. For $`L_\infty`$ routes,
$`L_g=C_g^{\rm raw}`$. For selective $`L_2`$ routes, the ideal-XOF sampler is
uniform over the accepted subset, conditional on sampler success, so raw
support is not the denominator.

For the catalog's $`D=128`$, weight-31 selective $`L_2`$ family, use
[LayerZero-Labs/akita at `e023b6a46143f7b23199ccecb2684192baf7b640`,
`scripts/operator_norm/d128/cert_d128_w31_gamma13.json`](https://github.com/LayerZero-Labs/akita/blob/e023b6a46143f7b23199ccecb2684192baf7b640/scripts/operator_norm/d128/cert_d128_w31_gamma13.json),
field `claimed.p0`. Its exact rational
certifies a contained subset accepted by the strict fixed-point predicate at
threshold 13. Define

```math
\begin{aligned}
C_{64}&=\binom{64}{41}\binom{41}{31}2^{41}
=361715808335458406596740595030299770880,\\
C_{128}&=\binom{128}{31}2^{31}
=1046949673839956231392050469602223718400,\\
C_{256}&=\binom{256}{23}2^{23}
=2873108161471518200301057969538203648000,\\
A_{128}&=\lceil C_{128}p_0\rceil
=502822610611069237129947393810277364817.
\end{aligned}
```

Thus $`|\mathcal C_{\rm accepted}|\geq A_{128}`$ and
$`\log_2 A_{128}\geq128.5633`$. Two single-group rows, width-14 $`(26,1)`$ and
width-4 $`(25,4)`$, end in a $`D=64`$ selective $`L_2`$ terminal with 31 entries of
magnitude one and 11 of magnitude two. For it the audit uses
[LayerZero-Labs/akita at the same pin,
`scripts/operator_norm/d64/cert_d64_a31_b11_gamma18.json`](https://github.com/LayerZero-Labs/akita/blob/e023b6a46143f7b23199ccecb2684192baf7b640/scripts/operator_norm/d64/cert_d64_a31_b11_gamma18.json)
(threshold 18), giving
$`A_{64}=\lceil\binom{64}{42}\binom{42}{31}2^{42}p_0\rceil
=355332842887957753779026572075354372186`$, $`\log_2 A_{64}\geq128.0624`$.
No grouped Falcon production row has a $`D=64`$ $`L_2`$ route. The audit retains the certificate's exact
`claimed.p0` rational and derives the ceiling, with no external checkout
needed. It uses $`L_g=A_{128}`$ for selective $`L_2`$ folds and fails on an
uncertified $`L_2`$ family. The terminal fold has one witness source and uses
the same route-aware denominator. The rational upper bounds are reproduced by
[`scripts/falcon_msis_audit.py`](scripts/falcon_msis_audit.py); this arithmetic
does not rerun Akita's full certificate-generation and verification pipeline.

**Selected query convention: indexed-coordinate accounting.** The paper's
Lemma C.3 ("Sparse Folds with an Indexed Query Budget") uses independent
indexed XOF streams on each fold seed and coordinate index. Sparse atom
masses are bounded by $`1/C_{64}`$, $`1/C_{128}`$, $`1/C_{256}`$, or
$`1/A_{128}`$, each below $`2^{-128}`$ for the grouped production schedules.
Sparse-fold work is zero. Multiplicity is charged through the indexed
coordinate/query budget and the final verifier's coordinate completions.
Response-nonce trials count as ordinary transcript/sampler queries in $`Q`$,
with no additional search penalty. The conditional CWSS sums remain
diagnostics; they are not multiplied into the selected per-query rate.

The paper's conditional extraction and address-accounting argument for the
target construction requires groupwise CWSS and coordinate-fork extraction
premises. The current pinned Akita release does not supply a full
heterogeneous four-group theorem that establishes those premises and composes
them with the random-oracle reduction and each SIS reduction. Those upstream
theorem obligations remain open. Accordingly, the ledger records the rational
conditional CWSS upper bound and the symbolic MSIS sum
$`\sum_{I\in\mathcal I}\mathrm{Adv}^{\rm exp}_{{\rm MSIS},I}(T_I)`$ over the
native occurrence multiset $`\mathcal I`$ at the expected reduction times
$`T_I`$. It does not claim a proven numerical PCS extraction probability
from the planner's security gate. The
[Falcon-512 ledger](falcon-ledger.md#3-pcs-msis-and-assembled-error) and
[Falcon-1024 ledger](falcon-1024-ledger.md#3-pcs-msis-and-assembled-error)
include the additive dev-protocol terminal-digest charge with a separate
terminal-oracle query budget $`Q_H`$, plus the binding and joint-sampler
terms. The terminal charge covers both recovered blocks once and does not
change the indexed-coordinate work allocation.

## 3. Width basis

$`S_1`$ is centered modulo $`q=12{,}289`$, so every honest coefficient has absolute
value at most $`6{,}144`$. Width 14 is required: Falcon-512's norm bound
permits a coefficient of magnitude $`5{,}833`$, which exceeds the width-13
maximum $`4{,}095`$.

The four slack values satisfy
$`\sum_k e(k,i)^2=\delta\leq\lfloor\beta^2\rfloor`$. At Falcon-512,
$`\lfloor\beta^2\rfloor=34{,}034{,}726`$, so each component is at most
$`5{,}833`$. At Falcon-1024, $`\delta`$ may exceed $`8191^2=67{,}092{,}481`$,
reaching $`70{,}265{,}242`$. Taking the largest square first can produce a
component of $`8{,}382`$. The honest producer therefore caps its first
component at $`8{,}191`$ and skips choices whose remainder has an integer square
root above $`8{,}191`$. The exhaustive check is implemented by
[`scripts/falcon_slack_width_check.py`](scripts/falcon_slack_width_check.py).

All digit and budget polynomials fit width 4. The PIOP checks each
polynomial's exact alphabet: `a0` is in $`[-4,3]`$, `a1` and `a2` are in
$`[-8,7]`$, and `sigma` is in $`\{-1,1\}`$. The producer uses the 3/4/4-bit
split of the specification. Both budget low digits are in $`[-4,3]`$; `u2`
is in $`[-4,3]`$ at degree 512 and $`[-8,7]`$ at degree 1024.

The A/source basis is independent of the B/D opening basis. Width 4 fixes its
inner basis at log basis 4: one balanced base-16 digit represents the entire
interval $`[-8,7]`$. Width 14 delegates to
`Dense::inner_basis_range()` (raw range 3 through 16); Akita clips this
raw-source search to 3 through 14. Across the regenerated production profiles,
the width-14 source log basis / digit count is 5/3 at 13–15 variables, 14/1
at 16–19, and 8/2 at 20–26. The planner selects these counts without a three-digit cap; both
width-4 groups select log basis 4 and one digit in every row. The
representative basis selections are recorded in
[`falcon.md`](falcon.md), Section 5. Changing the source-range policy changes
the catalog policy digest and requires regeneration.

The opening parameters remain `log_basis = 3`, `log_open_bound = Some(64)`, and `opening_basis_range = (3, 6)`. Ring dimensions follow Akita fp64 `Dense`'s adaptive policy: exact A/B/D tuples for the first two fold levels (A over 64 through 2048, B and D over 64 through 256), then a monotone uniform suffix from $`\{64,128\}`$; each catalog row records its selection. The nominal 22 base-8 opening digits are unrelated to the source digit count. The presets' source widths remain `log_commit_bound = 14` and `4`.

## 4. Costs and catalog status

The width-14 catalog has 14 single-polynomial rows. The width-4 catalog has
33 rows: 14 single-group producer rows and 19 grouped production rows (13
first used by Falcon-512 and six additional Falcon-1024-specific keys).
Falcon-1024 uses 14 distinct grouped keys in total. The supported live batch
range is $`1`$ through $`65{,}536`$ for both parameter sets.
`FalconDegree::padded_records` rounds the live count up to a power of two and
then to the minimum $`4096/n`$: catalog rows cover padded counts
$`8,16,\ldots,65{,}536`$ and $`4,8,\ldots,65{,}536`$, respectively. Several padded counts share the same
group-shape key; the validator checks all 14 and 15 padded-count aliases.
The report records zero unsupported requests. Its full schedules,
planner estimates, group profiles, and exact SIS occurrence multisets are in
[`scripts/falcon_pcs_catalog_audit.json`](scripts/falcon_pcs_catalog_audit.json);
[`scripts/falcon_msis_audit.py`](scripts/falcon_msis_audit.py) validates that
report against its recorded Akita audit provenance (`e023b6a46`, the current
dependency pin) and the production group shapes. The D64 and D128
accepted-support certificates at that pin are linked in Section 2.

The grouped-opening planner estimates cover the shared four-group schedule.
The audit retains the payload and setup estimates recorded during catalog
regeneration. Its catalog-derived fields can be reproduced from the retained
artifacts without replanning. For production profiles, each group's complete B image has
$`s_g r_g D_g`$ coefficients, where $`s_g`$ is its outer slice count, $`r_g`$ its
outer matrix output rank, and $`D_g`$ its outer ring dimension. The audited
profiles use $`r_g=1`$, $`D_g\in\{64,128\}`$, and $`s_g\in\{1,2,4,8\}`$, so the
source images contain 128, 256, 512, or 1,024 field coefficients (1,024 through
8,192 bytes).
The pinned Q64 `CompressionChainPlan` uses ring dimensions $`[32,16]`$ and
terminates at 16 coefficients, or 128 bytes, for each group commitment. The
four commitment coefficient payloads therefore total 512 bytes; this excludes
wire framing and all opening-proof messages.

| Degree | Live batch range (padded minimum) | Grouped-opening payload estimate | Setup field-element estimate | Fold levels |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1–65,536 (8) | 58,926–64,842 B | 131,072–1,143,296 | 5–7 |
| 1024 | 1–65,536 (4) | 58,926–64,958 B | 131,072–1,503,744 | 5–7 |

These are Akita planner estimates for the grouped opening, not full Falcon
frame sizes or measured runtime. Per-shape grouped estimates, conditional
CWSS values, and exact SIS occurrence counts are recorded in the two
composition ledgers.

The audit contains 1,546 SIS matrix occurrences across its 47 rows (345 in
the width-14 family and 1,201 in width 4). These are catalog-row counts, not
the number of instances in one proof. A production proof's exact occurrence
counts are listed by shape in the ledgers. Each term remains symbolic as
$`\sum_{I\in\mathcal I}\mathrm{Adv}_{\mathrm{MSIS}}(I)`$: the ADPS16
128-bit planner gate is a modeled attack-cost threshold, not a probability.
The conditional CWSS upper bounds use exact rational arithmetic; they are
diagnostics and are not multiplied into the deployed rate. The paper's
assembled argument for the target construction is Theorem 5.4 ("128-Bit
Classical-ROM Knowledge Soundness"), with the conditional extraction and
address-accounting lemmas of Appendix C. Including the additive losses
specified in the ledgers, the classical-ROM knowledge bound is
```math
(Q+R_{\rm field}+R_{\rm coord})2^{-128}+\epsilon_{\rm terminal}(Q_H)
+\delta_X+\Delta_{\rm setup}+\epsilon_{\rm bind}+\delta_{\rm samp}
+\sum_I\mathrm{Adv}^{\rm exp}_{{\rm MSIS},I}(T_I).
```
The selected schedules cover every padded batch the verifier admits,
$`N^*=8,\ldots,2^{16}`$ at Falcon-512 and $`N^*=4,\ldots,2^{16}`$ at
Falcon-1024 (29 shapes). The ledgers define the separate transcript/sampler
budget $`Q`$ and terminal-digest budget $`Q_H`$, and the binding and nonideal
joint-sampler terms with their ideal-model specializations.
The $`2^{-128}`$ coefficient is the per-counted-query rate under the stated
extraction premises and scheduled Module-SIS assumptions at expected
reduction times $`T_I`$. The other terms stay explicit and the MSIS
advantages stay symbolic, so no unconditional numerical probability or
QROM security is claimed.
