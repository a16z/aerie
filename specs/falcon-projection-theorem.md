# Clear-projection soundness theorem for the Falcon protocol

Companion to `specs/falcon.md` (clear structured projections, projection
link, integer lifting and soundness). It discharges the projection-proof
obligation of that specification to the extent
stated in the Status below. Numerical claims are checked by
`specs/scripts/falcon_projection_certificate.py` (exact integer or rational
arithmetic; `--check` compares a fresh run against the pinned report
`specs/scripts/falcon_projection_certificate.json`). A fact certified by
the script is labelled with its check name, `[check: name]`.

(F §x) refers to a section of `specs/falcon.md`.

## Status

- **Residual leg: proved here.** Theorem 5.3 bounds the probability that a
  residual table $`\Delta\not\equiv0\pmod q`$ passes all 31 divisibility tests. The
  proof is elementary: a Gaussian even-moment bound for integer wrap and a
  fresh-core annihilation bound over $`\mathbb F_q`$. It imports nothing.
- **Exact lift and residual bound: proved here** (Lemmas 4.1 and 4.2). One
  global energy bound lifts every record's norm identity with no union over
  records. The resulting residual bound holds whether or not the signature
  is ring-valid.
- **Norm leg: proved here** (Theorem 3.1). The proof is elementary and
  imports nothing. It has four parts: half-propagation of large coordinates
  and exactness below thresholds (Lemmas A, B, E); a scalar Laplace bound
  through the Gaussian characteristic function and Hölder's inequality
  (Lemma C, Corollary C′); a comparison coupling on the unreduced integer
  chain (Lemma D); and the numeric certificate N1. The supremum
  $`\sup_wh_3(w)\leq21/40`$ behind Lemma C is certified in exact rational
  arithmetic.
- **Projection term: proved here, with numeric constants.** Theorem 6.1
  gives $`2^{-136}`$ per transcript for both parameter sets (certified value
  $`2^{-136.418}`$), with no union over records, fibers or cores.
- **Honest rejection: proved here, with numeric constants** (Section 7):
  at most $`2^{-129}+\varepsilon_{\mathsf{wrap}}`$.
- **Machine-checked in Lean** (Section 11): the statements of Sections 3 to
  7, for both parameter sets and every padded record count, with
  the bounds $`2^{-136}`$ for the projection term and $`2^{-129}+2^{-142}`$
  for honest rejection. The hypotheses H1 to H4 of Section 2 are hypotheses
  of the Lean theorems as well. The finer numeric values of this document
  are checked by the script only.

Provenance labels used below: **proved** (full proof here), **imported**
(external result with citation and status), **numeric** (verified by the
companion script), **modeled** (standard cryptographic modeling assumption,
for example the random oracle).

## 1. Exact obligation and success criterion

The obligation is a theorem for the exact projection distributions of
(F §6): 16 fresh ternary prefix cores and one fresh ternary terminal on the
norm source $`S`$, and three fresh uniform cores on $`\{-12,\ldots,12\}`$ on the
virtual residual $`\Delta`$. Each core is reused across its own tensor fibers. The
theorem must condition on all pre-seed commitments, cover every
$`N_{\rm sig}\leq2^{16}`$ and every padded $`N_{\rm sig}^\star`$, and give a bound that meets the
composition ledger without a union over records.

Success criterion: for every adversarial pre-seed commitment choice, the
probability over the transcript-derived cores that both public predicates of
(F §6) accept while the fixed source is not a Falcon witness is at most a
stated bound. "Falcon witness" means: every record satisfies the norm
identity (F §2.2) over $`\mathbb Z`$, every padded record is zero, and every
record satisfies the ring equation (F §2.2) over $`\mathbb Z`$ for some integer
quotient.

What would NOT count:

- a statement conditioned on honest witnesses (that is completeness,
  Section 7);
- a statement that assumes $`|S_1|\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor`$, the lifted norm
  identity or a PCS schedule source bound before the norm leg has been
  applied (circular; see Section 8);
- a divisibility test evaluated in $`\mathbb F_p`$, where $`q`$ is invertible.

## 2. Setting, notation and hypotheses

Throughout: $`p=2^{64}-59`$ (prime, **numeric**), $`q=12{,}289`$ (prime,
**numeric**), $`n\in\{512,1024\}`$, $`d_{\rm sig}=\log_2N_{\rm sig}^\star`$ with
$`2^{12}\leq nN_{\rm sig}^\star`$, $`N_{\rm sig}^\star\leq2^{16}`$, and $`\mathrm{ctr}_p:\mathbb F_p\to
(-p/2,p/2)`$ is the centered representative. Since $`p`$ is odd, the interval is
symmetric and $`\mathrm{ctr}_p(x)`$ is the unique integer of least
absolute value congruent to $`x`$. For a table $`X`$ over $`\mathbb F_p`$, its
energy is $`\mathcal E(X)=\lVert\mathrm{ctr}_p X\rVert_2^2`$.

| Constant | Falcon-512 | Falcon-1024 | Source |
| --- | ---: | ---: | --- |
| $`\lfloor\beta^2\rfloor`$ | $`34{,}034{,}726`$ | $`70{,}265{,}242`$ | (F §2.1) |
| $`\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor`$ | $`5{,}833`$ | $`8{,}382`$ | **numeric** |
| $`H_S=2^{16}\lfloor\beta^2\rfloor`$ | $`2^{41.02}`$ | $`2^{42.07}`$ | honest energy ceiling |
| $`\Theta`$ | $`5{,}092{,}300{,}183{,}991{,}442{,}303{,}232`$ | $`10{,}513{,}136{,}047{,}130{,}310{,}946{,}696`$ | (F §6.1), Section 7 |
| $`\eta_{\mathrm{proj}}`$ | $`2^{15}`$ | same | N1, Section 3.5 |
| $`Q_S=(\Theta+1)/\eta_{\mathrm{proj}}`$ | $`2^{57.1088}`$ | $`2^{58.1546}`$ | **numeric** |
| $`L_\Delta`$ | $`811{,}061{,}395`$ | $`1{,}648{,}068{,}606`$ | Lemma 4.2 |
| $`H_\Delta=2^{16}nL_\Delta^2`$ | $`2^{84.19}`$ | $`2^{87.24}`$ | Lemma 4.2 |

Here $`\mathsf{live}(i)=1`$ iff $`i\lt N_{\rm sig}`$, and is zero otherwise.

**Sources.** $`S_2`$ denotes the table $`\sigma M`$ on the hypercube. Per record $`i\lt N_{\rm sig}^\star`$ the norm source $`S`$ has $`2n`$ slots
(F §6): slot $`j`$ holds $`S_1(j,i)`$, slot $`n+\mathrm{int}(k)`$ holds $`e(k,i)`$ for $`k\in\{0,1\}^2`$,
and the other $`n-4`$ slots are zero. The residual source is the virtual table
(F §6)

```math
\Delta(j,i)=S_1(j,i)+(h_i\circledast S_2(\cdot,i))_j-c_{i,j}
\quad\text{over }\mathbb F_p .
\qquad(2.1)
```

Public inputs are zero-extended on padded rows. Their centered
coefficients satisfy $`|h_{i,j}|,|c_{i,j}|\leq6{,}144=(q-1)/2`$ (F §9).

**Projections.** The norm leg (F §6) applies, for each group
$`g\in\{0,1,2,3\}`$, four fresh prefix cores $`A_{g,1},\ldots,A_{g,4}`$ with 43
rows each, concatenates the four $`43`$-channel tables, and applies one fresh
terminal core $`T`$ with 156 rows: $`y_{\mathrm{norm}}\in\mathbb F_p^{156}`$. The residual leg
(F §6) applies three fresh cores $`B_1,B_2,B_3`$ with 31 rows each:
$`y_{\mathrm{res}}\in\mathbb F_p^{31}`$. Each core contracts its incoming channel with the
next radix block of the tensor index and is applied with the same
coefficients to every remaining fiber. "Fresh" means independent of every
other core and of the history.

**Hypotheses.** The theorem is a statement about the statistical projection
term alone. Every other failure mode is charged to its own ledger item
(F §9.5) and enters here as a conditioning hypothesis.

- **(H1) Core distribution (modeled).** Conditioned on the full pre-seed
  transcript, the specified cores are independent. Every entry of a norm core is
  independent with $`\Pr[-1]=\Pr[1]=\tfrac14`$, $`\Pr[0]=\tfrac12`$. Every entry
  of a residual core is independent and uniform on $`\{-12,\ldots,12\}`$. This
  models the per-row cSHAKE256 streams of (F §6.3) as a random oracle, with
  the exact bit-pair map (F §6.3) and the unbiased rejection map (F §6.3).
  In the Fiat–Shamir compilation the per-transcript bound proved below is
  multiplied by the adversary's random-oracle query count in the composition
  ledger.
- **(H2) Pre-seed source fixation (imported: Akita binding and extraction).**
  The witness commitments are absorbed before the seed (F §10.1). They bind fixed
  field tables for $`S_1`$, $`e`$, the $`S_2`$ source tables and the
  budget digits. No centered source bound is imported from any PCS schedule.
  The derived tables $`S_2`$, $`S`$ and $`\Delta`$ are therefore fixed before the
  seed.
- **(H3) PIOP extraction of alphabets and projections (charged to the
  range, sign, projection-link, reduction and ring-functional ledger items).** Conditioned on
  acceptance, except with those already ledgered errors: (a) the four $`S_2`$
  source tables lie in their exact alphabets, so
  $`|\mathrm{ctr}_pS_2(j,i)|\leq2{,}047`$ (F §4.1); (b) `projection_link`
  binds the revealed outputs to the true projections of the fixed sources
  over $`\mathbb F_p`$: $`y_{\mathrm{norm}}=\Pi_{\mathrm{norm}}(S)`$ and $`y_{\mathrm{res}}=\Pi_{\mathrm{res}}(\Delta)`$, where
  $`\Pi_{\mathrm{norm}}`$ and $`\Pi_{\mathrm{res}}`$ are the public linear maps of the two
  legs. Part (b) uses the downstream authentication of the three `projection_link`
  terminal claims: $`v_{S_1}`$ and $`v_e`$ in `s1_reduction`, `slack_reduction` and the PCS,
  $`v_\Delta`$ through the ring scalar and `ring_functional` against definition (2.1).
- **(H4) Post-seed identity extraction (charged to `norm_equality` and its
  sumcheck and batching ledger items).** Conditioned on acceptance, except
  with those errors, the norm identity (F §7.4) holds over
  $`\mathbb F_p`$ for every record $`i\lt N_{\rm sig}^\star`$:

```math
\sum_{j\lt n}S_1(j,i)^2+\sum_{j\lt n}S_2(j,i)^2+\sum_{k\in\{0,1\}^2}e(k,i)^2
=\lfloor\beta^2\rfloor\,\mathsf{live}(i)
\quad\text{in }\mathbb F_p .
\qquad(2.2)
```

Note on H3 and H4. H3(a) concerns pre-seed tables. H3(b) and H4 are used only
after the core probability has been bounded, to convert acceptance into the
two predicates on the true projections and into (2.2). Their failure while
the verifier accepts is charged uniformly over the seed, so the proof never
conditions the core distribution on later acceptance.

There is no quotient table. The protocol never commits, ranges or projects
an integer quotient $`V`$. Divisibility of $`\Delta`$ by $`q`$ replaces it.

## 3. Norm leg

This section proves the norm-leg bound from first principles. It imports no
analytic lemma. The chain is:

- Lemma A (half-propagation): a fresh ternary row keeps a large coordinate
  large with probability at least $`\tfrac12`$.
- Lemma B (exactness): a stage whose inputs are small reduces exactly
  modulo $`p`$.
- Lemma C and Corollary C′ (scalar Laplace bound): a fresh ternary core
  contracts the real energy of any fixed table by a factor below $`u`$ with
  probability at most $`K_{\mathrm L}^ce^{tu}`$, at $`t=3`$, $`K_{\mathrm L}=21/40`$.
- Lemma D (comparison coupling): independent comparison variables lie
  below the energy ratios of the *unreduced* integer chain.
- Lemma E (escape): Lemma A chained through a group's four prefix cores,
  and through the terminal.
- Theorem 3.1 (assembly): the case split under shared fibers and the
  numeric certificate N1.

Every numeric fact is labelled with the check of
`specs/scripts/falcon_projection_certificate.py` that certifies it, written
`[check: name]`. Per-degree checks appear under each configuration of the
pinned report; the others are global checks. No step is float-only.

### 3.1 Tables, stages and thresholds

Let $`x_1,\ldots,x_5`$ be the reshaped index of (F §6), with radices
$`r_1=2^{12}`$ and $`r_2,\ldots,r_5`$ from the radix rule there. For a group
$`g\lt 4`$ write

```math
T_{g,0}=S,
\qquad
T_{g,i}=A_{g,i}\,T_{g,i-1}\bmod p\quad(1\leq i\leq4),
\qquad
T_4=(T_{0,4}\,\|\,T_{1,4}\,\|\,T_{2,4}\,\|\,T_{3,4}),
\qquad
y_{\mathrm{norm}}=T\,T_4\bmod p .
\qquad(3.1)
```

Core $`A_{g,i}`$ contracts the incoming channel of $`T_{g,i-1}`$ ($`1`$ channel for
$`S`$, $`43`$ otherwise) together with the radix block $`x_i`$, and acts with the
same coefficients on every remaining fiber. The terminal $`T`$ contracts the
$`4\cdot43=172`$ channels of $`T_4`$ together with $`x_5`$. Call the table consumed
at stage $`i`$ the *stage-$`i`$ table* ($`S`$ at stage 0, $`T_{g,i}`$ at stages 1 to
3, $`T_4`$ at stage 4) and its consumer's fan-in $`f_i`$:

```math
(f_0,f_1,f_2,f_3,f_4)=(r_1,\;43r_2,\;43r_3,\;43r_4,\;172r_5).
\qquad(3.2)
```

Every core is a matrix with $`f_i`$ columns and $`43`$ (prefix) or $`156`$
(terminal) rows of independent ternary entries (H1). An output entry at
fiber $`\varphi`$ is $`\sum_{k\lt f_i}a_{j,k}X_{k,\varphi}`$, where $`k`$ runs over
(channel, digit of $`x_{i+1}`$) and $`X`$ is the input viewed as an
$`f_i\times(\text{fibers})`$ matrix.

The radices grow with $`N_{\rm sig}^\star`$, so over every supported $`N_{\rm sig}^\star\leq2^{16}`$
the fan-ins are largest at $`N_{\rm sig}^\star=2^{16}`$:

| | $`f_0`$ | $`f_1`$ | $`f_2`$ | $`f_3`$ | $`f_4`$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falcon-512 | $`4096`$ | $`688`$ | $`688`$ | $`688`$ | $`688`$ |
| Falcon-1024 | $`4096`$ | $`688`$ | $`688`$ | $`688`$ | $`1376`$ |

The script takes the maximum over every $`d_{\rm sig}`$ explicitly and checks that the
radix rule reproduces the geometry it certifies
[check: `spec_radices_match_geometry`].

**Thresholds.** Let $`{m_{\Theta}}=\lfloor\sqrt{\Theta}\rfloor`$ and

```math
\tau_5=2{m_{\Theta}}+1,
\qquad
\tau_i=2^{5-i}\tau_5\quad(0\leq i\leq5),
\qquad\text{so }\tau_4=2\tau_5 .
\qquad(3.3)
```

$`\tau_i`$ is the threshold of the stage-$`i`$ table, and $`\tau_5`$ that of $`y_{\mathrm{norm}}`$.
A table $`X`$ is *large* at threshold $`\tau`$ if some entry has
$`|\mathrm{ctr}_pX_{k,\varphi}|\geq\tau`$, and *small* otherwise. Write
$`L_{g,i}`$ for "$`T_{g,i}`$ is large at $`\tau_i`$", $`L_4`$ for "$`T_4`$ is large at
$`\tau_4`$" and $`L_5`$ for "$`y_{\mathrm{norm}}`$ is large at $`\tau_5`$". Since $`T_4`$ is a
concatenation, $`T_4`$ is small at $`\tau_4`$ exactly when every $`T_{g,4}`$ is.

| | Falcon-512 | Falcon-1024 |
| --- | ---: | ---: |
| $`\log_2{m_{\Theta}}`$ | $`36.054`$ | $`36.577`$ |
| $`\log_2\tau_0`$ | $`42.054`$ | $`42.577`$ |
| $`\log_2\tau_4=\log_2(2\tau_5)`$ | $`38.054`$ | $`38.577`$ |

The certificate checks, for $`i=0,\ldots,4`$ and both degrees,

```math
2(\tau_{i+1}-1)\lt \tau_i ,
\qquad(3.4)
```

```math
f_i(\tau_i-1)\leq\tfrac{p-1}2 ,
\qquad(3.5)
```

[check: `lemma_a_stage_0` to `lemma_a_stage_4` for (3.4),
`lemma_b_stage_0` to `lemma_b_stage_4` for (3.5)], and $`{m_{\Theta}}\lt \tau_5`$
[check: `accept_below_tau5`]. The margins
$`\log_2\frac{(p-1)/2}{f_i(\tau_i-1)}`$ of (3.5), stages 0 to 4, are
$`8.946,12.519,13.519,14.519,15.519`$ bits at $`n=512`$ and
$`8.423,11.996,12.996,13.996,13.996`$ bits at $`n=1024`$ (display of the exact
check). (3.4) holds with room to spare because each $`\tau_i`$ is exactly
twice $`\tau_{i+1}`$. The check guards the labelling: at stage 4 it fails as
soon as the terminal input threshold drops to $`\tau_4\leq2\tau_5-2`$. Since $`f_i\geq1`$, (3.5) also gives
$`\tau_i\leq(p+1)/2`$.

**Unreduced chain.** Let $`R^{(0)}=\mathrm{ctr}_pS`$, an integer table,
and for each group $`R_g^{(i)}=A_{g,i}R_g^{(i-1)}`$ over $`\mathbb Z`$. No
reduction and no threshold enters this definition.

### 3.2 Half-propagation and exactness

**Lemma A (half-propagation, proved).** Let $`X`$ be a table over
$`\mathbb F_p`$, fixed before a fresh ternary core with $`c`$ rows. Let
$`\tau_{\mathsf{in}}\leq(p+1)/2`$ and $`\tau_{\mathsf{out}}`$ satisfy
$`2(\tau_{\mathsf{out}}-1)\lt \tau_{\mathsf{in}}`$. If $`X`$ is large at
$`\tau_{\mathsf{in}}`$, the output is small at $`\tau_{\mathsf{out}}`$ with
probability at most $`2^{-c}`$.

*Proof.* Select, by a fixed rule, an entry $`(k,\varphi)`$ with
$`|\mathrm{ctr}_pX_{k,\varphi}|\geq\tau_{\mathsf{in}}`$. Fix one output
row $`j`$ and condition on its entries $`a_{j,k'}`$, $`k'\neq k`$. The row's value
at fiber $`\varphi`$ is $`v(s)=R+sX_{k,\varphi}`$ with $`s=a_{j,k}\in\{-1,0,1\}`$
and $`R`$ fixed. Let $`P=\{s:|\mathrm{ctr}_pv(s)|\leq\tau_{\mathsf{out}}-1\}`$.
If $`s`$ and $`s+1`$ both lie in $`P`$, then
$`d=\mathrm{ctr}_pv(s+1)-\mathrm{ctr}_pv(s)`$ is an integer
congruent to $`X_{k,\varphi}`$ with
$`|d|\leq2(\tau_{\mathsf{out}}-1)\lt \tau_{\mathsf{in}}\leq(p+1)/2`$. Hence
$`|d|\leq(p-1)/2`$, so $`d=\mathrm{ctr}_pX_{k,\varphi}`$ and
$`|\mathrm{ctr}_pX_{k,\varphi}|\lt \tau_{\mathsf{in}}`$, a contradiction. So
$`P`$ contains no two consecutive values: $`P`$ is one of $`\varnothing`$,
$`\{-1\}`$, $`\{0\}`$, $`\{1\}`$, $`\{-1,1\}`$. Under the law
$`(\tfrac14,\tfrac12,\tfrac14)`$ each has mass at most $`\tfrac12`$. So row $`j`$
is small at fiber $`\varphi`$ with probability at most $`\tfrac12`$. The rows
are independent and a small output needs every row small at $`\varphi`$.
$`\square`$

The lemma looks at one input entry and one fiber, so it holds however the
core is shared across fibers.

**Lemma B (exactness, proved).** If the stage-$`i`$ table $`X`$ is small at
$`\tau_i`$ and (3.5) holds, then for every realization of the consuming core
$`A`$, $`\mathrm{ctr}_p(AX\bmod p)=A\mathrm{ctr}_pX`$ over
$`\mathbb Z`$.

*Proof.* Each output entry of $`A\mathrm{ctr}_pX`$ is a sum of $`f_i`$
terms $`a_{j,k}\mathrm{ctr}_pX_{k,\varphi}`$ with $`|a_{j,k}|\leq1`$ and
$`|\mathrm{ctr}_pX_{k,\varphi}|\leq\tau_i-1`$. Its absolute value is at
most $`f_i(\tau_i-1)\leq(p-1)/2`$, so it is the centered representative of its
residue. $`\square`$

### 3.3 The scalar Laplace bound

For $`0\lt w\leq1`$ and $`t\gt 0`$ let $`G\sim \mathcal N(0,1)`$ and

```math
h_t(w)=\mathbb E\bigl|\cos\bigl(\sqrt{tw/2}\,G\bigr)\bigr|^{2/w} .
\qquad(3.6)
```

**Lemma C (Laplace bound, proved).** Let $`X\neq0`$ be a real
$`f\times {d_{\mathrm{col}}}`$ matrix with rows $`X_1,\ldots,X_f`$, and let $`a\in\{-1,0,1\}^f`$
have independent entries with law $`(\tfrac14,\tfrac12,\tfrac14)`$. Then

```math
\mathbb E\exp\Bigl(-t\,\frac{\lVert a^\top X\rVert_2^2}{\lVert X\rVert_2^2}\Bigr)
\leq\sup_{0\lt w\leq1}h_t(w).
\qquad(3.7)
```

*Proof.* For $`v\in\mathbb R^{d_{\mathrm{col}}}`$ and $`g\sim \mathcal N(0,I_{d_{\mathrm{col}}})`$,
$`\langle g,v\rangle\sim \mathcal N(0,\lVert v\rVert^2)`$, so
$`e^{-t\lVert v\rVert^2}=\mathbb E_g\cos(\sqrt{2t}\langle g,v\rangle)`$. Take
$`v=a^\top X/\lVert X\rVert`$ and put
$`\theta_k=\sqrt{2t}\langle g,X_k\rangle/\lVert X\rVert`$. Then
$`\cos(\sqrt{2t}\langle g,v\rangle)=\mathrm{Re}\prod_ke^{ia_k\theta_k}`$.
The integrand is bounded, so by Fubini we may average over $`a`$ first:
$`\mathbb E_ae^{ia_k\theta}=\tfrac12+\tfrac12\cos\theta=\cos^2(\theta/2)`$.
Hence the left side of (3.7) is
$`\mathbb E_g\prod_k\cos^2(\theta_k/2)`$.

Let $`w_k=\lVert X_k\rVert^2/\lVert X\rVert^2`$, so $`\sum_kw_k=1`$. Rows with
$`w_k=0`$ have $`\theta_k=0`$ and contribute a factor 1; drop them. Each
$`\theta_k/2`$ has law $`\sqrt{tw_k/2}\,G`$. The $`\theta_k`$ are correlated,
which the generalized Hölder inequality with exponents $`1/w_k`$ absorbs:

```math
\mathbb E_g\prod_k\cos^2(\theta_k/2)
\leq\prod_k\Bigl(\mathbb E\bigl|\cos(\theta_k/2)\bigr|^{2/w_k}\Bigr)^{w_k}
=\prod_kh_t(w_k)^{w_k}
\leq\sup_{0\lt w\leq1}h_t(w),
```

the last step because a weighted geometric mean is at most the largest
factor. $`\square`$

Two values of $`h_t`$ are explicit: $`h_t(1)=\mathbb E\cos^2(\sqrt{t/2}\,G)
=\tfrac12(1+e^{-t})`$, and $`h_t(w)\to(1+t)^{-1/2}`$ as $`w\to0`$ (Lemma C.3
below). The protocol uses

```math
t=3,
\qquad
K_{\mathrm L}=\frac{21}{40}.
\qquad(3.8)
```

At $`w=1`$, $`h_3(1)=(1+e^{-3})/2\lt 21/40`$ is equivalent to $`e^3\gt 20`$, which
holds because $`\sum_{j\leq10}3^j/j!\gt 20`$ [check: `e^3 > 20`]. The supremum
over all $`w`$ needs more: Lemma C.1 gives a series for $`h_t`$, and
Lemmas C.2 and C.3 turn it into a finite certificate. Write $`{\xi}=1/w\geq1`$
and $`\tilde h({\xi})=h_t(1/{\xi})=\mathbb E|\cos(\sqrt{t/(2{\xi})}\,G)|^{2{\xi}}`$.

**Lemma C.1 (Fourier series, proved).** For real $`{\xi}\geq1`$,

```math
\tilde h({\xi})=a_0({\xi})\Bigl[1+2\sum_{k\geq1}r_k({\xi})e^{-tk^2/{\xi}}\Bigr],
\qquad
r_k({\xi})=\prod_{j=1}^{k}\frac{{\xi}-j+1}{{\xi}+j},
\qquad
a_0({\xi})=\frac1\pi\int_0^\pi|\cos u|^{2{\xi}}du,
\qquad(3.9)
```

with $`|r_k({\xi})|\leq1`$, $`a_0(x+1)=a_0(x)\frac{2x+1}{2x+2}`$,
$`a_0(n)=\binom{2n}n4^{-n}`$ for integers $`n`$, and $`\log a_0`$ convex.

*Proof.* $`\phi(u)=|\cos u|^{2{\xi}}`$ is even, $`\pi`$-periodic and $`C^1`$ for
$`{\xi}\geq1`$, so its Fourier series
$`\phi(u)=a_0+2\sum_{k\geq1}a_k\cos(2ku)`$,
$`a_k=\frac1\pi\int_0^\pi\phi(u)\cos(2ku)\,du`$, converges absolutely and
uniformly. With $`\theta^2=t/(2{\xi})`$, $`\mathbb E\cos(2k\theta G)=e^{-2k^2\theta^2}
=e^{-tk^2/{\xi}}`$, and uniform convergence lets the expectation pass through the
sum. For the ratio, let $`\alpha=2{\xi}`$, $`I_k=\int_0^{\pi/2}\cos^\alpha u\cos(2ku)\,du`$
(so $`a_k=\frac2\pi I_k`$ by the symmetry $`u\mapsto\pi-u`$) and
$`\Psi(u)=\cos^{\alpha+1}u\,\sin((2k-1)u)`$. Product-to-sum formulas give

```math
\Psi'(u)=\tfrac12\cos^\alpha u\bigl[(\alpha+2k)\cos(2ku)-(\alpha-2k+2)\cos(2(k-1)u)\bigr],
```

and $`\Psi(0)=\Psi(\pi/2)=0`$, so $`(\alpha+2k)I_k=(\alpha-2k+2)I_{k-1}`$, that is
$`a_k/a_{k-1}=({\xi}-k+1)/({\xi}+k)`$. Each factor has absolute value at most 1:
$`0\leq {\xi}-j+1\leq {\xi}+j`$ when $`j\leq {\xi}+1`$, and $`j-{\xi}-1\lt {\xi}+j`$ otherwise. The
recurrence for $`a_0`$ is the Wallis reduction
$`\int_0^\pi|\cos|^{\alpha+2}=\frac{\alpha+1}{\alpha+2}\int_0^\pi|\cos|^\alpha`$, and
$`a_0(0)=1`$ gives the integer values. Log-convexity of
$`x\mapsto\int_0^\pi|\cos u|^{2x}du`$ is Hölder's inequality. $`\square`$

**Lemma C.2 (cells, proved).** For $`1\leq {\xi}_a\leq {\xi}\leq {\xi}_b`$,
$`\tilde h({\xi})\leq\sqrt{{\xi}_b/{\xi}_a}\,\tilde h({\xi}_a)`$.

*Proof.* Substituting $`u=\sqrt{t/(2{\xi})}\,g`$,
$`\tilde h({\xi})=\sqrt{{\xi}/(\pi t)}\int_{\mathbb R}|\cos u|^{2{\xi}}e^{-\xi u^2/t}du`$. Both
$`|\cos u|^{2{\xi}}`$ and $`e^{-\xi u^2/t}`$ are nonincreasing in $`{\xi}`$, and
$`\sqrt{{\xi}/(\pi t)}\leq\sqrt{{\xi}_b/{\xi}_a}\sqrt{{\xi}_a/(\pi t)}`$. $`\square`$

**Lemma C.3 (small $`w`$, proved).** For $`{\xi}\geq {\xi}_0\geq1`$,

```math
\tilde h({\xi})\leq(1+t)^{-1/2}+\sqrt{\frac{4t}{\pi^3{\xi}_0}}\;e^{-\pi^2{\xi}_0/(4t)} .
\qquad(3.10)
```

*Proof.* On $`|x|\leq\pi/2`$, $`\log\cos x\leq-x^2/2`$ because
$`(\log\cos)''=-\sec^2\leq-1`$. With $`x=\sqrt{t/(2{\xi})}\,G`$,
$`|\cos x|^{2{\xi}}\leq e^{-tG^2/2}`$ on $`|x|\leq\pi/2`$, and
$`\mathbb Ee^{-tG^2/2}=(1+t)^{-1/2}`$. Off that event $`|\cos x|^{2{\xi}}\leq1`$
and the event is $`|G|\gt a=\pi\sqrt{{\xi}/(2t)}`$. The Mills bound
$`\Pr[|G|\gt a]\leq\sqrt{2/\pi}\,e^{-a^2/2}/a`$ gives
$`\sqrt{4t/(\pi^3{\xi})}\,e^{-\pi^2{\xi}/(4t)}`$, which decreases in $`{\xi}`$. $`\square`$

As $`{\xi}\to\infty`$ the right side tends to $`(1+t)^{-1/2}`$; together with
dominated convergence this gives the limit quoted above.

**Certified supremum (numeric, exact rational arithmetic).** At $`t=3`$ the
script certifies

```math
\sup_{0\lt w\leq1}h_3(w)\leq U=\frac{262447}{500000}=0.524894,
\qquad
K_{\mathrm L}-U=\frac{53}{500000}\approx1.06\cdot10^{-4}
\qquad(3.11)
```

[check: `Lemma C: sup_(0<w<=1) h_t(w) <= U = 262447/500000 <= K = 21/40 at t = 3`]
as follows.

- $`U`$ is $`\max(\tilde h(1),(1+t)^{-1/2})`$ rounded up at $`10^{-6}`$.
  $`\tilde h(1)=(1+e^{-3})/2=0.5248935342\ldots`$; the certified upper
  enclosure of $`\tilde h(1)`$ is within $`2^{-108}`$ of it.
- $`{\xi}\geq4`$: (3.10) at $`{\xi}_0=4`$ gives at most $`0.511590\lt U`$ (pinned as
  `laplace.small_w_upper`). By hand:
  $`\tfrac12+\frac{\sqrt3}{\pi^{3/2}}e^{-\pi^2/3}<\tfrac12+\tfrac13\cdot\tfrac1{20}
  =\tfrac{31}{60}<\tfrac{21}{40}`$.
- $`1\leq {\xi}\leq4`$: 40 consecutive cells $`[{\xi}_a,{\xi}_b]`$ with dyadic endpoints on
  the $`2^{-40}`$ grid, starting at $`{\xi}_a=1`$, each with
  $`({\xi}_b/{\xi}_a)\,\overline h({\xi}_a)^2\leq U^2`$, where $`\overline h({\xi}_a)`$ is an
  upper bound for $`\tilde h({\xi}_a)`$. Lemma C.2 then bounds $`\tilde h`$ by $`U`$ on
  the cell. The cell list is pinned by its SHA-256 (`laplace.cells_sha256`).
- $`\overline h({\xi}_a)`$ evaluates (3.9) in `Fraction` arithmetic. $`a_0({\xi}_a)`$ is
  bounded above by shifting with the recurrence to $`{\xi}_a+2048`$ and taking the
  chord of the log-convex $`a_0`$ between the neighbouring integers (a chord
  of a convex function lies above it). Each term $`r_k e^{-tk^2/{\xi}}`$ uses an
  upper enclosure of $`e^{-y}`$ when $`r_k\gt 0`$ and a lower enclosure when
  $`r_k\lt 0`$. The sum stops at the first $`k_0`$ with $`tk_0^2/{\xi}\geq64`$ and
  $`tk_0/{\xi}\geq1`$; the rest is at most $`\sum_{k\geq k_0}e^{-tk_0k/{\xi}}`$, a
  geometric tail. $`e^y`$ is enclosed by halving to $`[0,1]`$, a 40-term Taylor
  sum with its Lagrange tail bound, and repeated squaring with outward
  rounding to the $`2^{-320}`$ grid. $`\pi`$ is enclosed by Machin's formula with
  alternating-series bounds, and square roots by `math.isqrt` rounded up.
  Self-tests compare the $`\pi`$ and $`e^3`$ enclosures with independent bounds
  [check: `pi enclosure lies inside (333/106, 355/113)`,
  `exp(3) enclosure is consistent with ...`].

The script binds $`t`$ and $`K_{\mathrm L}`$ into one `LaplacePair(t, K)`. Its constructor
recomputes the certificate at that $`t`$ and refuses any $`K_{\mathrm L}`$ below the
certified $`U(t)`$. Every function of the grid engine takes the pair, never
$`t`$ or $`K_{\mathrm L}`$ alone, so changing one without the other cannot pass.

**Corollary C′ (core contraction, proved; constants numeric).** Let $`X\neq0`$
be a real table fixed before a fresh ternary core $`A`$ with $`c`$ rows sharing
its coefficients across fibers, and let $`V=\lVert AX\rVert_2^2/\lVert
X\rVert_2^2`$ (Frobenius norms, $`X`$ viewed as an $`f\times(\text{fibers})`$
matrix). With $`t=3`$ and $`K_{\mathrm L}=21/40`$, for every $`u\geq0`$,

```math
\Pr[V\leq u]\leq F_c(u):=\min\bigl(1,K_{\mathrm L}^ce^{tu}\bigr).
\qquad(3.12)
```

*Proof.* $`V=\sum_{j\lt c}\lVert a_j^\top X\rVert^2/\lVert X\rVert^2`$ over the
independent rows $`a_j`$. By Lemma C and (3.11),
$`\mathbb Ee^{-tV}=\prod_j\mathbb E e^{-t\lVert a_j^\top X\rVert^2/\lVert
X\rVert^2}\leq K_{\mathrm L}^c`$. Markov's inequality on $`e^{-tV}`$ at $`e^{-tu}`$ gives
(3.12). $`\square`$

$`F_c`$ is a distribution function on $`[0,\infty)`$ with an atom $`K_{\mathrm L}^c`$ at $`0`$.
Write $`Z_c`$ for a random variable with law $`F_c`$.

### 3.4 Comparison coupling

**Lemma D (comparison coupling, proved).** Fix $`S`$ with $`\mathcal E(S)\gt 0`$. Enlarge
the probability space by one independent uniform variable $`\xi`$ per core,
independent of the cores. There are random variables $`Z^{(g,i)}`$
($`g\lt 4`$, $`1\leq i\leq4`$) and $`Z^T`$ such that:

1. the cores keep their H1 law;
2. $`Z^{(g,i)}\sim Z_{43}`$, $`Z^T\sim Z_{156}`$, and all 17 are independent;
3. $`Z^{(g,i)}`$ is a function of group $`g`$'s cores and uniforms only;
4. $`Z^T`$ is independent of all prefix cores and prefix uniforms;
5. $`W_g:=\lVert R_g^{(4)}\rVert^2/\lVert R^{(0)}\rVert^2\geq\prod_{i=1}^4Z^{(g,i)}`$
   for every $`g`$;
6. if $`\mathrm{ctr}_pT_4\neq0`$ then
   $`V_T:=\lVert T\mathrm{ctr}_pT_4\rVert^2/\lVert\mathrm{ctr}_pT_4\rVert^2\geq Z^T`$.

*Proof.* Process the prefix cores in a fixed order, then the terminal. At a
prefix core $`A_{g,i}`$ let $`\mathcal H`$ be everything processed before it
(cores and uniforms). Define
$`V_{g,i}=\lVert R_g^{(i)}\rVert^2/\lVert R_g^{(i-1)}\rVert^2`$ if
$`R_g^{(i-1)}\neq0`$ and $`V_{g,i}=+\infty`$ otherwise. If $`R_g^{(i-1)}=0`$, put
$`U_{g,i}=\xi_{g,i}`$. Otherwise $`R_g^{(i-1)}`$ is a nonzero function of
$`\mathcal H`$ and $`A_{g,i}`$ is independent of $`\mathcal H`$, so the conditional
distribution function $`G(\cdot\mid\mathcal H)`$ of $`V_{g,i}`$ depends on
$`\mathcal H`$ only through $`R_g^{(i-1)}`$. Corollary C′ applied to the fixed
real table $`R_g^{(i-1)}`$ gives $`G(u\mid\mathcal H)\leq F_{43}(u)`$ for all
$`u`$. Take the distributional transform

```math
U_{g,i}=G(V_{g,i}^-\mid\mathcal H)+\xi_{g,i}\bigl(G(V_{g,i}\mid\mathcal H)-G(V_{g,i}^-\mid\mathcal H)\bigr).
```

Conditional on $`\mathcal H`$, $`U_{g,i}`$ is uniform on $`(0,1)`$ and
$`G^{-1}(U_{g,i}\mid\mathcal H)=V_{g,i}`$ almost surely (Rüschendorf's
distributional transform). So $`U_{g,i}`$ is uniform and independent of
$`\mathcal H`$. Put $`Z^{(g,i)}=F_{43}^{-1}(U_{g,i})`$ with the left-continuous
quantile. In both cases $`Z^{(g,i)}\sim Z_{43}`$ independent of $`\mathcal H`$.
When $`R_g^{(i-1)}\neq0`$, $`G\leq F_{43}`$ pointwise gives
$`F_{43}^{-1}\leq G^{-1}`$, so $`Z^{(g,i)}\leq V_{g,i}`$; when $`R_g^{(i-1)}=0`$
this is trivial. Nothing about the cores is resampled, which is 1. Each new
variable is independent of $`\mathcal H`$, which contains every
earlier one, which is 2. $`G`$ depends only on $`R_g^{(i-1)}`$, a function of
group $`g`$'s cores (and the fixed $`S`$), and $`\xi_{g,i}`$ belongs to group $`g`$,
which is 3.

For 5: if every $`R_g^{(i-1)}`$, $`i\leq4`$, is nonzero then $`W_g=\prod_iV_{g,i}
\geq\prod_iZ^{(g,i)}`$. Otherwise let $`i`$ be the first stage with
$`R_g^{(i)}=0`$. Then $`V_{g,i}=0`$, so $`Z^{(g,i)}=0`$ and both sides are $`0`$.

The terminal is the same construction with $`V_T`$ (set to $`+\infty`$ when
$`\mathrm{ctr}_pT_4=0`$), the fixed input table $`\mathrm{ctr}_pT_4`$,
which is a function of the prefix cores, and $`F_{156}`$. That gives 6, and
since $`U_T`$ is independent of everything processed before it, 4. $`\square`$

The lemma is stated for the unreduced chain $`R_g^{(i)}`$ and for the actual
terminal input $`\mathrm{ctr}_pT_4`$, and it never conditions on an
exactness event. This is essential. Conditioning a core on "its output is
small", or on any other event that depends on the core, changes the core's
law. Corollary C′ would then no longer apply, and the small-ball
probability could be larger than $`F_c`$. Exactness enters only in the case
split below, as a deterministic implication on a fixed outcome.

### 3.5 Case split and the norm-leg bound

For a group $`g`$ let

```math
F_g=\bigcup_{i=1}^{4}\bigl(L_{g,i-1}\wedge\neg L_{g,i}\bigr),
\qquad
F_T=L_4\wedge\neg L_5 ,
\qquad(3.13)
```

where $`L_{g,0}`$ ("$`S`$ is large at $`\tau_0`$") is fixed. $`F_g`$ says that some
prefix core of group $`g`$ turns a large input into a small output, and $`F_T`$
says the terminal does.

**Lemma E (escape probabilities, proved).**
$`\Pr[F_g]\leq\varepsilon_F:=1-(1-2^{-43})^4`$ for each $`g`$, and
$`\Pr[F_T]\leq2^{-156}`$.

*Proof.* At stage $`i`$, conditional on the history before $`A_{g,i}`$, the
input $`T_{g,i-1}`$ is fixed and $`A_{g,i}`$ is fresh. By Lemma A with (3.4),
$`\Pr[L_{g,i-1}\wedge\neg L_{g,i}\mid\text{history}]\leq2^{-43}`$. So the
probability that no transition occurs at stage $`i`$, given none occurred
before, is at least $`1-2^{-43}`$. Multiplying over $`i=1,\ldots,4`$ gives
$`\Pr[\neg F_g]\geq(1-2^{-43})^4`$. For $`F_T`$, condition on the prefix cores;
$`T_4`$ is then fixed and $`T`$ fresh, and Lemma A with 156 rows gives
$`2^{-156}`$. $`\square`$

$`\varepsilon_F=2^{-41.000}`$ is pinned exactly as `laplace.escape_eps_F`, and
$`2^{-156}\lt 2^{-136}`$ [check: `Lemma A terminal escape 2^-156 fits 2^-136`].

**Case split under shared fibers.** All four groups read the same fixed
source $`S`$ (they share its fibers), and every core is reused across its own
fibers. Fix an outcome of the cores on which $`y_{\mathrm{norm}}`$ is accepted, that is
$`\mathcal E(y_{\mathrm{norm}})\leq \Theta`$.

- *Case 1: $`T_4`$ is large at $`\tau_4`$.* Acceptance gives
  $`|\mathrm{ctr}_p(y_{\mathrm{norm},j})|\leq m_{\Theta}\leq\tau_5-1`$ for every $`j`$, so $`y_{\mathrm{norm}}`$ is small
  at $`\tau_5`$ and the outcome lies in $`F_T`$.
- *Case 2: $`T_4`$ is small at $`\tau_4`$.* Then every $`T_{g,4}`$ is small, and
  by Lemma B at stage 4, $`\mathrm{ctr}_p(y_{\mathrm{norm}})=T\mathrm{ctr}_pT_4`$
  over $`\mathbb Z`$. So $`\mathcal E(y_{\mathrm{norm}})=V_T\lVert\mathrm{ctr}_pT_4\rVert^2`$ when
  $`\mathrm{ctr}_pT_4\neq0`$. Split the groups:
  - *Case 2a: $`\neg F_g`$.* $`T_{g,4}`$ is small. If $`T_{g,3}`$ were large the
    stage-4 transition would lie in $`F_g`$, so $`T_{g,3}`$ is small, and
    backwards in the same way $`T_{g,2}`$, $`T_{g,1}`$ and $`S`$ are small at their
    thresholds. Lemma B at stages $`0,\ldots,3`$ and induction on $`i`$ give
    $`\mathrm{ctr}_pT_{g,i}=R_g^{(i)}`$. In particular
    $`\lVert\mathrm{ctr}_pT_{g,4}\rVert^2=\lVert R_g^{(4)}\rVert^2
    =\mathcal E(S)\,W_g`$.
  - *Case 2b: $`F_g`$.* Nothing is claimed; the group's energy is dropped
    (it is nonnegative).

  Concatenation adds energies, so with $`W'_g=W_g\mathbf 1[\neg F_g]`$,
  $`\lVert\mathrm{ctr}_pT_4\rVert^2\geq \mathcal E(S)\sum_gW'_g`$, and

  $`\displaystyle \Theta\geq \mathcal E(y_{\mathrm{norm}})\geq V_T\,\mathcal E(S)\sum_gW'_g\geq Q_S\,Z^T\sum_g\Bigl(\mathbf 1[\neg F_g]\prod_{i=1}^4Z^{(g,i)}\Bigr)`$

  when $`\mathcal E(S)\geq Q_S`$, by Lemma D (5) and (6). If
  $`\mathrm{ctr}_pT_4=0`$ the sum $`\sum_gW'_g`$ is $`0`$ and the same
  conclusion holds. So $`Z^T\sum_g\tilde Z_g\leq \Theta/Q_S=\eta_{\mathrm{proj}}\,\Theta/(\Theta+1)\lt \eta_{\mathrm{proj}}`$,
  where $`\tilde Z_g=\mathbf 1[\neg F_g]\prod_iZ^{(g,i)}`$.

Case 1 does not use the prefix, and Case 2 uses only deterministic
consequences of smallness. The sharing of $`S`$ enters in two places. First,
whether $`S`$ is large is the same for all groups, and it is fixed, so it does
not correlate the groups. Second, Case 2a derives "$`S`$ is small" separately
for each group without assuming it. Sharing a core across its fibers is
harmless, because Lemma A looks at one fiber and Corollary C′ at the
whole table.

**Numeric fact N1 (numeric, exact rational arithmetic).** Let $`Z^T\sim Z_{156}`$
and let $`Y_0,\ldots,Y_3`$ be independent of it and of each other, each with
distribution function $`\min(1,F_P+\varepsilon_F)`$, where $`F_P`$ is the law of
a product of four independent $`Z_{43}`$. Then

```math
2^{-156}+\Pr\Bigl[Z^T\sum_{g\lt 4}Y_g\lt \eta_{\mathrm{proj}}\Bigr]\leq\tfrac78\,2^{-136},
\qquad
\eta_{\mathrm{proj}}=2^{15}.
\qquad(3.14)
```

The script certifies (3.14) at the larger threshold
$`\eta_{\mathrm{grid}}=(1/1024)^5(33/32)^{1470}=2^{15.2594}`$, a point of its
grid, and $`1470`$ is the largest grid exponent for which the certified bound
fits. Since $`\eta_{\mathrm{proj}}\leq\eta_{\mathrm{grid}}`$
[check: `eta_proj_at_most_grid_threshold`], the event in (3.14) is contained
in the certified one. A power of two keeps the threshold of the theorem
independent of the script's grid, at a cost of $`0.26`$ bits of lift slack.
The certified left side, at $`\eta_{\mathrm{grid}}`$, is $`2^{-136.4416}`$ (pinned exactly as `norm_err`;
the check is the internal invariant of `mixed_tail`, which fails the run if
it does not hold). At exponent $`1471`$ the bound is $`2^{-136.14}`$, above
$`\tfrac78\,2^{-136}=2^{-136.193}`$. The factor $`\tfrac78`$ is the script's
margin for choosing the exponent. Theorem 3.1 states the round bound
$`2^{-136}`$, which (3.14) implies.

The script computes its bound with an exact grid engine.
Each step rounds toward a *smaller* random variable, so every computed
probability of falling below a threshold is an upper bound:

- `distribution(c, lap)` defines a grid variable on
  $`\{0\}\cup\{AR^j:j\geq0\}`$, with $`A=1/1024`$ and $`R=33/32`$, and masses in
  units of $`2^{-192}`$. Let $`\mathrm{cdf}(j)`$ be the integer ceiling of
  $`2^{192}K_{\mathrm L}^ce^{tAR^j}`$, capped at $`2^{192}`$, with $`e^x`$ rounded up by a
  128-term Taylor sum and an exact tail bound. The zero atom is
  $`\mathrm{cdf}(0)`$ and the mass at $`AR^j`$ is
  $`\mathrm{cdf}(j+1)-\mathrm{cdf}(j)`$. So the grid distribution function on
  $`[0,A)`$ is at least $`F_c(A)`$ and on $`[AR^j,AR^{j+1})`$ at least
  $`F_c(AR^{j+1})`$; in both cases it is at least $`F_c`$, and the grid variable
  is stochastically below $`Z_c`$
  [check: `exp_grid_upper(j) >= exp_upper(t A R^j) at j in {0,100,250}`].
- The prefix product of grid values is exact on the grid
  $`A^4R^{\mathbb Z}`$, and exact convolution (packed integer multiplication,
  checked against schoolbook convolution) keeps it exact
  [check: `packed convolution matches schoolbook`].
- The boosted zero atom adds $`\lceil\varepsilon_F2^{768}\rceil`$ mass at zero
  and removes it from the largest grid values. Moving mass from the top
  down to zero turns a distribution function $`F`$ into exactly
  $`\min(1,F+\delta)`$, which dominates $`\min(1,F_P+\varepsilon_F)`$.
- `grid_sum` replaces $`AR^a+AR^b`$ by the largest grid point not above it,
  $`AR^{\max(a,b)+s}`$ with $`s`$ the largest integer satisfying
  $`R^s\leq1+R^{-|a-b|}`$ (for $`|a-b|\gt 200`$, $`s=0`$, which is correct because
  $`R^{-200}\lt R-1`$, asserted at import), and keeps $`0+x=x`$. This is a
  nondecreasing map that never increases the value.
- Monotone maps of independent variables preserve the stochastic order.
  The final product with the terminal is exact on $`A^5R^{\mathbb Z}`$.

**Theorem 3.1 (norm leg, proved; constants numeric).** Fix a norm source
$`S`$ with $`\mathcal E(S)\geq Q_S`$ before the seed. Then

```math
\Pr_{\text{cores}}\Bigl[\sum_{j\lt 156}\mathrm{ctr}_p(y_{\mathrm{norm},j})^2\leq \Theta\Bigr]
\leq2^{-136}
\quad\bigl(\text{certified }\varepsilon_S=2^{-136.4416}\bigr),
```

for both parameter sets, every $`N_{\rm sig}\leq2^{16}`$ and every radix tuple of
(F §6).

*Proof.* By the case split, acceptance implies $`F_T`$ or
$`Z^T\sum_g\tilde Z_g\lt \eta_{\mathrm{proj}}`$. Lemma E bounds $`\Pr[F_T]`$ by $`2^{-156}`$. For
the second event:

- The variables $`\tilde Z_g`$ are independent across $`g`$. By Lemma D (3),
  $`\tilde Z_g`$ is a function of group $`g`$'s cores and uniforms, since $`F_g`$
  depends only on those cores and the fixed $`S`$.
- For $`x\geq0`$, $`\Pr[\tilde Z_g\leq x]\leq\Pr[\prod_iZ^{(g,i)}\leq x]+\Pr[F_g]
  \leq F_P(x)+\varepsilon_F`$, by Lemma D (2) and Lemma e. So $`\tilde Z_g`$
  is stochastically above $`Y_g`$.
- $`Z^T`$ has law $`Z_{156}`$ and is independent of $`(\tilde Z_g)_g`$ by
  Lemma D (4).

Coupling each $`\tilde Z_g`$ above $`Y_g`$ by quantiles and using monotonicity
of $`(z,y_0,\ldots,y_3)\mapsto z\sum_gy_g`$ gives
$`\Pr[Z^T\sum_g\tilde Z_g\lt \eta_{\mathrm{proj}}]\leq\Pr[Z^T\sum_gY_g\lt \eta_{\mathrm{proj}}]`$. N1 bounds the sum
of the two terms. The fan-in checks (3.5) use the largest fan-ins over every
$`N_{\rm sig}^\star\leq2^{16}`$, the thresholds depend only on $`\Theta`$, and a radix-1
core (a zero-bit block at small $`N_{\rm sig}^\star`$) is still a fresh core of the
certified width. Lemmas A, B and D hold for any number of fibers.
$`\square`$

No union over records, fibers or coordinates appears. The only unions are
over the four prefix cores of a group (inside $`\varepsilon_F`$) and over
$`F_T`$.

## 4. Exact lift and residual bound

**Lemma 4.1 (exact lift, proved; constants numeric).** Assume H3(a) and
(2.2), and $`\mathcal E(S)\lt Q_S`$. Then for every record $`i`$ the identity (2.2) holds
over $`\mathbb Z`$ for the centered representatives. Consequently
$`|\mathrm{ctr}_pS_1(j,i)|,|\mathrm{ctr}_pe(k,i)|
\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor`$, $`\lVert\mathrm{ctr}_pS_2(\cdot,i)\rVert_2^2
\leq \lfloor\beta^2\rfloor`$, and every padded record $`i\geq N_{\rm sig}`$ has $`S_1(\cdot,i)`$,
$`S_2(\cdot,i)`$ and $`e(\cdot,i)`$ identically zero.

*Proof.* The integer left side of (2.2) for record $`i`$ is nonnegative and at
most

```math
\mathcal E(S)+n\cdot2{,}047^2\lt Q_S+n\cdot2{,}047^2\lt p,
\qquad(4.1)
```

where the last inequality is the certificate's `lift` check (**numeric**:
slack $`6.89`$ bits at $`n=512`$ and $`5.85`$ bits at $`n=1024`$). It is congruent
modulo $`p`$ to $`\lfloor\beta^2\rfloor\,\mathsf{live}(i)\in[0,p)`$, so the two integers are
equal. The coordinate bounds follow from $`\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor^2\leq
\lfloor\beta^2\rfloor<(\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor+1)^2`$. A padded record has a zero sum of integer
squares. $`\square`$

One global energy bound controls every record's nonnegative contribution,
so there is no factor $`N_{\rm sig}^\star`$ in (4.1) and no union over records.

**Lemma 4.2 (residual bound, proved; constants numeric).** Under the
conclusion of Lemma 4.1, write
$`s_{1,i,j}=\mathrm{ctr}_p(S_1(j,i))`$ and $`s_{2,i,j}=\mathrm{ctr}_p(S_2(j,i))`$, and define the integer residual $`\Delta^{\mathbb Z}(j,i)=s_{1,i,j}+(h_i\circledast s_{2,i})_j-c_{i,j}`$ over the
centered representatives. Then

```math
|\Delta^{\mathbb Z}(j,i)|
\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor+6{,}144\sqrt{n\lfloor\beta^2\rfloor}+6{,}144
\leq L_\Delta,
\qquad
\lVert \Delta^{\mathbb Z}\rVert_2^2\leq N_{\rm sig}^\star nL_\Delta^2\leq H_\Delta,
\qquad(4.2)
```

and $`\mathrm{ctr}_p\Delta=\Delta^{\mathbb Z}`$.

*Proof.* Each negacyclic coefficient is $`\pm`$ a sum of $`n`$ products
$`h_{i,k}s_{2,i,j-k}`$, so by Cauchy–Schwarz
$`|(h_i\circledast s_{2,i})_j|\leq\lVert h_i\rVert_2\lVert s_{2,i}\rVert_2
\leq6{,}144\sqrt n\cdot\sqrt{\lfloor\beta^2\rfloor}`$. Add $`|s_{1,i,j}|\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor`$
and $`|c_{i,j}|\leq6{,}144`$. The certificate takes $`L_\Delta`$ as the least integer
with $`(L_\Delta-\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor-6{,}144)^2\geq6{,}144^2n\lfloor\beta^2\rfloor`$, so the middle
inequality is checked by squaring nonnegative integers (**numeric**:
$`811{,}061{,}395`$ and $`1{,}648{,}068{,}606`$). Since $`L_\Delta\lt p/2`$, the field
table (2.1) is the reduction of $`\Delta^{\mathbb Z}`$ and its centered
representative is $`\Delta^{\mathbb Z}`$. $`\square`$

Lemma 4.2 does not assume $`\Delta\equiv0\pmod q`$. It is a bound on every
norm-valid source, ring-valid or not. That is what lets Section 5 test
divisibility with a pre-seed energy bound.

## 5. Residual leg

Throughout this section $`\Delta`$ is an integer table fixed before the seed with
$`\lVert \Delta\rVert_2^2\leq H`$, the three cores have widths $`c_1,c_2,c_3`$ and
i.i.d. entries uniform on $`\{-b,\ldots,b\}`$, $`2b\lt q`$, and $`A=B_3B_2B_1`$ is
their unreduced integer composition. Put

```math
\sigma^2=\frac{b(b+1)}{3},
\qquad
(x)_{t_{\mathrm{mom}}}=x(x+1)\cdots(x+{t_{\mathrm{mom}}}-1).
\qquad(5.1)
```

**Lemma 5.1 (per-core moment bound, proved).** Let a fresh core with $`c`$ rows
and entries i.i.d. symmetric with $`\mathbb E a^{2{d_{\mathrm{mom}}}}\leq(2{d_{\mathrm{mom}}}-1)!!\,s^{2{d_{\mathrm{mom}}}}`$
for all $`{d_{\mathrm{mom}}}\geq1`$ act with the same coefficients on every fiber of a fixed
real table $`X`$. Then for every integer $`{t_{\mathrm{mom}}}\geq1`$,

```math
\mathbb E\lVert AX\rVert_2^{2{t_{\mathrm{mom}}}}\leq(2s^2)^{t_{\mathrm{mom}}}(c/2)_{t_{\mathrm{mom}}}\lVert X\rVert_2^{2{t_{\mathrm{mom}}}}.
\qquad(5.2)
```

For the uniform law on $`\{-b,\ldots,b\}`$ the hypothesis holds with
$`s^2=\sigma^2`$. For the ternary law of H1 it holds with $`s^2=\tfrac12`$,
giving the factor $`(c/2)_{t_{\mathrm{mom}}}`$.

*Proof.* Write $`X`$ as a matrix whose columns are the fibers. First the
scalar moments. For the uniform law,
$`\mathbb E a^{2{d_{\mathrm{mom}}}}=\frac2{2b+1}\sum_{k=1}^bk^{2{d_{\mathrm{mom}}}}\leq
b^{2{d_{\mathrm{mom}}}-2}\frac2{2b+1}\sum_{k=1}^bk^2=b^{2{d_{\mathrm{mom}}}-2}\sigma^2`$, and
$`b^2\leq b(b+1)=3\sigma^2`$ gives $`b^{2{d_{\mathrm{mom}}}-2}\sigma^2\leq3^{{d_{\mathrm{mom}}}-1}\sigma^{2{d_{\mathrm{mom}}}}
\leq(2{d_{\mathrm{mom}}}-1)!!\,\sigma^{2{d_{\mathrm{mom}}}}`$. For the ternary law,
$`\mathbb E a^{2{d_{\mathrm{mom}}}}=\tfrac12\leq(2{d_{\mathrm{mom}}}-1)!!\,2^{-{d_{\mathrm{mom}}}}`$.

Let $`g`$ be an independent standard Gaussian vector indexed like $`AX`$. For
every fixed matrix $`Y`$, $`\mathbb E_g\langle g,Y\rangle^{2{t_{\mathrm{mom}}}}=(2{t_{\mathrm{mom}}}-1)!!\,
\lVert Y\rVert_2^{2{t_{\mathrm{mom}}}}`$. For fixed $`g`$, $`\langle g,AX\rangle=\sum_kA_kv_k`$
is linear in the independent entries $`A_k`$ with coefficients $`v_k`$ depending
only on $`g`$ and $`X`$. Expanding the $`2{t_{\mathrm{mom}}}`$-th power, odd moments vanish by
symmetry, and every surviving monomial has even exponents and a nonnegative
coefficient. Replacing each $`A_k`$ by an independent $`\mathcal N(0,s^2)`$ variable
$`G_k`$ can only increase each term. Hence
$`\mathbb E\lVert AX\rVert_2^{2{t_{\mathrm{mom}}}}\leq\mathbb E\lVert GX\rVert_2^{2{t_{\mathrm{mom}}}}`$.
Diagonalize $`XX^\top`$ with eigenvalues $`\nu_k\geq0`$,
$`\sum_k\nu_k=\lVert X\rVert_2^2`$. By rotation invariance,
$`\lVert GX\rVert_2^2=s^2\sum_k\nu_k\chi_k`$ with independent
$`\chi_k\sim\chi^2_c`$. The map
$`\nu\mapsto\mathbb E(\sum_k\nu_k\chi_k)^{t_{\mathrm{mom}}}`$ is convex, so on the
simplex of fixed trace it is maximized at a vertex, the rank-one case, where
it equals $`\lVert X\rVert_2^{2{t_{\mathrm{mom}}}}\,\mathbb E(\chi_c^2)^{t_{\mathrm{mom}}}=
\lVert X\rVert_2^{2{t_{\mathrm{mom}}}}2^{t_{\mathrm{mom}}}(c/2)_{t_{\mathrm{mom}}}`$. $`\square`$

**Lemma 5.2 (fresh-core annihilation, proved).** Let $`Y\neq0`$ be a table
over $`\mathbb F_q`$ fixed before a fresh core with $`c`$ rows and entries
uniform on $`\{-b,\ldots,b\}`$, $`2b\lt q`$, $`q`$ prime. Then
$`\Pr[\text{the output table is }0]\leq(2b+1)^{-c}`$.

*Proof.* Select, by a fixed rule, a fiber $`f`$ with $`Y_{\cdot,f}\neq0`$ and a
coordinate $`k`$ with $`Y_{k,f}\neq0`$. In one output row, condition on every
entry except the one multiplying coordinate $`k`$. The row's value at fiber
$`f`$ is $`a\,Y_{k,f}+\text{const}`$. Since $`2b\lt q`$ the $`2b+1`$ values of $`a`$ are
distinct modulo $`q`$, and since $`q`$ is prime and $`Y_{k,f}\neq0`$ they give
$`2b+1`$ distinct values. At most one is zero, so the row vanishes at $`f`$ with
probability at most $`1/(2b+1)`$. Rows are independent. The output is zero only
if every row vanishes at $`f`$. $`\square`$

**Theorem 5.3 (residual leg, proved).** For $`\Delta`$, $`H`$ and the cores as above,
if $`\Delta\not\equiv0\pmod q`$ then

```math
\Pr\bigl[\forall j\lt 31:\ q\mid\mathrm{ctr}_p(y_{\mathrm{res},j})\bigr]
\leq
\underbrace{\Bigl(\frac{4H}{p^2}\Bigr)^{t_{\mathrm{mom}}}\prod_{i=1}^{3}(2\sigma^2)^{t_{\mathrm{mom}}}(c_i/2)_{t_{\mathrm{mom}}}}
_{\varepsilon_{\mathsf{wrap}}}
+
\underbrace{\sum_{i=1}^{3}(2b+1)^{-c_i}}_{\varepsilon_{\mathsf{kill}}}
\qquad(5.3)
```

for every integer $`{t_{\mathrm{mom}}}\geq1`$, where $`y_{\mathrm{res}}=A\Delta\bmod p`$.

*Proof.* Wrap event. Condition on each preceding integer intermediate and
apply Lemma 5.1 core by core. The tower rule gives
$`\mathbb E\lVert A\Delta\rVert_2^{2{t_{\mathrm{mom}}}}\leq H^{t_{\mathrm{mom}}}\prod_i(2\sigma^2)^{t_{\mathrm{mom}}}(c_i/2)_{t_{\mathrm{mom}}}`$.
Markov's inequality at $`p^2/4`$ bounds
$`\Pr[\lVert A\Delta\rVert_2^2\geq p^2/4]`$ by $`\varepsilon_{\mathsf{wrap}}`$. The
reduction modulo $`p`$ in the implementation does not change $`y_{\mathrm{res}}`$.

Outside the wrap event every integer output has $`|(A\Delta)_j|\lt p/2`$, so
$`\mathrm{ctr}_p(y_{\mathrm{res},j})=(A\Delta)_j`$. Acceptance then means $`A\Delta\equiv0\pmod q`$,
that is, the chain reduced modulo $`q`$ maps the nonzero table $`\Delta\bmod q`$ to
zero. There is a first core whose input is nonzero and whose output is zero.
Conditioning on the history at that core, Lemma 5.2 bounds its probability by
$`(2b+1)^{-c_i}`$. A zero table stays zero, so summing over the three cores
bounds the event by $`\varepsilon_{\mathsf{kill}}`$. $`\square`$

The union is over three fresh cores, not over coordinates, fibers, records
or candidate witnesses. No small-ball bound on the source is used.

**Instance (numeric).** Profile $`b=12`$, $`(c_1,c_2,c_3)=(31,31,31)`$, so
$`\sigma^2=52`$ and $`2b+1=25\lt q`$. With $`H=H_\Delta`$ from Lemma 4.2:

| | Falcon-512 | Falcon-1024 |
| --- | ---: | ---: |
| $`\varepsilon_{\mathsf{kill}}=3\cdot25^{-31}`$ | $`2^{-142.37}`$ | $`2^{-142.37}`$ |
| $`\varepsilon_{\mathsf{wrap}}`$ (best $`{t_{\mathrm{mom}}}`$) | $`2^{-437.7}`$ ($`{t_{\mathrm{mom}}}=136`$) | $`2^{-153.75}`$ ($`{t_{\mathrm{mom}}}=60`$) |

The rejected profile $`b=17`$, $`(28,28,28)`$ passes at $`n=512`$ but gives
$`\varepsilon_{\mathsf{wrap}}=2^{-45.6}`$ at $`n=1024`$, because $`H_\Delta`$ grows by
about $`2^3`$. The certificate script keeps that as a negative check.

## 6. Main theorem

**Theorem 6.1 (projection term, proved; constants numeric).** Fix any $`N_{\rm sig}\leq2^{16}`$, either parameter set and
an adversarial pre-seed transcript satisfying H2. Sample the cores as in H1
and let $`y_{\mathrm{norm}},y_{\mathrm{res}}`$ be the true projections of the fixed sources. Define

```math
\mathsf{Bad}=\{\mathcal E(S)\geq Q_S\}
\;\cup\;
\{\mathcal E(S)\lt Q_S,\ (2.2)\text{ holds for every record},\ \Delta^{\mathbb Z}\not\equiv0\ (\mathrm{mod}\ q)\}.
```

Then

```math
\Pr_{\text{cores}}\bigl[\mathsf{Bad}\wedge y_{\mathrm{norm}}\text{ passes (F §6.1)}\wedge
y_{\mathrm{res}}\text{ passes (F §6.1)}\bigr]
\leq\max\bigl(\varepsilon_S,\ \varepsilon_{\mathsf{wrap}}+\varepsilon_{\mathsf{kill}}\bigr)
\leq2^{-136},
```

since $`\varepsilon_S\leq2^{-136}`$ by Theorem 3.1 and
$`\varepsilon_{\mathsf{wrap}}+\varepsilon_{\mathsf{kill}}\lt 2^{-142}`$ by the instance of
Section 5. The certified values are $`\varepsilon_S=2^{-136.4416}`$ and, for the
sum $`\varepsilon_S+\varepsilon_{\mathsf{wrap}}+\varepsilon_{\mathsf{kill}}`$ that the ledgers
charge, $`2^{-136.418}`$ for both parameter sets (**numeric**, check
`total_at_least_136_bits`).

*Proof.* H2 fixes $`S`$ and $`\Delta`$ before the seed, so which part of
$`\mathsf{Bad}`$ occurs is decided before the cores are sampled; no case
selection inspects the cores. In the first part, Theorem 3.1 bounds the
probability that $`y_{\mathrm{norm}}`$ passes. In the second part, H3(a), (2.2) and $`\mathcal E(S)\lt Q_S`$
give Lemmas 4.1 and 4.2, hence $`\lVert \Delta^{\mathbb Z}\rVert_2^2\leq H_\Delta`$, and
Theorem 5.3 with $`H=H_\Delta`$ bounds the probability that $`y_{\mathrm{res}}`$ passes. The two
parts are disjoint, so the probability is at most the larger bound. $`\square`$

The second part does not use the norm leg's randomness, and the first does
not use the residual leg's. The two legs' cores are independent (H1) but the
proof does not need that: it never multiplies their probabilities.

**Corollary 6.2 (Falcon extraction).** Conditioned on acceptance, outside the
event of Theorem 6.1 and the H2–H4 errors charged to their ledger items, the
fixed source satisfies for every record:

- $`|s_{1,i,j}|,|e(k,i)|\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor`$ and the norm
  identity (F §2.2) holds over $`\mathbb Z`$ (Lemma 4.1);
- every padded record is zero (Lemma 4.1);
- $`\Delta^{\mathbb Z}(\cdot,i)\equiv0\pmod q`$, so $`v_i=\Delta^{\mathbb Z}(\cdot,i)/q`$ is
  an integer polynomial with $`s_{1,i}+h_is_{2,i}-qv_i=c_i`$ in
  $`\mathbb Z[X]/(X^n+1)`$, which is (F §2.2), and
  $`|v_{i,j}|\leq\lfloor L_\Delta/q\rfloor`$ ($`65{,}998`$ at $`n=512`$, $`134{,}109`$ at
  $`n=1024`$).

The quotient $`v_i`$ is a consequence of the proof. It is never committed and
the protocol never needs its bound.

*Proof.* Under H3(b), acceptance of the public predicates on the revealed
outputs is acceptance on the true projections. Outside $`\mathsf{Bad}`$ and
with (2.2), $`\mathcal E(S)\lt Q_S`$ and $`\Delta^{\mathbb Z}\equiv0\pmod q`$. Lemmas 4.1 and 4.2
give the rest. $`\square`$

## 7. Honest completeness

**Theorem 7.1 (honest norm rejection, proved; constants numeric).** For an
honest source ($`\mathcal E(S)\leq H_S=N_{\rm sig}^\star \lfloor\beta^2\rfloor\leq2^{16}\lfloor\beta^2\rfloor`$, since every record has
$`\lVert s_{1,i}\rVert^2+\lVert e_i\rVert^2\leq \lfloor\beta^2\rfloor`$ and padded rows are
zero), and every $`{t_{\mathrm{mom}}}\geq1`$,

```math
\Pr\Bigl[\sum_j\mathrm{ctr}_p(y_{\mathrm{norm},j})^2\gt \Theta\Bigr]
\leq\frac{H_S^{t_{\mathrm{mom}}}M_{t_{\mathrm{mom}}}}{\Theta^{t_{\mathrm{mom}}}},
\qquad
M_{t_{\mathrm{mom}}}=(78)_{t_{\mathrm{mom}}}\,{t_{\mathrm{mom}}}!\,[u^{t_{\mathrm{mom}}}]\Bigl(\sum_{k=0}^{{t_{\mathrm{mom}}}}\mu_k\frac{u^k}{k!}\Bigr)^4,
\quad
\mu_k=\bigl((43/2)_k\bigr)^4.
\qquad(7.1)
```

*Proof.* For honest sources every intermediate is a small integer table and
$`|\mathrm{ctr}_p(x)|\leq|x|`$ for every integer $`x`$, so it suffices to
bound the unreduced integer outputs. Lemma 5.1 with the ternary law gives
$`\mathbb E[e_{\mathsf{out}}^k\mid\text{input}]\leq(c/2)_kE_{\mathsf{in}}^k`$
per core. Along one four-core prefix the tower rule gives
$`\mathbb E[e_g^k]\leq\mu_k\mathcal E(S)^k`$. Conditional on $`S`$ the four groups are
independent, so the multinomial expansion of $`\mathbb E(\sum_gE_g)^{t_{\mathrm{mom}}}`$ is at
most $`\mathcal E(S)^{t_{\mathrm{mom}}}t_{\mathrm{mom}}![u^{t_{\mathrm{mom}}}](\sum_k\mu_ku^k/k!)^4`$. The terminal multiplies by
$`(156/2)_{t_{\mathrm{mom}}}=(78)_{t_{\mathrm{mom}}}`$. Markov's inequality on the $`{t_{\mathrm{mom}}}`$-th moment finishes.
$`\square`$

At $`{t_{\mathrm{mom}}}=46`$ the certificate chooses $`\Theta`$ as the least integer making (7.1) at
most $`2^{-129}`$ (**numeric**; orders 1 to 200 are searched and the best cap
in that range is recorded; global optimality is not claimed).

**Honest residual rejection.** An honest $`\Delta`$ is $`q`$ times the honest
quotient, so every integer output is divisible by $`q`$. Rejection needs the
wrap event, bounded by $`\varepsilon_{\mathsf{wrap}}`$ with $`H=H_\Delta`$
(Theorem 5.3, first part).

**Corollary 7.2.** The two predicates can fail on the same honest source, so
the bounds add: honest rejection is at most
$`2^{-129}+\varepsilon_{\mathsf{wrap}}`$, which is $`2^{-129.0}`$ at $`n=512`$ and
$`2^{-128.99999995}`$ at $`n=1024`$, both below $`2^{-128.9}`$ (**numeric**, check
`honest_at_least_128_9_bits`). The prover evaluates both predicates before
sending (F §6). There is no retry loop and no grinding allowance.

## 8. Adversarial audit

- **Circularity.** The hypotheses use source fixation, the $`S_2`$ alphabet
  bound, public $`h,c`$ and the field definition (2.1). They do not use a
  centered PCS source bound or the Falcon norm. The norm leg gives the
  energy bound before either integer lift; the residual bound comes after
  the norm lift and before any use of the ring equation.
- **Seed independence of conditioning.** H2 and H3(a) are properties of
  pre-seed tables. The events that they fail while the verifier accepts are
  bounded by their ledger items uniformly over the seed. H3(b) and H4 are
  used only after the probability bound.
- **Why $`e_i`$ is in the norm source.** The norm identity is first proved
  in $`\mathbb F_p`$ and cannot rule out its own wrap. With $`S_1=S_2=0`$ and
  $`e=(13,103,5833,2^{32})`$, the squared norm is exactly $`\lfloor\beta^2\rfloor+p`$
  (an integer-wrap counterexample).
- **Why $`S_2`$ is not in the norm source.** Its alphabet bound is already
  proved (H3(a)) and is what (4.1) adds as $`n\cdot2{,}047^2`$.
- **Why the divisibility test is on centered integers.** In $`\mathbb F_p`$,
  $`q`$ is invertible, so "$`q`$ divides $`y_{\mathrm{res},j}`$" holds for every $`y_{\mathrm{res},j}`$. The
  verifier tests $`q\mid\mathrm{ctr}_p(y_{\mathrm{res},j})`$ as an integer (F §6.1).
- **Why the rejection sampler matters.** Lemma 5.2 needs $`2b+1`$ values
  distinct modulo $`q`$, and Lemma 5.1 needs the stated symmetric law. Reducing
  a byte modulo 25 without rejection gives neither the exact law nor the
  certified constants.
- **Core reuse.** Reuse across a core's own fibers is covered: Corollary C′
  and Lemma 5.1 hold for any number of fibers, and Lemmas A and 5.2 select
  one fiber. Reuse across groups or legs is not covered: Lemma D needs one
  fresh comparison variable per core, Lemma E needs fresh prefix cores, and
  Theorem 5.3 needs fresh cores.
- **Exactness is not conditioned on.** Lemma D compares against the
  unreduced integer chain and never conditions a core on its output being
  small. Exactness (Lemma B) enters only as a deterministic implication in
  the case split of Section 3.5. The events where a large table turns small
  are charged separately: $`\varepsilon_F`$ per group in the zero atoms of N1
  and $`2^{-156}`$ for the terminal.
- **Degenerate shapes.** Padded rows are zero by Lemma 4.1. Zero-bit cores
  have radix 1 and are still fresh cores of the certified width. An all-zero
  source has $`\mathcal E(S)=0\lt Q_S`$, but its live rows violate (2.2) because
  $`\lfloor\beta^2\rfloor\neq0`$, so its acceptance is an H4 failure and is charged there.
- **Boundary of $`\mathrm{ctr}_p`$.** $`p`$ is odd, so no field element has
  centered value $`\pm p/2`$; the `u128` accumulator of (F §6.1) holds each
  square ($`\lt 2^{126}`$) and stops once the running sum exceeds $`\Theta\lt 2^{74}`$.
- **Tightness.** The residual kill term is tight up to the union over three
  cores: a source supported on one coordinate is annihilated by a core with
  probability about $`25^{-31}`$. The norm term is dominated by the
  comparison certificate N1; no claim of optimality is made.

## 9. Open items

1. **Machine-checked proofs.** The statements of Sections 3 to 7 are proved
   in Lean with the bounds $`2^{-136}`$ and $`2^{-129}+2^{-142}`$ (Section 11).
   Section 11.3 lists what the Lean theorems do not cover. In particular the
   finer numeric values are checked in exact arithmetic by the script only,
   and the correctness of the script is not formally verified.
2. **Ledger integration.** The projection item is at most $`2^{-136}`$; the
   ledgers charge the certified sum $`2^{-136.418}`$.
   The complete stage and monomial output-batching accounting is defined by
   `specs/falcon.md` Section 9.5 and the two composition ledgers. It is
   separate from this projection theorem.

## 10. Constant table

All verified by `specs/scripts/falcon_projection_certificate.py`.

| Constant | Falcon-512 | Falcon-1024 | Source |
| --- | ---: | ---: | --- |
| $`\Theta`$ (order-46 cap) | $`5{,}092{,}300{,}183{,}991{,}442{,}303{,}232`$ ($`2^{72.11}`$) | $`10{,}513{,}136{,}047{,}130{,}310{,}946{,}696`$ ($`2^{73.15}`$) | Theorem 7.1 |
| $`\eta_{\mathrm{proj}}`$ | $`2^{15}`$ | same | N1 |
| $`\eta_{\mathrm{grid}}`$ | $`(1/1024)^5(33/32)^{1470}=2^{15.2594}`$ | same | N1 |
| $`\log_2Q_S`$ | $`57.1088`$ | $`58.1546`$ | Section 3.5 |
| $`\sup_wh_3(w)\leq U`$, $`K_{\mathrm L}`$ | $`U=0.524894`$, $`K_{\mathrm L}=21/40`$ | same | Section 3.3 |
| Exactness margins, stages 0 to 4 | $`8.946,12.519,13.519,14.519,15.519`$ bits | $`8.423,11.996,12.996,13.996,13.996`$ bits | (3.5) |
| Lift slack $`\log_2(p-n\cdot2047^2)-\log_2Q_S`$ | $`6.89`$ bits | $`5.85`$ bits | Lemma 4.1 |
| $`\varepsilon_S`$ | $`2^{-136.4416}`$ | $`2^{-136.4416}`$ | Theorem 3.1 |
| $`L_\Delta`$ | $`811{,}061{,}395`$ | $`1{,}648{,}068{,}606`$ | Lemma 4.2 |
| $`\log_2H_\Delta`$ | $`84.19`$ | $`87.24`$ | Lemma 4.2 |
| $`\varepsilon_{\mathsf{kill}}`$ | $`2^{-142.37}`$ | $`2^{-142.37}`$ | Theorem 5.3 |
| $`\varepsilon_{\mathsf{wrap}}`$ | $`2^{-437.7}`$ ($`t_{\mathrm{mom}}=136`$) | $`2^{-153.75}`$ ($`t_{\mathrm{mom}}=60`$) | Theorem 5.3 |
| Projection term | $`2^{-136.418}`$ | $`2^{-136.418}`$ | Theorem 6.1 |
| Honest rejection | $`2^{-129.0}`$ | $`2^{-128.99999995}`$ | Corollary 7.2 |
| $`\lfloor L_\Delta/q\rfloor`$ | $`65{,}998`$ | $`134{,}109`$ | Corollary 6.2 |

## 11. Lean formalization

The statements of Sections 3 to 7 are formalized and proved in Lean 4 in the
library certified-projections, in the directory `CertifiedProjections/Aerie`.
They rest on a general library for projections applied in several stages,
`CertifiedProjections/MultiStage`. certified-projections is developed outside
this repository and is not public yet.
The Lean kernel checks every proof. The proofs use only the three standard
axioms of Lean (`propext`, `Classical.choice`, `Quot.sound`), and the file
`CertifiedProjections/Tests/AerieAxioms.lean` checks this for every
declaration named below.

The script `specs/scripts/falcon_projection_lean.py` generates the Lean file
`CertifiedProjections/Tests/AerieSpecConstants.lean` from the report of the
certificate script. In that file Lean checks each constant of the report
against its Lean definition. `specs/scripts/falcon_projection_lean.json`
records the SHA-256 of the file. In spec CI, `falcon_projection_lean.py
--check` compares the constants of the Rust verifier with the report, and
checks that the file generated from the current report has the recorded
SHA-256. This repository does not record a commit of certified-projections.

### 11.1 The final theorems

Fix a parameter set and a record count $`N_{\rm sig}^\star`$ that the geometry
of (F §6.2) admits: $`2^{2+R}`$ with $`R\leq14`$ at $`n=512`$, or $`2^{1+R}`$ with
$`R\leq15`$ at $`n=1024`$. The padded record counts of the protocol, which
satisfy $`2^{12}\leq nN_{\rm sig}^\star`$, are those with $`R\geq1`$. Fix
public inputs whose coefficients have absolute value at most $`6{,}144`$, and
integer tables that represent $`S_1`$, $`S_2`$ and $`e`$. The
probability is over the 20 cores, with independent entries distributed as in
H1. The outputs are the true projections of the source: Lean computes them
from the source and the cores by the contractions of (F §6.3) and reduces
them to centered representatives modulo $`p`$ (`Geometry.outputs`).

- **Soundness** (`falcon512_projection_sound`,
  `falcon1024_projection_sound`). Assume H3(a) and (2.2) for every record. If
  the centered source is not a Falcon witness in the sense of Section 1, then
  the outputs pass both predicates of (F §6.1) with probability at most
  $`2^{-136}`$. This is the success criterion of Section 1. It combines
  Theorem 6.1 and Corollary 6.2.
- **Completeness** (`falcon512_projection_complete`,
  `falcon1024_projection_complete`). If the source is a Falcon witness, then
  the outputs fail one of the two predicates with probability at most
  $`2^{-129}+2^{-142}`$. This is Corollary 7.2 with
  $`\varepsilon_{\mathsf{wrap}}`$ bounded by $`2^{-142}`$.

### 11.2 Correspondence

Files are relative to `CertifiedProjections/`.

| This document | Lean declarations | File |
| --- | --- | --- |
| Parameters and thresholds (Sections 2, 3.1 and 10) | `falcon512`, `falcon1024`, `falcon512_valid`, `falcon1024_valid` | `Aerie/Params.lean` |
| Radices and reshaping (F §6.2) | `falcon512Geometry`, `falcon1024Geometry`, `Geometry.normPlacement`, `Geometry.ringPlacement`, `falcon512Geometry_fanInLe`, `falcon1024Geometry_fanInLe` | `Aerie/Geometry.lean` |
| Contractions by the cores (F §6.3) | `normProjection`, `ringProjection`, `Geometry.outputs` | `Aerie/Cores.lean`, `Aerie/Falcon.lean` |
| H1, as independent core entries with the two laws | `Geometry.CoreSeeds`, `projectionCheck_eq_map` | `Aerie/Cores.lean`, `Aerie/Falcon.lean` |
| Byte maps (F §6.3), for one byte | `map_ternaryOfByte`, `filter_byte_lt`, `map_uniformOfByte` | `Aerie/Sampling.lean` |
| Public predicates (F §6.1) | `NormAccept`, `RingAccept` | `Aerie/NormCheck.lean`, `Aerie/RingCheck.lean` |
| Falcon witness (Section 1) | `FalconWitness` | `Aerie/Falcon.lean` |
| Lemma A | `stage_collapse_le` | `MultiStage/Lemmas/Collapse.lean` |
| Lemma B | `ctr_mul_eq_mul_ctr` | `MultiStage/Lemmas/NoOverflow.lean` |
| Lemma C and the supremum $`21/40`$ | `stage_mgf_le`, `EntryLaw.ternary_mgfConstant` | `MultiStage/Lemmas/MGF.lean`, `Certificates/Families/MultiStage/CosineSup.lean` |
| Corollary C′ | `stage_smallBall_le`, `stage_stochGe` | `MultiStage/Compose/Domination.lean` |
| Lemma D | replaced by stochastic domination: `StochGe`, `normCheck_accept_le_comparison` | `MultiStage/Compose/Domination.lean`, `Aerie/NormCheck.lean` |
| Lemma E | `collapseFailBound`, used in `normCheck_accept_le_comparison` | `MultiStage/Compose/Domination.lean`, `Aerie/NormCheck.lean` |
| Certificate N1 (Section 3.5) | `normComparison_exists` | `Certificates/Families/Aerie/NormComparison.lean` |
| Theorem 3.1 | `normCheck_accept_le` | `Aerie/NormCheck.lean` |
| Lemma 4.1 | `normIdentity_lift`, `padding_eq_zero` | `Aerie/Lift.lean` |
| Lemma 4.2 | `sqNorm_residual_le` | `Aerie/Lift.lean` |
| Lemma 5.1 | `stage_moment_le` | `MultiStage/Lemmas/Moments.lean` |
| Lemma 5.2 | `stage_zero_mod_le` | `MultiStage/Lemmas/FalseZero.lean` |
| Theorem 5.3 | `ringCheck_accept_le`, `falcon512_ringCheckError_le`, `falcon1024_ringCheckError_le` | `Aerie/RingCheck.lean` |
| Theorem 6.1 | `projection_sound` | `Aerie/Main.lean` |
| Corollary 6.2 | `falconWitness_of_not_projectionBad` | `Aerie/Falcon.lean` |
| Theorem 7.1 | `normCheck_reject_le`, `falcon512_honest_le`, `falcon1024_honest_le` | `Aerie/Main.lean` |
| Corollary 7.2 | `projection_complete` | `Aerie/Main.lean` |

The Lean text uses other words for some objects of this document. A group
is a branch of the norm check, a prefix core is the core of a branch stage,
the terminal core is the core of the final stage, the norm leg is the norm
check, and the residual leg is the ring-equation check.

Lean proves the certificate N1 with its own tables of exact finite laws,
which the kernel evaluates. It does not read the report of the certificate
script.

### 11.3 What the Lean theorems do not cover

1. **H1.** The theorems take the entries of the cores to be independent with
   the two laws of H1. The three statements on the byte maps are about one
   byte, and the final theorems do not use them. The derivation of the bytes
   from the transcript (F §6.3) is not formalized: the cSHAKE256 streams
   modeled as a random oracle, the order of the bytes within a row of a core,
   and the bytes read after a rejected byte.
2. **H2.** The theorems fix the source and then sample the cores. That the
   committed source is fixed before the seed is the binding and extraction
   property of the commitment scheme. The Fiat–Shamir compilation and its
   query-count factor are in the composition ledger.
3. **H3 and H4.** H3(a) and (2.2) are hypotheses of the soundness theorems.
   H3(b) enters through the outputs: the theorems are about the true
   projections of the source, and that the revealed outputs equal them is
   the claim of `projection_link`.
4. **Finer numeric values.** Lean proves the bounds $`2^{-136}`$ and
   $`2^{-129}+2^{-142}`$. The values $`2^{-136.4416}`$ and $`2^{-136.418}`$ of
   Theorem 6.1, the second of which the ledgers charge, and the values of
   Corollary 7.2 are checked by the certificate script only.
5. **The implementation.** The Rust verifier is tested, not formally
   verified. `falcon_projection_lean.py --check` compares its constants with
   the report. It does not compare its code with the Lean definitions.
6. **The Falcon relation.** A Falcon witness in the sense of Section 1 is a
   statement about the committed integer tables and the public inputs. Its
   relation to Falcon signature verification (the encodings and the hash to
   the target polynomial) is specified in (F §2) and is outside the Lean
   development.
