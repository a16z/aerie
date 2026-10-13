# Aerie Falcon protocol

## Status

| Field | Value |
| --- | --- |
| Status | normative standalone specification; transcript implementation complete |
| Audience | Aerie protocol and implementation contributors |
| Applies to | homogeneous Falcon-512 or Falcon-1024 aggregates, wire version 1 |
| Implementation status | three-stage PIOP, native v1 transcript with scheduled grinding, and four native groups implemented (Section 12.3) |
| Projection theorem | [falcon-projection-theorem.md](falcon-projection-theorem.md) |
| Last reviewed | 2026-09-27 |

Section 10 defines the first frozen v1 format. Current development proofs implement its transcript and grinding requirements. The acceptance criteria in Section 13 gate implementation conformance.

The uppercase terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are normative requirements as defined by BCP 14 when, and only when, they appear in uppercase.

## 1. Scope

Aerie proves Falcon verification for a batch of public records. Each record has one public key, message, and public 40-byte salt. A proof establishes the ring equation, Falcon squared-norm bound, and compressed-signature length bound for every live record.

Falcon symbols retain their Falcon meaning: $`n,q,h,m,r,c,s_1,s_2,\lfloor\beta^2\rfloor`$, and $`\mathsf{sbytelen}`$. The salt $`r`$ appears only in the public statement. The two parameter sets are homogeneous: an aggregate MUST use one parameter set.

The protocol uses four Akita commitment groups, one clear structured projection, three batched sumcheck stages, and one Akita opening. The final opening authenticates nine evaluations of committed polynomials at three group points.

## 2. Falcon statement and parameters

### 2.1 Parameter sets

The protocol operates over $`R_q=\mathbb Z_q[X]/(X^n+1)`$, with $`q=12{,}289`$.

The batch count MUST satisfy $`1\le N_{\rm sig}\le2^{16}`$. Define
$`N_{\rm sig}^\star=\max(\mathrm{next\_power\_of\_two}(N_{\rm sig}),N_{\min})`$,
where $`N_{\min}=8`$ at Falcon-512 and $`N_{\min}=4`$ at Falcon-1024.
Implementations MUST pad to this minimum even for a single live record.
Define $`d_{\rm coef}=\log_2 n`$, $`d_{\rm sig}=\log_2N_{\rm sig}^\star`$,
and $`d=d_{\rm coef}+d_{\rm sig}`$.
Equivalently, $`n\cdot N_{\rm sig}^\star\ge2^{12}`$. The smallest admitted
Akita catalog shape commits $`2^{12}`$ coefficients per column, and the
projection's fixed 4096-column first block (Section 6) needs
$`2n\cdot N_{\rm sig}^\star\ge4096`$, which the catalog floor implies.
The projection profile and its order-46 moment-cap certificate are unchanged.
The verifier MUST reject a resulting shape with no admitted Akita catalog row.
It MUST NOT increase $`N_{\rm sig}^\star`$ further to reach another catalog row.

| Parameter | Falcon-512 | Falcon-1024 |
| --- | ---: | ---: |
| $`n`$ | 512 | 1,024 |
| $`d_{\rm coef}=\log_2 n`$ | 9 | 10 |
| $`\lfloor\beta^2\rfloor`$ | 34,034,726 | 70,265,242 |
| $`\mathsf{sbytelen}`$, padded signature bytes | 666 | 1,280 |
| Fixed compressed-body cost, bits | 4,608 | 9,216 |
| Unary budget $`L=8\mathsf{sbytelen}-328-9n`$ | 392 | 696 |
| Budget offset $`\kappa`$ | 292 | 548 |
| Encoding claim $`L-\kappa`$ | 100 | 148 |
| $`u_2`$ alphabet | $`[-4,3]`$ | $`[-8,7]`$ |
| Public-key bytes | 897 | 1,793 |
| Public-key header | 0x09 | 0x0A |
| Padded-signature header | 0x39 | 0x3A |
| Parameter-set byte | 9 | 10 |
| Reference batch profile | $`N_{\rm sig}=2^{15}`$ | $`N_{\rm sig}=2^{14}`$ |

Here $`\mathsf{sbytelen}`$ is the padded signature length including its header
and salt. The compressed body has $`8\mathsf{sbytelen}-328`$ bits: 328 bits
account for the one-byte header and the 40-byte salt. The fixed compressed
body cost is $`9n`$ bits: one sign bit, seven low magnitude bits, and one
unary terminator per coefficient.

The protocol supports padded batch sizes $`N_{\rm sig}^\star\leq2^{16}`$, subject to an
admitted Akita catalog row for all four group shapes. The projection geometry
also requires $`d_{\rm sig}\geq b_{\rm split}=12-(d_{\rm coef}+1)`$. Catalog admission is determined
by the regenerated multi-polynomial catalogs.

### 2.2 Falcon verification relation

To verify record $`i`$, the canonical Falcon byte layer parses the selected
public-key header and its $`n`$ coefficients, derives
$`\mathsf{hpk}_i=\mathrm{SHAKE256}(\mathsf{pk}_i,64)`$, and computes
```math
c_i=\mathsf{HashToPoint}\bigl(
 r_i\Vert\mathsf{hpk}_i\Vert\texttt{0x00}\Vert
 \texttt{0x00}\Vert m_i\bigr)\in R_q.
```
The two zero bytes select raw-message mode with an empty context. HashToPoint
reads big-endian 16-bit candidates from SHAKE256, rejects candidates at
least $`5q=61{,}445`$, and reduces each accepted candidate modulo $`q`$ until it
has $`n`$ coefficients. The byte layer accepts only the parameter-set header
and exact lengths in this table:

| Parameter set | Public key | Padded signature |
| --- | --- | --- |
| Falcon-512 | header 0x09 and 896 coefficient bytes; total 897 bytes | header 0x39, 40-byte salt, and 625 compressed-body bytes; total 666 bytes |
| Falcon-1024 | header 0x0A and 1,792 coefficient bytes; total 1,793 bytes | header 0x3A, 40-byte salt, and 1,239 compressed-body bytes; total 1,280 bytes |

Public-key coefficients use the canonical Falcon 14-bit format. Signature
decoding MUST reject malformed coefficient codes, an invalid header,
truncated input, excess input, and a compressed body longer than the
parameter's padded signature length. It MUST use Falcon's canonical
most-significant-bit-first bit packing and centered reduction.

The implementation byte-layer definitions are in
[preprocess.rs](../crates/aerie-falcon-native/src/preprocess.rs) and
[source.rs](../crates/aerie-falcon-native/src/source.rs). These files define the
byte-for-byte Falcon codec and HashToPoint framing for this specification.
Raw messages and empty contexts are the only supported message mode.

The verifier accepts a compressed $`s_{2,i}`$ only if decoding succeeds and its encoded body is within $`\mathsf{sbytelen}`$ bytes. It sets
```math
s_{1,i}=c_i-h_i s_{2,i}\pmod q
```
using centered coefficients, and checks
```math
\lVert s_{1,i}\rVert_2^2+\lVert s_{2,i}\rVert_2^2\leq\lfloor\beta^2\rfloor.
```
For $`N_{\rm sig}`$ records, the prover supplies integer polynomials $`s_{1,i},s_{2,i}`$ and a slack vector $`e_i\in\mathbb Z^4`$ for every live row $`i\lt N_{\rm sig}`$. The protocol proves
```math
s_{1,i}+h_i\circledast s_{2,i}-c_i\equiv0\pmod q,
\qquad(2.1)
```
```math
\lVert s_{1,i}\rVert_2^2+\lVert s_{2,i}\rVert_2^2+\lVert e_i\rVert_2^2
=\lfloor\beta^2\rfloor,
\qquad(2.2)
```
and that $`s_{2,i}`$ has a valid compressed encoding of at most $`\mathsf{sbytelen}`$ bytes. Here $`\circledast`$ is negacyclic multiplication in $`R_q`$. Equation (2.2) is equivalent to the Falcon norm inequality because every nonnegative integer is a sum of four squares.

### 2.3 Public records and byte layer

The canonical byte layer MUST accept exactly the selected Falcon parameter set. It uses the existing Falcon decoding, key parsing, rejection sampling, byte order, centered reduction, and HashToPoint rules. The public key and message are statement inputs. The proof transports the $`N_{\rm sig}`$ live 40-byte salts in record order, or one shared 40-byte salt in common-salt mode (Section 10.2).

The transcript binds the ordered public records with one ParallelHash256 digest. For each record define
```math
\mathsf{hpk}_i=\mathrm{SHAKE256}(\mathsf{pk}_i,64).
```
Let $`\mathrm{LE64}(x)`$ be the unsigned eight-byte little-endian encoding of $`x`$. The logical input is
```math
X=\mathrm{LE64}(N_{\rm sig})\Vert
\big\Vert_{i=0}^{N_{\rm sig}-1}
\left(
\mathsf{hpk}_i\Vert r_i\Vert
\mathrm{LE64}(|m_i|)\Vert m_i
\right).
\qquad(2.3)
```
The digest is NIST SP 800-185 ParallelHash256 with block size 16,384 bytes and output length 256 bits. Per-record mode uses customization string ASCII aerie/falcon/v1/public-records-binding. Common-salt mode MUST instead use ASCII aerie/falcon/v1/public-records-binding/common-salt. Both modes hash the same logical input $`X`$, including the expanded salt for every record; equation (2.4) shows the per-record customization.

Partition $`X`$ into consecutive blocks $`X_0,\ldots,X_{b-1}`$, each 16,384 bytes except possibly the last. There is no record-level or block-level padding in $`X`$. Compute
```math
\mathsf{CV}_j=\mathrm{SHAKE256}(X_j,64),
```
```math
Z=\mathrm{left\_encode}(16{,}384)\Vert
\mathsf{CV}_0\Vert\cdots\Vert\mathsf{CV}_{b-1}
\Vert\mathrm{right\_encode}(b)
\Vert\mathrm{right\_encode}(256),
```
```math
\mathsf{public\_records}=
\mathrm{cSHAKE256}
(Z,256,\texttt{ParallelHash},
\texttt{aerie/falcon/v1/public-records-binding}).
\qquad(2.4)
```
The implementation MUST stream the same logical byte string, preserve record order, include empty messages with a zero length, and reuse the HashToPoint public-key hash. Raw messages and salts MAY cross a 16-KiB boundary. A verifier MUST reject a statement shape disagreement before proving or verifying.

## 3. Notation, domains, and witness tables

### 3.1 Fields and multilinear extensions

The base field is $`\mathbb F_p`$, where $`p=2^{64}-59`$. Sumcheck challenges lie in $`K=\mathbb F_{p^2}`$. Committed values are represented in $`\mathbb F_p`$. An extension element is encoded as its two canonical $`\mathbb F_p`$ limbs, each an eight-byte little-endian integer, in the pinned Akita basis order.

Let $`N_{\rm sig}^\star=2^{d_{\rm sig}}`$ be the minimum-padded row count defined in Section 2.1, with $`N_{\rm sig}\leq N_{\rm sig}^\star`$, and let $`\mathsf{live}(i)=1`$ for $`i\lt N_{\rm sig}`$ and $`0`$ otherwise. The coefficient domain is $`\{0,1\}^{d_{\rm coef}+d_{\rm sig}}`$. A coefficient index $`j`$ occupies the first $`d_{\rm coef}`$ variables, least significant bit first; the row index $`i`$ occupies the remaining $`d_{\rm sig}`$ variables, least significant bit first. The point convention is $`r=(r_{\rm coef},r_{\rm sig})`$. The signature domain is $`\{0,1\}^{d_{\rm sig}}`$. The total coefficient-domain dimension is $`d=d_{\rm coef}+d_{\rm sig}`$. The slack domain is $`\{0,1\}^{2+d_{\rm sig}}`$, with the two slack-slot variables followed by the row variables.

For any arity $`d_{\rm loc}`$, an evaluation point $`z\in K^{d_{\rm loc}}`$ and $`x\in\{0,1\}^{d_{\rm loc}}`$, define
```math
\mathrm{eq}(z,x)=
\prod_{t=0}^{d_{\rm loc}-1}\bigl(z_tx_t+(1-z_t)(1-x_t)\bigr).
\qquad(3.1)
```
For a Boolean table $`T`$,
```math
T(z)=\sum_{x\in\{0,1\}^{d_{\rm loc}}}\mathrm{eq}(z,x)T(x).
\qquad(3.2)
```
We use the same symbol for a table and its multilinear extension.

A local sumcheck instance may use its own variable order. It applies each stage challenge to its next local variable. An instance with fewer variables than a stage is inactive during the leading rounds and uses the corresponding suffix of the stage point. Each terminal point below is defined by that instance's own order.

Unless an instance is explicitly restricted to a smaller domain, its sum is
over the full Boolean table in its stated domain. Public tables are zero on
padded rows, except where Section 3.4 specifies canonical nonzero padding for
the digit and budget witnesses.

### 3.2 Witness tables

| Table | Status | Domain | Honest values | Meaning |
| --- | --- | --- | --- | --- |
| $`S_1(j,i)`$ | committed | coefficient | centered Falcon coefficient, hence $`\vert S_1\vert \leq6{,}144`$ | coefficients of $`s_{1,i}`$ |
| $`e(k,i)`$ | committed | slack | $`\vert e\vert \leq8{,}191`$ | four-square slack $`e_i`$ |
| $`a_0(j,i),a_1(j,i),a_2(j,i)`$ | committed | coefficient | $`a_1,a_2\in[-8,7]`$, $`a_0\in[-4,3]`$ | balanced digits of $`\vert s_2\vert `$ |
| $`\sigma(j,i)`$ | committed | coefficient | $`\{-1,1\}`$ | sign of $`s_2`$ |
| $`u_0(i),u_1(i)`$ | committed | signature | $`[-4,3]`$ | balanced budget digits |
| $`u_2(i)`$ | committed | signature | $`[-4,3]`$ at 512; $`[-8,7]`$ at 1024 | high budget digit |
| $`C(j,i)`$ | public | coefficient | centered Falcon representative | target polynomial $`c_i`$ |
| $`h(j,i)`$ | public | coefficient | centered Falcon representative | public key polynomial |
| $`M(j,i)`$ | virtual | coefficient | $`[0,2{,}047]`$ after digit checks | magnitude of $`s_2`$ |
| $`S_2(j,i)`$ | virtual | coefficient | $`[-2{,}047,2{,}047]`$ after digit and sign checks | signed decoded coefficient |
| $`\Delta(j,i)`$ | virtual | coefficient | $`\mathbb F_p`$ | ring residual |

The slack slot is $`k\in\{0,1\}^2`$, and $`e_i=(e(k,i))_k`$ is the per-signature slack vector. Budget digits are $`u_\ell`$, with $`\ell\in\{0,1,2\}`$; their folded claims are $`v^P_{u_\ell}`$.

The digit tables reconstruct
```math
M=(a_0+4)+8(a_1+8)+128(a_2+8)=a_0+8a_1+128a_2+1{,}092,
\qquad
S_2=\sigma M.
\qquad(3.3)
```
The public table order is $`C(j,i)=c_{i,j}`$ and $`h(j,i)=h_{i,j}`$, with
coefficient index first and row index second. Their padded rows are zero.
Define the residual over $`\mathbb F_p`$ by
```math
\Delta(j,i)=S_1(j,i)+(h_i\circledast S_2(\cdot,i))_j-C(j,i).
```
The public coefficient convention is the one used in the negacyclic
functional in Section 8.
The constant $`1{,}092`$ centers the three unsigned digits. In particular,
$`(a_0+4)+8(a_1+8)`$ is the low seven-bit magnitude and $`a_2+8`$ is the unary high part.

For each record, define the unused unary budget
```math
u(i)=u_0(i)+8u_1(i)+64u_2(i)+\kappa.
\qquad(3.4)
```
The required budget identity is
```math
\sum_j(a_2(j,i)+8)+u(i)=L.
\qquad(3.5)
```
Its field claim is
```math
\sum_{j,i}\mathrm{eq}(r_{\rm sig},i)(a_2(j,i)+8)
+\sum_i\mathrm{eq}(r_{\rm sig},i)
(u_0(i)+8u_1(i)+64u_2(i))=L-\kappa.
\qquad(3.6)
```
The constants $`L`$, $`\kappa`$, and $`L-\kappa`$ are in the table of Section 2.1.

### 3.3 Range polynomials

For a digit $`Z`$, define the folded digit table $`P_Z(x)=Z(x)(Z(x)+1)`$ for Boolean $`x`$. The symbol $`P_Z`$ at a non-Boolean point denotes the multilinear extension of this table, not the product of the extensions of $`Z`$ and $`Z+1`$. On Boolean points it satisfies $`4P_Z+1=(2Z+1)^2`$. The map is two-to-one on a balanced alphabet because $`Z`$ and $`-1-Z`$ have the same image.

For $`Z\in[-4,3]`$, the image lies in $`\{0,2,6,12\}`$, and
```math
Q_8(P)=P(P-2)(P-6)(P-12).
\qquad(3.7)
```
For $`Z\in[-8,7]`$, the image lies in $`\{0,2,6,12,20,30,42,56\}`$. Define
```math
Q_{16}^{\rm lo}(P)=Q_8(P),\qquad
Q_{16}^{\rm hi}(P)=(P-20)(P-30)(P-42)(P-56).
\qquad(3.8)
```
Then $`Z\in[-8,7]`$ exactly when $`Q_{16}^{\rm lo}(P_Z)Q_{16}^{\rm hi}(P_Z)=0`$ on the Boolean domain.

For a base-16 digit, $`A_Z^{\rm lo}`$ and $`A_Z^{\rm hi}`$ are virtual Boolean tables with values $`Q_{16}^{\rm lo}(P_Z(x))`$ and $`Q_{16}^{\rm hi}(P_Z(x))`$. These virtual tables, like $`P_Z`$, are separately multilinear-extended. The extension of a pointwise product is not computed by multiplying the source extensions.

### 3.4 Canonical padded witness values

Every padded row $`i\geq N_{\rm sig}`$ has semantic $`S_1=e=M=S_2=0`$. The committed digits MUST use
```math
a_0=-4,\quad a_1=-8,\quad a_2=-8,\quad \sigma=1.
\qquad(3.9)
```
These digits reconstruct $`M=0`$. The budget digits MUST be
```math
(u_0,u_1,u_2)=(-4,-3,2)\quad\text{at Falcon-512},
```
```math
(u_0,u_1,u_2)=(-4,3,2)\quad\text{at Falcon-1024}.
\qquad(3.10)
```
They reconstruct $`u(i)=L`$. Thus the padded row satisfies the same budget identity as every live row. Padded digits are generally nonzero. Implementations MUST NOT require every committed padded digit to be zero.

## 4. Exact Falcon encodability

### 4.1 Coefficient and length conditions

Falcon compresses each coefficient of $`s_2`$ as one sign bit, seven low magnitude bits, and a unary high part with a terminator. An integer polynomial is encodable exactly when
```math
|s_{2,i,j}|\leq2{,}047\quad\text{for every }j
\qquad(4.1)
```
and
```math
\sum_{j=0}^{n-1}\left\lfloor\frac{|s_{2,i,j}|}{128}\right\rfloor
\leq L.
\qquad(4.2)
```
The term $`L=8\mathsf{sbytelen}-328-9n`$ is 392 at Falcon-512 and 696 at Falcon-1024.

The checked digit alphabets imply (4.1). The range of $`a_2`$ gives
```math
a_2+8=\left\lfloor\frac{M}{128}\right\rfloor\in[0,15].
\qquad(4.3)
```
Equation (3.5), together with $`u(i)\geq0`$ from its checked digits, implies (4.2).

### 4.2 Budget digit ranges and integer identity

The digits $`u_0,u_1`$ are in $`[-4,3]`$. The $`u_2`$ range is $`[-4,3]`$ at 512 and $`[-8,7]`$ at 1024. Thus the minimum of (3.4) is zero: $`-4-32-256+292=0`$ at 512, and $`-4-32-512+548=0`$ at 1024. The budget proof therefore gives $`u(i)\geq0`$.

The identity in (3.5) is proved over $`\mathbb F_p`$. Its nonnegative left side is at most $`15n+1{,}023`$, which is below $`p`$ for both parameter sets. It therefore lifts to the integer identity. The identity constrains the decoded signature length; the declared Akita width does not.

## 5. Akita commitment groups and widths

Akita commits any number of same-shape polynomials in one group. Each group has one commitment and one opening point. The groups in this specification have no power-of-two plane-count requirement.

| Group | Polynomials | Variables per polynomial | Honest signed width | Opening point |
| --- | --- | ---: | ---: | --- |
| $`[S_1]`$ | $`S_1`$ | $`d_{\rm coef}+d_{\rm sig}`$ | 14 | $`r^{(3)}`$ |
| $`[e]`$ | $`e`$ | $`2+d_{\rm sig}`$ | 14 | $`r^{(3)}_e`$ |
| $`[a_0,a_1,a_2,\sigma]`$ | $`a_0,a_1,a_2,\sigma`$ | $`d_{\rm coef}+d_{\rm sig}`$ | 4 | $`r^{(3)}`$ |
| $`[u_0,u_1,u_2]`$ | $`u_0,u_1,u_2`$ | $`d_{\rm sig}`$ | 4 | $`r^{(3)}_{\rm sig}`$ |

A signed width $`w`$ represents the interval $`[-2^{w-1},2^{w-1}-1]`$. These declarations size the honest producer and PCS parameters. They are not range proofs.

The implementation uses `Fp64SignedWidth14` and `Fp64SignedWidth4` at Akita revision `e023b6a46143f7b23199ccecb2684192baf7b640` with Akita's `dev-protocol` feature (catalogs regenerated and audited at that revision; Section 12.3 describes the pin), with Akita fp64 `Dense`'s adaptive ring-dimension policy: the planner searches exact A/B/D dimension tuples for the first two fold levels (A over 64 through 2048, B and D over 64 through 256), then uses a monotone sequence of uniform dimensions from $`\{64,128\}`$. Each trusted catalog row records the dimensions it selected; no single ring dimension applies to every level. A/source decomposition is independent of the B/D opening decomposition. Width 4 uses exactly one balanced base-16 digit (`inner.log_basis = 4`). Width 14 delegates its inner basis range to Akita's `Dense::inner_basis_range()` (3 through 16); the raw width-14 bound clips the source search to 3 through 14. The catalog selects inner log basis / digit count 5/3 at 13–15 variables, 14/1 at 16–19, and 8/2 at 20–26. These selections apply to either width-14 group at that physical shape; the planner determines the count without a three-digit cap.

The opening parameters remain `log_basis = 3`, `log_open_bound = Some(64)`, and opening basis range 3 through 6. The nominal opening decomposition uses 22 base-8 digits; this does not determine the A/source digit count. The commitment bounds are 14 and 4. The physical table of each polynomial is zero-extended independently to at least $`2^{13}`$ cells. Polynomial count is part of the catalog key and never adds plane coordinates to an opening point.

Width 14 is required for $`S_1`$. Falcon centers $`s_1`$ modulo $`q`$, so $`|s_1|\leq6{,}144`$. At Falcon-512, the norm permits a coefficient as large as $`\lfloor\sqrt{34{,}034{,}726}\rfloor=5{,}833`$, greater than the positive width-13 maximum 4,095. At Falcon-1024 the Falcon centering bound remains 6,144.

Width 14 for $`e`$ is also required. The four slack squares sum to
```math
\delta=\lfloor\beta^2\rfloor-\lVert s_1\rVert_2^2-\lVert s_2\rVert_2^2.
\qquad(5.1)
```
At Falcon-512 each component is at most 5,833. At Falcon-1024, $`\delta`$ can exceed $`8{,}191^2=67{,}092{,}481`$, and a greedy largest-square-first decomposition may produce a component as large as 8,382. The honest producer MUST start its first component at $`\min(\lfloor\sqrt\delta\rfloor,8{,}191)`$ and MUST skip a choice if the remaining value has integer square root greater than 8,191. The exhaustive check over $`(8{,}191^2,70{,}265{,}242]`$ finds a four-square decomposition with every $`|e_a|\leq8{,}191`$. The standard-library-only check is specs/scripts/falcon_slack_width_check.py; its optional range arguments select a subrange. The full exhaustive run is not part of spec CI.

Every polynomial in the digit group and budget group fits width 4. Their exact alphabets are proved by the range predicates of Sections 7.2 and 7.3. Width 4 does not replace those predicates.

The norm predicate, digit range checks, and sign check establish the source bounds used by soundness. A PCS width alone MUST NOT be used as an extracted range guarantee. The final Akita group MUST be the width-4 budget group $`[u_0,u_1,u_2]`$. The budget group preserves canonical commitment order in Akita execution. These are estimates for the complete grouped opening, not complete Falcon frame sizes.

| Degree | Padded batch | Grouped-opening payload bytes | Setup field elements | $`S_1`$ inner log basis / digits | $`e`$ inner log basis / digits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 62,956 | 131,072 | 5 / 3 | 5 / 3 |
| 1024 | 4 | 62,956 | 131,072 | 5 / 3 | 5 / 3 |
| 512 | 1,024 | 64,909 | 262,144 | 14 / 1 | 5 / 3 |
| 1024 | 1,024 | 65,737 | 327,680 | 8 / 2 | 5 / 3 |
| 512 | 65,536 | 68,872 | 1,048,576 | 8 / 2 | 14 / 1 |
| 1024 | 65,536 | 68,988 | 1,469,952 | 8 / 2 | 14 / 1 |

Both width-4 groups use inner log basis 4 and one source digit in every row.

## 6. Clear structured projections

### 6.1 Sources and verifier predicates

The norm projection source is the virtual concatenation $`S=S_1\Vert e`$, with $`2n`$ slots per record:
```math
S(j+na,i)=
\begin{cases}
S_1(j,i),&a=0,\\
e(\mathrm{bits}_2(j),i),&a=1,\ j\lt 4,\\
0,&a=1,\ j\geq4.
\end{cases}
\qquad(6.1)
```
For $`j\lt 4`$, $`\mathrm{bits}_2(j)=(k_0,k_1)`$ is defined by $`j=k_0+2k_1`$, with $`k_0,k_1\in\{0,1\}`$. The flat index is $`j+na+2ni`$. The four $`e`$ values occupy their native slots; no dense epsilon table is committed.

The virtual residual is
```math
\Delta(j,i)=S_1(j,i)+(h_i\circledast S_2(\cdot,i))_j-c_{i,j}
\quad\text{in }\mathbb F_p.
\qquad(6.2)
```
It is not committed. Its flat index is $`j+ni`$.

A transcript-derived seed defines the structured public linear maps
```math
\Pi_{\rm norm}:\mathbb F_p^{2nN_{\rm sig}^\star}\to\mathbb F_p^{156},
\qquad
\Pi_{\rm res}:\mathbb F_p^{nN_{\rm sig}^\star}\to\mathbb F_p^{31}.
```
The prover sends
```math
y_{\rm norm}=\Pi_{\rm norm}(S),\qquad
y_{\rm res}=\Pi_{\rm res}(\Delta).
\qquad(6.3)
```
Let $`\mathrm{ctr}_p`$ be the centered integer representative in
$`(-p/2,p/2)`$. The verifier checks
```math
\sum_{o\lt 156}\mathrm{ctr}_p(y_{{\rm norm},o})^2\leq \Theta
\qquad(6.4)
```
and
```math
q\mid\mathrm{ctr}_p(y_{{\rm res},o})\quad\text{for all }o\lt 31.
\qquad(6.5)
```
The checks are over centered integers. The projection theorem in the companion document bounds the soundness error. The norm projection bounds the centered energy of $`S_1\Vert e`$. The residual projection proves divisibility of $`\Delta`$ by $`q`$. For honest projections, the rejection probability is less than $`2^{-128.9}`$ for both parameter sets; on failure the prover returns the distinct projection-rejection error and MUST NOT retry.

### 6.2 Tensor geometry

Let
```math
b_{\rm split}=12-(d_{\rm coef}+1),\qquad R=d_{\rm sig}-b_{\rm split}.
\qquad(6.6)
```
Write the row index as $`i=i_{\lt b_{\rm split}}+2^{b_{\rm split}} i_R`$. The geometry requires $`d_{\rm sig}\geq b_{\rm split}`$.

The norm source is reshaped with its low 12 bits first: coefficient index $`j`$, lane bit $`a`$, and the low $`b_{\rm split}`$ row bits $`i_{\lt b_{\rm split}}`$. Its first radix is 4,096. The remaining $`R`$ row bits are taken consecutively from least to most significant and split over four further cores: each of the first three chunks takes $`\min(4,\text{remaining bits})`$ bits, and the fourth takes the remainder. A zero-bit chunk has radix one and remains in the profile.

The residual source is reshaped with low 11 bits first: $`j`$ and $`i_{\lt b_{\rm split}}`$, so its first radix is 2,048. Its remaining bits are taken consecutively from least to most significant and split between two cores, with $`\lfloor R/2\rfloor`$ bits in the second block and $`\lceil R/2\rceil`$ in the third.

| Parameter set at $`N_{\rm sig}^\star=2^{16}`$ | $`b_{\rm split}`$ | $`R`$ | Norm radices | Residual radices |
| --- | ---: | ---: | --- | --- |
| Falcon-512 | 2 | 14 | $`(4{,}096,16,16,16,4)`$ | $`(2{,}048,128,128)`$ |
| Falcon-1024 | 1 | 15 | $`(4{,}096,16,16,16,8)`$ | $`(2{,}048,128,256)`$ |

The first norm and residual blocks cover the same coefficient and low-row bits; the norm block also has the lane bit. This alignment is used by the projection-link verifier.

### 6.3 Core distributions and derivation

The profile uses 20 independent cores: 16 norm-prefix cores, one norm-terminal core, and three residual cores. This count matches the frozen projection certificate.

For each norm group $`g\in\{0,1,2,3\}`$, reshape its source by the five radix blocks of Section 6.2 as $`S[x_1,\ldots,x_5]`$. Four independent prefix cores have entries in $`\{-1,0,1\}`$ and shapes
```math
A_{g,1}\in\{-1,0,1\}^{43\times r_1},\qquad
A_{g,t}\in\{-1,0,1\}^{43\times(43r_t)}\quad(t=2,3,4).
```
They contract by
```math
Y_{g,1}[c,x_2,\ldots,x_5]=\sum_{x_1\lt r_1}A_{g,1}[c,x_1]S[x_1,\ldots,x_5],
```
```math
Y_{g,t}[c,x_{t+1},\ldots,x_5]=\sum_{c'\lt 43}\sum_{x_t\lt r_t}A_{g,t}[c,c'+43x_t]Y_{g,t-1}[c',x_t,\ldots,x_5],\quad t=2,3,4.
```
Concatenate the four final channels by $`(g,c)\mapsto43g+c`$. A fresh terminal core $`T`$ has shape $`156\times(172r_5)`$ and gives
```math
y_{{\rm norm},o}=\sum_{g\lt 4}\sum_{c\lt 43}\sum_{x_5\lt r_5}T[o,(43g+c)+172x_5]Y_{g,4}[c,x_5].
```
The terminal column is $`(43g+c)+172x_5`$. Every norm entry has
```math
\Pr[-1]=\tfrac14,\qquad\Pr[0]=\tfrac12,\qquad\Pr[1]=\tfrac14.
\qquad(6.7)
```
Within each byte, four consecutive entries consume bit pairs from least to most significant. Pair $`00`$ maps to $`-1`$, pair $`11`$ to $`1`$, and pairs $`01`$ and $`10`$ to $`0`$.

For the residual leg, reshape $`\Delta[x_1,x_2,x_3]`$ by the three radices. Three fresh cores with 31 rows have shapes $`31\times r'_1`$, $`31\times(31r'_2)`$, and $`31\times(31r'_3)`$. Their entries are independent uniform values in $`\{-12,\ldots,12\}`$. For $`t=2,3`$, use column $`c'+31x_t`$ and contract the incoming channel $`c'`$ with radix coordinate $`x_t`$; the first core uses column $`x_1`$. The output is
```math
y_{{\rm res},o}=\sum_{c'\lt 31}\sum_{x_3\lt r'_3}B_3[o,c'+31x_3]
\sum_{c''\lt 31}\sum_{x_2\lt r'_2}B_2[c',c''+31x_2]
\sum_{x_1\lt r'_1}B_1[c'',x_1]\Delta[x_1,x_2,x_3].
```
An octet $`b_{\rm byte}`$ is accepted only if $`b_{\rm byte}\lt 250`$, and maps to $`(b_{\rm byte}\bmod25)-12`$. Values 250 through 255 are discarded and sampling continues. This rejection rule gives the exact uniform distribution.

The transcript samples one 32-byte seed after the four commitments. For each core row, form $`F_{\rm core}`$ as the ASCII label aerie/falcon/v1/projection/core-v1, followed by wire version LE16, parameter-set byte, $`N_{\rm sig}`$ and $`N_{\rm sig}^\star`$ as LE64, leg/group/layer as U8, row and column counts as LE64, radix count as U8, all radices as LE64, seed length as LE64, and the seed. Derive row $`a`$ as $`\mathrm{SHAKE256}(F_{\rm core}\Vert\mathrm{LE64}(a))`$. The norm prefix cores use leg 0, group bytes 0 through 3, and layer bytes 1 through 4. The norm terminal uses leg 0, group 255, layer 5. Residual cores use leg 1, group 0, layers 1 through 3. Each core row has an independent SHAKE input. A core is shared only across its own fibers; implementations MUST NOT reuse a core or row across groups or legs or replace the row streams with one monolithic stream.

### 6.4 Projection profile

The exact profile record is bound after $`N_{\rm sig}^\star`$ and before the group descriptors. Its serialization is the ASCII label aerie/falcon/v1/projection/profile, then the norm fields in order: LE64(4), four LE64(43) prefix widths, LE64(156), five LE64 radices, U8(0) for the ternary distribution, U8(0) for source order $`S_1\Vert e`$, and LE128($`\Theta`$). Next encode the residual fields: three LE64(31) widths, three LE64 radices, U8(1)||U8(12) for the uniform-symmetric distribution and LE64(12,289) for the modulus. The parameter-set byte, $`n`$, $`N_{\rm sig}`$, and $`N_{\rm sig}^\star`$ are bound by the enclosing transcript context.

| Parameter set | Norm radices | Residual radices | Norm cap $`\Theta`$ |
| --- | --- | --- | ---: |
| Falcon-512 | derived from $`b_{\rm split}=2`$, $`R=d_{\rm sig}-2`$ | derived from $`b_{\rm split}=2`$, $`R=d_{\rm sig}-2`$ | $`5{,}092{,}300{,}183{,}991{,}442{,}303{,}232`$ |
| Falcon-1024 | derived from $`b_{\rm split}=1`$, $`R=d_{\rm sig}-1`$ | derived from $`b_{\rm split}=1`$, $`R=d_{\rm sig}-1`$ | $`10{,}513{,}136{,}047{,}130{,}310{,}946{,}696`$ |

The radix derivation is Section 6.2. At $`N_{\rm sig}^\star=2^{16}`$ the norm radices are $`(4096,16,16,16,4)`$ at Falcon-512 and $`(4096,16,16,16,8)`$ at Falcon-1024; residual radices are $`(2048,128,128)`$ and $`(2048,128,256)`$, respectively. Both caps are order-46 moment caps. Any profile change requires a new certificate and companion-theorem update. The verifier MUST reject $`N_{\rm sig}^\star\gt 2^{16}`$.

## 7. The three-stage PIOP

### 7.1 Commit and project

Before any challenge, bind the full instance of Section 10.1, including the salt mode, ordered public records, and setup/catalog/schedule identities. The prover commits to $`[S_1]`$, $`[e]`$, $`[a_0,a_1,a_2,\sigma]`$, and $`[u_0,u_1,u_2]`$. All four commitments precede the projection seed.

The transcript samples the projection seed and derives the cores of Section 6.3. The prover computes both projections, checks (6.4) and (6.5), and sends the canonical output message. It MUST return a distinct projection-rejection error if an honest predicate fails. It MUST NOT resample the seed.

After the outputs are absorbed, sample $`\mu\in K^8`$. For output index $`o\lt 187`$, let $`o_t`$ be bit $`t`$ of $`o`$, and define
```math
\lambda_o=\prod_{t=0}^{7}\mu_t^{o_t},\qquad 0\leq o\lt 187.
\qquad(7.1)
```
Split $`\lambda=(\lambda_{\rm norm},\lambda_{\rm res})`$ according to the first 156 and last 31 outputs. Sample $`\tau\in K^{d_{\rm coef}+d_{\rm sig}}`$ and let $`\tau_{\rm sig}`$ be its final $`d_{\rm sig}`$ coordinates. Each stage is one sumcheck of $`d_{\rm coef}+d_{\rm sig}`$ rounds for a random linear combination of that stage’s instances. For stage $`s`$, sample a fresh vector $`\gamma^{(s)}`$ with one independent uniform coefficient in $`K`$ per instance, in the canonical instance order of Sections 7.2–7.4; do not derive coefficients as powers of one scalar. An instance with fewer variables sits out the leading $`d_{\rm inactive}`$ rounds and binds the remaining challenges in its own variable order. In stages 1 and 3, lift every local summand $`f`$ of arity $`v`$ to the full $`d`$ variables as $`2^{-(d-v)}f`$ on the active suffix, independently of the leading variables. This rule also applies when $`f`$ includes a local equality factor. It preserves the local Boolean sum; each inactive ordinary round halves that term's contribution, and its terminal contribution retains the factor $`2^{-(d-v)}`$. The displayed local identities below give the unscaled sums, and their scalar input claims are unchanged. Stage 2 instead multiplies each inner relation by the full shared equality table and uses the normalized recurrence of Section 10.3; only stage 2 omits this scale factor.

### 7.2 Stage 1: basis16_product and projection_link

Stage 1 has $`d_{\rm coef}+d_{\rm sig}`$ rounds and maximum individual degree 3. It batches basis16_product instances followed by projection_link. The $`\gamma^{(1)}`$ order is the basis16_product sources $`a_1,a_2`$, then $`u_2`$ at Falcon-1024, then the one projection_link identity. The round challenge vector is $`r^{(1)}`$, and each round sends three $`K`$ elements. A local instance applies each active round challenge in its own variable order.

For each source $`Z`$, set $`\tau_Z=\tau`$ for coefficient sources and $`\tau_Z=\tau_{\rm sig}`$ for $`u_2`$. For every $`Z\in\{a_1,a_2\}`$ and, at Falcon-1024, $`u_2`$, basis16_product proves
```math
0=\sum_x\mathrm{eq}(\tau_Z,x)
A_Z^{\rm lo}(x)A_Z^{\rm hi}(x).
\qquad(7.2)
```
At Falcon-1024, $`Z=u_2`$ is added with the signature point $`\tau_{\rm sig}`$. Thus this stage has two basis-16 instances at Falcon-512 and three at Falcon-1024. Their outputs are $`A_Z^{\rm lo}(r^{(1)})`$ and $`A_Z^{\rm hi}(r^{(1)})`$, or the signature suffix for $`u_2`$. The prover sends $`A_Z^{\rm lo}`$ and $`A_Z^{\rm hi}`$ at the corresponding stage-one terminal point. This splits the degree-eight basis-16 root polynomial and keeps the next stage at degree 5.

Projection_link proves that the clear outputs are the projection of the committed sources. Let $`w_{S_1},w_e,w_\Delta`$ be the public tables obtained by applying the transposes of the two projection maps to $`\lambda`$. It proves the one identity
```math
\langle\lambda_{\rm norm},y_{\rm norm}\rangle+\langle\lambda_{\rm res},y_{\rm res}\rangle
=\sum_{j,i}w_{S_1}(j,i)S_1(j,i)
+\sum_{k,i}w_e(k,i)e(k,i)
+\sum_{j,i}w_\Delta(j,i)\Delta(j,i).
\qquad(7.3)
```
The three terms have unit weight because they are parts of one identity. There are no selector variables. The $`e`$ instance has $`2+d_{\rm sig}`$ variables. It uses local order $`i_{b_{\rm split}},\ldots,i_{d_{\rm sig}-1}`$, then the two bits $`k_0,k_1`$ of slack slot $`k`$, least significant first, then $`i_0,\ldots,i_{b_{\rm split}-1}`$. It is inactive during the first $`d_{\rm coef}-2`$ stage-one rounds. Its active challenge coordinates are reindexed into canonical $`(k_0,k_1,i_0,\ldots,i_{d_{\rm sig}-1})`$ order to define $`r_{\Pi,e}`$.

The coefficient-domain projection_link instance uses local order $`i_{b_{\rm split}},i_{b_{\rm split}+1},\ldots,i_{d_{\rm sig}-1}`$, then $`j_0,j_1,\ldots,j_{d_{\rm coef}-1}`$, then $`i_0,i_1,\ldots,i_{b_{\rm split}-1}`$. Its challenge sequence is reindexed into canonical $`(j_0,\ldots,j_{d_{\rm coef}-1},i_0,\ldots,i_{d_{\rm sig}-1})`$ order before defining $`r_\Pi`$ and evaluating public tables. The slack instance uses its own local variable order. Implementations MUST NOT infer that $`r_{\Pi,e}`$ is the last two coefficient coordinates followed by the same row coordinates of $`r_\Pi`$; the local variable orders can permute those coordinates. The verifier evaluates the public transposed projection weights by tensor contraction, without materializing a full $`2^{d_{\rm coef}+d_{\rm sig}}`$-entry weight table. For the norm leg, contract $`\lambda_{\rm norm}`$ backward through the terminal core, then cores 4, 3, and 2, keeping all radix fibers. Contract the resulting $`172`$ channels through each group’s first core to obtain weights for the norm source. The $`a=0`$ lane gives $`w_{S_1}(j,i)`$; lane $`a=1`$ gives $`w_e(k,i)`$ only at the four native slots and is zero elsewhere. For the residual leg, contract $`\lambda_{\rm res}`$ backward through cores 3 and 2, then core 1, obtaining $`w_\Delta(j,i)`$. Both transposes use the fixed column orders of Section 6.3.

The stage-one terminal claims are $`v_{S_1}=S_1(r_\Pi)`$, $`v_e=e(r_{\Pi,e})`$, and $`v_\Delta=\Delta(r_\Pi)`$, together with the four or six basis16_product child claims. The verifier checks the basis16_product terminal as $`\mathrm{eq}(\tau_Z,r_Z^{(1)})A_Z^{\rm lo}(r_Z^{(1)})A_Z^{\rm hi}(r_Z^{(1)})`$, where coefficient sources use $`(\tau,r^{(1)})`$ and $`u_2`$ uses their signature suffixes. The signature-only $`u_2`$ basis16_product terminal additionally carries the factor $`2^{-d_{\rm coef}}`$. It checks projection_link from $`w_{S_1}(r_\Pi)v_{S_1}+w_\Delta(r_\Pi)v_\Delta+2^{-(d_{\rm coef}-2)}w_e(r_{\Pi,e})v_e`$. After absorbing those claims, the verifier computes the public target evaluation $`C(r_\Pi)`$ and sets
```math
v_{\rm ring}=v_\Delta-v_{S_1}+C(r_\Pi).
\qquad(7.4)
```
This is a claim about the public ring functional on $`S_2=\sigma M`$.

### 7.3 Stage 2: range_leaf

After the stage-one claims are absorbed, sample fresh independent $`\gamma^{(2)}`$ coefficients for range_leaf instances in this order: $`a_1`$ low and high, $`a_2`$ low and high, $`u_2`$ low and high at Falcon-1024, then base-8 sources $`a_0,u_0,u_1,u_2`$ (omitting $`u_2`$ at Falcon-1024). Stage 2 has $`d_{\rm coef}+d_{\rm sig}`$ rounds, ends at $`r^{(2)}`$, and has maximum individual degree 5. It sends four $`K`$ elements per round.

For each base-16 source $`Z`$, stage 1 supplied the two child claims. Stage 2 proves
```math
v_Z^b=\sum_x\mathrm{eq}(r_Z^{(1)},x)
Q_{16}^b(P_Z(x)),
\qquad b\in\{\mathrm{lo},\mathrm{hi}\},
\qquad(7.5)
```
The base-16 sources are $`a_1,a_2`$, and additionally $`u_2`$ at Falcon-1024. For every range_leaf source $`Z`$, set $`r_Z^{(1)}=r^{(1)}`$ on the coefficient domain and $`r_Z^{(1)}=r^{(1)}_{\rm sig}`$ on the signature domain.

For every base-8 source $`Z`$, stage 2 proves
```math
0=\sum_x\mathrm{eq}(r_Z^{(1)},x)Q_8(P_Z(x)).
\qquad(7.6)
```
The base-8 sources are $`a_0,u_0,u_1,u_2`$ at Falcon-512 and $`a_0,u_0,u_1`$ at Falcon-1024. Signature sources use their suffix point $`r_Z^{(1)}=r^{(1)}_{\rm sig}`$. There are eight range_leaf instances at Falcon-512 and nine at Falcon-1024.

The output claims are the six folded-digit evaluations
```math
v^P_{a_0},v^P_{a_1},v^P_{a_2}
\quad\text{at }r^{(2)},\qquad
v^P_{u_0},v^P_{u_1},v^P_{u_2}
\quad\text{at }r^{(2)}_{\rm sig}.
\qquad(7.7)
```
For stage 2, extend each signature-domain inner relation independently of the leading coefficient variables. Multiply it by the full shared equality table: summing its leading equality coordinates gives one, so this lift uses no $`2^{-d_{\rm coef}}`$ factor. The normalized shared-equality recurrence consumes the equality point $`r^{(1)}`$ (or its signature suffix). The stage-two terminal is $`Q_8(v^P_Z)`$ or $`Q_{16}^b(v^P_Z)`$; it has no extra equality factor. The prover and verifier absorb these six claims before sampling $`\gamma^{(3)}`$.

### 7.4 Stage 3: remaining relations

Stage 3 has $`d_{\rm coef}+d_{\rm sig}`$ rounds and maximum individual degree 3. Coefficient instances use canonical coefficient-then-row order, slack instances use $`(k_0,k_1,i_0,\ldots,i_{d_{\rm sig}-1})`$, and signature instances use row order, all least significant bit first. In $`\gamma^{(3)}`$ order it batches range_source_link for $`a_0,a_1,a_2,u_0,u_1,u_2`$, then sign_check, norm_equality, encoding_budget, ring_functional, s1_reduction, and slack_reduction.

Range_source_link ties the stage-two folded-digit claims to the committed digits:
```math
v^P_Z=\sum_x\mathrm{eq}(r^{(2)},x)Z(x)(Z(x)+1),
\quad Z\in\{a_0,a_1,a_2\},
\qquad(7.8)
```
and
```math
v^P_{u_t}=\sum_i\mathrm{eq}(r^{(2)}_{\rm sig},i)u_t(i)(u_t(i)+1),
\quad t\in\{0,1,2\}.
\qquad(7.9)
```
Sign_check proves
```math
0=\sum_x\mathrm{eq}(r^{(2)},x)(\sigma(x)^2-1).
\qquad(7.10)
```
Together with the digit ranges, this gives $`S_2=\sigma M`$ and
$`|S_2|\leq2{,}047`$ on every coefficient.

Norm_equality proves
```math
\sum_{j,i}\mathrm{eq}(r^{(2)}_{\rm sig},i)
\bigl(S_1(j,i)^2+M(j,i)^2\bigr)
+\sum_{k,i}\mathrm{eq}(r^{(2)}_{\rm sig},i)e(k,i)^2
=
\lfloor\beta^2\rfloor
\sum_{i\lt N_{\rm sig}}\mathrm{eq}(r^{(2)}_{\rm sig},i).
\qquad(7.11)
```
The right side is public and is evaluated from $`N_{\rm sig}`$ in $`O(d_{\rm sig})`$ operations. Equation (7.10) implies $`\sigma^2=1`$ on the Boolean domain, so $`S_2^2=M^2`$ in the norm identity. The $`e`$ term ends at its native slack point.

Encoding_budget proves
```math
\begin{aligned}
&\sum_{j,i}\mathrm{eq}(r^{(2)}_{\rm sig},i)(a_2(j,i)+8)\\
&\quad+\sum_i\mathrm{eq}(r^{(2)}_{\rm sig},i)
(u_0(i)+8u_1(i)+64u_2(i))
=L-\kappa.
\end{aligned}
\qquad(7.12)
```
The checked ranges make the unused budget nonnegative. This proves the unary length bound.

For ring_functional, define the public table $`w_{\rm ring}`$ from Section 8. The stage-one claims give (7.4), and stage 3 proves
```math
v_{\rm ring}=\sum_{j,i}w_{\rm ring}(j,i)\sigma(j,i)M(j,i).
\qquad(7.13)
```
The summand agrees on the Boolean domain with the linear functional of $`S_2`$ and has individual degree 3.

The two reduction instances move the stage-one $`S_1`$ and $`e`$ claims to their stage-three opening points:
```math
v_{S_1}=\sum_{j,i}\mathrm{eq}(r_\Pi,(j,i))S_1(j,i),
\qquad(7.14)
```
```math
v_e=\sum_{k,i}\mathrm{eq}(r_{\Pi,e},(k,i))e(k,i).
\qquad(7.15)
```
These reductions ensure that each committed group needs one opening point.

### 7.5 Terminal evaluations and Akita opening

At the stage-three terminal, the verifier checks each instance after applying the corresponding equality factor: $`\mathrm{eq}(r^{(2)},r^{(3)})`$ for coefficient range links and sign_check, $`\mathrm{eq}(r^{(2)}_{\rm sig},r^{(3)}_{\rm sig})`$ for signature and norm/budget terms, $`\mathrm{eq}(r_\Pi,r^{(3)})`$ for s1_reduction, and $`\mathrm{eq}(r_{\Pi,e},r^{(3)}_e)`$ for slack_reduction. The ring_functional terminal is $`w_{\rm ring}(r^{(3)})\sigma(r^{(3)})M(r^{(3)})`$.

The ordinary-stage lifting factors of Section 7.1 also apply at this terminal: multiply each signature range_source_link and the signature term of encoding_budget by $`2^{-d_{\rm coef}}`$, and multiply the slack term of norm_equality and slack_reduction by $`2^{-(d_{\rm coef}-2)}`$. Coefficient-domain terms have factor one. Local equality factors do not remove these scales.

The prover sends nine committed-polynomial evaluations in canonical group order:
```math
S_1(r^{(3)}),\quad e(r^{(3)}_e),\quad
a_0(r^{(3)}),a_1(r^{(3)}),a_2(r^{(3)}),\sigma(r^{(3)}),
\quad u_0(r^{(3)}_{\rm sig}),u_1(r^{(3)}_{\rm sig}),u_2(r^{(3)}_{\rm sig}).
\qquad(7.16)
```
Here $`r^{(3)}_e`$ and $`r^{(3)}_{\rm sig}`$ are the terminal points in the corresponding local variable orders. They are suffix points of the stage-three challenge sequence. The final batched sumcheck terminal uses the products with the equality factors stated above; one opening authenticates these values at their canonical group points. The verifier checks the batched stage-three terminal equation from these evaluations, the public values, the projection weights, and the equality weights. For range_source_link it evaluates $`Z(r^{(3)})(Z(r^{(3)})+1)`$ and its signature analogue; for norm_equality it evaluates the products of the opened values for $`S_1`$, $`M`$, and $`e`$ at their own group points; for encoding_budget it evaluates the affine combination of the opened digits.

The multilinear extension of $`M`$ is the affine combination
```math
M(r^{(3)})=
a_0(r^{(3)})+8a_1(r^{(3)})+128a_2(r^{(3)})+1{,}092.
\qquad(7.17)
```
The verifier computes $`w_{\rm ring}(r^{(3)})`$ from the public keys as specified in Section 8, after reindexing all stage-three coordinates to canonical coefficient, row, and slack-slot order. All stage-three terminal claims are evaluations of committed polynomials or of the public affine combination $`M`$. One Akita opening proves the evaluations for the four groups, with one point per group.

## 8. Public negacyclic functional

### 8.1 Ring weight

Write $`r_\Pi=(r_{\Pi,\rm coef},r_{\Pi,\rm sig})`$ in canonical coefficient-then-row order. This point is obtained by reindexing the projection_link challenge sequence. For $`z\in K^{d_{\rm coef}}`$, define
```math
\omega_z(k)=\mathrm{eq}(z,\mathrm{bits}_{d_{\rm coef}}(k)).
\qquad(8.1)
```
For public key row $`h_i`$, define
```math
K_{h_i,z}[j]=\sum_{a=0}^{n-1}h_{i,a}(-1)^{\lfloor(a+j)/n\rfloor}\omega_z((a+j)\bmod n).
\qquad(8.2)
```
Then
```math
(h_i\circledast S_{2,i})(z)=\sum_{j=0}^{n-1}K_{h_i,z}[j]S_2(j,i).
\qquad(8.3)
```
The full public coefficient weight is
```math
w_{\rm ring}(j,i)=\mathrm{eq}(r_{\Pi,\rm sig},i)K_{h_i,r_{\Pi,\rm coef}}[j].
\qquad(8.4)
```
Padded public-key rows are zero. Therefore
$`(h\circledast S_2)(r_\Pi)=\sum_{j,i}w_{\rm ring}(j,i)S_2(j,i)`$.

### 8.2 Prover materialization

The prover MAY materialize $`w_{\rm ring}`$ while stage 2 runs. It MUST compute each kernel with this factorized negacyclic transform, not with a quadratic scan for each coefficient. for $`0\leq t\lt d_{\rm coef}`$ define
```math
f_t(0)=1-r_{\Pi,\rm coef,t},\qquad f_t(1)=r_{\Pi,\rm coef,t},
```
```math
w^{(t)}[u]=\prod_{b=t}^{d_{\rm coef}-1}f_b(\mathrm{bit}_{b-t}(u)).
\qquad(8.5)
```
Thus $`w^{(0)}[k]=\omega_{r_{\Pi,\rm coef}}(k)`$ and $`w^{(d_{\rm coef})}=[1]`$. Define
```math
h_i^\dagger[0]=h_{i,0},\qquad h_i^\dagger[j]=-h_{i,n-j}\quad(1\leq j\lt n).
\qquad(8.6)
```
Then $`K_{h_i,r_{\Pi,\rm coef}}=h_i^\dagger\circledast w^{(0)}`$. For transform length $`D=2^{d_{\rm coef}-t}`$, let $`\mathcal T_t(A)=A\circledast w^{(t)}`$. Write $`A(X)=A_0(Y)+XA_1(Y)`$ with $`Y=X^2`$, and let $`\mathrm{rot}(A_1)=YA_1\bmod(Y^{D/2}+1)`$. Set $`f_0=f_t(0)`$ and $`f_1=f_t(1)`$. Then
```math
\begin{aligned}
(\mathcal T_t(A))_{\rm even}&=\mathcal T_{t+1}(f_0A_0+f_1\mathrm{rot}(A_1)),\\
(\mathcal T_t(A))_{\rm odd}&=\mathcal T_{t+1}(f_1A_0+f_0A_1).
\end{aligned}
\qquad(8.7)
```
The base case is $`\mathcal T_{d_{\rm coef}}([a])=[a]`$. The recurrence follows from
$`w^{(t)}(X)=(f_t(0)+f_t(1)X)w^{(t+1)}(X^2)`$ in $`K[X]/(X^D+1)`$. It costs $`O(n\,d_{\rm coef})`$ operations per live key row and $`O(nN_{\rm sig}^\star)`$ output storage.

### 8.3 Verifier evaluation at the stage-three point

The verifier MUST evaluate $`w_{\rm ring}(r^{(3)})`$ without materializing its $`nN_{\rm sig}^\star`$ entries. For coefficient index $`a`$, define
```math
\Lambda_a=\sum_{j=0}^{n-1}\mathrm{eq}(r^{(3)}_{\rm coef},j)(-1)^{\lfloor(a+j)/n\rfloor}\mathrm{eq}(r_{\Pi,\rm coef},(a+j)\bmod n).
\qquad(8.8)
```
Compute all $`\Lambda_a`$ together in a binary prefix trie. Let $`l_t=(r_{\Pi,\rm coef})_t`$ and $`n_t=(r^{(3)}_{\rm coef})_t`$. Initialize the state at the empty prefix to $`(D_0(\emptyset,0),D_0(\emptyset,1))=(1,0)`$. At bit $`t`$, set
```math
p_t=n_tl_t,\qquad g_t=1-n_t-l_t+2p_t,\qquad l_t^-=l_t-p_t,\qquad n_t^-=n_t-p_t.
\qquad(8.9)
```
For every prefix $`a_{\lt t}`$ with state $`(d_0,d_1)`$, update its two child states by
```math
\begin{aligned}
D_{t+1}(a_{\lt t}\Vert1,0)&=l_t^-d_0,&
D_{t+1}(a_{\lt t}\Vert1,1)&=n_t^-d_0+g_td_1,\\
D_{t+1}(a_{\lt t}\Vert0,0)&=g_td_0+l_t^-d_1,&
D_{t+1}(a_{\lt t}\Vert0,1)&=n_t^-d_1.
\end{aligned}
\qquad(8.10)
```
After $`d_{\rm coef}`$ levels, $`\Lambda_a=D_{d_{\rm coef}}(a,0)-D_{d_{\rm coef}}(a,1)`$. The two states encode the final carry 0 or 1 in $`a+j`$; the second state supplies the negacyclic wrap sign. The trie computes all values in $`O(n)`$ field operations.

For each row set
```math
J_i=\mathrm{eq}(r_{\Pi,\rm sig},i)\mathrm{eq}(r^{(3)}_{\rm sig},i).
```
The terminal weight is
```math
w_{\rm ring}(r^{(3)})=\sum_{i\lt N_{\rm sig}^\star}J_i\sum_{a=0}^{n-1}h_{i,a}\Lambda_a.
\qquad(8.11)
```
The verifier forms the row factors in $`O(N_{\rm sig}^\star)`$ operations and streams each public-key inner product. Padded key rows are zero. Total work is $`O(n+N_{\rm sig}^\star+nN_{\rm sig})`$ field operations and storage is $`O(n+N_{\rm sig}^\star)`$.

## 9. Integer lifting and soundness

### 9.1 Projection energy bound

Define the centered energy
```math
\mathcal E(S)=\sum_{u\lt 2nN_{\rm sig}^\star}\mathrm{ctr}_p(S[u])^2,\qquad
Q_S=\frac{\Theta+1}{\eta_{\rm proj}},
\qquad
\eta_{\rm proj}=2^{15}.
\qquad(9.1)
```
The companion projection theorem proves that an accepted norm predicate with
$`\mathcal E(S)\geq Q_S`$ has probability at most $`2^{-136}`$ (certified $`2^{-136.4416}`$), including its terminal escape term. Its proof uses the certified Laplace pair $`t=3,K_{\rm L}=21/40`$, exactness thresholds, and one global energy bound. It does not assume an Akita source range.

For Falcon-512, $`\log_2 Q_S=57.1088`$, and the norm-lift slack is 6.89 bits. For Falcon-1024, $`Q_S\lt 2^{58.16}`$, and the lift slack is 5.85 bits. The cap $`\Theta`$, and therefore $`Q_S`$, is bound to the parameter set by Section 6.4.

### 9.2 Exact norm lift

The range checks give $`|S_2|\leq2{,}047`$, hence
```math
\lVert s_{2,i}\rVert_2^2\leq n\cdot2{,}047^2.
\qquad(9.2)
```
For a fixed record, its complete nonnegative norm sum is less than
$`Q_S+n\cdot2{,}047^2`$. This is less than $`p`$ at both parameter sets:
```math
Q_S+512\cdot2{,}047^2
\lt 155{,}404{,}670{,}846{,}297{,}723\lt p,
\qquad(9.3)
```
```math
Q_S+1{,}024\cdot2{,}047^2
\leq320{,}835{,}454{,}947{,}826{,}963\lt p.
\qquad(9.4)
```
Thus the field norm identity lifts to the integers. For live rows it proves Falcon's norm bound and
```math
|S_1|,|e|\leq\lfloor\sqrt{\lfloor\beta^2\rfloor}\rfloor,
```
which is 5,833 at 512 and 8,382 at 1024. For padded rows, it forces every semantic value $`S_1,e,S_2`$ to zero. One global energy bound controls every row; there is no union bound over records.

The extracted $`S_1`$ polynomial need not be centered. Centering its coefficients modulo $`q`$ cannot increase their absolute values and yields the canonical Falcon representative, bounded by 6,144. The width declarations of Section 5 constrain honest producer input and PCS sizing only; they do not impose the malicious prover's source range. In particular, the maliciously extracted slack values can be as large as 8,382 at Falcon-1024 even though the honest producer uses the width-14 interval ending at 8,191.

### 9.3 Residual bound

For both parameter sets, centered public-key and target coefficients have absolute value at most 6,144. By Cauchy-Schwarz,
```math
|(\,h_i\circledast s_{2,i}\,)_j|
\leq6{,}144\sqrt{n\lfloor\beta^2\rfloor}.
\qquad(9.5)
```
Let $`\Delta^{\mathbb Z}(j,i)=\mathrm{ctr}_p(S_1(j,i))+(h_i\circledast s_{2,i})_j-c_{i,j}`$ be the integer residual, where $`s_{2,i,j}=\mathrm{ctr}_p(S_2(j,i))`$ and $`h_i,c_i`$ use their centered modulo-$`q`$ representatives. In particular, form this residual before centering the extracted $`S_1`$ modulo $`q`$. Then
```math
|\Delta^{\mathbb Z}(j,i)|
\leq |\mathrm{ctr}_p(S_1(j,i))|+
6{,}144\sqrt{n\lfloor\beta^2\rfloor}+6{,}144
\leq L_\Delta.
\qquad(9.6)
```
The parameter values are:

| Parameter set | $`L_\Delta`$ | $`H_\Delta=2^{16}nL_\Delta^2`$ | $`\lfloor L_\Delta/q\rfloor`$ |
| --- | ---: | ---: | ---: |
| Falcon-512 | 811,061,395 | approximately $`2^{84.19}`$ | 65,998 |
| Falcon-1024 | 1,648,068,606 | approximately $`2^{87.24}`$ | 134,109 |

Both $`L_\Delta\lt p/2`$, so the field table $`\Delta`$ is the reduction of
$`\Delta^{\mathbb Z}`$, and its centered representatives are those integers.

### 9.4 Divisibility and extracted ring equation

Conditioned on the norm lift, the residual theorem bounds acceptance of a source with $`\Delta^{\mathbb Z}\not\equiv0\pmod q`$ by a wrap term and an annihilation term:
```math
\varepsilon_{\rm kill}=3\cdot25^{-31}\approx2^{-142.37}.
\qquad(9.7)
```
The wrap term is at most $`2^{-437.7}`$ at Falcon-512 and less than $`2^{-153.74}`$ at Falcon-1024. Outside these events,
```math
v_i=\Delta_i^{\mathbb Z}/q\in\mathbb Z[X]/(X^n+1),
\qquad |v_{i,j}|\leq\lfloor L_\Delta/q\rfloor.
\qquad(9.8)
```
Therefore $`s_{1,i}+h_i s_{2,i}-qv_i=c_i`$ over the integers, where $`s_{1,i}`$ has coefficients $`\mathrm{ctr}_p(S_1(j,i))`$. Centering $`s_{1,i}`$ modulo $`q`$ afterward changes it by a multiple of $`q`$ and cannot increase its norm, so it yields the canonical Falcon verification relation. The quotient is recovered during extraction and is never committed.

The clear-projection theorem's total statistical term is at most $`2^{-136}`$ per transcript for both parameter sets (certified $`2^{-136.418}`$). Its case split is over the fixed pre-seed sources. It has no union loss over records, groups, fibers, coordinates, or cores. See [falcon-projection-theorem.md](falcon-projection-theorem.md), Theorem 6.1 and Corollary 6.2.

### 9.5 PIOP soundness ledger

Let $`R_s=d_{\rm coef}+d_{\rm sig}`$ be the rounds in stage $`s`$. A degree-$`d_s`$ sumcheck contributes at most $`d_sR_s/|K|`$. The monomial projection batching has error at most $`8/|K|`$: if the 187 outputs are wrong, their weighted difference is a nonzero polynomial of degree at most 8 in $`\mu\in K^8`$.

For the reference profiles in Section 2.1, each stage has 24 rounds. The union bound over these checks, without proof-of-work, is:

| Source | Numerator over $`\vert K\vert `$ |
| --- | ---: |
| Stage sumchecks, $`(3+5+3)\cdot24`$ | 264 |
| Basis-16 zero checks at $`\tau`$ | at most 62 |
| Base-8 zero check $`a_0`$ at $`r^{(1)}`$ | 24 |
| Remaining base-8 zero checks at $`r^{(1)}`$ | at most 45 |
| Sign, norm, and budget zero checks at $`r^{(2)}`$ | at most 54 |
| Monomial output batching | 8 |
| Three stage batching challenges | 3 |
| Total | at most 460 |

This union bound is $`460/|K|\lt 2^{-119}`$. It is an algebraic check on the stage structure, not the deployed PIOP soundness term. The transcript grinds every PIOP field site to its Section 10.1 difficulty, so each site has weight $`2^{-g_r}e_r\leq2^{-128}`$, and the ledgers charge the sites through the per-query classical-ROM bound ([falcon-ledger.md](falcon-ledger.md), Section 3). The 460 check applies to the 24-round reference profiles only. Other admitted batch sizes have parameter-dependent round counts and MUST NOT reuse 460; Section 10.1 allocates work for every admitted shape.

The complete composition ledger also includes the projection term by reference, all stage batching errors, the Akita extraction and PCS terms, and Fiat-Shamir composition. The ledgers in [falcon-ledger.md](falcon-ledger.md) and [falcon-1024-ledger.md](falcon-1024-ledger.md) record per-profile accounting. Their PIOP terms follow the three stages. PCS and catalog terms use the four-group catalogs.

### 9.6 Stage soundness and point reuse

The four group commitments fix every source before the projection seed and all sumcheck challenges. The public statement fixes $`h_i,c_i`$. Therefore $`\Delta`$, $`S`$, all digits, and all budget tables are fixed before the projection randomness.

Stage 1 proves the basis-16 product checks and the projection link. Stage 2 uses the unpredictable point $`r^{(1)}`$ to test base-8 range predicates and bind the stage-one virtual $`A_Z`$ claims. Stage 3 reuses $`r^{(2)}`$ for the folded-digit link, sign, norm, and encoding checks. This reuse is sound because the tested source tables were committed before $`r^{(2)}`$ was sampled.

The ring chain is linear in $`S_2`$. Given the correct stage-one claim on $`S_1`$, the stage-three ring_functional identity is correct exactly when
```math
v_{\rm ring}=(h\circledast S_2)(r_\Pi)
=\sum_{j,i}w_{\rm ring}(j,i)S_2(j,i).
```
On the Boolean domain, $`S_2=\sigma M`$. The summand $`w_{\rm ring}\sigma M`$ has degree 3, so the stage-three error for this identity is at most $`3(d_{\rm coef}+d_{\rm sig})/|K|`$. At the terminal, $`M(r^{(3)})`$ is the affine combination in (7.17). This authenticates $`v_\Delta`$ against $`\Delta=S_1+h\circledast S_2-C`$. The residual predicate then proves the ring equation modulo $`q`$.

## 10. Transcript and wire format

### 10.1 Transcript registry

This section freezes the first version-1 transcript, including work messages.
It specifies the target of steps E (exact-cardinality grind pricing and public
identities), F (native PIOP transcript and PCS session export), and G (PIOP
grinding). Section 12.3 records implementation status. Earlier development
proofs are not this format; the version remains 1 because no format was public.

#### Encoding and state

All integers are unsigned little endian unless explicitly stated otherwise.
`B(x) = LE64(|x|) || x` frames a byte string. ASCII strings have no terminator.
A base element is its canonical eight-byte residue in `[0,p)`, with
$`p=2^{64}-59`$. An extension element is `LE64(c0) || LE64(c1)` in the basis of
Section 4; it is not an integer reduced modulo $`p^2`$. Vectors concatenate
atoms without a wire length. The public shape fixes their counts.

The event-header and field-work nonce encoding below is normative for the Aerie PIOP stream only. The PCS uses Akita's native proof stream and native proof grammar without Aerie event headers.

The backend is Spongefish `ef9741346a150039427d9e0c02d6b2d73e93ec81`
`Blake2b512`, the pinned Akita default. Backend substitution, including Keccak,
is not a v1 option. Initialize the PIOP with `DomainSeparator::new(tag)`,
`.session(B("aerie/falcon/v1"))`, `.instance(B(instance_digest))`, then
`to_prover(Blake2b512::default())` or the corresponding verifier constructor.
Here `tag` is ASCII `aerie/falcon/v1/piop/blake2b` zero-padded to 64 bytes;
`B` denotes an encoding wrapper, not a second serialization by Spongefish.
Prover and verifier use separate native states and the same public duplex
operations. Private prover randomness never enters this public state.

Every event first absorbs one header as a single public message:

```
U8(kind) || B(site) || LE64(atom_count) || LE64(payload_bytes)
```

Kinds are 0 (public bytes), 1 (proof bytes), 2 (challenge), 3 (work nonce),
and 4 (state export). Thus a header is `25 + |site|` bytes. The site is the
literal ASCII identifier in the registry. An indexed suffix `/i` uses the
base-10 index with no leading zeros, starting at zero. Headers are derived,
not transmitted. For kind 0 or 1, absorb the payload next as one byte-string
message, without an implicit length prefix. Proof bytes are also appended to
the argument string. No per-atom absorb boundaries are inserted inside that
payload. Kind 2 has no absorbed payload: its byte count describes the decoded
challenge, not the variable number of sampler bytes. For kind 3, the header's
byte count is the public maximum native nonce length, not its realized
length: `ceil((g+7)/7)` for nonzero work. Its payload is one canonical
unsigned LEB128 integer, with no length prefix on the wire. Kind 4 carries
no payload and requests a 32-byte squeeze.
No diagnostic label may substitute for this absorbed header.

For a field challenge, first absorb its kind-2 header. If its scheduled
work difficulty $`g`$ is positive, absorb a kind-3 header with site
`site || "/work"`, count 1, and byte count `ceil((g+7)/7)`. This public
maximum is fixed before searching, so all candidates start from the same
state. Use Akita's native proof-of-work transition on that state:

1. For a candidate integer $`c`$ in $`[0,2^{g+7})`$, clone the public duplex
   state, absorb the canonical unsigned LEB128 bytes of $`c`$, and squeeze
   exactly 32 predicate bytes.
2. Accept the candidate if the first $`g`$ bits are zero, counting least
   significant bit first within each byte. The honest prover searches upward
   from zero. Failed previews MUST NOT change the live state, argument
   bytes, private randomness, or plan cursor.
3. Commit the selected native nonce once at this inline proof position,
   absorb the same bytes, and perform the same 32-byte predicate squeeze
   on the live state. The verifier receives and range-checks the integer,
   reproduces this transition, and rejects a failed predicate.
4. Draw the protected field challenge separately from the resulting state,
   using the exact sampler below. None of the predicate bytes is reused
   as a field challenge or rejection candidate.

Thus the predicate hashes the complete prechallenge transcript state, the
absorbed challenge and work site headers, and the native nonce bytes under
the stream's fixed backend. The instance binds the shape, schedule, and
work plan before this transition. It is not a hash of the nonce alone.
The predicate and protected challenge are distinct queries, as in Akita's
`specs/transcript-grinding.md`; their outputs have different roles.
Akita's native nonce grammar and absorption are retained for PCS, including
after exact-cardinality repricing. The event headers here are the existing
Aerie transcript-framing requirement, not a replacement nonce codec.

When $`g=0`$, emit no nonce bytes, omit the kind-3 header and predicate
squeeze, and sample the challenge directly after its kind-2 header. The
site remains in the public plan. The verifier derives every difficulty
from the trusted public shape and schedule; proof bytes cannot override it.
All candidate queries, including failed work and rejection sampling, count
in the ROM experiment. Concrete compressed prefixes and sponge realization
retain the binding and sampling assumptions stated below.

Each $`K`$ challenge samples limb 0, then limb 1. For each limb, squeeze exactly
eight successive bytes, interpret them as LE64, accept if less than $`p`$, and
otherwise consume another eight bytes. No high bits are masked for this
64-bit modulus. No bytes are reduced, skipped, reused, or discarded between
limbs. A field vector samples elements in increasing index order, under one event
header and, only when $`g\gt 0`$, one work nonce. This is Akita's exact native per-limb sampler,
without the old 32-byte squeeze and 16-byte modular reduction. Sampling
retries do not add proof bytes or new work nonces. The projection seed is
one unground kind-2 event consuming exactly 32 bytes.

#### Initial instance and identities

Let `H(label,x) = SHAKE256(B(ASCII(label)) || B(x), 32)`. These hashes are
public-input computations, not extra verifier challenges. The following
canonical instance record is computed by both parties before initialization:

```
LE16(1) || U8(parameter_set) || LE32(layout) || LE32(n)
|| LE64(N_sig) || LE64(N_sig_star) || U8(salt_mode)
|| public_records_digest
|| projection_profile || group_descriptors
|| setup_id || catalog_id || schedule_id
```

The layout is 5 or 6 for Falcon-512 or Falcon-1024. The salt-mode byte is 0
for per-record and 1 for common-salt. The 32-byte public-record digest uses
Section 2.3, including customization
`aerie/falcon/v1/public-records-binding/common-salt` in common-salt mode and
`aerie/falcon/v1/public-records-binding` otherwise. Both modes hash the full
expanded ordered records; the mode byte is bound in addition to this digest.

The ordered public records are the Fiat-Shamir statement. The verifier derives
$`h_i=\mathrm{Decode}(pk_i)`$ and
$`C_i=\mathrm{HashToPoint}(r_i\mathbin\|hpk_i\mathbin\|0x00\mathbin\|0x00\mathbin\|m_i)`$
from those records, never from prover-supplied rows. Under collision resistance,
$`hpk_i=\mathrm{SHAKE256}(pk_i,64)`$ fixes the canonical key bytes. HashToPoint
is a fixed function of the bound bytes; signing simulation reprograms it only
at fresh salts on which no earlier query can depend. No additional row digest
is included. Each row has exactly one public consumer: $h$ enters the ring
functional and $C$ the target evaluation. The verifier derives each row once,
inside that evaluation, and never materializes either table.

The profile is the 198-byte record in Section 6.4. Each group descriptor is
`U8(role) || U8(polynomials) || LE64(logical_cells_per_polynomial) || U8(family)`,
11 bytes. In order these are `(0,1,n*N*,0)`, `(1,1,4*N*,0)`,
`(2,4,n*N*,1)`, `(3,3,N*,1)`, where family 0 is width 14 and family 1 is width 4.
The complete instance record is therefore 398 bytes; its digest is
`H("aerie/falcon/v1/instance", record)`. All identities below are 32 bytes.

| Byte range | Field |
| --- | --- |
| `[0,2)` | Version LE16 |
| `[2,3)` | Parameter set U8 |
| `[3,7)` | Layout LE32 |
| `[7,11)` | Degree LE32 |
| `[11,19)` | Live records LE64 |
| `[19,27)` | Padded records LE64 |
| `[27,28)` | Salt mode U8 |
| `[28,60)` | Public-records digest |
| `[60,258)` | Projection profile |
| `[258,302)` | Four 11-byte group descriptors |
| `[302,334)` | Setup identity |
| `[334,366)` | Catalog identity |
| `[366,398)` | Schedule identity |

Use the pinned Akita identity exports byte-for-byte, with Blake2b-256
configured for a 32-byte output (not truncation of Blake2b-512):

- `setup_id = setup_seed_digest(setup_seed)`, the hash of the one-byte
  `Shake256PagedV1` derivation tag followed by the actual 32-byte public seed.
  This identifies the deterministic matrix stream. The selected profiles and
  expanded schedule identify its dimensions and uses. Local matrix caches,
  provisioned capacity, and pointer identity are not setup identity. Only
  this transparent setup derivation is admitted by v1.
- `catalog_id = catalog.catalog_digest()`, the hash of ASCII
  `AKITA-TRUSTED-SCHEDULE-CATALOG-V1 || B(family_name) || policy_digest ||
  LE64(row_count) || row_digest[0] || ... || row_digest[row_count-1]`, in the
  trusted catalog's canonical resolved row order. `policy_digest` and every
  row digest are 32 bytes. The catalog is the merged signed-width family
  selected by Aerie, not a prover-supplied file name or a hash of JSON text.
- `schedule_id = selection.row_digest`, the hash of ASCII
  `akita/schedule-row/v3 || U8(1) || LE64(4) || B(profile[0]) || ... ||
  B(profile[3]) || B(effective_schedule)`. The profile and effective schedule
  bytes are Akita's `canonical_descriptor_bytes`, after expanding the
  selected `.aks` row. Their exact bytes, not a schedule name or estimator
  score, define the identity. The verifier resolves them from its catalog.

These identity functions were introduced at Akita revision
`515a21668bd2fb7f8e3b0a41c325b7c253b87aed` and are carried at the current
pin `e023b6a46143f7b23199ccecb2684192baf7b640`; their encodings above define
Aerie v1. Step E provides exact-cardinality field-site pricing and the three
public identity functions. The setup seed serialization tag is `U8(1)`, not
the field-modulus tag. The verifier derives the selected schedule and its
grinding allocation from trusted public inputs before initializing the PIOP.
A catalog change, effective schedule change, or setup-seed change changes
the instance digest. Proof bytes cannot select an identity, schedule, or
difficulty.

#### Ordered PIOP events

The table lists every event after domain initialization. `A` means an absorb
(kind 0 for the instance and kind 1 otherwise); `C` means a challenge. Work
subevents are inserted by the rule above at every protected `C`. Let
$`b=2`$ for Falcon-512 and $`b=3`$ for Falcon-1024, and let
$`m=(3,8,12)`$ or $`(4,9,12)`$ be the stage combiner widths.

| Order | Site | Operation and canonical payload | Payload or decoded challenge bytes |
| ---: | --- | --- | ---: |
| 0 | `aerie/falcon/v1/instance` | A, full instance record, one byte-string atom | 398 |
| 1 | `aerie/falcon/v1/commit/s1` | A, 16 canonical base-field coefficients | 128 |
| 2 | `aerie/falcon/v1/commit/slack` | A, 16 canonical base-field coefficients | 128 |
| 3 | `aerie/falcon/v1/commit/s2-sources` | A, 16 canonical base-field coefficients | 128 |
| 4 | `aerie/falcon/v1/commit/budget` | A, 16 canonical base-field coefficients | 128 |
| 5 | `aerie/falcon/v1/projection/seed` | C, one 32-byte seed, no nonce | 32 |
| 6 | `aerie/falcon/v1/projection/output` | A, one canonical Rice section, including its two parameters and padding | $`2+\lceil(L(k_y)+L(k_z))/8\rceil`$ |
| 7 | `aerie/falcon/v1/mu` | C, all eight independent extension coordinates | 128 |
| 8 | `aerie/falcon/v1/tau` | C, all $`d`$ coordinates, coefficient bits then signature bits | $`16d`$ |
| 9 | `aerie/falcon/v1/stage1/gamma` | C, independent weights for $`a_1,a_2,[u_2]`$, projection link | $`16m_1`$ |
| 10 | `aerie/falcon/v1/stage1/round/t/message` | A, $`(g_0,g_2,g_3)`$; repeat for $`t=0,\ldots,d-1`$ | 48 |
| 11 | `aerie/falcon/v1/stage1/round/t/challenge` | C, $`r^{(1)}_t`$ immediately after that round's message | 16 |
| 12 | `aerie/falcon/v1/stage1/terminal` | A, $`2b`$ basis claims then three projection claims | $`16(2b+3)`$ |
| 13 | `aerie/falcon/v1/stage2/gamma` | C, the $`m_2`$ independent weights in the order below | $`16m_2`$ |
| 14 | `aerie/falcon/v1/stage2/round/t/message` | A, $`(q_1,q_2,q_3,q_4)`$; repeat for $`t=0,\ldots,d-1`$ | 64 |
| 15 | `aerie/falcon/v1/stage2/round/t/challenge` | C, $`r^{(2)}_t`$ immediately after that round's message | 16 |
| 16 | `aerie/falcon/v1/stage2/terminal` | A, six folded digits in Section 10.3 order | 96 |
| 17 | `aerie/falcon/v1/stage3/gamma` | C, twelve independent weights in the order below | 192 |
| 18 | `aerie/falcon/v1/stage3/round/t/message` | A, $`(g_0,g_2,g_3)`$; repeat for $`t=0,\ldots,d-1`$ | 48 |
| 19 | `aerie/falcon/v1/stage3/round/t/challenge` | C, $`r^{(3)}_t`$ immediately after that round's message | 16 |
| 20 | `aerie/falcon/v1/stage3/terminal` | A, the nine evaluations of (7.16) | 144 |
| 21 | `aerie/falcon/v1/pcs/session` | kind 4, state export after all terminal checks | 32 squeezed bytes |

In the indexed sites, replace the literal `t` by its canonical decimal index.
A round-message/challenge pair is repeated before proceeding to the next row
of the protocol, not as two separate batches of messages and challenges.
Atom counts are the displayed field-element counts, except byte strings have
count 1. A challenge vector's count is its number of extension elements.

Stage-two weights are ordered as low/high for $`a_1`$, low/high for $`a_2`$,
then low/high for $`u_2`$ at Falcon-1024, then low for $`a_0,u_0,u_1`$,
and finally low for $`u_2`$ at Falcon-512. Stage-three weights are ordered as
six source links $`a_0,a_1,a_2,u_0,u_1,u_2`$, sign, norm, encoding budget,
ring functional, $`S_1`$ reduction, and slack reduction. Deriving $`\lambda`$,
reindexing points, reconstructing omitted coefficients, computing public
claims, and padding PCS points with zeros are deterministic operations;
they create no challenge or proof message.

To export the session, clone the PIOP duplex state after event 20, absorb the
kind-4 header for `aerie/falcon/v1/pcs/session` with count 1 and length 32,
and squeeze exactly 32 bytes. Those bytes, without hex encoding or another
length prefix, are the PCS API's `session` argument. Neither the clone nor
the digest is transported. This is a domain-separated digest of the full
PIOP state, including the instance, commitments, all accepted work nonces,
all challenge/rejection history, Rice bytes, round messages, and terminal
claims. The fixed string `aerie/falcon/v1` is not a valid PCS session.
Pass the exported bytes to Akita's native PCS stream, which binds its own
complete opening statement. PCS operations are outside this ordered PIOP
event registry.

#### Akita PCS stream

The PCS is Akita's native proof stream at the pinned revision, with backend
tag `akita-pcs/native-proof-stream/v7/blake2b`. Akita initializes it with the
32-byte session exported by PIOP event 21 and Akita's canonical instance
descriptor for the trusted setup, selected schedule including its grinding
plan, opening layout, and basis. Akita binds commitments, points, openings,
and its native nonces using its own absorption rules. The verifier derives
the descriptor and schedule from trusted public inputs, never from proof data.

The PCS stream is opaque to Aerie event framing. Its backend tag and native
grammar are pinned with Akita; Aerie does not add site headers to PCS
operations. The selected schedule fixes all PCS work sites and difficulties.
Akita's terminal response grammar carries the scheduled Rice-coded response
coordinates with the schedule's `z_coords`, `z_rice_low_bits`,
`z_linf_cap`, and `z_payload_bytes` bounds. Akita's native
`send_bounded_bytes` and `receive_bounded_bytes` functions in
`akita-transcript/src/proof_stream.rs` encode the payload length as LE32, then carry
that many bytes. Akita's `akita-types/src/proof/tail_segments.rs` decodes the
scheduled signed coordinates with canonical Golomb-Rice encoding and enforces
the scheduled caps and byte limit. Akita's field-work nonces use canonical
unsigned LEB128 in `akita-transcript/src/proof_stream/nonce.rs`. These are
properties of the pinned Akita grammar, not additional Aerie PCS framing. In
this Aerie wire frame the complete native PCS argument remains length-prefixed
by LE64 as specified in Section 10.2.

Aerie builds Akita with its `dev-protocol` feature, which selects Akita's
recompute-the-last-block terminal (`RECOMPUTE_LAST_BLOCK` in
`akita-params/src/layout/tail_segments.rs`). For a terminal with $`m`$ live
blocks, the response carries the opening rings $`E_b`$ and inner-image rings
$`\mathbf t_b`$ of the first $`m-1`$ blocks only. In place of each omitted
last block the prover sends a 32-byte digest of its canonical coefficients,
at the transcript position where the block itself would have been absorbed.
The verifier solves both terminal ring relations for the last block, checks
the relations with the recovered block, and compares its digest with the one
received. A proof in this grammar is not accepted by a build without the
feature, and conversely.

The terminal response encodes signed coordinates in scheduled ring-major,
coefficient-minor order. For each coordinate $`v`$, Akita maps it to zigzag
$`u=2v`$ for $`v\geq0`$ and $`u=-2v-1`$ otherwise. With scheduled
$`k=\mathrm{z\_rice\_low\_bits}`$, it emits $`\lfloor u/2^k\rfloor`$ one bits,
one zero bit, then the low $`k`$ bits of $`u`$, least significant bit first. It
concatenates codes least significant bit first within each byte and pads with
the minimum zero bits to a byte boundary. The payload contains no parameter
bytes because $`k`$ comes from the schedule. Akita's implementation is in
`akita-params/src/golomb_rice.rs` and
`akita-types/src/proof/tail_segments.rs`.

For bounded decoding, set $`B=\mathrm{z\_linf\_cap}`$ when present and
$`B=32767`$ otherwise. The signed zigzag width is 1 when $`B=0`$ and
$`1+\mathrm{bitlength}(B)`$ otherwise. The decoder rejects a unary
quotient greater than $`\lfloor2B/2^k\rfloor`$, a value outside that signed
width or outside i16, and any $`|v|\gt B`$ when a cap is present. When the cap is
absent, Akita's scheduled $`L_2`$ response bound still applies. Decode exactly
`z_coords` values, reject nonzero final padding and extra bytes, and require
the payload length to be at most `z_payload_bytes`. Akita emits and checks
the native unsigned LEB128 nonce with its bounded u32 and canonical-encoding
rules in `akita-transcript/src/native/nonce.rs`.

#### Work allocation

Use indexed-coordinate accounting, the paper's Lemma C.3 ("Sparse Folds with an Indexed Query Budget"). For each nonseed field site
$`r`$, let $`e_r`$ denote its extraction-tree debit. Set

```math
g_r=\min\{g\in\mathbb Z_{\geq0}:2^{128}e_r\leq2^g\}.
```

All comparisons MUST use integers or exact rationals. In particular,
$`|K|=(2^{64}-59)^2`$, never $`2^{128}`$. A field vector is one event and one
nonce, not one per limb. PIOP debits are $`8/|K|`$ for `mu`, $`d/|K|`$ for
`tau`, $`m_s/|K|`$ for `stage{s}/gamma`, and respectively
$`3/|K|,5/|K|,3/|K|`$ for each round of the three stages. These are the
extraction-tree debits, not the ideal-table RBR escape numerators. No global
site-count multiplier enters the difficulty. The projection seed and every
sparse-fold coordinate have zero work.

For PCS, use Akita's exact-cardinality grinding plan at the pinned revision.
The following debit table states the expected field-site allocation for every
admitted level. Let $`v`$ be the incoming opening arity, $`P`$ its polynomial
count, $`w`$ the outgoing witness length, and $`D'`$ the successor inner ring
dimension. Set
$`r=\log_2(D'\mathrm{nextpow2}(\lceil w/D'\rceil))`$. Let $`G`$ be the
number of groups, $`a_g`$ their inner output ranks, $`s_g`$ their outer slice
counts, $`b_g`$ their outer output ranks, and $`o`$ the open-matrix output rank.
Set $`u=\lceil\log_2(1+\sum_g(1+a_g+s_gb_g)+o+\chi\,2(G+1))\rceil`$,
where $`\chi=1`$ for a compressed opening payload and 0 for raw.

| PCS field family | Occurrences and challenge shape | Numerator $`L`$ in $`e=L/\vert K\vert `$ |
| --- | --- | ---: |
| `EORSplitPoint` | one scalar before each evaluation-trace opening reduction, including terminal | 1 |
| `EORClaimBatch` | absent; multi-claim EOR is not admitted by this table | not charged |
| `EORSumcheck` | $`v-1`$ scalar rounds, degree 2 | 2 |
| `EvaluationBatch` | one grouped-root site for nine independent extension coefficients; absent for single-polynomial recursive folds | number of polynomials $`P`$ (9 at the root) |
| `RingSwitchAlpha` | one scalar per nonterminal level; $`D_g`$ is the inner ring dimension of group $`g`$, for both opening methods | $`\max_g(2D_g-1)`$ |
| `Tau0Point` | one complete $`K^r`$ vector per nonterminal level | $`r`$ |
| `Tau1Point` | one complete $`K^u`$ vector per nonterminal level | $`u`$ |
| `Stage1RangeSumcheck` | $`r`$ rounds per product/leaf stage of full degree $`D_s`$ | $`D_s`$ |
| `Stage1InterstageBatch` | one scalar after each product stage with $`c`$ child claims, powers $`1,x,\ldots,x^{c-1}`$ | $`c-1`$ |
| `L2NormMerge` | one scalar on each physical-$`L_2`$ route | 1 |
| `PhysicalL2Sumcheck` | $`r`$ rounds on that route, full degree leaf-degree plus one | leaf-degree plus one |
| `L2VirtualBatch` | one scalar after physical-$`L_2`$ virtual evaluations | 1 |
| `CompressionBinary` | one scalar on each compressed-payload level | 1 |
| `Stage2Batch` | one scalar per nonterminal level | 1 |
| `Stage2Sumcheck` | $`r`$ scalar rounds per nonterminal level | 3 |

The last group supplies the opening digit basis. For `log_basis_open`
2, 3, 4, 5, 6, the product arity lists are respectively `[]`, `[]`, `[2]`,
`[4]`, `[2,4]`. A product stage of arity $`a`$ has full sumcheck degree $`a+1`$;
its child count is the cumulative product of arities so far. The final leaf
has degree 2 for log basis 2 and degree 4 otherwise, plus one for its equality
factor. An $`L_\infty`$ path appends this leaf stage. An $`L_2`$ path instead uses
the physical-$`L_2`$ sumcheck. These 16 schedules have no setup-prefix stage 3.
The root `EvaluationBatch` site has loss factor $`L=9`$, the number of
batched polynomials, and tree factor 10. The prover sends the nine claimed
evaluations $`v\in K^9`$ before the independent batching vector
$`\eta\in K^9`$ is sampled, and the batched opening proves a claim about
$`\langle\eta,v\rangle`$. The extractor branches this site on a flat
coordinate family: a base vector $`\eta`$ and nine alternatives
$`\eta^{(k)}`$, each differing from $`\eta`$ only in coordinate $`k`$.
Descendant extraction in each child yields sources for the four groups. By
weak binding these agree across the ten children or yield a scheduled
Module-SIS collision. With common sources, let $`\delta_k`$ be the difference
between $`v_k`$ and the true evaluation of the $`k`$th extracted polynomial at
its native point. Each child authenticates its batched claim, so
$`\langle\eta,\delta\rangle=\langle\eta^{(k)},\delta\rangle=0`$; subtracting
gives $`(\eta^{(k)}_k-\eta_k)\delta_k=0`$, hence $`\delta=0`$. This is the
coordinate-wise tree charge for independent batching coefficients in the
Akita paper (IACR ePrint 2026/1983, eq. (208) and (222)-(223), Theorem 10.31):
debit $`9/|K|`$ and ten children. Since $`9\cdot2^{128}/|K|`$ lies in
$`(8,16]`$, the exact $`p^2`$ rule gives $`g=4`$. One child does not suffice.
The bound $`1/|K|`$ for $`\langle\eta,\delta\rangle=0`$ presumes that
$`\delta`$ is fixed before $`\eta`$, but the opening is extracted from the
subtree below $`\eta`$ and binding is only computational, so with one child
no collision can be formed, so a fixed-discrepancy charge ($`L=1`$, one
child, $`g=1`$) does not apply; see the paper's Lemma C.2 ("Checkpoint Structure of the Four-Group Opening"). Akita's
exact-cardinality plan supplies this difficulty; Section 12.3 records the
pin status. All other PCS field challenges use the same
exact per-site rule.

For sparse folds, Akita uses an independent indexed XOF stream for each
coordinate. The support for a coordinate in group $`g`$ at level $`j`$ has weight
$`1/C_{jg}`$. Use

```math
C_{64}=\binom{64}{31}\binom{33}{10}2^{41}
=361715808335458406596740595030299770880,
```
```math
C_{128}=\binom{128}{31}2^{31}
=1046949673839956231392050469602223718400,
```
```math
C_{256}=\binom{256}{23}2^{23}
=2873108161471518200301057969538203648000.
```

These are the raw supports for the $`(31,10)`$ signed D64 packing shell, the
$`(31,0)`$ signed D128 shell and the $`(23,0)`$ signed D256 packing shell; D128 coefficient packing and D128 evaluation-trace
$`L_\infty`$ routes use $`C_{128}`$. On D128 $`L_2`$ routes replace $`C_{128}`$ by the
certified accepted-support lower bound
$`A_{128}=502822610611069237129947393810277364817`$. The catalog audit checks
the norm-filter admission. The grouped rows of the admitted Falcon shapes have no D64 $`L_2`$ route.
All four supports exceed $`2^{128}`$, so sparse
coordinates require zero proof-of-work. Their multiplicity contributes to
query and verifier-completion counts, not to per-coordinate difficulty.

Pinned Akita's native fold response nonce, indexed sampler, coefficient
packing, support filter, and transcript updates are used as supplied by
`akita-challenges/src/sampler/mod.rs` and the native prover and verifier
protocol modules. Aerie does not add PCS headers or replace Akita's nonce
codec. Response-nonce trials and internal sampling retries count among the
random-oracle queries $`Q`$; they have no separate work predicate. The verifier
regenerates the selected coordinates and checks the scheduled folded-response
bounds.

Step E supplies exact-cardinality field-site pricing and the public setup,
catalog, and schedule identities. Akita's exact-cardinality plan owns PCS
field-site work. Step F exports the PIOP session and passes it to Akita's
native PCS stream. Step G derives and enforces the PIOP work plan and inline
nonce path. Sparse-fold coordinates retain Akita's native indexed sampler and
have zero work.

Let $`R_{\rm field}`$ bound the field-family addresses, including the
projection seed, completed by final verification without an original prover
query, and let $`R_{\rm coord}`$ bound its analogous sparse-coordinate
completions. The table gives conservative counts for one full verification:
$`R_{\rm field}=3d+5+F+1`$ and
$`R_{\rm coord}=\sum_{j,g}W_{jg}`$, where $`F`$ is the PCS field-site count.
A fold retains its flat extraction tree with one reference and one
alternative per coordinate; indexed accounting does not turn it into
sequential binary rounds. For the specified native allocation, let $`e_r`$ and
$`g_r`$ denote the extraction debit and difficulty at each nonseed field site,
including root `EvaluationBatch` values $`L=9`$ and $`g=4`$ with ten children. With
$`\vartheta\leq2^{-135}`$, set

```math
\eta=\max(3\cdot2^{-256},(4/3)\vartheta),\qquad
\rho=\max\{\eta,\max_r2^{-g_r}e_r,
\max_{j,g}C_{jg}^{-1}\}
\leq2^{-128}.
```

The resulting classical-ROM knowledge bound uses this native allocation.
The paper's Theorem 5.4 states it for every admitted padded
batch: $`N^*=8,\ldots,2^{16}`$ at Falcon-512 and $`N^*=4,\ldots,2^{16}`$ at
Falcon-1024.

```math
(Q+R_{\rm field}+R_{\rm coord})2^{-128}
+\delta_X+\Delta_{\rm setup}
+\sum_I\mathrm{Adv}^{\rm exp}_{{\rm MSIS},I}(T_I).
```

Here $`Q`$ counts the original prover's hash evaluations, including failed
work, response-nonce trials, and coordinate sampling. The theorem's expected
reduction times $`T_I`$, expansion bad-event term $`\delta_X`$, and setup
exceptional-event term $`\Delta_{\rm setup}`$ remain explicit. Compressed
records and transcript prefixes, including fold seeds, additionally incur
the relevant binding collision advantage at the extractor's expected time;
a nonideal joint sampler adds its error over the complete query experiment.
This is a rate per counted hash evaluation, not a bound with all verifier
completions replaced by one, and it makes no QROM claim.
Whole-fold accounting remains a proved alternative in the paper. The root
`EvaluationBatch` site is covered by this bound with $`L=9`$, $`g=4`$, and ten
children, using the flat coordinate branching over $`\eta`$ described above
(the paper's Lemma C.2 ("Checkpoint Structure of the Four-Group Opening") and Appendix C).

PIOP field-work nonces use Akita's native bounded `u32` contract.
For $`g\gt 0`$ the semantic width is $`w=g+7`$, and production policy MUST reject
$`g\gt 25`$. Encode the accepted integer as canonical unsigned LEB128: low
seven-bit groups first, high continuation bit set only on nonfinal bytes,
and zero encoded as `00`. There is no nonce count, section prefix, or length
prefix. The decoder MUST reject unterminated input, overflow beyond `u32`,
a redundant final zero group, and an integer outside $`[0,2^w)`$. It MUST NOT
advance its input cursor on a malformed native encoding. An in-range nonce
with a failed predicate also rejects. A zero-bit site has no nonce field.

The verifier accepts any in-range winner; it MUST NOT require the first
winner. Exhausting the $`2^{g+7}`$ candidates fails proving, with ideal
per-site honest exhaustion probability
$`(1-2^{-g})^{2^{g+7}}\leq\exp(-128)`$. No wraparound or fallback to unground
challenges is permitted. Omitted, extra, reordered, truncated, or invalid
work rejects. Acceptance requires complete plan consumption and no trailing
native argument bytes. For PIOP, these rules use Akita's native transition in
`specs/transcript-grinding.md` and its canonical unsigned LEB128 nonce
encoding. The PCS retains the pinned native Akita transition and grammar.

<!-- BEGIN TRANSCRIPT GRINDING TABLE -->
| Shape | PIOP field-site work bits | PIOP field sites | PCS field sites | PCS site-family work bits (g:events) | $`R_{\mathrm{field}}`$ | Indexed sparse-coordinate completions (zero work) | PIOP nonce fields | Worst-case PIOP nonce bytes | Expected PIOP nonce bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Falcon-512-Nstar-8 | mu 4; tau 4; gamma 2,4,4; sumcheck S1 2x12; S2 3x12; S3 2x12 | 41 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 350 | 48 | 41 | 82 | 41.00 |
| Falcon-512-Nstar-16 | mu 4; tau 4; gamma 2,4,4; sumcheck S1 2x13; S2 3x13; S3 2x13 | 44 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 353 | 48 | 44 | 88 | 44.00 |
| Falcon-512-Nstar-32 | mu 4; tau 4; gamma 2,4,4; sumcheck S1 2x14; S2 3x14; S3 2x14 | 47 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 356 | 50 | 47 | 94 | 47.00 |
| Falcon-512-Nstar-64 | mu 4; tau 4; gamma 2,4,4; sumcheck S1 2x15; S2 3x15; S3 2x15 | 50 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,9:1,10:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 359 | 52 | 50 | 100 | 50.00 |
| Falcon-512-Nstar-128 | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x16; S2 3x16; S3 2x16 | 53 | 312 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:50; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:109; Stage2Batch 1:4; Stage2Sumcheck 2:73; Tau0Point 5:4; Tau1Point 2:2,3:2 | 366 | 55 | 53 | 106 | 53.02 |
| Falcon-512-Nstar-256 | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x17; S2 3x17; S3 2x17 | 56 | 395 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:67; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:4,3:2; Stage1RangeSumcheck 2:71,3:92; Stage2Batch 1:5; Stage2Sumcheck 2:92; Tau0Point 5:5; Tau1Point 2:1,3:4 | 452 | 80 | 56 | 112 | 56.02 |
| Falcon-512-Nstar-512 | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x18; S2 3x18; S3 2x18 | 59 | 378 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:111; Stage2Batch 1:5; Stage2Sumcheck 2:93; Tau0Point 5:5; Tau1Point 2:1,3:4 | 438 | 85 | 59 | 118 | 59.02 |
| Falcon-512-1K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x19; S2 3x19; S3 2x19 | 62 | 378 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:111; Stage2Batch 1:5; Stage2Sumcheck 2:93; Tau0Point 5:5; Tau1Point 2:1,3:4 | 441 | 81 | 62 | 124 | 62.02 |
| Falcon-512-2K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x20; S2 3x20; S3 2x20 | 65 | 401 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,11:1; Stage1InterstageBatch 1:3,2:1,3:2; Stage1RangeSumcheck 2:54,3:112; Stage2Batch 1:5; Stage2Sumcheck 2:94; Tau0Point 5:5; Tau1Point 2:1,3:4 | 467 | 129 | 65 | 130 | 65.02 |
| Falcon-512-4K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x21; S2 3x21; S3 2x21 | 68 | 401 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,11:1; Stage1InterstageBatch 1:3,2:1,3:2; Stage1RangeSumcheck 2:54,3:112; Stage2Batch 1:5; Stage2Sumcheck 2:94; Tau0Point 5:5; Tau1Point 2:1,3:4 | 470 | 139 | 68 | 136 | 68.02 |
| Falcon-512-8K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x22; S2 3x22; S3 2x22 | 71 | 388 | CompressionBinary 1:3; EORSplitPoint 1:4; EORSumcheck 2:70; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:116; Stage2Batch 1:5; Stage2Sumcheck 2:97; Tau0Point 5:5; Tau1Point 2:2,3:3 | 460 | 179 | 71 | 142 | 71.02 |
| Falcon-512-16K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x23; S2 3x23; S3 2x23 | 74 | 388 | CompressionBinary 1:3; EORSplitPoint 1:4; EORSumcheck 2:70; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,9:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:116; Stage2Batch 1:5; Stage2Sumcheck 2:97; Tau0Point 5:5; Tau1Point 2:2,3:3 | 463 | 249 | 74 | 148 | 74.02 |
| Falcon-512-32K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x24; S2 3x24; S3 2x24 | 77 | 452 | CompressionBinary 1:5; EORSplitPoint 1:5; EORSumcheck 2:88; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,9:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:135; Stage2Batch 1:6; Stage2Sumcheck 2:117; Tau0Point 5:6; Tau1Point 2:1,3:5 | 530 | 419 | 77 | 154 | 77.02 |
| Falcon-512-64K | mu 4; tau 5; gamma 2,4,4; sumcheck S1 2x25; S2 3x25; S3 2x25 | 80 | 455 | CompressionBinary 1:5; EORSplitPoint 1:5; EORSumcheck 2:89; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:136; Stage2Batch 1:6; Stage2Sumcheck 2:118; Tau0Point 5:6; Tau1Point 2:1,3:5 | 536 | 386 | 80 | 160 | 80.02 |
| Falcon-1024-Nstar-4 | mu 4; tau 4; gamma 3,4,4; sumcheck S1 2x12; S2 3x12; S3 2x12 | 41 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 350 | 48 | 41 | 82 | 41.00 |
| Falcon-1024-Nstar-8 | mu 4; tau 4; gamma 3,4,4; sumcheck S1 2x13; S2 3x13; S3 2x13 | 44 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 353 | 48 | 44 | 88 | 44.00 |
| Falcon-1024-Nstar-16 | mu 4; tau 4; gamma 3,4,4; sumcheck S1 2x14; S2 3x14; S3 2x14 | 47 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 356 | 50 | 47 | 94 | 47.00 |
| Falcon-1024-Nstar-32 | mu 4; tau 4; gamma 3,4,4; sumcheck S1 2x15; S2 3x15; S3 2x15 | 50 | 308 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:49; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,9:1,10:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:107; Stage2Batch 1:4; Stage2Sumcheck 2:72; Tau0Point 5:4; Tau1Point 2:2,3:2 | 359 | 52 | 50 | 100 | 50.00 |
| Falcon-1024-Nstar-64 | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x16; S2 3x16; S3 2x16 | 53 | 312 | CompressionBinary 1:2; EORSplitPoint 1:3; EORSumcheck 2:50; EvaluationBatch 4:1; L2NormMerge 1:1; L2VirtualBatch 1:1; PhysicalL2Sumcheck 3:17; RingSwitchAlpha 8:2,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:109; Stage2Batch 1:4; Stage2Sumcheck 2:73; Tau0Point 5:4; Tau1Point 2:2,3:2 | 366 | 55 | 53 | 106 | 53.02 |
| Falcon-1024-Nstar-128 | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x17; S2 3x17; S3 2x17 | 56 | 395 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:67; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:4,3:2; Stage1RangeSumcheck 2:71,3:92; Stage2Batch 1:5; Stage2Sumcheck 2:92; Tau0Point 5:5; Tau1Point 2:1,3:4 | 452 | 80 | 56 | 112 | 56.02 |
| Falcon-1024-Nstar-256 | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x18; S2 3x18; S3 2x18 | 59 | 378 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:111; Stage2Batch 1:5; Stage2Sumcheck 2:93; Tau0Point 5:5; Tau1Point 2:1,3:4 | 438 | 85 | 59 | 118 | 59.02 |
| Falcon-1024-Nstar-512 | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x19; S2 3x19; S3 2x19 | 62 | 378 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:111; Stage2Batch 1:5; Stage2Sumcheck 2:93; Tau0Point 5:5; Tau1Point 2:1,3:4 | 441 | 81 | 62 | 124 | 62.02 |
| Falcon-1024-1K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x20; S2 3x20; S3 2x20 | 65 | 401 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,11:1; Stage1InterstageBatch 1:3,2:1,3:2; Stage1RangeSumcheck 2:54,3:112; Stage2Batch 1:5; Stage2Sumcheck 2:94; Tau0Point 5:5; Tau1Point 2:1,3:4 | 467 | 129 | 65 | 130 | 65.02 |
| Falcon-1024-2K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x21; S2 3x21; S3 2x21 | 68 | 401 | CompressionBinary 1:4; EORSplitPoint 1:4; EORSumcheck 2:68; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,11:1; Stage1InterstageBatch 1:3,2:1,3:2; Stage1RangeSumcheck 2:54,3:112; Stage2Batch 1:5; Stage2Sumcheck 2:94; Tau0Point 5:5; Tau1Point 2:1,3:4 | 470 | 137 | 68 | 136 | 68.02 |
| Falcon-1024-4K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x22; S2 3x22; S3 2x22 | 71 | 388 | CompressionBinary 1:3; EORSplitPoint 1:4; EORSumcheck 2:70; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,10:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:116; Stage2Batch 1:5; Stage2Sumcheck 2:97; Tau0Point 5:5; Tau1Point 2:2,3:3 | 460 | 179 | 71 | 142 | 71.02 |
| Falcon-1024-8K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x23; S2 3x23; S3 2x23 | 74 | 388 | CompressionBinary 1:3; EORSplitPoint 1:4; EORSumcheck 2:70; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:3,9:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:116; Stage2Batch 1:5; Stage2Sumcheck 2:97; Tau0Point 5:5; Tau1Point 2:2,3:3 | 463 | 248 | 74 | 148 | 74.02 |
| Falcon-1024-16K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x24; S2 3x24; S3 2x24 | 77 | 452 | CompressionBinary 1:5; EORSplitPoint 1:5; EORSumcheck 2:88; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,9:1,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:135; Stage2Batch 1:6; Stage2Sumcheck 2:117; Tau0Point 5:6; Tau1Point 2:1,3:5 | 530 | 418 | 77 | 154 | 77.02 |
| Falcon-1024-32K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x25; S2 3x25; S3 2x25 | 80 | 455 | CompressionBinary 1:5; EORSplitPoint 1:5; EORSumcheck 2:89; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:4,11:2; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:136; Stage2Batch 1:6; Stage2Sumcheck 2:118; Tau0Point 5:6; Tau1Point 2:1,3:5 | 536 | 383 | 80 | 160 | 80.02 |
| Falcon-1024-64K | mu 4; tau 5; gamma 3,4,4; sumcheck S1 2x26; S2 3x26; S3 2x26 | 83 | 457 | CompressionBinary 1:5; EORSplitPoint 1:5; EORSumcheck 2:89; EvaluationBatch 4:1; L2NormMerge 1:2; L2VirtualBatch 1:2; PhysicalL2Sumcheck 3:34; RingSwitchAlpha 8:5,11:1; Stage1InterstageBatch 1:2,2:1,3:2; Stage1RangeSumcheck 2:34,3:137; Stage2Batch 1:6; Stage2Sumcheck 2:119; Tau0Point 5:6; Tau1Point 2:1,3:5 | 541 | 666 | 83 | 166 | 83.02 |
<!-- END TRANSCRIPT GRINDING TABLE -->

The table is generated and checked by
[`falcon_transcript_grinding.py`](scripts/falcon_transcript_grinding.py).
It covers all 29 padded shapes that Section 2.1 admits for $`1\le N_{\rm sig}\le2^{16}`$:
Falcon-512 at $`N^*=8,16,\ldots,2^{16}`$ and Falcon-1024 at $`N^*=4,8,\ldots,2^{16}`$,
with K=1024. The script derives this set from the padding rule and requires
the native catalog audit and the paper evidence to contain exactly these
shapes; the `aerie-falcon` test
`grinding_plan_covers_exactly_the_shapes_the_verifier_supports` requires every
record count that `FalconVerifierSetup::supports` admits to map to a planned shape.
It MUST NOT be extrapolated to other catalog rows without regenerating the
complete event inventory and checking tree admission. The selected padded
shape fixes the plan; the live count remains bound in the instance.
`--check` reads the paper's `indexed_allocation` evidence, pinned by
SHA-256 in the script, from the gzip copy vendored at
[`evidence/whole-protocol-tree-checks.json.gz`](evidence/whole-protocol-tree-checks.json.gz)
and checks the hash of the decompressed bytes; `--paper-json` selects
another copy. A missing or
changed file fails the check. Every site, including the root
`EvaluationBatch` with $`L=9,g=4`$ and ten children, must match the paper
allocation exactly.

### 10.2 Header and record transport

The transcript root is aerie/falcon/v1. The proof frame begins with 27 bytes: a 16-byte magic, wire version LE16 equal to 1, parameter-set byte 9 or 10, and $`N_{\rm sig}`$ as LE64. The per-record PCS frame magic is aerie/falcon/v1p. The common-salt PCS frame magic is aerie/falcon/v1c. A decoder checks the magic, version, and parameter-set byte in that order, then rejects $`N_{\rm sig}^\star\gt 2^{16}`$. Public keys and messages are statement inputs. Immediately after the header, the per-record transport frame contains the $`N_{\rm sig}`$ live 40-byte salts in record order, with no per-record length fields. In common-salt mode, this frame is exactly one 40-byte salt. The decoder MUST expand it to $`N_{\rm sig}`$ copies before statement binding and target derivation. A common-salt encoder MUST reject empty or unequal salt vectors and vectors whose length differs from $`N_{\rm sig}`$. The magic selects the mode without a length heuristic, and the verifier MUST use that mode's public-record customization from Section 2.3. Re-encoding a proof body under the opposite mode MUST fail verification, even when all expanded salts are equal. All subsequent fields use the same grammar in both modes. The canonical public keys and raw messages remain statement inputs; private witness bytes are not transported.

The codec derives $`N_{\rm sig}^\star`$, the profile, group shapes, and projection profile from the parameter set and $`N_{\rm sig}`$. It rejects missing, trailing, malformed, or noncanonical data. The final Akita section is length-prefixed as LE64 length followed by the canonical Akita argument string. Its length MUST be nonzero, at most $`2^{20}`$, and equal to the number of remaining frame bytes.

The production body, after the salt transport, is exactly:

```
commitment[0] || commitment[1] || commitment[2] || commitment[3]
|| rice_outputs
|| nonce_mu || nonce_tau || nonce_gamma1
|| (stage1_round[t] || nonce_stage1_round[t]) for t = 0..d-1
|| stage1_terminal
|| nonce_gamma2
|| (stage2_round[t] || nonce_stage2_round[t]) for t = 0..d-1
|| stage2_terminal
|| nonce_gamma3
|| (stage3_round[t] || nonce_stage3_round[t]) for t = 0..d-1
|| stage3_terminal
|| LE64(pcs_argument_length) || pcs_argument
```

Each `nonce` position in this PIOP body carries a canonical unsigned LEB128
integer when its site's $`g\gt 0`$, and is absent when $`g=0`$. Challenges, event
headers, instance/setup/catalog identities, predicate bytes, and the PCS
session digest are recomputed and add no transport bytes. The PCS argument
retains Akita's native inline nonces, including their grammar and bytes.
Exact-cardinality repricing can change a site's difficulty or whether work
is present, but does not re-encode any nonce. Only PCS-backed proof frames are
implemented.

Let $`\ell(c)`$ be the canonical unsigned LEB128 length of $`c`$, with
$`\ell(0)=1`$. The PIOP work bytes in a built proof are
$`L_{\rm PIOP}=\sum_{r:g_r\gt 0}\ell(c_r)`$, bounded by
```math
B_{\rm PIOP}=\sum_{r:g_r\gt 0}\ell(2^{g_r+7}-1)
=\sum_{r:g_r\gt 0}\left\lceil\frac{g_r+7}{7}\right\rceil.
```
Every PIOP site in the 16 admitted shapes has $`1\leq g_r\leq5`$, so
$`B_{\rm PIOP}=2(3d+5)`$ bytes. More generally the two-byte bound applies
for $`1\leq g\leq7`$; zero-bit sites contribute zero bytes. The generated
table gives each shape's nonce count, worst-case bound, and expected size
rounded to two decimal places.

For ascending first-success search, put $`a_r=1-2^{-g_r}`$ and
$`H_r=2^{g_r+7}`$. Conditional on finding a winner before exhaustion, the
expected byte length at a nonzero site is
```math
\mathbb E[\ell(c_r)\mid c_r\lt H_r]
=1+\sum_{k=1}^{\lceil(g_r+7)/7\rceil-1}
\frac{a_r^{2^{7k}}-a_r^{H_r}}{1-a_r^{H_r}}.
```
Sum these expectations across the PIOP sites. For the current two-byte
maximum this is $`1+(a_r^{128}-a_r^{H_r})/(1-a_r^{H_r})`$ per site.
These are ideal independent-predicate geometric-search expectations,
conditional on successful bounded searches, not parser bounds. The verifier
accepts other in-range winners with lengths up to the stated maximum.

Let $`L_y`$ be the Rice-section byte length and $`L_A`$ the actual complete
native PCS argument length, including all its field-work and fold-response
nonces. The full frame length is
```math
27+S+512+L_y+160d+16(2b+18)+L_{\rm PIOP}+8+L_A,
```
where $`S=40N_{\rm sig}`$ or 40 for the two salt modes, and $`b=2`$ or 3.
At both reference profiles $`d=24`$, the PIOP has 77 nonce fields and adds at
most 154 bytes; its expected size is given in the table. No formula based
only on the number of PCS sites gives the realized PCS byte delta relative
to pinned Akita. That delta MUST be measured from built native proofs under
the compared plans. Schedule selection prices each present native nonce at
its LEB128 maximum `ceil(semantic_nonce_width/7)`; this deterministic
objective is distinct from realized winning-nonce lengths. Aerie MUST NOT
replace or re-encode PCS nonces. The PCS length cap includes all native
nonce bytes. All length arithmetic MUST be checked before allocation or
slicing.

### 10.3 Claims and round messages

Stage-one basis16_product claims are encoded in canonical source order: low and high claims for $`a_1`$, then $`a_2`$, and at Falcon-1024 then $`u_2`$. Stage-one projection_link claims follow as $`v_{S_1},v_e,v_\Delta`$. Stage two carries the six claims in order $`a_0,a_1,a_2,u_0,u_1,u_2`$. Stage three carries the nine evaluations in the order of (7.16).

Every $`K`$ element is 16 bytes in the canonical limb order above. Each round carries 3 elements in stage 1, 4 in stage 2, and 3 in stage 3. The reference profiles have 24 rounds per stage and therefore carry 240 round-message elements, excluding claims. Ordinary compressed rounds omit the linear coefficient. Shared-equality rounds use the normalized equality recurrence and omit the constant coefficient. Implementations MUST use the same round convention when producing, absorbing, and verifying each stage. The wire carries every sumcheck round message, all required stage claims, the nine terminal evaluations, and the final Akita opening.

For an ordinary cubic round, write $`g(X)=g_0+g_1X+g_2X^2+g_3X^3`$ and let $`v`$ be the current claim. Send $`(g_0,g_2,g_3)`$. The verifier reconstructs $`g_1=v-2g_0-g_2-g_3`$, samples the round challenge $`t`$, and updates the claim to $`g(t)`$.

For a normalized stage-two round, let $`\alpha`$ be the next coordinate of $`r^{(1)}`$ and write the inner quartic as $`q(X)=q_0+q_1X+\cdots+q_4X^4`$. The recurrence is
```math
v=(1-\alpha)q(0)+\alpha q(1),\qquad
q_0=v-\alpha(q_1+q_2+q_3+q_4).
```
Send $`(q_1,q_2,q_3,q_4)`$. After reconstructing $`q_0`$, sample $`t`$ and update the normalized claim to $`q(t)`$. This consumes the equality coordinate without division, including when $`\alpha`$ is zero or one. After the final round, compare the claim with the batched range-polynomial values from Section 7.3, without another equality factor.

The catalog descriptor fixes $`[u_0,u_1,u_2]`$ as the final Akita group, after the three precommitted groups in canonical order. No plane-coordinate challenges are sampled for PCS group claims. This is the first frozen wire version 1; old development encodings require regeneration.

### 10.4 Canonical Rice projection section

The output section is absorbed byte-for-byte at `aerie/falcon/v1/projection/output`, after canonical decoding and both projection checks and before `mu` work. The two parameter bytes and zero padding are included. It carries no work nonce of its own. Decoded residues are used in the PIOP equations but are not absorbed a second time.

The output section consists of one-byte $`k_y`$, one-byte $`k_z`$, and one bit stream. The stream packs bits least significant first within each byte and has zero padding to the next byte. It contains the 156 norm outputs followed by the 31 residual outputs.

A norm code carries $`v=\mathrm{ctr}_p(y_{{\rm norm},j})`$ and the decoder rejects $`|v|\gt (p-1)/2`$. A residual code carries $`v=\mathrm{ctr}_p(y_{{\rm res},j})/q`$, so values failing (6.5) have no encoding. The residual magnitude is bounded by
```math
U=\left\lfloor\frac{p-1}{2q}\right\rfloor
=750{,}538{,}858{,}886{,}384.
\qquad(10.1)
```
Map signed $`v`$ to $`z=2v`$ for $`v\geq0`$ and $`z=-2v-1`$ for $`v\lt 0`$. With parameter $`k`$, encode $`\lfloor z/2^k\rfloor`$ one bits, a zero bit, and the low $`k`$ bits of $`z`$, all least significant bit first.

Each of $`k_y,k_z`$ is in $`[0,63]`$ and MUST be the smallest value minimizing its leg's bit length
```math
L(k)=\sum_j\left(k+1+\left\lfloor z_j/2^k\right\rfloor\right).
\qquad(10.2)
```
The decoder recomputes both values, rejects a nonminimal parameter, an overflowing unary run, an out-of-range decoded value, a nonzero padding bit, or trailing data. This gives one encoding for each accepted vector.

Any decodable projection section is at most 1,471 bytes. The section is at most 893 bytes for accepted Falcon-512 outputs and 904 bytes for accepted Falcon-1024 outputs. These bounds follow from (6.4), Cauchy-Schwarz, and the residual bound (10.1). They carry no probability term. Honest output sizes depend on $`N_{\rm sig}`$ and the projection seed.

## 11. Prover and verifier algorithms

### 11.1 Prover

For every live record, the prover decodes the Falcon witness, checks the ring equation and norm as witness-builder sanity checks, and computes the four-square slack. It decomposes $`s_2`$ into the four coefficient tables and chooses budget digits satisfying (3.5). For the slack vector, the Falcon-1024 producer uses the capped search in Section 5.

The prover extends tables to $`N_{\rm sig}^\star`$, using (3.9) and (3.10) for padded digit rows and zero values for $`S_1,e`$. It commits once to each of the four groups. It computes the structured projections, rejects an honest predicate failure without resampling, then executes stages 1, 2, and 3 in transcript order. It computes the direct ring functional from (8.7) and (8.8), sends the nine terminal evaluations, and produces the four-group Akita opening.

### 11.2 Verifier

The verifier parses the parameter set and batch size, derives the public-record hash and Falcon targets, and checks that the selected profile and catalog row are admitted. It reconstructs the transcript through the four commitments, derives the projection cores, decodes the Rice section, and checks (6.4) and (6.5) on centered integers.

It verifies the three stages, derives (7.4), and evaluates the public terminal ring weight by (8.9) through (8.11). It checks the nine terminal evaluations and verifies the Akita opening. It rejects a changed group descriptor, a noncanonical output section, a transcript reorder, a false source claim, or a proof outside the selected parameter set.

## 12. Tests and open obligations

### 12.1 Required tests

Tests MUST cover:

- Falcon-512 and Falcon-1024 parameter constants, headers, and public-record binding.
- Every endpoint of the $`a_0,a_1,a_2,\sigma,u_0,u_1,u_2`$ alphabets and exhaustive decoding of all 2,048 magnitude digit triples with both signs (4,096 signed coefficient tuples).
- Both representations of zero magnitude and canonical extraction to Falcon bytes.
- Every root of $`Q_8,Q_{16}^{\rm lo},Q_{16}^{\rm hi}`$, including rejection of values outside each exact alphabet.
- The budget totals at and around $`L`$, and the padded rows in (3.9) and (3.10).
- The capped four-square producer over the entire Falcon-1024 range $`(8{,}191^2,70{,}265{,}242]`$; the standalone check is specs/scripts/falcon_slack_width_check.py and accepts an optional subrange.
- The projection geometry for every admitted $`N_{\rm sig}^\star`$, all ternary byte mappings, and every rejection-sampler byte.
- Distinct core-row SHAKE inputs, profile binding, and both projection predicates.
- Stage-one basis16_product claims and projection_link equality, including the independent local point order for $`r_{\Pi,e}`$.
- Stage-two range_leaf claims and the six square terminals.
- Stage-three range_source_link, sign_check, norm_equality, encoding_budget, ring_functional, and both claim reductions.
- The ring functional against a direct negacyclic reference at random extension-field points for both degrees.
- Point reuse at $`r^{(2)}`$, verifying that all source tables were committed before the point was sampled.
- The nine terminal evaluations opened at the three group points through Akita.
- Norm-lift and residual inequalities with exact integer arithmetic.
- Rice output round trips, minimal $`k`$, accepted-size bounds, and rejection of malformed lengths, nonzero padding bits, and trailing bytes.
- Honest and adversarial signatures against the reference Falcon verifier.

### 12.2 Open obligations

The written clear-projection theorem is in [falcon-projection-theorem.md](falcon-projection-theorem.md). Its statements are proved in Lean in the library certified-projections, which is not public yet, with the bounds $`2^{-136}`$ for the projection term and $`2^{-129}+2^{-142}`$ for honest rejection; Section 11 of the theorem document gives the correspondence and lists what the Lean theorems do not cover (the hypotheses H1 to H4, the finer numeric values, and the implementation). The frozen certificate script and JSON report remain part of spec CI, together with `specs/scripts/falcon_projection_lean.py --check`, which ties the report to the Rust constants and to the Lean definitions.

The current implementation uses four native Akita groups, with the width-4 budget group final, and catalogs containing 14 width-14 rows and 33 width-4 rows. Section 5 records the group parameters and estimates; the extraction report and ledgers account for the catalog terms. The implementation uses the three-stage PIOP, direct ring functional, native Spongefish PIOP stream with scheduled proof-of-work, and Akita's native PCS stream. Akita owns PCS work and its proof grammar.

The $`460/|K|`$ union bound of Section 9.5 is an algebraic check at the reference batch profiles, not the deployed soundness term. The deployed accounting is the indexed-coordinate allocation of Section 10.1 and the paper's classical-ROM theorem, recorded in the ledgers; their scheduled Module-SIS and concrete transcript-binding assumptions remain explicit.

### 12.3 Implementation status

Step E pinned Akita to
`515a21668bd2fb7f8e3b0a41c325b7c253b87aed` for exact-cardinality field-site
pricing and the public setup, catalog, and schedule identity functions. The
previous pin was `cc2645763f7f16d94f49dbd78e23ec1376e79447`. Revision `3720078d` keeps those
functions and adds Dense's adaptive ring-dimension policy used by both catalogs; `362dfa929` adds only the
stage-1 and stage-2 prover sumcheck changes (Akita PRs 109 and 110). `cc2645763` prices claim-batching
sites at their coordinate-wise tree charge (PR 130, grinding query policy revision 4), plans root
openings once per interchangeable producer class with a full-search fallback (PR 127), admits only
catalog rows whose opening methods can execute (PR 116), stops balanced-digit series once they
saturate (PR 113), and carries prover and refactoring changes (PRs 87, 121, 126, 129, 139). It moves
Akita's jolt-field pin to `1618f4b6dc97effcec3568ca7cf2b7420ac9e75f`.

The previous pin was `3d69096eadb129c097e4fc2e4f2d289e82e0ea1a`, the commit on Akita `dev`
that merges Akita `main` through PR 195 into `dev` (PR 196). From `main` it brings the split of
parameter-side types into `akita-params` (PR 134), the transcript and sumcheck renames that drop the
`native` prefix (PRs 128, 133, 136), `AkitaError::Internal` and the removal of prover panic sites,
the setup-offloading prover (PR 179), witness chunk alignment (PR 175), and the removal of Policy A
and the uniform scheduler mode (PR 192). From `dev` it brings the dev protocol (PR 177), which Aerie
selects with the `dev-protocol` feature: the terminal response omits the last live block of `e` and
of `t` and carries a 32-byte digest of each (Section 10.1, Akita PCS stream). Both catalogs are regenerated at this
pin with the feature on. Every row keeps its fold count and its inner decomposition. Every row except
the single-group rows width-14 $`(26,1)`$ and width-4 $`(25,4)`$ keeps its schedule apart from the
terminal `e` and `t` lengths; those two rows select different parameters. The pin uses jolt-field
`c7c2d08d5bb0c7c80501f29f723e595af7fcff3c`, which Aerie matches. Akita states the terminal variant's
soundness loss over the full-block terminal, in the classical random-oracle model with $`Q`$ digest
queries, as at most $`(\binom{Q}{2}+2Q)/2^{256}`$; the ledgers of this repository do not yet
carry that term.

The current pin is `e023b6a46143f7b23199ccecb2684192baf7b640`, the head of Akita `dev` on
2026-10-10 (PR 228). Relative to the previous pin it brings PR 209, which prices the ring-switch
challenge by the native inner ring for both opening methods, $`2d_{\max}-1`$ with $`d_{\max}`$ the
largest inner ring dimension of the level's groups. Under the previous pin, coefficient-packing levels
were priced by the challenge-subring dimension; at 58 of the 140 ring-switch sites of the admitted
shapes this ground fewer bits than the specified debit of Section 10.1 (worst per-query weight
about $`2^{-124}`$). At this pin every PCS field site of the 29 admitted shapes grinds exactly the
specified difficulty. The repricing changes the planner's choices, so both catalogs are regenerated at
this pin: 3 of the 14 width-14 rows select different commitment profiles, 8 width-4 rows are re-keyed
with those producers and 4 more change schedule; every row keeps its fold count. Three levels
(Falcon-512 at $`N^*=2^{13}`$ and $`2^{16}`$, Falcon-1024 at $`N^*=2^{12}`$) now use the $`(23,0)`$
signed D256 packing family, whose raw support $`C_{256}`$ exceeds $`2^{129}`$ (Section 10.1). The
catalog identity, row identity and setup-seed digest encodings are unchanged, as are the D128 and D64
operator-norm certificates. The other PRs in range are LaBinius, field and prover additions that Aerie
does not select. Step F
implements the v1 PIOP transcript, derives the 398-byte instance record before
initialization, uses canonical Rice projection bytes and exact extension
sampling, and exports the PIOP session digest to Akita's native PCS stream.
Akita's PCS stream and grammar are used as supplied by the pinned revision.

Step G implements PIOP proof-of-work. The shape-derived plan assigns the
Section 10.1 difficulties to `mu`, `tau`, the three gamma sites, and every
sumcheck round. Each nonzero site emits its kind-3 work header and one inline
canonical unsigned-LEB128 nonce; the prover searches Akita's bounded native
transition and the verifier checks its range and predicate. Akita continues to
enforce its exact-cardinality PCS plan, indexed sparse sampler, and native
LEB128 nonce grammar.

The root `EvaluationBatch` uses the coordinate-wise tree charge of Section
10.1: $`L=9`$, ten children, and $`g=4`$. The checker retains the pinned paper
evidence as an independent comparison and requires exact agreement at every
site, including the root. Aerie takes the root difficulty from Akita's plan,
which prices `EvaluationBatch` with loss factor equal to the number of
polynomials from `cc2645763` (PR 130) on, so the implemented root site has
$`g=4`$.

## 13. Acceptance criteria

The implementation conforms to frozen version 1 only when all of the following hold:

- The prover and verifier use the four same-shape Akita groups and one opening point per group.
- The groups have widths 14, 14, 4, and 4; the final Akita group uses the width-4 family.
- Falcon-1024 four-square production succeeds for every admissible slack value with components bounded by 8,191.
- The exact Falcon digit ranges, sign check, and budget identity prove both coefficient magnitude and body length.
- The clear projections, profile, core derivation, and centered-integer predicates match Section 6.
- Stage 1 batches basis16_product and projection_link; stage 2 proves range_leaf; stage 3 proves all seven named relations and ends in nine committed evaluations.
- The ring functional uses $`S_2=\sigma M`$, the affine terminal (7.17), and the public kernel of Section 8.
- The norm identity lifts to integers and the residual predicate proves divisibility by $`q`$, with the stated projection error.
- The PIOP and PCS composition ledgers account for the pinned native plan for every admitted batch shape. The root `EvaluationBatch` uses the coordinate-wise tree charge of Section 10.1: debit $`9/|K|`$, ten children, and $`g=4`$, as priced by the pinned Akita plan.
- The canonical transcript and version-1 wire grammar pass differential and adversarial tests, including changed site IDs, instance identities, salt mode, PIOP prefix, and nonces, and golden vectors are regenerated.
- The checker compares every site of all 29 admitted padded shapes exactly with the pinned paper indexed allocation, including the root; Aerie enforces the framed exact-sampling PIOP contract and Akita enforces its native exact-cardinality PCS plan and indexed sparse sampler.
- The projection certificate check and all spec CI checks pass.

## Appendix A. Code identifiers

| Specification symbol or module | Code identifier |
| --- | --- |
| $`S_1`$ | `FalconWitness::s1` |
| $`e`$ | `FalconWitness::slack`, native table from `slack_group` |
| $`a_0,a_1,a_2,\sigma`$ | `SourceTables::a0`, `a1`, `a2`, `sigma` |
| $`u_0,u_1,u_2`$ | `SourceTables::u0`, `u1`, `u2` |
| $`M`$ | `EncodedS2Coefficient::magnitude`, affine stage-three magnitude |
| $`\Delta`$ | `projection::sources::delta_record`, `v_delta` |
| $`[S_1]`$, $`[e]`$ | `GroupRole::S1`, `GroupRole::Slack` |
| $`[a_0,a_1,a_2,\sigma]`$, $`[u_0,u_1,u_2]`$ | `GroupRole::S2Sources`, `GroupRole::Budget` |
| Signed width-14 and width-4 PCS families | `Fp64SignedWidth14`, `Fp64SignedWidth4` |
| $`Q_8,Q_{16}^{\rm lo},Q_{16}^{\rm hi}`$ | `QuarticPolynomial::Low`, `QuarticPolynomial::High` |
| Stage 1 | `protocol::stage1::{basis16_product, projection_link}` |
| Stage 2 | `protocol::stage2::range_leaf` |
| Stage 3 | `protocol::stage3::{range_source_link, sign_check, norm_equality, encoding_budget, ring_functional, s1_reduction, slack_reduction}` |
| $`(7.16)`$ terminal evaluations | `Stage3Proof::terminal_evaluations` |
| Four-group opening | `OpeningLedger::groups` |
| PIOP transcript | `protocol::transcript::{ProtocolTranscript, VerifierProtocolTranscript}` |
| Canonical instance record | `protocol::instance::record` |

The arithmetic kernels are in `crates/aerie-falcon/src/protocol/kernels/`.
