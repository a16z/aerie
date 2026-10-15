# Compact range kernels and normalized rounds

The three-stage Falcon prover reuses the compact arithmetic kernels in
`crates/aerie-prover/src/protocol/kernels/`. The normative identities and
round encodings are in [Falcon Sections 7 and 10.3](../falcon.md).

## Basis-16 product

`ProductGroup` starts from signed digits and the two quartic range images.
Its compact pair-pattern tables retain integer information until challenges
require extension-field values. Stage one weights its quadratic inner
polynomial by the current equality factor, producing an ordinary cubic
round. The transmitted coefficients are the constant, quadratic, and cubic
coefficients; the running claim determines the linear coefficient.

A signature-domain instance is inactive during the coefficient rounds.
Its contribution halves in each such round and retains the corresponding
power-of-two scale at the terminal.

## Range leaf

`QuarticGroup` operates on the virtual images $`P_Z=Z(Z+1)`$. Compact pattern
lookup and equality-weight bucketing avoid repeatedly evaluating the same
integer images. Later rounds fold extension-field tables.

Stage two uses one shared equality tensor and the division-free normalized
recurrence. It sends the four nonconstant coefficients of the quartic inner
polynomial. If the current equality coordinate is $`\alpha`$, the running
claim determines the constant coefficient as
$`q_0=\mathrm{claim}-\alpha(q_1+q_2+q_3+q_4)`$.
This works at $`\alpha=0`$ and $`\alpha=1`$ as well as at general field points.
Signature-domain sources add no nonconstant coefficients during their
inactive prefix. The terminal check contains the quartic evaluations
without another equality factor.

## Projection link

Stage one's projection link retains the tensor contraction of the public
cores. Signed source values stay compact through their first active fold.
The rest-coordinate folds operate on whole channel fibers; the first-core
contraction then supplies coefficient weights. Signed limb accumulators are
reduced per fiber pair, before field aggregation, to preserve their integer
headroom at every admitted batch size.

## Source links and stage-three predicates

`range_source_kernel::QuadraticGroup` links the six range images to their
committed digits. The named sign, norm, budget, ring and reduction provers
own their relation state directly. Signed sources fold into extension-field
values after their first active challenge. Affine factors such as `sigma + 1`
are evaluated inside the kernel rather than stored as extra tables.

Split equality contractions reduce each inner block before multiplying its
outer equality weight. Signed first rounds use separate positive and negative
unreduced accumulators; later rounds use unreduced field products. Large
contractions distribute outer blocks across Rayon when `parallel` is enabled.

Leading inactive rounds halve the local claim and preserve that scale in
all subsequent round messages and the terminal check. The ring relation
uses the public negacyclic weight, sign, and affine magnitude. Its public
weight is materialized with the factorized transform, while the verifier
uses the carry recurrence in Section 8.
