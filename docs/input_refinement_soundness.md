# Input-ball refinement: soundness, exactness and limitations

This document states what input-ball refinement of poison-certified training guarantees, proves it,
lists the assumptions the proof rests on and how the code enforces them, and records the one
limitation the guarantee has in practice: floating-point rounding.

It is written in the notation of *Certified Robustness to Data Poisoning in Gradient-Based Training*
(Sosnin, Müller, Baader, Tsay & Wicker, [arXiv:2406.05670](https://arxiv.org/abs/2406.05670)), and
reads as an addendum to that paper's Appendix A. Everything in sections 1 to 5 concerns a *single*
iteration of that paper's Algorithm 1; the certificate for a whole training run follows by the same
induction over iterations as in the paper, because refinement hands the next iteration a single
parameter interval exactly as the unrefined algorithm does.

Implementation: [`abstract_gradient_training/input_refinement.py`](../abstract_gradient_training/input_refinement.py).
Tests: [`tests/test_input_refinement.py`](../tests/test_input_refinement.py) (section 8 maps claims to tests).

## Summary

- **What refinement does.** For each sample, the feature-poisoning ball is cut into a grid of $L$
  sub-boxes ("leaves"). Each leaf is bounded by the ordinary bounding pass, and the per-sample
  parameter-gradient bounds are hulled over the leaves before clipping and aggregation.
- **Soundness (Proposition 1).** If the leaves cover the ball, the hulled bounds enclose every
  gradient the adversary can cause, so Theorem 3.4 of the paper applies unchanged. This needs only
  that each sample's gradient does not depend on the other samples (A1).
- **Never wider (Proposition 2).** If the leaves lie inside the ball and the bounding pass is
  inclusion-monotone, the refined bound is contained in the unrefined one.
- **Exactness (Theorem 3).** Hulling $L$ leaves per sample gives the *same* update interval as
  enumerating all $L^b$ assignments of a leaf to every sample of a batch of $b$. Refinement is
  exponential in the number of coordinates cut and flat in the batch size.
- **Limitation (section 7).** The guarantee is a statement in real arithmetic. The implementation,
  like the AGT library it extends, computes interval endpoints in round-to-nearest floating point
  without outward rounding. Section 7 lists every operation where outward rounding would be needed.

---

## 1. Setting

$f^\theta$ is the model, $\mathcal{L}$ the element-wise loss,
$\mathcal{B} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{b}$ the current batch, and $[\theta^L, \theta^U]$ the
parameter interval carried into the current iteration, which accounts for poisoning of all previously
seen batches. Write $\theta \in \mathbb{R}^{d}$, index parameter coordinates by $r \in [d]$, and write
$[u]_r$ for the $r$-th coordinate of $u$. Inequalities between vectors are element-wise.

Fix a bounded adversary $\langle n, \epsilon, p, \nu, q \rangle$ acting on the current batch: it may
choose up to $n$ samples and move each one's features within
$\mathbb{B}_p(x^{(i)}, \epsilon) = \{ \tilde{x} : \lVert x^{(i)} - \tilde{x} \rVert_p \le \epsilon \}$
and its label within $\mathbb{B}_q(y^{(i)}, \nu)$. In the implementation $p = \infty$ (`epsilon`,
`k_poison` $= n$) and the feature set is a box; the proof uses only that the set can be partitioned.

### 1.1 The bounding oracle

Both gradient bounds required by Theorem 3.4 of the paper are produced by one routine,
`BoundedModel.bound_backward_combined`, abstracted here as an oracle $\Gamma^{(i)}$: given a feature
set $\mathcal{X}$ and a label set $\mathcal{Y}$ it returns an interval

$$\Gamma^{(i)}(\mathcal{X}, \mathcal{Y}) = \Big[\Gamma^{(i)}_L, \Gamma^{(i)}_U\Big] \supseteq \Big\{ \nabla_{\tilde{\theta}} \mathcal{L}\big( f^{\tilde{\theta}}(\tilde{x}), \tilde{y} \big) \;\Big|\; \tilde{\theta} \in [\theta^L, \theta^U],\; \tilde{x} \in \mathcal{X},\; \tilde{y} \in \mathcal{Y} \Big\}.$$

The two families of per-sample bounds of Theorem 3.4 are

$$\big[\delta^{(i)}_L, \delta^{(i)}_U\big] = \Gamma^{(i)}\big( \{x^{(i)}\}, \{y^{(i)}\} \big), \qquad \big[\tilde{\delta}^{(i)}_L, \tilde{\delta}^{(i)}_U\big] = \Gamma^{(i)}\big( \mathbb{B}_p(x^{(i)}, \epsilon), \mathbb{B}_q(y^{(i)}, \nu) \big),$$

the effect of previous poisoning only, and its combination with poisoning of the current batch.

### 1.2 The aggregator

Theorem 3.4 aggregates the per-sample bounds into the descent-direction interval

$$\Delta\theta^U = \frac{1}{b}\left( \underset{n}{\operatorname{SEMax}} \Big\{ \tilde{\delta}^{(i)}_U - \delta^{(i)}_U \Big\}_{i=1}^{b} + \sum_{i=1}^{b} \delta^{(i)}_U \right), \qquad \Delta\theta^L = \frac{1}{b}\left( \underset{n}{\operatorname{SEMin}} \Big\{ \tilde{\delta}^{(i)}_L - \delta^{(i)}_L \Big\}_{i=1}^{b} + \sum_{i=1}^{b} \delta^{(i)}_L \right),$$

where $\operatorname{SEMax}_n$ and $\operatorname{SEMin}_n$ sum the largest and smallest $n$ of their
arguments *at each index* of the parameter vector (`PoisoningGradientAccumulator`). Name the upper
endpoint as a function of the perturbed bounds alone, holding the unperturbed bounds fixed:

$$A^U_r(t_1, \dots, t_b) = \frac{1}{b}\left( \underset{n}{\operatorname{SEMax}} \Big\{ t_i - \big[\delta^{(i)}_U\big]_r \Big\}_{i=1}^{b} + \sum_{i=1}^{b} \big[\delta^{(i)}_U\big]_r \right),$$

so that $[\Delta\theta^U]_r = A^U_r\big( [\tilde{\delta}^{(1)}_U]_r, \dots, [\tilde{\delta}^{(b)}_U]_r \big)$,
and dually $A^L_r$ with $\operatorname{SEMin}_n$ and $\delta^{(i)}_L$. Two properties of the aggregator
carry the exactness argument; both are stated in the paper rather than being new assumptions.

- **A2: it is separable across parameter indices.** $A^U_r$ reads coordinate $r$ of the per-sample
  bounds and nothing else (the paper: "this update is applied independently to each parameter index").
- **A3: it is monotone.** $A^U_r$ and $A^L_r$ are nondecreasing in each argument (Lemma 1).

---

## 2. Refinement

**Definition 1 (leaf family).** For each $i \in [b]$, a finite family
$\mathcal{P}^{(i)} = \{ \mathcal{P}^{(i)}_1, \dots, \mathcal{P}^{(i)}_L \}$ of subsets of the input
space is

- **covering** for sample $i$ if $\bigcup_{\ell \in [L]} \mathcal{P}^{(i)}_\ell \supseteq \mathbb{B}_p(x^{(i)}, \epsilon)$;
- **inner** for sample $i$ if $\mathcal{P}^{(i)}_\ell \subseteq \mathbb{B}_p(x^{(i)}, \epsilon)$ for every $\ell$.

The grid built by `leaf_boxes` is both (section 6). The leaf bounds are the oracle applied to each
leaf, with the label set left whole:

$$\big[\tilde{\delta}^{(i,\ell)}_L, \tilde{\delta}^{(i,\ell)}_U\big] = \Gamma^{(i)}\Big( \mathcal{P}^{(i)}_\ell, \mathbb{B}_q\big(y^{(i)}, \nu\big) \Big), \qquad \ell \in [L].$$

Two indices must be kept apart. $\ell \in [L]$ is a **per-sample leaf index**: it names one leaf of
*one* sample's ball. $\vec{\ell} = (\ell_1, \dots, \ell_b) \in [L]^{b}$ is a **joint leaf index**: it
fixes a leaf for every sample at once, naming the region
$\mathcal{P}^{(1)}_{\ell_1} \times \cdots \times \mathcal{P}^{(b)}_{\ell_b}$ of the adversary's feature
choice set.

**Per-sample refinement** (what the implementation does). Hull each sample's own leaves back into one
interval, coordinatewise in $r$ and separately for each $i$:

$$\tilde{\delta}^{(i),\mathrm{ps}}_U = \max_{\ell \in [L]} \tilde{\delta}^{(i,\ell)}_U, \qquad \tilde{\delta}^{(i),\mathrm{ps}}_L = \min_{\ell \in [L]} \tilde{\delta}^{(i,\ell)}_L,$$

then apply Theorem 3.4 to those $b$ intervals, giving $\Delta\theta^U_{\mathrm{ps}}$ and
$\Delta\theta^L_{\mathrm{ps}}$. Cost: $L$ bounding passes over the batch.

**Joint refinement** (the enumeration one might think is required). Apply Theorem 3.4 *inside* each
joint leaf $\vec{\ell}$, with sample $i$'s perturbed bound taken to be
$[\tilde{\delta}^{(i,\ell_i)}_L, \tilde{\delta}^{(i,\ell_i)}_U]$, to obtain $\Delta\theta^U(\vec{\ell})$
and $\Delta\theta^L(\vec{\ell})$, then hull over all joint leaves:

$$\big[\Delta\theta^U_{\mathrm{joint}}\big]_r = \max_{\vec{\ell} \in [L]^{b}} \big[\Delta\theta^U(\vec{\ell})\big]_r, \qquad \big[\Delta\theta^L_{\mathrm{joint}}\big]_r = \min_{\vec{\ell} \in [L]^{b}} \big[\Delta\theta^L(\vec{\ell})\big]_r.$$

Cost: $L^{b}$ bounding passes. Section 4 shows the two agree exactly.

---

## 3. Soundness

**A1 (per-sample separability).** For every $\tilde{\theta}$, the gradient of sample $i$,
$\nabla_{\tilde{\theta}} \mathcal{L}(f^{\tilde{\theta}}(\tilde{x}^{(i)}), \tilde{y}^{(i)})$, depends on
$(\tilde{x}^{(i)}, \tilde{y}^{(i)})$ alone and not on the rest of the batch.

This is the paper's standing assumption that the loss is computed element-wise, together with the
absence of any batch-coupled layer in $f$. Its consequence is that the leaf bound
$[\tilde{\delta}^{(i,\ell)}_L, \tilde{\delta}^{(i,\ell)}_U]$ encloses sample $i$'s gradient on
$\mathcal{P}^{(i)}_\ell$ *whatever leaf every other sample occupies*.

**Proposition 1 (soundness).** Assume A1 and let $\mathcal{P}^{(i)}$ be covering for every $i$. Then
$[\tilde{\delta}^{(i),\mathrm{ps}}_L, \tilde{\delta}^{(i),\mathrm{ps}}_U]$ satisfies the hypothesis of
Theorem 3.4 on $\tilde{\delta}^{(i)}_L, \tilde{\delta}^{(i)}_U$, and therefore
$\Delta\theta^L_{\mathrm{ps}} \le \Delta\theta \le \Delta\theta^U_{\mathrm{ps}}$ for every
$\theta \in [\theta^L, \theta^U]$ and every poisoned batch the adversary can produce.

*Proof.* Fix $i$ and take any $\tilde{\theta} \in [\theta^L, \theta^U]$,
$\tilde{x} \in \mathbb{B}_p(x^{(i)}, \epsilon)$ and $\tilde{y} \in \mathbb{B}_q(y^{(i)}, \nu)$. By
covering there is an $\ell$ with $\tilde{x} \in \mathcal{P}^{(i)}_\ell$, so by A1 and the definition of
the oracle,
$\tilde{\delta}^{(i,\ell)}_L \le \nabla_{\tilde{\theta}} \mathcal{L}(f^{\tilde{\theta}}(\tilde{x}), \tilde{y}) \le \tilde{\delta}^{(i,\ell)}_U$.
The hull only widens this:
$\tilde{\delta}^{(i),\mathrm{ps}}_L \le \tilde{\delta}^{(i,\ell)}_L$ and
$\tilde{\delta}^{(i,\ell)}_U \le \tilde{\delta}^{(i),\mathrm{ps}}_U$. So the hulled interval encloses
the set Theorem 3.4 requires it to, and the theorem applies verbatim with it substituted in. $\square$

Proposition 1 needs no monotonicity of the aggregator, no separability across parameter indices, and
no relation between the leaf families of different samples. It also holds for any clipping method
whose interval propagation is sound, because clipping is applied to the hulled interval, which
encloses the true per-sample gradient.

**Proposition 2 (never wider).** Assume in addition that $\Gamma^{(i)}$ is *inclusion-monotone*,
$\mathcal{X} \subseteq \mathcal{X}' \Rightarrow \Gamma^{(i)}(\mathcal{X}, \mathcal{Y}) \subseteq \Gamma^{(i)}(\mathcal{X}', \mathcal{Y})$,
and let $\mathcal{P}^{(i)}$ be inner. Then
$\tilde{\delta}^{(i)}_L \le \tilde{\delta}^{(i),\mathrm{ps}}_L \le \tilde{\delta}^{(i),\mathrm{ps}}_U \le \tilde{\delta}^{(i)}_U$:
refinement never loosens the bound of Theorem 3.4.

*Proof.* Each leaf bound is an oracle call on a subset of $\mathbb{B}_p(x^{(i)}, \epsilon)$, hence
contained in the unrefined bound; a hull of subsets of an interval is a subset of it. $\square$

Interval bound propagation is inclusion-monotone by construction. The same argument shows that a grid
obtained by cutting every cell of another grid again gives a bound that is no wider (2 cuts to 4 per
coordinate, say). Grids that are not nested (2 cuts against 3) are not ordered.

---

## 4. Exactness

**Lemma 1 (the aggregator is monotone).** For every $r$, $A^U_r$ and $A^L_r$ are nondecreasing with
respect to the product order on $\mathbb{R}^{b}$.

*Proof.* Write the order statistics as optima over subsets:
$\operatorname{SEMax}_n\{u_i\} = \max_{|S| = n} \sum_{i \in S} u_i$ and
$\operatorname{SEMin}_n\{u_i\} = \min_{|S| = n} \sum_{i \in S} u_i$. For each fixed $S$ the sum is
nondecreasing in every $u_i$, and a pointwise maximum or minimum of nondecreasing functions is
nondecreasing. In $A^U_r$ the map $t_i \mapsto t_i - [\delta^{(i)}_U]_r$ is increasing, the terms
$[\delta^{(i)}_U]_r$ are constants (oracle calls at the clean pair, which depend on no leaf), and
$1/b > 0$. The argument for $A^L_r$ is identical. $\square$

Note that $\operatorname{SEMin}_n$ is nondecreasing, not nonincreasing: it selects the $n$ smallest
arguments, but raising any argument can only raise, or leave alone, that sum.

**A5 (the choice set is a product).** The admissible joint leaf indices are all of $[L]^{b}$: fixing
the leaf occupied by one sample places no constraint on the leaf occupied by any other. This is a
property of the threat model. The bounded adversary perturbs each selected point independently within
its own ball, and its only condition linking samples, the count budget $n$, is absorbed into the
aggregator by $\operatorname{SEMax}_n$.

**Lemma 2 (monotone optimisation over a product).** Let $A : \mathbb{R}^{b} \to \mathbb{R}$ be
nondecreasing in the product order, let $[L]$ be finite and nonempty, and let
$u_1, \dots, u_b : [L] \to \mathbb{R}$. Then

$$\max_{\vec{\ell} \in [L]^{b}} A\big( u_1(\ell_1), \dots, u_b(\ell_b) \big) = A\Big( \max_{\ell} u_1(\ell), \dots, \max_{\ell} u_b(\ell) \Big), \qquad \min_{\vec{\ell} \in [L]^{b}} A\big( u_1(\ell_1), \dots, u_b(\ell_b) \big) = A\Big( \min_{\ell} u_1(\ell), \dots, \min_{\ell} u_b(\ell) \Big).$$

*Proof.* The set of argument vectors at which $A$ is evaluated,
$\mathcal{U} = \{ (u_1(\ell_1), \dots, u_b(\ell_b)) : \vec{\ell} \in [L]^{b} \}$, factorises as
$u_1([L]) \times \cdots \times u_b([L])$, because the $i$-th component of $\vec{\ell}$ ranges over all
of $[L]$ whatever the other components are. Each factor is a finite nonempty set of reals, so it has
a greatest element, attained at some $\ell^\star_i$. Hence $\mathcal{U}$ has a *greatest* element
$u^\star = (\max_\ell u_1(\ell), \dots, \max_\ell u_b(\ell))$, attained at
$\vec{\ell}^\star = (\ell^\star_1, \dots, \ell^\star_b)$. Since $A$ is nondecreasing,
$A(u) \le A(u^\star)$ for every $u \in \mathcal{U}$, and $A(u^\star)$ is itself one of the values
being maximised, so it is the maximum. The second identity is the same argument with the least element
of $\mathcal{U}$. $\square$

**Theorem 3 (per-sample refinement is exact).** Assume A1, A2, A3 and A5. Then for every parameter
index $r$,

$$\big[\Delta\theta^U_{\mathrm{ps}}\big]_r = \big[\Delta\theta^U_{\mathrm{joint}}\big]_r, \qquad \big[\Delta\theta^L_{\mathrm{ps}}\big]_r = \big[\Delta\theta^L_{\mathrm{joint}}\big]_r.$$

Hulling $L$ per-sample leaves and hulling $L^{b}$ joint leaves give the same interval, coordinate for
coordinate.

*Proof.* Fix $r$. By A2,
$[\Delta\theta^U(\vec{\ell})]_r = A^U_r\big( [\tilde{\delta}^{(1,\ell_1)}_U]_r, \dots, [\tilde{\delta}^{(b,\ell_b)}_U]_r \big)$.
Set $u_i(\ell) = [\tilde{\delta}^{(i,\ell)}_U]_r$. By Lemma 1, $A^U_r$ is nondecreasing, so Lemma 2
gives

$$\max_{\vec{\ell}} \big[\Delta\theta^U(\vec{\ell})\big]_r = A^U_r\Big( \max_{\ell} \big[\tilde{\delta}^{(1,\ell)}_U\big]_r, \dots, \max_{\ell} \big[\tilde{\delta}^{(b,\ell)}_U\big]_r \Big) = \big[\Delta\theta^U_{\mathrm{ps}}\big]_r.$$

The lower endpoint is the $\min$ identity applied to $A^L_r$. $\square$

**In words.** The expensive enumeration cannot beat the cheap one because the worst joint leaf *is*
the combination of each sample's own worst leaf, and that combination is always available: the
adversary's choice set is a product, so nothing forbids every sample from sitting in its own worst
leaf at once.

**A4 (the certificate is a box).** The maximiser $\vec{\ell}^\star$ depends on $r$: different
parameter indices are attained by different joint leaves. This is not a gap, because Algorithm 1
propagates an *interval* $[\theta^L, \theta^U]$, so the hull over joint leaves is itself taken index
by index. It is the relaxation the paper already accepts when it notes that "the $n$ worst-case points
for one parameter index are unlikely to be the same for all indices". Theorem 3 says refinement adds
no further looseness of this kind.

**Worked example.** Take $b = 2$, $n = 1$, $L = 2$, $\delta^{(i)}_U = 0$, and two parameter indices
$r$ and $r'$, with leaf upper bounds

| | $[\tilde{\delta}^{(i,1)}_U]_r$ | $[\tilde{\delta}^{(i,2)}_U]_r$ | $[\tilde{\delta}^{(i,1)}_U]_{r'}$ | $[\tilde{\delta}^{(i,2)}_U]_{r'}$ |
|---|---|---|---|---|
| sample 1 | 3.0 | 1.0 | 0.2 | 4.0 |
| sample 2 | 0.5 | 2.0 | 1.0 | 0.0 |

The per-sample hulls are $(3.0, 4.0)$ and $(2.0, 1.0)$ over $(r, r')$, so
$b \cdot \Delta\theta^U_{\mathrm{ps}} = (3.0, 4.0)$. Over the four joint leaves
$(1,1), (1,2), (2,1), (2,2)$ the update at $r$ is $3.0, 3.0, 1.0, 2.0$ and at $r'$ is
$1.0, 0.2, 4.0, 4.0$: the joint hull is $(3.0, 4.0)$ as well. Index $r$ is attained at $(1, 2)$ and
$r'$ at $(2, 1)$; no single joint leaf attains both, and A4 means none needs to.

### Clipping

Gradient clipping sits between the leaf bounds and the aggregator. Under `clip_method="clamp"`,
$\operatorname{Clip}_\kappa : t \mapsto \max(-\kappa, \min(\kappa, t))$ is applied element-wise to the
interval endpoints.

**Corollary 4.** $\operatorname{Clip}_\kappa$ is nondecreasing and acts element-wise, so Lemma 1
survives composition with it and Theorem 3 holds for the clipped update rule. Moreover it commutes
with the hull, $\max_\ell \operatorname{Clip}_\kappa(t_\ell) = \operatorname{Clip}_\kappa(\max_\ell t_\ell)$
and likewise for $\min$, so clipping after the hull (as the implementation does) and clipping inside
each leaf give identical numbers.

Under `clip_method="norm"` the clipping factor $\kappa / \lVert \delta^{(i)} \rVert_2$ couples every
index of a sample's gradient, so A2 fails. Per-sample hulling remains sound (Proposition 1) but is
strictly looser than clipping inside each leaf.

---

## 5. The assumptions, and how each one fails

| | Assumption | Holds because | Fails under | Cost of failure |
|---|---|---|---|---|
| A1 | per-sample separability | losses use `reduction="none"`; no batch-coupled layer | batch norm; a loss with a batch-level term | **soundness** |
| A2 | aggregator separable across parameter indices | per-index top-$n$ in `PoisoningGradientAccumulator`; clamp clipping | `clip_method="norm"` | tightness only |
| A3 | aggregator monotone | Lemma 1 | a non-monotone aggregator (median, trimmed mean) | **soundness**: the per-sample hull no longer bounds the joint enumeration, and Theorem 3.4 would have to be re-derived for that aggregator |
| A4 | the certificate is a box | Definition 3.1 of the paper | a relational abstraction of the parameter set | tightness only |
| A5 | the adversary's choice set is a product | the bounded threat model | an adversary whose choices are coupled across samples, such as a shared backdoor trigger | tightness only |

Certifying against the product set remains sound for a coupled adversary (its choice set is a subset
of the product), so losing A5 costs exactness, not soundness.

`validate_input_refinement` enforces the two a user can break from the configuration: it rejects
batch-coupled layers and any loss whose reduction is not `"none"` (A1), and warns on norm clipping
(A2).

---

## 6. How the implementation meets the hypotheses

**The grid covers the ball and lies inside it** (`leaf_boxes`). Coordinate `dims[i]` of each sample's
box $[x - \epsilon, x + \epsilon]$ is cut into `splits[i]` intervals and every other coordinate keeps
its full range. Covering and inner are properties of the computed floating-point corners, not only of
the ideal grid:

- the outermost corners are `batch - epsilon` and `batch + epsilon`, the very expressions the
  unrefined bound uses, so the grid's extent is bit-identical to the unrefined ball;
- two neighbouring leaves use one shared expression for their common corner, so there is no gap
  between them whatever that corner rounds to;
- every interior corner is clamped to at most `batch + epsilon`.

Interior corners therefore need no directed rounding: any value is sound as long as both neighbours
use it. (An earlier version computed the top corner as `lower + n_splits * width`, which rounds below
`batch + epsilon` in about a fifth of coordinates and left a sliver of the ball uncovered; the test
`test_leaves_tile_the_ball_exactly` guards this.)

**The grid is shared across the batch, which is a convenience and not a hypothesis.** Neither
proposition relates the leaf families of different samples. Giving every sample the same pattern of
offsets around its own centre is what lets one leaf be a single batched bounding pass over $b$ rows.

**The split coordinates may change from step to step.** With `strategy="sensitivity"` they are
re-selected each iteration from the nominal first-layer weights. Any choice of coordinates gives a
covering, inner family, so the choice affects tightness only.

**The hull is taken at the per-sample parameter gradient**, before clipping and before the order
statistic. Later is impossible, since the order statistic is the reduction whose leaves must be kept
separate. A single interval $[\theta^L, \theta^U]$ is carried into the next iteration, as Algorithm 1
requires, so leaf counts do not compound across a run.

**Chunking** groups leaves into one bounding call and hulls group by group; $\min$ and $\max$ are
associative and exact, so the grouping cannot change the result.

**Sharding** gives each rank of a `torch.distributed` process group a contiguous block of leaf ids
and combines the per-rank hulls with MIN and MAX all-reduces, which are exact. The hull then mixes
leaves bounded on different ranks, which is sound only if every rank bounded them under the same
parameter interval and the same batch. This is enforced rather than assumed: rank 0's parameters,
parameter bounds, batch and split coordinates are broadcast to every rank at the start of every step,
and rank 0's final model at the end.

---

## 7. Floating point: the limitation, and where outward rounding is needed

Sections 1 to 6 are statements in real arithmetic. An interval $[l, u]$ computed in floating point
encloses the true range only if every operation that produces $l$ rounds down and every operation
that produces $u$ rounds up (*outward rounding*). The AGT library computes all endpoints in IEEE
round-to-nearest, and this extension inherits that. So the certificate is rigorous up to rounding
error, and **not rigorous in the strict sense**: a computed endpoint can sit on the wrong side of the
true one by an accumulated rounding error.

### What is done instead

- `interval_arithmetic.validate_interval` checks $l \le n \le u$ for the nominal value $n$ after most
  operations and logs any violation, at a level set by its size: nothing below $10^{-6}$, then debug,
  info ($> 10^{-5}$), warning ($> 10^{-4}$) and error ($> 10^{-3}$). It never raises.
- The experiment harness (`scripts/refinement_paper/harness.count_violations`) records the worst
  logged violation of every run. At the `WARNING` log level the experiments use, it sees violations
  above $10^{-4}$; a recorded `violation` of `0.0` means "none above $10^{-4}$", not "none". A run
  above $10^{-3}$ is not reported, and its float64 rerun is used in its place.
- Half-moons and MAGIC run in float64 throughout. CIFAR-10 and OCT-MNIST run in float32, where the
  margin is thin: on OCT-MNIST the same schedule at 12 PCA features violates its bounds by up to
  $10^{-1}$ in float32 and by nothing in float64.

This check is necessary, not sufficient. It detects the nominal value leaving its own bounds; it
cannot detect a bound that is too tight for some non-nominal parameter or input.

### Where refinement itself needs rounding control: nowhere new

Refinement adds three kinds of operation, and none of them creates a rounding obligation of its own:

- **leaf corners.** Only the outermost corners must enclose the ball, and they are the unrefined
  expressions `batch - epsilon` and `batch + epsilon` (item 1 below, shared with the unrefined path).
  Interior corners need no directed rounding (section 6).
- **the hull**, `amin`/`amax` over leaves and `torch.minimum`/`torch.maximum` across chunks: exact.
- **the cross-rank reduction**, MIN/MAX `all_reduce`, and the broadcasts: exact.

So a refined run needs outward rounding in exactly the places an unrefined run does, applied once per
leaf.

### Where outward rounding would be needed (inherited from AGT)

For a strictly rigorous certificate, every item below would have to round its lower endpoint down and
its upper endpoint up. Items are in the order the pipeline executes them.

| # | Where | Operations | What rigour needs |
|---|---|---|---|
| 1 | `training_utils.compute_batch_gradients`, `input_refinement.leaf_boxes` | `batch - epsilon`, `batch + epsilon` | one subtraction rounded down, one addition rounded up |
| 2 | `interval_arithmetic.propagate_matmul_rump` (default), `_exact`, `_nguyen`, via `propagate_affine`: every `Linear` layer of the forward pass, and the backward pass products in `IntervalBoundedModel.bound_backward` | midpoint and radius `(u ± l) / 2`; the matrix products `A_mu @ B_mu`, `\|A_mu\| @ B_r`, `A_r @ \|B_mu\|`, `A_r @ B_r`; `H_mu ± H_r` | the radius rounded up at every step, and the rounding error of the midpoint product added to the radius. The matrix products are GPU GEMM calls with fused multiply-adds and an unspecified summation order: **this is the hot loop, and the place a custom kernel would go** |
| 3 | `interval_arithmetic.propagate_elementwise` (products of two intervals: the backward pass through activations, the per-sample weight gradient, norm clipping) | four products, then `min`/`max` | each product rounded toward its endpoint's side; `min`/`max` are exact |
| 4 | ReLU and its derivative interval | `max(x, 0)`, comparisons with 0 | exact: nothing needed |
| 5 | `bounded_losses`: `BoundedCrossEntropyLoss.bound_backward` through `interval_arithmetic.propagate_softmax`; `BoundedBCEWithLogitsLoss.bound_backward` (`torch.sigmoid(logit) - target`) | `exp`, sums, a division; `sigmoid`; a subtraction | rigorous enclosures of `exp`/`sigmoid`. GPU math libraries bound their error in ulps but do not round in a chosen direction, so this needs either a directed implementation or inflation by the documented error bound |
| 6 | `training_utils.propagate_clipping` | clamp: `min`/`max`. Norm: squares, a sum, a square root, a division, products | clamp is exact. Norm clipping needs every listed operation directed (`propagate_norm`, then item 3) |
| 7 | `gradient_accumulation.PoisoningGradientAccumulator` | sums over the batch (`sum(dim=0)`, thousands of terms, unspecified order); the differences `iwp - wp`; top-$n$ selection; the sum of the selected $n$; division by the batch size | selection is exact; the sums, differences and division need directing |
| 8 | `bounded_optimizers.BoundedSGD.step` | the learning-rate schedule; `param_l -= lr * grad_u`, `param_u -= lr * grad_l`; the L2 term when used | product and subtraction directed. This is where one step's error enters the interval the next step starts from |
| 9 | `test_metrics.test_accuracy` and `test_cross_entropy` | a bounding forward pass (items 2 to 4) on the test points, then the sign of the worst-case logit margin | items 2 to 4; the comparison is exact |

Two things lie outside the certificate altogether, by construction rather than by rounding: the fixed
feature map applied before training (standardisation, PCA projection and scaling), since the
certificate is about the feature vectors as computed; and the nominal trajectory, which is the
floating-point run itself.

### How it could be made rigorous

Dynamic rounding modes do not exist on GPUs, but three techniques cover the table.

- **One-ulp inflation for correctly rounded scalar operations** (items 1, 3, 8, and the scalar parts
  of 6 and 7). IEEE addition, subtraction, multiplication, division and square root are correctly
  rounded, so the true result lies within one unit in the last place of the computed one. Replacing a
  lower endpoint by `torch.nextafter(l, -inf)` and an upper endpoint by `torch.nextafter(u, +inf)`
  after each such operation is rigorous, needs no custom kernel, and costs one extra elementwise pass.
- **A-priori error bounds for sums and dot products** (items 2 and 7). For a length-$m$ dot product
  evaluated in any order, with or without fused multiply-add,
  $\lvert \mathrm{fl}(a^\top b) - a^\top b \rvert \le \gamma_m \lvert a \rvert^\top \lvert b \rvert$
  with $\gamma_m = m u / (1 - m u)$ and $u$ the unit roundoff (plus an underflow term). Adding
  $\gamma_m \, \lvert A \rvert \lvert B \rvert$ to the radius in Rump's algorithm makes item 2
  rigorous with stock GEMM, at the price of one more matrix product and a slightly wider interval.
  The same bound with $b = \mathbf{1}$ covers the batch sums of item 7.
- **Custom CUDA kernels with directed rounding** (item 2 above all). CUDA exposes per-operation
  directed-rounding intrinsics for addition, multiplication and fused multiply-add. An interval GEMM
  written with them would compute the lower endpoint with every accumulation rounded down and the
  upper with every accumulation rounded up, giving the tightest rigorous result. It would replace the
  four GEMM calls of `propagate_matmul_rump` and the reductions of item 7, and is the only one of the
  three techniques that needs new kernels. Item 5 would additionally need `exp` and `sigmoid` with a
  guaranteed direction or a proven error bound.

None of these is implemented. The a-priori bounds are the cheapest route to a rigorous certificate;
the custom kernels are the route to a rigorous certificate that is as tight as the current one.

---

## 8. How the tests check each claim

| Claim | Test in `tests/test_input_refinement.py` |
|---|---|
| The leaves cover the ball exactly, with no gaps (Definition 1) | `test_leaves_tile_the_ball_exactly` |
| Proposition 1: refined gradient bounds enclose every attainable per-sample gradient | `test_sampled_gradients_lie_inside_the_refined_bounds` (cross-entropy, binary cross-entropy, MSE; interior points and vertices of the parameter box and the ball) |
| Proposition 1, end to end: a model trained on poisoned data lies inside the refined box | `test_models_trained_on_poisoned_data_lie_inside_the_refined_box` (model-blind poisoning and an adaptive gradient-directed attack; clamp and norm clipping; with a trusted clean batch in `test_poisoned_models_lie_inside_the_refined_box_with_clean_data`) |
| That test can fail: the attack escapes a box certified for a weaker adversary, and one built from too few leaves | `test_the_adaptive_attack_escapes_an_under_certified_box`, `test_the_adaptive_attack_escapes_a_box_built_from_too_few_leaves` |
| Proposition 2: never wider; nested grids are ordered | `test_refinement_is_never_wider`, `test_a_nested_finer_grid_is_never_wider`, `test_refined_training_is_valid_and_no_wider` |
| Theorem 3 with Corollary 4: per-sample hull equals the joint enumeration, bit for bit | `test_per_sample_hull_equals_joint_leaf_enumeration` |
| Chunking and sharding do not change the result | `test_chunking_does_not_change_the_bound`, `test_sharded_bound_matches_single_process`, `test_sharded_training_matches_single_process` |
| A1 and the cost guard are enforced | `test_entry_rejects_batch_coupled_layer`, `test_entry_rejects_over_max_leaves` |

The end-to-end tests use short runs on purpose. The certified box compounds from step to step much
faster than any real attack does: after one or two steps the adaptive attack reaches the edge of a
correctly certified box and escapes an under-certified one, and the fraction of the box it reaches
falls as the run gets longer. An attack that cannot get near the boundary cannot expose an unsound
bound.
