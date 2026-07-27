# Two-phase C_GG protocol (stochastic compositional Balls & Urns)

Locked single-point test of $\mathrm{C}_{GG}$ emergence via curriculum + patching.
This document incorporates the design-review fixes (approx $\mathrm{C}_{GG}$, Phase-1
$g$/$f$ Bayes gates, $z$_decode, measurement locks).

**Code:** [`composition/compositional_urns/`](.).

---

## Locked settings

| Knob | Value |
|------|--------|
| Alphabets | $\|X\|=20$, $\|Z\|=6$, $\|Y\|=20$ (bottleneck heuristic: $\|Z\| < \|X\|/2$; **not** claimed as write-up $\Delta K$) |
| Vocab | $X \sqcup Z \sqcup Y$, size 46, contiguous offsets |
| $D$ | 64 train tasks; 200 OOD tasks; seed 1 |
| $C$ / pairs | sequence length 64 → $L=32$ pairs |
| Arch | **2 layers, 4 heads, $d=128$, $d_{ff}=512$**, context 128 (Wurgaft 1L/64d is out of scope) |
| Opt | AdamW lr $10^{-3}$, wd $0.01$, linear warmup 1000, then constant; bs 256 |
| Patch layer | residual **after layer 0**; also log layer 1 |
| Patch design | **cross-task**: source/destination residuals from distinct tasks $d_{\mathrm{src}}\ne d_{\mathrm{dst}}$ |
| Distance | **symmetrized KL** on $Y$-slice softmax |
| Soft 3-way weight | $d_Q^{\mathrm{rel}} = d(h,Q) / \sum_{Q'\in\{M,G,\mathrm{C}_{GG}\}} d(h,Q')$ |
| Success margin $\varepsilon$ | $d_{\mathrm{C}_{GG}}^{\mathrm{rel}}$ strictly smallest **and** $\ge \varepsilon$ below next-best with $\varepsilon=0.05$ |
| Patch margin | mean $\mathrm{KL}_{\mathrm{route}} + 0.05 \le \mathrm{KL}_{\mathrm{transplant}}$ **and** $\le \mathrm{KL}_{\mathrm{unaffected}}$, over 200 accepted triples |
| $z$_decode gate | accuracy $\ge 0.4$ (chance $=1/6\approx 0.167$) |

---

## Task generative model

Task $d = (w_g^{(d)}, w_f^{(d)})$:

- $w_g \in \mathbb{R}^{20\times 6}$ row-stochastic; each row $\sim\mathrm{Dirichlet}(\mathbf{1}_6)$
- $w_f \in \mathbb{R}^{6\times 20}$ row-stochastic; each row $\sim\mathrm{Dirichlet}(\mathbf{1}_{20})$

Composite marginal: $P(y\mid x) = \sum_z w_g[x,z]\, w_f[z,y]$.

### Sequence types (inputs Uniform on their alphabet)

- **g-only:** emit $(x_1,z_1,\ldots,x_L,z_L)$, $z_i\sim w_g[x_i]$
- **f-only:** emit $(z_1,y_1,\ldots,z_L,y_L)$, $y_i\sim w_f[z_i]$
- **comp:** emit $(x_1,y_1,\ldots,x_L,y_L)$; sample hidden $z_i\sim w_g[x_i]$, then $y_i\sim w_f[z_i]$

Loss CE masked to output positions only (odd indices).

---

## Bayesian predictors (explicit algorithms)

Implemented in [`predictors.py`](predictors.py). All predictives are distributions over $Y$
(or $Z$ for the $g$-predictor) given a typed prefix.

### $g$-predictor (Phase-1 gate; g-only prefixes)

Exact Dirichlet–Categorical on rows of $w_g$:

- Prior $\alpha_g[x,:]=\mathbf{1}_{|Z|}$
- After counts $n_{x,z}$ from observed $(x,z)$ pairs:
  $P(z\mid x,\mathrm{prefix}) = (\alpha_g[x,z]+n_{x,z}) / (|Z| + n_{x,\cdot})$

### $f$-predictor (Phase-1 gate; f-only prefixes)

Same for rows of $w_f$ with prior $\mathbf{1}_{|Y|}$ on $(z,y)$ pairs.

### $M$ (memorizing; discrete prior over train pool)

Uniform prior over $d=1..D$. Soft likelihood on a **comp** prefix
$(x_i,y_i)_{i=1}^{n}$:

$$
\ell_d = \prod_{i=1}^{n} \sum_z w_g^{(d)}[x_i,z]\, w_f^{(d)}[z,y_i]
$$

Posterior $\propto \ell_d$. Predictive at query $x$:

$$
P_M(y\mid x) = \sum_d P(d\mid\mathrm{prefix})\, \sum_z w_g^{(d)}[x,z]\, w_f^{(d)}[z,y]
$$

### Atomic $G$ / direct-Dirichlet (misspecified stand-in)

**Not** the true prior predictive under $T_{\mathrm{true}}$. Treats each $x$ as having
a direct categorical $P(y\mid x)$ with $\mathrm{Dirichlet}(\mathbf{1}_{|Y|})$ prior,
updated by observed $(x,y)$ counts only (ignores factor structure).

### Approximate $\mathrm{C}_{GG}$ (mean-field / online soft counts)

Used for all **comp** behavioral scoring ($L=32$). Maintains Dirichlet counts
$\alpha_g[x,z]$, $\alpha_f[z,y]$ (init ones). For each prefix pair $(x,y)$ in order:

1. $p_g(z) = \alpha_g[x,z] / \sum_{z'}\alpha_g[x,z']$
2. $p_f(z) \propto \alpha_f[z,y] / \sum_{y'}\alpha_f[z,y']$ (row-normalized then reweight)
3. Soft responsibility $r(z) \propto p_g(z)\, p_f^{\mathrm{row}}(z)$ with $p_f^{\mathrm{row}}(z)=\alpha_f[z,y]/\sum_{y'}\alpha_f[z,y']$
4. $\alpha_g[x,:] \mathrel{+}= r$; $\alpha_f[:,y] \mathrel{+}= r$ (add $r(z)$ to $\alpha_f[z,y]$)

Predictive: $P(y\mid x)=\sum_z \hat w_g[x,z]\, \hat w_f[z,y]$ with row means of $\alpha$.

**Unit test:** for $L\le 4$, compare approx predictive to expensive **exact** sum over
$z_{1:L}$ of Dirichlet–Multinomial marginals ([`test_predictors.py`](test_predictors.py)).

### Exact $\mathrm{C}_{GG}$ (short prefixes only)

$$
P(y_{1:n}\mid x_{1:n})
= \sum_{z_{1:n}}
\Biggl(\prod_x \frac{\prod_z \Gamma(1+n_{x,z})\, \Gamma(|Z|)}{\Gamma(|Z|+n_{x,\cdot})\, \Gamma(1)^{|Z|}}\Biggr)
\Biggl(\prod_z \frac{\prod_y \Gamma(1+m_{z,y})\, \Gamma(|Y|)}{\Gamma(|Y|+m_{z,\cdot})\, \Gamma(1)^{|Y|}}\Biggr)
$$

with $n_{x,z}=\#\{i:x_i=x,z_i=z\}$, $m_{z,y}=\#\{i:z_i=z,y_i=y\}$.
Predictive via ratio of joint marginals with an extra query pair.

---

## Phase 1: g+f pretraining

- Mix: $p_g=0.5$, $p_f=0.5$, $p_{\mathrm{comp}}=0$
- Eval every 5k; cap $N_1^{\max}=10^5$
- **Gate (both must hold on ID, output positions only):**
  - $\mathrm{CE}_{\mathrm{model}}^{\mathrm{g\text{-}only}} - \mathrm{CE}_{g\mathrm{-}Bayes} \le 0.05$
  - $\mathrm{CE}_{\mathrm{model}}^{\mathrm{f\text{-}only}} - \mathrm{CE}_{f\mathrm{-}Bayes} \le 0.05$
- Keep **earliest** checkpoint that passes; if none by cap → **inconclusive**, stop

Helpers: [`phase1_gate.py`](phase1_gate.py).

---

## Phase 2: comp + replay

- Init from Phase-1 ckpt
- Mix: $p_{\mathrm{comp}}=0.85$, $p_g=p_f=0.075$
- Loss: $\mathcal{L}=\mathcal{L}_{\mathrm{comp}}+2\mathcal{L}_g+2\mathcal{L}_f$
- Steps: 60k; save at $\{1\text{k},3\text{k},10\text{k},30\text{k},60\text{k}\}$

### Per-checkpoint measurements

**Behavioral (ID and OOD):**

- Comp CE; g-only / f-only CE (forgetting)
- Soft 3-way $d_Q^{\mathrm{rel}}$ for $Q\in\{M,G,\mathrm{C}_{GG}\}$ on **comp** (ID + OOD)
- “$\mathrm{C}_{GG}$ wins” requires ID **and** OOD, with margin $\varepsilon$

**Mechanistic (200 triples), cross-task:**

Same-task patching (one shared task, two queries) cannot distinguish genuine
routing from shortcut task-recognition: with a single $(w_g,w_f)$, both
candidate answers are values that task already produces on its own for some
query. Trials are therefore **cross-task**: source and destination residuals
come from two *distinct* pretraining tasks, so the routed target is a value
neither task produces alone.

1. Sample distinct tasks $d_{\mathrm{src}} \ne d_{\mathrm{dst}}$ and independent
   queries $x_{\mathrm{src}}, x_{\mathrm{dst}} \sim \mathrm{Unif}(X)$.
   $z_{\mathrm{src}} = \arg\max_z w_g^{(d_{\mathrm{src}})}[x_{\mathrm{src}}]$ (modal intermediate).
2. Three candidates: $\mathrm{routed}=f_{\mathrm{dst}}(z_{\mathrm{src}})$ (row lookup);
   $\mathrm{transplant}=f_{\mathrm{src}}(g_{\mathrm{src}}(x_{\mathrm{src}}))$ (source's own finished answer);
   $\mathrm{unaffected}=f_{\mathrm{dst}}(g_{\mathrm{dst}}(x_{\mathrm{dst}}))$ (destination's own finished answer).
3. **Discriminating-trial filter**: accept only if the three candidates'
   $\arg\max$ are pairwise distinct (retry, bounded); report accept rate.
4. Independent prefixes for source and destination (same length so query
   position aligns); capture $h_{\mathrm{src}}$ at layer 0, query position, from
   the source run; **patch into the destination run** at the same position.
5. $\mathrm{KL}_{\mathrm{route}}$, $\mathrm{KL}_{\mathrm{transplant}}$, $\mathrm{KL}_{\mathrm{unaffected}}$: patched
   output vs each candidate.
6. Success needs $\mathrm{KL}_{\mathrm{route}}+0.05\le\mathrm{KL}_{\mathrm{transplant}}$ **and**
   $\mathrm{KL}_{\mathrm{route}}+0.05\le\mathrm{KL}_{\mathrm{unaffected}}$ (route beats both controls)
7. Also: **clean** (unpatched) KLs on the destination run; **$z$_decode** from
   $h_{\mathrm{src}}$ (LM-head $Z$ slice) vs $z_{\mathrm{src}}$

Implementation: [`probe.py`](probe.py).

---

## Baseline

From-scratch **comp-only** ($p_{\mathrm{comp}}=1$), same $D$, 60k steps, same diagnostics.
Framework expects **no** routing signature.

---

## Success / failure / inconclusive

**$\mathrm{C}_{GG}$ emerged** (all):

1. Soft $d_{\mathrm{C}_{GG}}^{\mathrm{rel}}$ smallest among $\{M,G,\mathrm{C}_{GG}\}$ (ID+OOD) with margin $\varepsilon$
2. $\mathrm{KL}_{\mathrm{route}} + 0.05 \le \mathrm{KL}_{\mathrm{transplant}}$ **and** $\mathrm{KL}_{\mathrm{route}} + 0.05 \le \mathrm{KL}_{\mathrm{unaffected}}$
3. $z$_decode $\ge 0.4$
4. Baseline fails (2) or (3)

**Falsified:** Phase-1 gate passed; every Phase-2 ckpt has $\mathrm{KL}_{\mathrm{route}}\ge\min(\mathrm{KL}_{\mathrm{transplant}},\mathrm{KL}_{\mathrm{unaffected}})$ despite low OOD comp CE.

**Inconclusive:** Phase-1 gate fails; or patching ambiguous (route fails to clear the 0.05 margin over both controls) with weak $z$_decode; or the discriminating-trial filter's accept rate is too low to reach 200 triples in a reasonable budget.
