# Codex review: bias and noise of the conditional KDE stream likelihood (2026-10-17)

Question: see JOURNAL 2026-10-17 (n_eff study). Codex CLI, effort high, read-only.

**Use replicated randomized integration first, a corrected delta method second, and a surrogate only after those pass calibration. Do not trust proposal (a) as written.**

The proposed \(+\tfrac12\operatorname{Var}(\log L_i)\) is generally **not** the bias correction for the implemented likelihood ratio. Also, its particle-variance formula assumes an i.i.d. sample that this release design does not provide: epochs are deterministic, offsets are random, and particles are subsequently selected by trajectory-dependent \(\chi>0\). [`grid4_common.py:55–57`](bin/streams/grid4_common.py:55), [`spray.py:58–61`](src/ocen_dm/streams/spray.py:58), [`score.py:282–286`](src/ocen_dm/streams/score.py:282).

My priority order is **(b), corrected (a), (c), (e), (d)**.

**(a): correct the estimator before estimating its uncertainty.**

For unbiased estimators \(\widehat A_i,\widehat B_i\) of the mixture numerator and denominator, write
\[
\widehat\ell_i=\log\widehat A_i-\log\widehat B_i.
\]
The second-order correction is
\[
\widehat\ell_{i,\mathrm{bc}}
=\widehat\ell_i+
\frac12\left[
\frac{\widehat{\operatorname{Var}}(\widehat A_i)}{\widehat A_i^2}
-\frac{\widehat{\operatorname{Var}}(\widehat B_i)}{\widehat B_i^2}
\right].
\]
In contrast,
\[
\operatorname{Var}(\widehat\ell_i)\approx
\frac{V_{A_i}}{A_i^2}+\frac{V_{B_i}}{B_i^2}
-\frac{2C_{A_iB_i}}{A_iB_i}.
\]
Covariance belongs in the **noise estimate**; it cancels from this leading **log-ratio bias correction**. These expressions differ for the ratio actually calculated. [`score.py:327–329`](src/ocen_dm/streams/score.py:327).

There is another complication: both densities divide by the random number of accepted particles. Simply treating their sample means as unbiased skips this selection-normalization bias. [`score.py:282–286`](src/ocen_dm/streams/score.py:282), [`score.py:313–324`](src/ocen_dm/streams/score.py:313).

A cleaner calculation preserves contributions from **every eligible release epoch**, including rejected ones. Let \(I_j\) indicate acceptance, \(G_{ij}\) be its joint kernel and \(H_{ij}\) its along-stream kernel. Define
\[
X_{ij}=I_j[(1-\epsilon)G_{ij}+\epsilon g_ug_{w,i}],\qquad
Y_{ij}=I_j[(1-\epsilon)H_{ij}+\epsilon g_u].
\]
Then \(\sum_jX_{ij}/\sum_jY_{ij}\) equals the existing pooled likelihood whenever its foreground bin exists. Apply bias and covariance calculations to these raw sums. This retains the fluctuating acceptance normalization explicitly.

For deterministic epochs, variation across particles includes variation in their epoch-dependent expectations. Consequently, ordinary sample variance divided by particle count need not estimate seed variance. Particle leave-one-out jackknifing inherits that problem. Contiguous split halves also change the release-age distribution.

I would start with **8–16 independent complete replicates**, retaining raw sums, and estimate the correction from replicate covariance. Use a matched, interleaved delete-group jackknife only as a cheaper candidate to validate against those replicates. A split estimator
\[
2\ell_{\mathrm{full}}-\tfrac12(\ell_{\mathrm{half1}}+\ell_{\mathrm{half2}})
\]
can cancel leading \(1/N\) bias under the appropriate sampling design; it does not guarantee cancellation in sparse tails.

**At \(n_{\rm eff}=3\)–10, neither delta correction nor jackknifing deserves an accuracy claim without replication.** Effective count alone does not characterize skewness, denominator variation or background transitions. The implemented diagnostic measures only foreground joint-kernel concentration. [`score.py:325–334`](src/ocen_dm/streams/score.py:325).

For total noise, preserve cross-star covariance:
\[
\operatorname{Var}(\widehat\ell)\approx
\operatorname{Var}\!\left[
\sum_i\left(\frac{\delta A_i}{A_i}-\frac{\delta B_i}{B_i}\right)
\right].
\]
Summing per-star variances is insufficient. Estimate this total directly across replicates, including variability introduced by the correction itself.

**(b): highest priority, but benchmark equal computational cost.**

Use scrambled Sobol points over **release time plus the four Gaussian escape variables**, with independent scrambles for error estimation. Alternatively, retain the deterministic time grid and first test stratified offsets. The four underlying normal draws are explicit in the release code. [`spray.py:58–61`](src/ocen_dm/streams/spray.py:58).

Use power-of-two Sobol nets and estimate variance between independent scrambles, not between their dependent points. Randomized integration can give unbiased raw integral estimates; applying a logarithm or ratio still introduces bias. [SciPy’s Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).

Antithetic **Gaussian deviations about their prescribed means** are worth testing; opposite launch arms are not substitutes for antithetic samples of the trailing-arm distribution. The release code assigns those arms different signed offsets. [`spray.py:55–65`](src/ocen_dm/streams/spray.py:55).

Use common random numbers for local comparisons. Measure their actual reduction in **difference variance**; they do not reduce marginal bias. Preserve random-variable identities when changing particle count—the current successive array-sized draws do not supply a nested realization of every offset component. [`spray.py:58–61`](src/ocen_dm/streams/spray.py:58).

**(c): useful after identifying influential release regions; “every member above 30” is the wrong objective.**

For proposal \(q(t)\) and target release law \(p(t)\),
\[
\widehat f_i=N^{-1}\sum_j[p(t_j)/q(t_j)]G_{ij}
\]
is unbiased for the corresponding *unnormalized integral*. Self-normalization, selection conditioning, ratios and logarithms reintroduce bias. Carry weights through the raw \(X,Y\) sums above.

Parameter-dependent proposals are valid if their densities and weights are correct. They need not be parameter-independent. For operational simplicity, learn a frozen mixture from a broad pilot design, including a substantial uniform component—say 20%—to preserve support.

Allocate effort using contribution to **total-score variance per unit cost**, rather than minimum Kish count. Some members should remain background-dominated; forcing their foreground effective counts upward can waste most of the computation. The existing mixture explicitly permits that regime. [`score.py:327–329`](src/ocen_dm/streams/score.py:327).

**(d): hold bandwidth fixed during numerical convergence tests.**

Define the target as the infinite-particle likelihood at a specified bandwidth. Otherwise numerical convergence and changes to the statistical model become inseparable.

Measurement-error convolution is principled, but it does not determine additional smoothing in release time or sky position. The implementation already adds Gaia covariance and radial-velocity error variance to its kernel widths; setting those widths directly equal to measurement errors would add another smoothing contribution. [`score.py:308–321`](src/ocen_dm/streams/score.py:308).

Choose additional widths through predictive checks and physical-model assumptions, then report bandwidth sensitivity. Do not enlarge them solely to obtain affordable numerical noise.

**(e): appropriate for exploration, with two restrictions.**

First, a GP nugget cannot absorb systematic estimator bias. Train on corrected scores only after validating that correction across the design domain.

Second, common random numbers create correlated simulation errors across design points. A diagonal heteroscedastic nugget then misrepresents their covariance. Either retain replicate identities and estimate that covariance, or use independent training randomizations and reserve common random numbers for local contrasts.

For tempering,
\[
\ell_\beta=\beta\ell,\quad
b_\beta=\beta b,\quad
V_\beta=\beta^2V.
\]
For segment powers, use \(\beta^\mathsf T\Sigma\beta\), retaining cross-segment covariance. Calibrate powers against observational/model misspecification and recovery coverage, separately from simulation noise. Holmes–Walker provides an information-matching rationale, not an automatic remedy for correlated members or simulation error. [Holmes & Walker](https://arxiv.org/html/1701.08515).

Require direct replicated evaluations near posterior modes and between competing modes. A raw bias tolerance of 10 may suffice for exploration; at \(\beta=1\), a parameter-dependent error of 10 can distort posterior odds by \(e^{10}\).

**There is no simple reformulation that makes the current target unbiased.**

Unbiased joint-density estimates do not make their ratio unbiased. Even unbiased \(L_i\) estimates would not generally give an unbiased product when they share spray particles. Exponentiating an unbiased log-score is also biased.

Exact pseudo-marginal inference requires a nonnegative unbiased estimate of the **whole likelihood**. Correlated pseudo-marginal methods can tolerate large marginal log noise when the proposal-ratio noise is controlled, so “sd must equal one” is too categorical; the present estimator nevertheless fails the unbiased-likelihood prerequisite. [Deligiannidis, Doucet & Pitt](https://www.stats.ox.ac.uk/~doucet/DeligiannidisDoucetPitt_correlatedpseudomarginal.pdf).

Russian-roulette/Poisson constructions require separate proofs of unbiasedness, sign, finite variance and cost. They are a research project here. Signed-estimator complications are documented by [Lyne et al.](https://arxiv.org/abs/1306.4032). Synthetic likelihood or ABC changes the inferential target and may discard useful distributional information. An analytically evaluable conditional mixture could remove repeated KDE integration, but introduces approximation error requiring its own validation.

**1. Verified versus assumed**

Verified by reading the four requested files fully and inspecting relevant tests:

- The study explicitly uses pooled scoring. The wrapper itself does **not** default to pooled scoring; a new exploration must request it explicitly. [`neff_study.py:43–44`](bin/streams/neff_study.py:43), [`grid4_common.py:80–86`](bin/streams/grid4_common.py:80), [`score.py:233`](src/ocen_dm/streams/score.py:233).
- Recalculation from the stored 40 runs reproduces the reported score means. At 256k, individual score SDs are **41.5 for A and 22.9 for B**; **56.3** describes their paired difference. The difference’s standard errors across the four resolutions are **42.4, 35.0, 25.2, 17.6**, rather than approximately 20 throughout. [`neff_study_20261017.json:1`](results/plot_data/neff_study_20261017.json:1).
- The convergence test checks one pair of realizations at two sizes; the effective-count test checks identical contributors and score invariance. Neither establishes a bias bound or calibrated noise estimate. [`test_score_conditional.py:113–137`](tests/test_score_conditional.py:113).
- Resumption identifies completed runs only by model label, size and seed, then saves current model parameters alongside retained runs. Changed models or scoring code can silently mix incompatible results. Add configuration/data/code fingerprints before expanding the campaign. [`neff_study.py:51–60`](bin/streams/neff_study.py:51).

I assume the intended target is the continuous-release, infinite-particle, fixed-bandwidth conditional score; that the catalogue’s member selection permits this conditional likelihood; and that simulator draws are computational integration variables rather than a physical finite population. Those assumptions need scientific validation.

I could not execute the tests: the available Python lacks NumPy and pytest. Nothing was modified.

**2. Strongest argument against my recommendation**

Randomized integration may deliver little benefit where trajectory dynamics and the hard \(\chi>0\) selection produce sharp boundaries. Those boundaries are present in this calculation. [`spray.py:156–163`](src/ocen_dm/streams/spray.py:156), [`score.py:282`](src/ocen_dm/streams/score.py:282).

Splitting a fixed budget among replicates also leaves each replicate sparser. Higher-order bias and correction noise could outweigh the benefit. A larger conventional spray with targeted importance sampling might therefore be cheaper and more reliable. Decide using measured wall-time efficiency, not the nominal convergence rate.

**3. Checks needed before trusting it**

These are proposed acceptance criteria:

1. **Analytic fixtures:** Gaussian and Gaussian-mixture streams with known convolved densities; include correlated PM errors, missing/measured velocities, background transitions and random acceptance. Across repeated samples, require  
   \[
   |\widehat b|+1.96\,\mathrm{SE}(\widehat b)<10
   \]
   for total corrected score, with reference uncertainty included.

2. **Real-model references:** audit at least 20 points spanning both candidate modes, sparse regions and parameter extremes. Build independently randomized, substantially higher-resolution references. Require reference SE below 2, a 95% bound on further-resolution drift below 3, and the same bias criterion above. Increase repetitions until the bound is resolved; five seeds are insufficient.

3. **Noise calibration:** use approximately 200 independent evaluations at selected audit points. Require empirical variance divided by mean predicted variance in **0.8–1.25**, standardized residual mean within **0.2** of zero, and nominal 95% interval coverage in **92–98%**. Check total scores, segment scores and paired parameter differences separately.

4. **Equal-cost comparisons:** compare conventional sampling, stratification, scrambled Sobol, antithetics and importance sampling using wall time and **MSE against the reference**, including correction noise. Promote a method only if its MSE reduction is statistically resolved.

5. **Posterior validation:** on independent simulator evaluations near posterior support, require weighted log-score **contrast** errors below **0.5**, including uncertainty. Test posterior coverage on mocks with realistic selection and model discrepancy; 90% intervals should cover in approximately **86–94%** of at least 200 recovery experiments. A finite audit supports accuracy over the tested domain, not a uniform guarantee over all nine dimensions.