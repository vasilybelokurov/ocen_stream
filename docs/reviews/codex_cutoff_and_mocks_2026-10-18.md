# Codex review: score cutoff speed-up and mock-data design (2026-10-18)

Codex CLI, effort high, read-only. Adopted: absolute-error tests on real sprays (cutoff 7), full-count normalisation, one-pass group jackknife, mock support redraw (measured), fixed-bin weights, several truth seeds/catalogues, independent scores at truth and maxima, angle-10 truth moved inside, per-mock output directories.


**B is not directly compatible with the current scorer:** foreground Gaussian draws can leave `BG_BOUNDS` or `V_BOUNDS`, and the scorer then raises instead of assigning zero background density. Rejecting or clipping those draws would change the proposed distribution. Fix that before calling the mock exact. (`src/ocen_dm/streams/score.py:265–279`)

**A: implement the cutoff, starting at 6, with an error budget and exact fallback.**

1. **Highest priority: correct the error claim.** A sky-only cut is sufficient to bound the *absolute omitted density*, assuming valid covariance matrices and positive bandwidths. The PM Gaussian is bounded by its peak, which can exceed one. Thus an excluded joint contribution is bounded by
   \[
   e^{-c^2/2}\big/[(2\pi h_u h_x)(2\pi\sqrt{\det\Sigma_i})],
   \]
   multiplied by the velocity peak when measured. This is relative to the kernel’s peak, **not relative to the retained density or likelihood**. The implemented normalisations require those factors. (`src/ocen_dm/streams/score.py:306–324`)

   Far particles can supply essentially all foreground density. Background dominance limits the resulting log error only when the omitted numerator is small relative to the background floor; “small foreground” alone is insufficient. Truncating `f_u` also changes the denominator, so score error can have either sign. Bound both errors against their respective mixture terms and enlarge the cutoff or fall back when necessary. (`src/ocen_dm/streams/score.py:327–329`)

2. **Preserve normalisation and empty-bin behaviour.** Divide by full `n_k`, retain fixed `K`, and compute the u marginal independently of x. Fixed-edge bins retain empty-bin weights; legacy bins discard bins below `nmin`; pooled scoring has its own minimum-count rule. (`src/ocen_dm/streams/score.py:282–305`) Sum squared **weighted** contributions for `n_eff`. (`src/ocen_dm/streams/score.py:323–334`)

3. **Vectorise after measuring neighbour occupancy.** Chunked flattened member/particle pairs with `bincount(minlength=...)` avoid `reduceat`’s empty-segment traps. Bound pair-array memory. A further opportunity is accumulating kernel sums by jackknife group once, then reconstructing leave-outs with adjusted bin counts: the present jackknife recomputes the complete score ten times. (`src/ocen_dm/streams/score_bias.py:82–91`)

4. **Test absolute log errors, not relative totals.** Keep `cutoff=None` bit-identical. For finite cutoffs, suggested initial gates are maximum per-star error `<10⁻⁵` and total error `<0.1`, including the **corrected** total; tighten if scientifically relevant differences are smaller. Jackknife can amplify bounded constituent errors by up to 19 for `G=10`. (`src/ocen_dm/streams/score_bias.py:75–91`)

   Cover pooled/fixed/legacy bins, sparse and empty bins, cutoff boundaries, zero neighbours, correlated PM errors, measured/missing velocities, components and `n_eff`. Existing score tests exercise binning and `n_eff`, but their fixtures lack measured velocities and correlated PM covariance. (`tests/test_score_conditional.py:9–14`, `tests/test_score_conditional.py:97–137`) Test density errors separately where background dominance makes log errors insensitive.

**B: the sampling algebra is correct for the raw finite-spray conditional KDE, after resolving support and matching binning.**

1. **Match the actual training target.** Training uses fixed edges `0:5:200`; the old generator dynamically drops bins with fewer than 20 particles. Do not reuse its `prep()` unchanged. Use `1/(K n_k)` for occupied fixed bins, retain empty-bin weights in `f_u`, and normalise the particle-selection probabilities only when sampling the foreground. (`bin/streams/training_set.py:58`, `bin/streams/d2_mock_recovery.py:24–29`, `src/ocen_dm/streams/score.py:288–295`)

   Your background probability and Gaussian covariance then reproduce the raw conditional formula. They do **not** generate exactly from an infinite-particle model or a normalised jackknife-corrected likelihood; jackknife adjusts log scores. (`src/ocen_dm/streams/score.py:308–328`, `src/ocen_dm/streams/score_bias.py:91`)

2. **Keep real u, errors and velocity availability.** That is appropriate for conditional recovery, assuming selection and missingness can be treated as fixed. It does not validate those assumptions. Freeze the real background bounds rather than estimating them from each mock; the current bounds are derived from real observations. (`bin/streams/grid4_common.py:15–18`, `bin/streams/grid4_common.py:86–89`) Specify truth PMRA and PMDEC too: the emulator is nine-dimensional. (`bin/streams/emulator_design.py:29–39`)

3. **Separate recovery from emulator calibration.** Independent truth seeds make this a useful internal closure test, but shared simulator physics cannot establish physical correctness. Use several truth-spray seeds and mock catalogues. Start at matching particle count, then increase truth particles until recovery conclusions stabilise.

   Reject “within 1000 of peak” and training-point rank as primary acceptance criteria. Calibrate likelihood-ratio regions across repeated mocks; independently score truth and proposed maxima to distinguish interpolation failure from sampling variation. Separately assess held-out score residuals using predictive **plus simulation** variance, as the current validation does. (`bin/streams/emulator_design.py:107–140`) The angle-10 truth lies on the design boundary, so symmetric parameter-error criteria are inappropriate. (`bin/streams/emulator_design.py:36`)

4. **Protect results.** Write mock scores to separate directories with catalogue hash, truth seed, cutoff and likelihood settings. The emulator loader consumes `total_jk` and `var_jk`; it checks simulation identity and within-batch manifest consistency, but does not explicitly enforce a common mock-catalogue identity across batches. (`bin/streams/emulator_design.py:43–81`)

**1. Verified versus assumed.** Verified by reading all five requested files and the cited training/tests: formulas, support rejection, binning, correction and loading behaviour. Mathematical bounds above are deductions assuming finite inputs, positive bandwidths and valid covariances. I did not verify the stated timings, spray counts or realised recovery performance, and ran no tests.

**2. Strongest argument against this recommendation.** Six bandwidths may already make errors negligible on every relevant spray; fallback machinery could add complexity without measurable benefit. Conversely, a few successful closure mocks could conceal poor recovery elsewhere.

**3. Needed for confidence.** Exact-versus-cutoff benchmarks on representative and deliberately mismatched sprays; corrected-score error and memory measurements; generator normalisation/support tests; independent scores around recovered maxima; repeated-mock coverage; and particle-count/seed convergence. Existing jackknife and emulator tests establish toy behaviour, not this end-to-end recovery. (`tests/test_score_bias.py:30–44`, `tests/test_emulator.py:8–41`)