# Codex review: stream paper plan (2026-10-16)

Question: critical review of the proposed MNRAS storyline (data, model, no fit without a bar, bar constraints) and figure plan. Ordinary Codex, effort high, read-only.

The repository contradicts two premises in the proposed storyline. **The angle refinements are not converged profile fits**, and **34–35 km/s/kpc is not a general constraint once bar size and angle vary**: the saved sweep’s best sampled 28-degree model uses 36, and its best sampled 36-degree model uses 37. [docs/STREAM_PM_REFINEMENT_2026-10-13.md:107](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L107), [results/plot_data/bar3d_sweep/w0.jsonl:50](../../results/plot_data/bar3d_sweep/w0.jsonl#L50), [results/plot_data/bar3d_sweep/w4.jsonl:102](../../results/plot_data/bar3d_sweep/w4.jsonl#L102)

**I would reject “impossible without a bar” as the paper’s claim.** The defensible paper is a comparison: tested axisymmetric models miss the observed bend and PM gradients; barred models offer an explanation worth testing, but parameter inference remains conditional on unresolved model and selection errors. The existing axisymmetric failures are documented, while the later barred candidates retain PM residuals and external-measurement tension. [docs/STREAM_KNEE_INVESTIGATION.md:9](../../docs/STREAM_KNEE_INVESTIGATION.md#L9), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:119](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L119), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:128](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L128)

**Claims and reasoning that need correction**

- **“Independent detection” needs qualification.** The Gaia check avoids orbit templates, but derives its query footprint, CMD locus, and on-stream centre from STREAMFINDER members. It also includes those members in the Gaia sample. Its reported significance is the maximum over a searched PM map, without a trial correction in this script. Call it an alternative selection check, and repeat it excluding catalogue members with control-field calibration. [bin/streams/m5_independent_selection.py:38](../../bin/streams/m5_independent_selection.py#L38), [bin/streams/m5_independent_selection.py:51](../../bin/streams/m5_independent_selection.py#L51), [bin/streams/m5_independent_selection.py:62](../../bin/streams/m5_independent_selection.py#L62)

- **The mocks establish limited recovery, not absence of angle bias.** They use two truths from the same spray family, add the scorer’s kernel widths, omit contaminants, fix candidate distance, and search only four angles. They do not calibrate false bar detections, continuous confidence intervals, or recovery under model misspecification. [bin/streams/d2_mock_recovery.py:3](../../bin/streams/d2_mock_recovery.py#L3), [bin/streams/d2_mock_recovery.py:19](../../bin/streams/d2_mock_recovery.py#L19), [bin/streams/d2_mock_recovery.py:32](../../bin/streams/d2_mock_recovery.py#L32)

- **“Density-free” overstates the implemented score.** The foreground conditional cancels its along-stream marginal, but foreground/background odds still depend on that marginal. Also, 5% is the joint mixture coefficient, not a fixed conditional contamination fraction. The density audit preserves the low-angle ranking for the tested candidates; it does not validate the background distribution. [src/ocen_dm/streams/score.py:239](../../src/ocen_dm/streams/score.py#L239), [src/ocen_dm/streams/score.py:278](../../src/ocen_dm/streams/score.py#L278), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:152](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L152)

- **Keep score definitions separate.** The older score adds sky, PM, and RV log terms; the adopted score evaluates their joint conditional mixture. The quoted distance-sensitivity experiment uses the older robust chi-KDE. Neither its numerical changes nor an older sky/PM decomposition should be presented as measurements under the final objective. Separate marginal diagnostic scores also do not add up to the joint score. [src/ocen_dm/streams/score.py:67](../../src/ocen_dm/streams/score.py#L67), [src/ocen_dm/streams/score.py:276](../../src/ocen_dm/streams/score.py#L276), [bin/streams/ocen_state_sensitivity.py:36](../../bin/streams/ocen_state_sensitivity.py#L36), [bin/streams/obs_split.py:27](../../bin/streams/obs_split.py#L27)

**The minimum fair axisymmetric test**

Fit a nested Hunter comparison with amplitude \(A=0\) versus \(A>0\). Jointly optimize cluster distance and both PMs, halo flattening, disc normalization, and host normalization under identical external constraints, release assumptions, sample, bandwidths, and numerical budgets. Start both fits from multiple locations and inspect boundaries. The existing 4D grid fixes PMs, and the broader size sweep fixes distance, amplitude, and catalogue PMs; neither substitutes for this comparison. [bin/streams/grid4.py:20](../../bin/streams/grid4.py#L20), [bin/streams/bar3d_sweep.py:29](../../bin/streams/bar3d_sweep.py#L29), [bin/streams/bar3d_sweep.py:41](../../bin/streams/bar3d_sweep.py#L41)

There is an implementation trap: the direct `hunter24_axi` branch returns before applying `axi_variant` or `mass_scale`. Use the common amplitude-zero construction, or explicitly verify that the null actually changes when its host parameters change. Add an amplitude-zero force-equivalence test. [src/ocen_dm/streams/restricted.py:80](../../src/ocen_dm/streams/restricted.py#L80), [src/ocen_dm/streams/restricted.py:99](../../src/ocen_dm/streams/restricted.py#L99), [src/ocen_dm/streams/restricted.py:110](../../src/ocen_dm/streams/restricted.py#L110)

Then require:

1. Stable score differences and sky-conditioned residuals across paired seeds and increasing resolution.
2. Axisymmetric-truth mocks processed through **both complete fits**, measuring false bar preference.
3. Barred-truth mocks measuring power and interval coverage, including contaminants and plausible release-model departures.
4. Prediction of withheld contiguous stream segments, with the split declared before fitting.

Even successful tests would reject the specified axisymmetric family under stated assumptions, not prove impossibility for every unbarred potential.

**Bar constraints, angle tension, and remaining misfit**

Report current parameter values as **preferred sampled solutions**, without confidence errors. Use paired seeds during comparisons, then independent seeds to validate selected candidates. Demonstrate stability of differences, not merely equal release counts: changing resolution also changes release-time spacing and which chi bins pass the particle threshold. [bin/streams/grid4_common.py:55](../../bin/streams/grid4_common.py#L55), [src/ocen_dm/streams/score.py:252](../../src/ocen_dm/streams/score.py#L252), [bin/streams/seed_test.py:11](../../bin/streams/seed_test.py#L11)

The block bootstrap is a sensitivity diagnostic. It resamples unequal-population spatial blocks and computes its score contrast between two models selected on the original data. Its variance inflation is not a universal \(N_{\rm eff}\), and its sign frequency is not a calibrated model-selection probability. Do not turn it into a likelihood temperature or combine it with seed scatter without a crossed calibration experiment. [bin/streams/bootstrap_grid4.py:17](../../bin/streams/bootstrap_grid4.py#L17), [bin/streams/bootstrap_grid4.py:30](../../bin/streams/bootstrap_grid4.py#L30)

Present the low-angle preference alongside nuisance pulls and residuals. The recorded 28-degree candidate uses \(d=5.801\) kpc; **the 16-degree candidate also shifts to \(5.616\) kpc**, about 3.7 adopted distance errors above 5.43 kpc. Thus the external tension is not confined to the literature-angle solution. Neither candidate is a converged optimum. [results/plot_data/bar_profile_refinement_20261013/fit_an28.json:540](../../results/plot_data/bar_profile_refinement_20261013/fit_an28.json#L540), [results/plot_data/bar_profile_refinement_20261013/fit_an16.json:387](../../results/plot_data/bar_profile_refinement_20261013/fit_an16.json#L387), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:21](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L21), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:107](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L107)

Show unpenalized scores, external-measurement penalties, and prior-conditioned results separately. The older free-angle fitter uses a 0.15-kpc distance prior and an optional angle prior, whereas the later refinement defaults to 0.05 kpc. Those are different inference setups. [bin/streams/fit_bar_ocen.py:28](../../bin/streams/fit_bar_ocen.py#L28), [bin/streams/fit_bar_ocen.py:35](../../bin/streams/fit_bar_ocen.py#L35), [bin/streams/refine_bar_profiles.py:169](../../bin/streams/refine_bar_profiles.py#L169)

Do not claim that slowdown, another bar, or another escape law is excluded: several comparisons freeze nuisance parameters, and the escape-law note explicitly limits its conclusion to that pilot. Likewise, the RV encounter statistic is a diagnostic, not an ancestry probability. [bin/streams/phase2_variants.py:20](../../bin/streams/phase2_variants.py#L20), [docs/STREAM_PM_REFINEMENT_2026-10-13.md:211](../../docs/STREAM_PM_REFINEMENT_2026-10-13.md#L211), [bin/streams/backtrack_barred_release.py:4](../../bin/streams/backtrack_barred_release.py#L4)

**Correctness and reproducibility blockers**

- The background uses constant density everywhere without implementing finite support. Specify a proper normalized background and treatment of selection before interpreting likelihood ratios inferentially. [src/ocen_dm/streams/score.py:249](../../src/ocen_dm/streams/score.py#L249), [src/ocen_dm/streams/score.py:278](../../src/ocen_dm/streams/score.py#L278)
- The documented `fit_bar_ocen.py --conditional` invocation is rejected by its parser: the flag is inspected in `sys.argv` but never registered before `parse_args()`. [bin/streams/fit_bar_ocen.py:34](../../bin/streams/fit_bar_ocen.py#L34), [bin/streams/fit_bar_ocen.py:40](../../bin/streams/fit_bar_ocen.py#L40)
- Cache existence is sometimes treated as configuration identity. For example, grid2’s filename omits release count and release window despite exposing both as arguments. Require validated configuration/source/input fingerprints before reusing publication results; this is a contamination risk, not evidence that contamination occurred. [bin/streams/spray_bar_grid2.py:35](../../bin/streams/spray_bar_grid2.py#L35), [bin/streams/spray_bar_grid2.py:45](../../bin/streams/spray_bar_grid2.py#L45)
- The inspected conditional-score tests exercise no measured RVs or nonzero PM correlations. Add analytic correlated-covariance, RV marginalization, background-normalization, invalid-input, and CLI regression tests. [tests/test_score_conditional.py:9](../../tests/test_score_conditional.py#L9), [tests/test_score_conditional.py:24](../../tests/test_score_conditional.py#L24)

**I recommend six figures**, with validation details and chi ordering in an appendix.

| Figure | Required content |
|---|---|
| 1 | Data: sky and PMs; RV locations; alternative-selection ridge with its limits clearly marked. |
| 2 | Central comparison: fairly refitted axisymmetric, literature-angle barred, and freely fitted barred models against the same data. |
| 3 | Residuals: sky-conditioned PM predictions, sky residuals, RVs, and regional contributions; include model spread and numerical variation. |
| 4 | Profile maps: angle–size–speed degeneracy, boundaries, external pulls, and raw versus penalized scores. |
| 5 | Calibration: axisymmetric-null recovery, barred recovery, and paired-seed/resolution stability. |
| 6 | Selected robustness results: physical residual changes plus score changes, distinguishing frozen pilots from refitted comparisons. |

Use common plotting scales and enclosed-probability bands. Existing profile plots normalize every column to its peak, which can visually promote sparse branches; the conditional plotting script also omits measurement-error convolution. State that distinction in captions or plot observation-convolved predictions. [bin/streams/plot_profile_stage1.py:41](../../bin/streams/plot_profile_stage1.py#L41), [bin/streams/plot_conditional_model.py:6](../../bin/streams/plot_conditional_model.py#L6)

Use the MNRAS sequence Introduction → Data → Methods → Model comparison → Conditional bar constraints → Discussion → Conclusions. Put residual misfit beside the principal comparison, rather than postponing it to a robustness section.

1. **Verified versus assumed.** I read the cited implementation, tests, notes, and selected saved results; I checked saved sweep maxima and refinement candidates. I did not reproduce integrations, validate raw catalogue membership, or independently verify literature values and novelty claims. The attempted score-test execution stopped because the available Python lacks NumPy; no tests passed in this review. Nothing was modified.

2. **Strongest argument against my answer.** I may be demanding too much calibration for a paper whose main contribution is a qualitative dynamical explanation. The documented axisymmetric failures span several model choices, so a focused comparison paper could be justified before precise inference is ready. That supports a narrower paper; it still does not support “impossible” or calibrated bar measurements. [docs/STREAM_KNEE_INVESTIGATION.md:9](../../docs/STREAM_KNEE_INVESTIGATION.md#L9)

3. **What I would need to check to be sure.** The matched null/bar fits and mock calibration above; replay of publication results from frozen inputs; membership-excluded control-field detection; tracer validation at the final fitted states; and physical admissibility of the scaled host, including density positivity and bar-force strength. The amplitude parameter multiplies a full-minus-axisymmetric potential term, so it should not be interpreted directly as bar mass. [src/ocen_dm/streams/restricted.py:95](../../src/ocen_dm/streams/restricted.py#L95), [src/ocen_dm/streams/restricted.py:106](../../src/ocen_dm/streams/restricted.py#L106)