# Codex review: score-version consistency, provenance and the pooled default (2026-10-17)

Question: the four audit issues and proposed fixes (docs/kb/models/stream_likelihood_versions.md). Codex CLI, effort high, read-only.

One audit premise is incorrect: `phase2_variants.py` recomputes the requested variants rather than skipping their scores. It still permits mixing: it loads the existing dictionary and replaces only the requested keys. [phase2_variants.py:38–65](bin/streams/phase2_variants.py:38)

**My priority order is (3), (4), (2), then (1). Adopt S4 for paper inference only after validating and refitting under S4.**

1. **(3): enforce provenance before resumption, cache reuse, and comparison.** Saving `settings` alone is insufficient. Paper display resumption keys only on model name and seed; convergence resumption keys only on model label, particle count, and seed. Changing the parameters behind those names can silently retain stale results even without a score-version change. [paper_models.py:109–123](bin/streams/paper/paper_models.py:109), [particle_convergence.py:35–39,79–89](bin/streams/particle_convergence.py:79)

2. **(4): remove import-time writes immediately.** A `main()` guard fixes the demonstrated overwrite path. Keep reusable model definitions in an inert module and import `single_scores` directly from its defining module. Currently the knee script imports D1, whose top-level body writes both the D1 JSON and plot. [knee_spread_and_frame.py:16](bin/streams/knee_spread_and_frame.py:16), [d1_segments.py:21–45](bin/streams/d1_segments.py:21)

3. **(2): share the estimator, while distinguishing marginals from additive contributions.** The present split implementation independently fixes cuts, bin weights, bandwidths, and background widths. That duplication must disappear. [obs_split.py:11–29](bin/streams/obs_split.py:11)

4. **(1): centralize explicit configurations; avoid making every scientific script follow a changing default.** The current wrapper fixes the real catalogue and chord frame. D2 scores simulated observables; the frame checks use alternative coordinates and subsets. Passing those callers through `score(m)` unchanged would change the experiment. Extract a configurable scoring layer accepting data, frame, and resolved score configuration; keep the current wrapper as its convenience interface. [grid4_common.py:15–18,83–94](bin/streams/grid4_common.py:83), [d2_mock_recovery.py:55–58](bin/streams/d2_mock_recovery.py:55), [rescore_conditional.py:31–38](bin/streams/rescore_conditional.py:31)

**The correct per-observable definition**

Let \(F_A(u,y_A)\) be the foreground joint density marginalized to observable group \(A\), and \(G_A(y_A)\) the corresponding background marginal. Then

\[
p(y_A\mid u)=
\frac{(1-\epsilon)F_A(u,y_A)+\epsilon g_u(u)G_A(y_A)}
     {(1-\epsilon)f_u(u)+\epsilon g_u(u)}.
\]

Use identical particles, weights, cuts, and \(f_u\) for every marginal. The conditional background probability is

\[
\beta(u)=\frac{\epsilon g_u(u)}
{(1-\epsilon)f_u(u)+\epsilon g_u(u)},
\]

rather than a fresh constant 0.05 for each observable. This follows directly from the implemented joint numerator and denominator. [score.py:316–331](src/ocen_dm/streams/score.py:316)

These are proper marginal densities under normalized component distributions. **Their log scores generally do not sum to the joint log score.** Sharing a debris particle and foreground/background membership induces dependence even when each particle’s kernels factorize.

For exact addition, expose a separately named chain decomposition:

\[
\log p(x,\boldsymbol\mu,v\mid u)=
\log p(x\mid u)+
\log p(\boldsymbol\mu\mid u,x)+
\log p(v\mid u,x,\boldsymbol\mu).
\]

Compute successive ratios of mixture marginals; this automatically updates particle and background probabilities after each conditioning step. The attribution depends on ordering. For missing velocity, omit the velocity factor, matching the existing joint implementation. [score.py:321–328](src/ocen_dm/streams/score.py:321)

The plotting proposal also needs qualification. `return_components` currently returns only the evaluated joint density and \(u\) marginal. The plots require arbitrary evaluation grids, individual PM marginals, and intrinsic distributions without per-star errors. The original-coordinate plot additionally reweights particles and smooths in different coordinates; it is an approximate visualization, not simply another evaluation of the joint likelihood. Share particle selection, weights, and marginal evaluators, while retaining that distinction. [score.py:330–331](src/ocen_dm/streams/score.py:330), [plot_conditional_model.py:35–41](bin/streams/plot_conditional_model.py:35), [plot_conditional_orig.py:33–51](bin/streams/plot_conditional_orig.py:33)

**Minimal robust provenance**

Use immutable manifests with content-derived identifiers:

- **Score manifest:** schema version; all effective defaults and overrides; weighting; actual bounds and window; bandwidths; cuts; epsilon; hashes of scoring/projection source and relevant dependency versions.
- **Data manifest:** hashes of ordered evaluated arrays, errors/covariances, missing-value masks, member identities, selection, units, and frame inputs.
- **Spray manifest:** complete resolved physical and numerical parameters, seed, release schedule/count, reference state, potential/profile input hashes, generator code/environment, and output-array hash.

Store manifests beside numerical results under `results/plot_data/` or with spray artifacts under `results/streams/`. Each score record should reference its score/data/spray identifiers. Sidecars for `.npy` and `.npz` must include the artifact hash.

The existing `settings` records only supplied options and a short commit with an optional `-dirty` suffix. Different dirty trees can therefore have identical provenance; omitted defaults are implicit; changes outside the checked paths are missed. Use actual source-content hashes as identity and retain Git information as descriptive metadata. [grid4_common.py:98–123](bin/streams/grid4_common.py:98)

Use separate keys:

- **Comparison identity:** score definition, data, and frame; validate controlled particle resolution and sampling conventions.
- **Job identity:** comparison identity plus complete model/spray configuration and seed.

Validate existing records **before** building `done`, returning a cached result, or writing anything. Missing provenance means unknown provenance and should fail closed. A new score filename does not validate an old spray cache.

I would omit `--force-mix` initially. If retained, require per-record identities and explicit partition selection by readers; it must never authorize unlabeled mixed comparisons. Use atomic replacement and a single writer or locking for shared outputs. The paper JSON already uses atomic replacement, but its read/modify/write sequence has no synchronization. [paper_models.py:43–48,108–123](bin/streams/paper/paper_models.py:43)

Do not treat the historical index as proof. It assigns versions using modification times and inferred commits; its embedded-settings inspection retains version tags without checking complete configuration agreement. Existing escape pilots already store source hashes, but their resume path does not validate those hashes. [score_provenance.py:71–90,101–125](bin/streams/score_provenance.py:101), [compare_escape_recipes.py:77–92](bin/streams/compare_escape_recipes.py:77)

**What must be regenerated for an S4 paper**

- **Refit the null and barred candidates**, then repeat held-out scoring and model selection. Rescoring S2-selected parameters establishes their S4 scores, not their S4 optimality. The paper currently obtains barred parameters from old refinement files and selects the null from saved fit scores. [paper_models.py:55–57,101–110](bin/streams/paper/paper_models.py:101)
- **Regenerate Figure 2’s likelihood annotations, Figure 3’s cumulative likelihood panel, Figure 5’s sweep rankings, and every likelihood difference in Figure 6**, including each matched baseline. These directly consume saved scores or per-star arrays. [fig2_models.py:36–46](bin/streams/paper/fig2_models.py:36), [fig3_residuals.py:55–63](bin/streams/paper/fig3_residuals.py:55), [fig5_bar_constraints.py:30–40](bin/streams/paper/fig5_bar_constraints.py:30), [fig6_robustness.py:28–61](bin/streams/paper/fig6_robustness.py:28)
- **Repeat mock recovery, seed/particle convergence, and uncertainty diagnostics supporting S4 claims.** Update both mock generation and scoring: D2 currently generates foreground-only mocks using legacy bin weights. Test full-mixture recovery separately. [d2_mock_recovery.py:24–41,55–58](bin/streams/d2_mock_recovery.py:24)
- **Reuse verified sprays when their generating inputs remain appropriate.** Data-only Figure 1 needs no score-driven regeneration. Fixed-parameter morphology and residuals may remain; regenerate them when refitting changes their parameters. Figure 4 uses variants at one shared fitted cluster state, so it does not establish independently optimized alternatives. [fig1_data.py:40–73](bin/streams/paper/fig1_data.py:40), [paper_models.py:127–139](bin/streams/paper/paper_models.py:127)

Historical experiments can remain documented S2. They cannot substantiate S4-specific calibration or optimality claims. The manuscript’s current method explicitly describes equal-weight chi bins and quotes corresponding recovery and scatter results; those passages require revision if adopting S4. [main.tex:164–190](paper/stream_bar/main.tex:164)

**Tests required**

For **no silent mixing**:

- Resume and append integration tests: independently change weighting, background, an implicit default, data order/value/error/mask, frame, model parameters, seed, resolution, and dirty source contents. Require rejection with output bytes unchanged.
- Reject missing provenance, stale spray caches, and mixed worker files. Test readers as well as writers.
- Demonstrate identical jobs resume without recomputation, explicit/default-equivalent settings share identity, and interrupted writes cannot create completed jobs.
- Test that importing D1 performs no computation or writes.

For **marginals matching the joint definition**:

- An independent tiny-particle analytic oracle, with unequal bin populations, correlated PM errors, and mixed velocity availability.
- Numerical marginalization and normalization tests, including Gaussian foreground mass outside the background box.
- Background-only, \(\epsilon=1\), \(\epsilon=0\) where defined, empty-selection, and far-from-stream cases.
- Verify identical \(u\)-conditioned background probability across marginals; verify chain terms sum per star to the joint score.
- Include a correlated example where the standalone marginal log scores deliberately **do not** sum to the joint.
- Check diagnostic modes leave joint scores unchanged and preserve explicitly frozen S2 replay values.

Existing tests cover background fallback, several estimator properties, and version tags. They do not establish these provenance or marginal/chain guarantees; the convergence test compares only two sample sizes. [test_score_conditional.py:24–137](tests/test_score_conditional.py:24), [test_spray_subset.py:106–122](tests/test_spray_subset.py:106)

**1. Verified versus assumed.** I verified the control flow, likelihood algebra, provenance weaknesses, and figure dependencies cited above. I assume your historical paper artifacts were produced under S2; I did not independently establish their execution history. I could not run tests: the available Python lacks `pytest` and NumPy. Nothing was modified.

**2. Strongest argument against my recommendation.** Pooling changes the scientific weighting, not merely numerical convergence: legacy bins give particles weight \(1/(K n_k)\), while pooling weights all selected particles equally. Different branches at the same \(u\) can therefore change relative probability, and contamination remains dependent on \(f_u\). A convergent estimator can converge to an inappropriate spray distribution. Preserving a homogeneous, explicitly finite-resolution S2 analysis could be more defensible than migrating before validating S4. [score.py:285–299,323–328](src/ocen_dm/streams/score.py:285)

**3. What I would check to be sure.** Reconstruct artifact-level provenance; reproduce historical S2 scores; measure S4 ranking stability across independent seeds and resolution; run injection/recovery under plausible release histories and contamination; refit with multiple starts; and execute the integration and mathematical tests above. The threshold-based bin selection demonstrates finite-sample instability, but does not by itself prove universal asymptotic non-convergence of every legacy-bin model. [score.py:297–299](src/ocen_dm/streams/score.py:297), [test_score_conditional.py:113–125](tests/test_score_conditional.py:113)