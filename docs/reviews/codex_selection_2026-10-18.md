# Codex review: selection normalisation of the conditional stream likelihood (2026-10-18)

Codex CLI, effort high, read-only. Outcome: no change to the score; the selection question is open (user: the model PDF at fixed phase is the likelihood); mock sampler fixed to the current score's mixture.


The code contradicts two premises: the final sample contains **stream 54 only**, and the mock generator is **not** an exact sampler of the proposed corrected likelihood. It chooses foreground/background once, then retries only foreground draws. ([spray_bar_grid2.py:25–31](bin/streams/spray_bar_grid2.py:25), [mock_data.py:62–79](bin/streams/mock_data.py:62))

**My recommendation: implement neither the denominator change nor a closure claim without first specifying the selection being tested.** The proposed denominator is mathematically correct for an observed-space box selection, but that does not establish its correctness for this catalogue.

1. **High priority — the diagnosis is conditional on the selection assumption.** Both scoring paths use the unrestricted foreground \(u\)-marginal; bounds validate members and normalize the background without restricting foreground particles. Thus they evaluate the stated unselected mixture conditional, rather than its box-selected counterpart. This is a selection mismatch **if the data were selected by that box**; it is not an intrinsically invalid density over the full foreground support. ([score.py:274–290](src/ocen_dm/streams/score.py:274), [score.py:329–344](src/ocen_dm/streams/score.py:329), [score.py:389–403](src/ocen_dm/streams/score.py:389))

   The box is constructed from member extrema; the explicit catalogue cuts instead involve stream identity, latitude and longitude. Containing every member does not make the box the selection function. Treat it as a sensitivity experiment, not an established correction. Padding-dependent normalization can move model preferences, particularly near edges. ([grid4_common.py:82–89](bin/streams/grid4_common.py:82), [spray_bar_grid2.py:25–31](bin/streams/spray_bar_grid2.py:25))

2. **High priority — use kernel-integrated acceptance, including PM correlation.** For an observed-space selection probability \(a(u,w)\), the correct expression is
   \[
   L_i^a=
   \frac{a(u_i,w_i)[(1-\epsilon)f_i(u_i,w_i)+\epsilon g(u_i,w_i)]}
   {\int a(u_i,w)[(1-\epsilon)f_i(u_i,w)+\epsilon g(u_i,w)]\,dw}.
   \]
   For exactly the background box, your proposed denominator follows. For a different footprint, the background integral generally changes too.

   Consistency with the implemented numerator requires integrating each Gaussian kernel over the selection. The PM rectangle probability is **bivariate**, not a product of marginal Gaussian probabilities when covariance is nonzero. Dependence on \(C_i\) is correct when selection acts on measured PMs. An indicator on particle centres describes a different, latent-space selection and cannot replace this integral while leaving the numerator unchanged. ([score.py:317–340](src/ocen_dm/streams/score.py:317))

3. **High priority — the mock mismatch survives the proposed fix.** Write
   \[
   D_i=(1-\epsilon)f_u+\epsilon g_u,\qquad
   b_i=\epsilon g_u/D_i,\qquad F_i^S=\int_S f_i(u_i,w)\,dw.
   \]
   The current sampler generates
   \[
   q_{\rm mock}(w\mid u_i)
   =(1-b_i)\frac{f_i(u_i,w)}{F_i^S}+b_i g_w(w),
   \quad w\in S,
   \]
   because rejection occurs after fixing the component. Your corrected likelihood instead requires background probability
   \[
   b_i^S=\frac{\epsilon g_u}{(1-\epsilon)F_i^S+\epsilon g_u}.
   \]
   These agree only in special cases. Either use \(b_i^S\), or retry the **entire mixture draw**, including component selection, after rejection. ([mock_data.py:60–80](bin/streams/mock_data.py:60))

4. **Selection details — \(u\) cancels; a velocity cut needs evidence.**
   
   A selection depending only on the conditioned \(u_i\) cancels. Do not trim model particles to the observed \(u\) interval merely because observations stop there: kernels centred outside it still contribute inside. However, the implementation uses the observed range to set \(g_u\), so changing that range changes foreground/background mixing. ([grid4_common.py:18](bin/streams/grid4_common.py:18), [score.py:272–273](src/ocen_dm/streams/score.py:272), [score.py:332–343](src/ocen_dm/streams/score.py:332))

   A velocity-box factor belongs only in measured-velocity members’ denominators **if that subset was actually selected by observed velocity**. The loader applies an error-quality rule, not a \(0<v<400\) cut; that interval is supplied as background support. Its existence alone does not justify truncating foreground velocities. If catalogue membership depends on velocity even for missing-\(v\) objects, their likelihoods require integrating that selection too. ([spray_bar_grid2.py:29–31](bin/streams/spray_bar_grid2.py:29), [grid4_common.py:89–105](bin/streams/grid4_common.py:89))

5. **Consequences — rankings and emulator calibration require revalidation.** Holding the numerator fixed, the box correction adds
   \[
   \Delta\log L=\sum_i\log\frac{(1-\epsilon)f_u(u_i)+\epsilon g_u}
   {(1-\epsilon)F_i^S+\epsilon g_u}\ge0.
   \]
   The increase is model-dependent; aggregate fractions of debris outside the box cannot predict ranking changes. The relevant quantities are kernel- and bin-weighted acceptance probabilities at individual members. ([score.py:329–343](src/ocen_dm/streams/score.py:329))

   The emulator consumes stored jackknife totals and variances. A changed score requires rescoring every leave-out sample, refitting both emulator mean and noise, and repeating validation. Existing batch code saves full sprays, allowing rescoring without rerunning the dynamics. ([emulator_design.py:51–58](bin/streams/emulator_design.py:51), [emulator_design.py:84–94](bin/streams/emulator_design.py:84), [stream_batch.py:95–109](bin/streams/stream_batch.py:95))

6. **Simpler alternatives — apply known cuts accurately before approximating unknown selection.** Transform the explicit sky cuts into acceptance at each \(u\), then integrate the kernels over that region. A verified footprint polygon improves geometric selection, but does not establish PM-dependent membership completeness. Restricting particle centres before smoothing is an approximation whose boundary error must be measured. The actual coordinates are a rotated longitude and polynomial-subtracted latitude, so the Galactic cuts do not generally become a constant across-stream interval. ([spray_bar_grid2.py:28](bin/streams/spray_bar_grid2.py:28), [path.py:127–130](src/ocen_dm/streams/path.py:127))

**1. Verified versus assumed.** I verified the formulas, correlated kernels, component-first rejection, selection cuts and emulator input path by reading the cited code. Existing tests check background-only normalization, cutoff agreement and jackknife agreement; those checks do not establish selected-foreground normalization or sampler closure. ([test_score_conditional.py:68–86](tests/test_score_conditional.py:68), [test_score_cutoff.py:29–47](tests/test_score_cutoff.py:29))

I did not run tests or recompute the quoted debris fractions, bounds or velocity count. The selection derivation assumes acceptance acts on observed quantities and that uncertainties and velocity availability can be conditioned on without an additional selection model.

**2. Strongest argument against my recommendation.** The current likelihood also lacks a justified STREAMFINDER selection model. If the box closely approximates actual acceptance and acceptance varies substantially between models, leaving the score unchanged could cause more bias than the approximation introduces. That argues for promptly measuring sensitivity, even before the complete selection function is available.

**3. Needed for confidence.**

- Numerically integrate the corrected density to one at fixed \(u\), using correlated PM errors, edge particles and mixed velocity availability.
- Check mock component frequencies against \(b_i^S\), then test repeated-catalogue parameter recovery and coverage under known selection, both with and without correction.
- Repeat with misspecified selection and varied, independently fixed box boundaries.
- Recheck exact/cutoff agreement and every jackknife normalization after correction.
- Rescore existing competitors and inspect acceptance and score changes per member before interpreting emulator profiles.
- Bound rejection attempts and preserve previous mock outputs: the current loop is unlimited, and catalogue/JSON outputs are overwritten on rerun. ([mock_data.py:71–79](bin/streams/mock_data.py:71), [mock_data.py:94–99](bin/streams/mock_data.py:94))