# Reviews

Independent reviews of the stream work by OpenAI Codex (read-only access to the repository) and Fable, newest first.
They were written in oCen_dm before the stream work moved here on 2026-10-25: code paths `src/ocen_dm/streams/X.py` are now `src/ocen_stream/X.py`; links to `results/` point to local run outputs that are not in the repository.

| Date | Review | Subject |
|---|---|---|
| 2026-10-27 | [final_run_fable_codex_20261027](final_run_fable_codex_20261027.md) | Final GPU run: kernel fixes, mock criteria, temperature, distance prior, N 600 vs 1200 |
| 2026-10-26 | [sampler_fable_codex_20261026](sampler_fable_codex_20261026.md) | Sampler for the GPU fit: emcee vs SMC vs dynesty; mode test; target; distance prior flag |
| 2026-10-24 | [sky1comp_fable_codex_20261024](sky1comp_fable_codex_20261024.md) | Single-component sky fit: two likelihood bugs; local fixed-knot fits, main band only |
| 2026-10-24 | [twocomp_modelling_fable_codex_20261024](twocomp_modelling_fable_codex_20261024.md) | Two-component measurement and the modelling likelihood (density likelihood, not knots) |
| 2026-10-23 | [observable_format_fable_codex_20261023](observable_format_fable_codex_20261023.md) | Format of the observables for model comparison (joint Poisson count cube) |
| 2026-10-23 | [iteration_readiness_fable_codex_20261023](iteration_readiness_fable_codex_20261023.md) | dm estimator choice and the sky+PM <-> CMD iteration protocol |
| 2026-10-23 | [mask_smoothing_fable_codex_20261023](mask_smoothing_fable_codex_20261023.md) | Mask track smoothing: spline gap overshoot, clip, orbit-start masks |
| 2026-10-23 | [dm_iteration_fable_codex_20261023](dm_iteration_fable_codex_20261023.md) | dm(phi1) refit by iteration: background normalisation, null-gated dm estimator |
| 2026-10-23 | [cmd_mask_fable_codex_20261023](cmd_mask_fable_codex_20261023.md) | CMD mask plan: Fable and Codex (2026-10-23) |
| 2026-10-23 | [fable_bar_slowdown_20261023](fable_bar_slowdown_20261023.md) | Fable: bar slowdown and the omega Cen stream -- importance, rewinding lessons, strategy (2026-10-23) |
| 2026-10-23 | [fable_member_reselection_20261023](fable_member_reselection_20261023.md) | Fable: STREAMFINDER selection imprints, Gaia DR3 re-selection, selection-aware modelling (2026-10-23) |
| 2026-10-23 | [masks_fable_codex_20261023](masks_fable_codex_20261023.md) | Mask borders for the Gaia cross-selection: Fable and Codex (2026-10-23) |
| 2026-10-22 | [fable_demonstration_design_20261022](fable_demonstration_design_20261022.md) | Fable: how to demonstrate (1) no-bar fails, (2) the bar must rotate, (3) the bar is non-standard (2026-10-22) |
| 2026-10-22 | [fable_note_review_20261022](fable_note_review_20261022.md) | Fable: review of the stream-fitting note (paper/stream_fits_note, local only; 2026-10-22) |
| 2026-10-22 | [fable_posterior_revisited_20261022](fable_posterior_revisited_20261022.md) | Fable review: path to a global posterior, revisited after Stage 0 (2026-10-22) |
| 2026-10-21 | [fable_bar_force_20261021](fable_bar_force_20261021.md) | Fable review: is the outer bar-force demand robust, and what next (2026-10-21) |
| 2026-10-21 | [fable_bar_model_20261021](fable_bar_model_20261021.md) | Fable review: why the 25-deg bar scatters its debris, and a targeted alternative-bar plan (2026-10-21) |
| 2026-10-21 | [fable_selection_tests_20261021](fable_selection_tests_20261021.md) | Fable review: selection-aware score and a Gaia search for the 25-deg second branch (2026-10-21) |
| 2026-10-20 | [codex_solar_frame_20261020](codex_solar_frame_20261020.md) | Codex review: self-consistent solar frame (2026-10-20) |
| 2026-10-20 | [fable_global_fit_20261020](fable_global_fit_20261020.md) | Fable review: global fitting design for stream 54 (2026-10-20) |
| 2026-10-20 | [fable_step2_design_20261020](fable_step2_design_20261020.md) | Fable review: step-2 design (2026-10-20) |
| 2026-10-19 | [codex_profile_plan_2026-10-19](codex_profile_plan_2026-10-19.md) | Codex review: direct per-angle profile by optimisation on sprays (2026-10-19) |
| 2026-10-19 | [codex_spray_speed_2026-10-19](codex_spray_speed_2026-10-19.md) | Codex review: speeding up the AGAMA spray (2026-10-19) |
| 2026-10-18 | [codex_emulator_conclusions_2026-10-18](codex_emulator_conclusions_2026-10-18.md) | Conclusions of the emulator campaign: what the evidence supports, over-claims, caveats |
| 2026-10-18 | [codex_emulator_review_2026-10-18](codex_emulator_review_2026-10-18.md) | Stochastic-kriging emulator and refinement design: duplicate picks, loader provenance, fold-wise noise model, seeds |
| 2026-10-18 | [codex_cutoff_and_mocks_2026-10-18](codex_cutoff_and_mocks_2026-10-18.md) | Codex review: score cutoff speed-up and mock-data design (2026-10-18) |
| 2026-10-18 | [codex_emulator_bias_2026-10-18](codex_emulator_bias_2026-10-18.md) | Codex review: low-angle bias of the emulator in the mock recovery test (2026-10-18) |
| 2026-10-18 | [codex_search_plan_2026-10-18](codex_search_plan_2026-10-18.md) | Codex review: history matching + stratified search plan (2026-10-18) |
| 2026-10-18 | [codex_selection_2026-10-18](codex_selection_2026-10-18.md) | Codex review: selection normalisation of the conditional stream likelihood (2026-10-18) |
| 2026-10-17 | [codex_training_set_plan_2026-10-17](codex_training_set_plan_2026-10-17.md) | 9-D emulator training set: calibration tranche first, float64 sprays, per-star score components, provenance hashes |
| 2026-10-17 | [codex_score_provenance_2026-10-17](codex_score_provenance_2026-10-17.md) | Stream-score version consistency, provenance and the pooled default (adopt S4 only after refitting) |
| 2026-10-17 | [codex_unbiased_likelihood_2026-10-17](codex_unbiased_likelihood_2026-10-17.md) | Bias and noise of the conditional KDE stream likelihood: replicated randomized integration, delta method, power likelihood |
| 2026-10-16 | [codex_stream_paper_plan_2026-10-16](codex_stream_paper_plan_2026-10-16.md) | Stream paper plan: rejects "impossible without a bar"; matched null-vs-bar fit and mocks needed; six-figure plan |
| 2026-10-11 | [codex_speed_review_2026-10-11](codex_speed_review_2026-10-11.md) | Particle-spray speed-ups |
| 2026-10-10 | [codex_stepback_2026-10-10](codex_stepback_2026-10-10.md) | Step-back audit of the knee investigation |
| 2026-10-10 | [codex_profile_likelihood_plan_2026-10-10](codex_profile_likelihood_plan_2026-10-10.md) | Profile likelihood for the bar angle |
| 2026-10-10 | [codex_frame_decomp_next_2026-10-10](codex_frame_decomp_next_2026-10-10.md) | Initial-condition versus projection frame decomposition |
| 2026-10-10 | [codex_distances_and_misfit_2026-10-10](codex_distances_and_misfit_2026-10-10.md) | Parallax offset and CMD distance along the stream |
| 2026-10-10 | [codex_diagnostic_experiments_2026-10-10](codex_diagnostic_experiments_2026-10-10.md) | Ranked causes of the knee misfit |
| 2026-10-10 | [codex_code_review_2026-10-10](codex_code_review_2026-10-10.md) | Stream code review (handedness, age cut) |
| 2026-10-10 | [codex_T1T2_next_steps_2026-10-10](codex_T1T2_next_steps_2026-10-10.md) | Solar-frame and bar-size experiments (T1/T2) |
| 2026-10-09 | [codex_track_objective_2026-10-09](codex_track_objective_2026-10-09.md) | Failure mode of the track objective |
| 2026-10-09 | [codex_likelihood_calibration_2026-10-09](codex_likelihood_calibration_2026-10-09.md) | Block bootstrap and seed noise of the stream likelihood |
| 2026-10-09 | [codex_conditional_kde_2026-10-09](codex_conditional_kde_2026-10-09.md) | Conditional chi-KDE likelihood design |
| 2026-10-08 | [codex_research_knee_2026-10-08](codex_research_knee_2026-10-08.md) | Web research pass on the stream-54 PM knee (contains wrong DOIs; see Literature) |
| 2026-10-08 | [codex_research_fit_campaign_2026-10-08](codex_research_fit_campaign_2026-10-08.md) | Web research for the stream fit campaign (two arXiv-ID errors and wrong DOIs; see Literature) |
- [error_floors_fable_codex_20261028.md](error_floors_fable_codex_20261028.md) — S5-style likelihood floors: Codex (adopt, validate first) vs Fable (no: floors switch off pmdec; misfit sits in four end knots)
- [pm_windows_fable_codex_20261028.md](pm_windows_fable_codex_20261028.md) — PM local windows instead of spline knots: both yes in principle; pmdec windows all pass (far-end misfit stays, +0.19 +- 0.05); pmra interior degenerate without a fixed width; Fable pragmatic vs Codex full validation
- [cross_machine_fable_codex_20261029.md](cross_machine_fable_codex_20261029.md) — Colab vs CSD3 chi2 not bit-identical (sd ~1 = roundoff amplified by chaos); one machine per run + environment fingerprint; Codex: revalidate on CSD3, Fable: cheap rescoring check
