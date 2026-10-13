The code does not establish a definite handedness, arm-sign, or observable-conversion bug. It does expose a more immediate issue for interpreting the results: **the grid score is designed to ignore stream density along the tail**, while it excludes model particles older than 700 Myr. That makes the 700 Myr cut a relevant scoring sensitivity, but the score cannot by itself explain the visible thinning as a count mismatch. ([score.py:233](src/ocen_dm/streams/score.py:233), [score.py:250](src/ocen_dm/streams/score.py:250))

## Findings

- **No clear reflection error in the observable round trip.** `to_model` flips the Astropy x position and velocity; `observables` flips them back before transforming to ICRS. That is internally consistent in the code. ([frames.py:31](src/ocen_dm/streams/frames.py:31), [frames.py:48](src/ocen_dm/streams/frames.py:48))

- **No obvious sign inversion in the release basis.** `rj_vj_R` builds its second axis as `L × r`, and `release_ic` offsets the `arm=+1` particle to positive radial x with a positive tangential velocity offset for a positive-Lz orbit. That is consistent with the stated outer/trailing convention, but the comments and equations do not verify the result dynamically. ([spray.py:25](src/ocen_dm/streams/spray.py:25), [spray.py:31](src/ocen_dm/streams/spray.py:31), [spray.py:49](src/ocen_dm/streams/spray.py:49), [spray.py:57](src/ocen_dm/streams/spray.py:57))

- **The trailing-arm selection is not independently validated by the scoring path.** The grid keeps `arm == 1`, while `score_conditional` separately drops particles with `chi <= 0`. If those labels disagree, the selected model stream is silently reduced or altered. ([grid4.py:36](bin/streams/grid4.py:36), [score.py:250](src/ocen_dm/streams/score.py:250))

  **Minimal test:** on representative and circular synthetic orbits, report the fraction of each arm with `chi > 0` and the sign of its `psi`. Check that arm `+1` is consistently the trailing branch over the release window before scoring.

- **The 700 Myr age cut can affect angle ranking.** Grid sprays release over 1000 Myr, but scoring retains only particles with `age < 700`; the model’s effective age range can also depend on which chi bins meet `nmin=20`. ([grid4.py:27](<code>/oCen%20dm/bin/streams/grid4.py:27), [score.py:250](src/ocen_dm/streams/score.py:250), [score.py:253](src/ocen_dm/streams/score.py:253))

  **Minimal test:** rescore the existing 1000 Myr sprays with age limits 500, 700, and 1000 Myr; report the angle ranking and per-star likelihood changes by phi1. This is cheaper and more informative than first generating longer sprays.

- **The conditional score cannot test the observed thinning as a count discrepancy.** It forms a density in observables conditional on along-stream position by dividing by the along-stream marginal. Where model support is weak, the background mixture takes over. Thus this score can compare track and kinematic agreement, but it is not evidence that the model predicts the observed number of stars along the tail. ([score.py:233](src/ocen_dm/streams/score.py:233), [score.py:240](src/ocen_dm/streams/score.py:240), [score.py:273](src/ocen_dm/streams/score.py:273))

  **Minimal test:** compare angle rankings from the current conditional score with an explicitly defined count or coverage diagnostic, while accounting for catalogue completeness and selection. Do not treat raw particle counts as observed counts without that model.

- **Data loading leaves a numerical-integrity check outstanding.** The cuts select stream 54 by sky position, but do not filter nonfinite PMs, PM errors, or correlations. Those values feed directly into the likelihood covariance calculation. ([spray_bar_grid2.py:27](bin/streams/spray_bar_grid2.py:27), [spray_bar_grid2.py:28](bin/streams/spray_bar_grid2.py:28), [spray_bar_grid2.py:30](bin/streams/spray_bar_grid2.py:30), [score.py:259](src/ocen_dm/streams/score.py:259))

  **Minimal test:** assert all selected PMs, errors, and correlations are finite, all uncertainties are positive, and every covariance determinant is positive. The source-ID equality assertion checks row alignment, not those conditions. ([spray_bar_grid2.py:26](bin/streams/spray_bar_grid2.py:26))

## Revised experiment order

1. Run the end-to-end checks still outstanding: `to_model`/`observables` round trip; synthetic circular-orbit release basis; arm versus chi sign; and catalogue finite-value/covariance checks. The packet says the derivative, bar-angle-time, and centre-orbit checks already passed, so repeating those is lower priority.
2. Rescore cached sprays across age limits and inspect per-star contributions by phi1. The current code supports this without another integration. ([grid4.py:31](bin/streams/grid4.py:31), [score.py:250](src/ocen_dm/streams/score.py:250))
3. Compare segment and observable residuals, then run mock recovery to test whether the angle preference is calibrated under the score. The grid holds the cluster PMs fixed while varying bar parameters and distance, so its angle maximum is conditional on that fixed state. ([grid4.py:20](bin/streams/grid4.py:20), [grid4.py:35](bin/streams/grid4.py:35))
4. Use the supplied 200k-tracer run as evidence against the spray release prescription being the sole cause of the miss, as the packet reports the same miss at 20 degrees. Test a different bar angle or centre state before repeating that comparison at 20 degrees.
5. Extend the release window beyond 1000 Myr only if existing sprays show that older particles would contribute; first establish the effect of the existing 700 Myr scoring cut.

## Verified, assumed, and remaining checks

**Verified by reading the code:** the x reflection is applied symmetrically in the conversion functions; the release basis uses `L × r`; the grid selects arm `+1`; the score filters on `chi > 0` and `age < age_max`; the conditional score divides by its along-stream marginal; and the data loader’s cuts do not include explicit finite-PM/covariance validation. ([frames.py:31](src/ocen_dm/streams/frames.py:31), [spray.py:31](src/ocen_dm/streams/spray.py:31), [grid4.py:36](bin/streams/grid4.py:36), [score.py:250](src/ocen_dm/streams/score.py:250), [score.py:273](src/ocen_dm/streams/score.py:273), [spray_bar_grid2.py:28](bin/streams/spray_bar_grid2.py:28))

**Assumptions:** I accept the packet’s reported astrometric residuals, catalogue size, 200k-tracer miss, and previous audit results as inputs; I did not reproduce those runs. I also assume the external Hunter potential assets use the intended model-frame orientation.

**Strongest argument against this assessment:** the most consequential remaining defect could be in the external potential’s rotation convention or in its mapping to the reflected model frame. The source documents the intended angle convention, but source inspection alone cannot prove that the external potential rotates in that sense. ([restricted.py:43](src/ocen_dm/streams/restricted.py:43), [restricted.py:51](src/ocen_dm/streams/restricted.py:51))

**To be sure:** run the four end-to-end/data checks above; inspect likelihood contributions by age and phi1; and verify the external potential’s rotation direction against an independently transformed positive-longitude near end. The supplied test coverage includes focused score and spin tests, but the visible test references do not cover the release-basis, arm/chi, or end-to-end frame checks proposed here. ([test_score_conditional.py:24](tests/test_score_conditional.py:24), [test_spray_spin.py:14](<home>/<cloud>/<repo>/tests/test_spray_spin.py:14))