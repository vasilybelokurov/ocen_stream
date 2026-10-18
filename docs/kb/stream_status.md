# Stream 54: status (2026-10-18)

This page summarises the stream work since 2026-10-13. Earlier results are on the
[PM refinement page](../STREAM_PM_REFINEMENT_2026-10-13.md) (historical). Every ln L below names its score version and
particle number. Totals from different versions or particle numbers are **not comparable**; the versions are defined on
[stream likelihood versions](models/stream_likelihood_versions.md).

## Summary

- **A bar is needed, but the fit is not yet satisfactory.** Barred models beat the best axisymmetric model by about 3300–4100
  in ln L. No model reproduces the steep μ<sub>δ</sub> of the turn at φ<sub>1</sub> > 22°.
- **Two objectives disagree.** Fits to the binned tracks halve the track χ<sup>2</sup> but lose 100–450 in the star-by-star
  ln L. The disagreement is a trade-off between the sky track and the PM track in the turn.
- **The original likelihood did not converge with particle number.** It is replaced by fixed χ bins with equal weights. A
  jackknife removes the small-particle-number bias.
- **A 9-parameter emulator is built and validated** (470 sprays). It does not yet constrain the bar. One independently
  checked point beats the paper models by about 600, but the turn is still not fitted.

## Model comparison

Scores S2 (2026-10-16) and S3 (2026-10-17), 32,000 release epochs. The host is the Hunter+2024 Milky Way
([2024A&A...692A.216H](https://ui.adsabs.harvard.edu/abs/2024A%26A...692A.216H/abstract)); the axisymmetric null uses the
same host with bar amplitude 0.

| Model | Δln L vs refined 16° bar, S2 (seeds 4–6) | ln L, S3 (seed 4) |
|---|---|---|
| Refined 16° bar | 0 | −22552.5 |
| Refined 28° bar | −292 ± 157 | −22946.9 |
| Axisymmetric null, ω Cen distance and PMs refitted | −3541 ± 196 | −26649.3 |
| Axisymmetric null, catalogue state | −7565 ± 86 | — |

- **Columns.** Δln L: mean and seed scatter over three seeds. S3 differs from S2 only in the normalisation of the uniform
  background; the ranking is unchanged (16° − 28°: +321 in S2 and +394 in S3 at seed 4).
- **The refitted null runs to the bounds**: distance 5.95 kpc (10.4σ from the adopted 5.43 ± 0.05 kpc), PMs 2.6–3.0σ off.
- **Host variants of the null** (halo flattening 0.8/1.2, disc scale 0.8/1.2, v<sub>c</sub>(R<sub>0</sub>) 220/240 km s<sup>−1</sup>,
  McMillan 2017) change ln L by −630 to +370. All remain 3300–4300 below the 16° bar, and none steepens μ<sub>δ</sub> in the
  turn (−8.3 to −9.0 mas yr<sup>−1</sup> at φ<sub>1</sub> = 25°, against −11.0 in the data).
- **Barred residuals.** μ<sub>δ</sub> is too shallow by 0.8 mas yr<sup>−1</sup> at φ<sub>1</sub> = 21° and by 1.2–1.8 at
  23–27°. Most of the barred gain over the null (about 2900 of 3250–3550) comes from φ<sub>1</sub> > 17°.
- **The 16° vs 28° gap shrinks with particle number.** Under the pooled score the 28° − 16° difference is −454 at 4,000
  epochs and −98 ± 18 at 512,000 (five seeds), and still trending towards zero. The bar angle is therefore not yet
  constrained by these comparisons.

## Track objective vs star-by-star likelihood

Fits to binned tracks (Δφ<sub>2</sub> and PM medians in 2° bins of φ<sub>1</sub>; `bin/streams/fit_track_objective.py`)
roughly halve the track χ<sup>2</sup> of the refined models. Scored star by star (S2, 32,000 epochs, seeds 5 and 6) they are
100–450 **worse** in ln L.

The disagreement comes from the turn (`bin/streams/plot_objective_disagreement.py`):

- the track-optimised long bar matches the PM medians to φ<sub>1</sub> ≈ 25° but puts the turn debris 1.5° below the members
  in Δφ<sub>2</sub> at φ<sub>1</sub> = 21–26°;
- the refined 28° bar has the sky track right at φ<sub>1</sub> = 18–24° but PMs that are too shallow.

No model found so far gets both. Separately, all barred models place the young debris near the cluster about 1° from the
observed track at φ<sub>1</sub> = 2–8°; the axisymmetric null does not.

## Likelihood: convergence, bias and noise

- **Legacy score (S2/S3) does not converge.** It gives every χ bin with at least 20 particles equal weight, so the number of
  bins grows with particle number. Model differences changed sign between 4,000 and 128,000 epochs.
- **Fixed χ bins (chosen 2026-10-17).** Bins of width 5 with fixed edges and equal weights, empty bins kept. Equal weight per
  χ interval keeps the model's density along the stream out of the score, because that density is not reliably modelled.
  The candidate range is χ = 0–200, which holds 93–96% of the model debris inside the data window; no member loses more than
  2 in ln L against the pooled score. The library default is still the pooled score until this range is confirmed.
- **Noise is shot noise.** The seed variance of each member's ln L follows 1/n<sub>eff</sub>, the effective number of model
  particles behind its density, for n<sub>eff</sub> ≳ 3. Neighbouring members share particles, so the noise of the total is
  2–3 times the independent-star value. Members with few particles sit in the far turn (φ<sub>1</sub> > 22°).
- **Bias: corrected with a jackknife.** The log of a noisy density is biased low, by about 45–140 at 256,000 epochs depending
  on the parameters. A delete-group jackknife (10 interleaved groups; `src/ocen_dm/streams/score_bias.py`) removes it:

  | Calibration point | Raw: 256k − 128k | Jackknife: 256k − 128k | Seed sd at 128k / 256k |
  |---|---|---|---|
  | 14 (near the centre of the design) | +73 ± 34 | −10 ± 35 | 85 / 43 |
  | 20 (largest bar amplitude × size) | +25 ± 27 | −5 ± 32 | 62 / 43 |

  Means and seed scatter over 8 independent seeds per resolution, fixed χ bins 0–200. The check is limited to about ±35
  by the number of seeds. A second-order (delta-method) correction removes only about half the bias.

## Emulator (2026-10-18)

**Data.** 470 sprays at 256,000 release epochs over 375 points of the 9-parameter space (`bin/streams/training_set.py`,
`bin/streams/stream_batch.py`), each scored with fixed χ bins 0–200 and jackknife-corrected:
- 256 Sobol design points;
- 32 points with two extra independent seeds, for the noise. Each repeat shifts the release-time grid by a random offset
  as well as redrawing the escape offsets; the original runs agree with the shifted repeats (mean difference −24 ± 20);
- two refinement rounds (100 and 60 runs) chosen by predicted ln L + 2σ, with exploration points, held-out points and
  repeats;
- 8 seeds at the emulator maximum.

**Model.** Stochastic kriging (`src/ocen_dm/streams/emulator.py`; Ankenman, Nelson & Staum 2010,
[2010OpRes..58..371A](https://ui.adsabs.harvard.edu/abs/2010OpRes..58..371A/abstract); Binois, Gramacy & Ludkovski,
[arXiv:1611.05902](https://arxiv.org/abs/1611.05902)):
- a GP for the mean, with each point's noise fixed;
- a noise surface ln s² = a + b ln v<sub>jk</sub> + GP fitted to the repeat variances (median noise sd 65 in ln L).

**Validation** (5-fold, noise model refitted in each fold; `results/plot_data/emulator_validate_final_20261018.json`):

| Subset | RMSE | Bias (true − predicted) | z sd | 95 % coverage |
|---|---|---|---|---|
| All 375 points | 1269 | −76 | 1.08 | 0.94 |
| Top 20 % | 1043 | +467 | 0.95 | 0.95 |
| 18 held-out points (never fitted) | 1123 | −413 | 1.01 | 16/18 |

- **Columns.** z = (measured − predicted)/total predicted sd. The uncertainties are roughly right overall but not yet
  established near the maximum: the top quintile is biased by +467 and only 18 points were held out.

**Results.**
- **The bar is not yet constrained.** The emulator's uncertainty is about 575 at the profile peaks and 806 at its optimised
  maximum, while each one-parameter profile (maximum over the other eight) varies by only 500–1350. Profiles are maxima over
  a finite candidate set with pointwise bands, not uncertainty-integrated. The emulator maximum moves between fits.
- **The two paper models are indistinguishable at current precision**: 16° −22174 and 28° −22171 (2 seeds each; difference
  3 ± ~70). They also differ in distance, amplitude and size, so this is not a clean test of the angle.
- **One independently replicated point is about 600 better**: −21569 ± 11 (8 seeds) at angle 17°, Ω<sub>b</sub> 32,
  amplitude 1.57, size 0.89, distance 5.79 kpc, v<sub>c</sub> 220 km s<sup>−1</sup>, q 1.02. Its slow bar, large distance and
  low v<sub>c</sub> describe this candidate, not a measurement. The score has no prior on ω Cen's distance (adopted
  5.43 ± 0.05 kpc). v<sub>c</sub> sits at the lower edge of the range; the latest emulator maximum also sits at the edges of
  μ<sub>δ</sub> and halo q.
- **Predicted maxima are optimistic.** The three checked predicted-best points came in 430–770 below their predictions.
- **The turn is still not fitted.** At φ<sub>1</sub> = 21° the best point improves the μ<sub>δ</sub> residual from −0.82 to
  −0.16 mas yr<sup>−1</sup> (paper 16°), but its Δφ<sub>2</sub> offset grows from 0.47° to 1.42°. This is a remaining
  mismatch, not shown to be unavoidable everywhere in the space.

**Open decisions:**
1. extend v<sub>c</sub> below 220 km s<sup>−1</sup> (and check the μ<sub>δ</sub> and halo-q edges);
2. add the distance measurement as a prior or keep the score likelihood-only;
3. denser local design around the top (a local GP is an experiment), keeping global exploration and fresh held-out points;
4. choose the tempering by held-out segments and mock coverage.

## Reviews

[Unbiased likelihood](../reviews/codex_unbiased_likelihood_2026-10-17.md),
[score provenance](../reviews/codex_score_provenance_2026-10-17.md),
[training-set plan](../reviews/codex_training_set_plan_2026-10-17.md) (Codex, 2026-10-17);
[emulator and design](../reviews/codex_emulator_review_2026-10-18.md),
[campaign conclusions](../reviews/codex_emulator_conclusions_2026-10-18.md) (Codex, 2026-10-18).
