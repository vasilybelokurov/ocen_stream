# Stream 54: status (2026-10-17)

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
- **A 9-parameter emulator training set is running** (307 sprays at 256,000 release epochs).

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

## Emulator training set (running)

`bin/streams/training_set.py`, started 2026-10-17, output `results/streams/training_9d_20261017/` (not in the repository).

- **Parameters and ranges:**
  - bar angle 10–35°, pattern speed 30–40 km s<sup>−1</sup> kpc<sup>−1</sup>, amplitude 0.6–2.0;
  - bar size 0.8 up to the corotation limit (size × 4.25 kpc ≤ R<sub>CR</sub> of the point's own host, at most 1.5);
  - ω Cen distance 5.25–5.95 kpc, PMs ±4σ;
  - v<sub>c</sub>(R<sub>0</sub>) 220–240 km s<sup>−1</sup> through the halo normalisation, halo flattening 0.8–1.2.
- **Design:** 256 scrambled-Sobol points, an independent seed per run, 256,000 release epochs.
- **Calibration runs:** two points with 8 seeds at 256k and 8 at 128k (above); seven more with 4 seeds each.
- **Storage:** every run keeps its full spray, so any score definition can be recomputed later.
- **Provenance:** source, input and data hashes are recorded. A run refuses to start if any source changed.

**Next steps:**
1. jackknife-correct every run;
2. fit a noise model from the replicate seeds;
3. fit the emulator to the corrected ln L, with each point's noise, and test it on held-out design points;
4. choose the posterior calibration (global or per-segment tempering, or Holmes & Walker 2017,
   [arXiv:1701.08515](https://arxiv.org/abs/1701.08515)) by held-out segments and mock coverage.

## Reviews

[Unbiased likelihood](../reviews/codex_unbiased_likelihood_2026-10-17.md),
[score provenance](../reviews/codex_score_provenance_2026-10-17.md),
[training-set plan](../reviews/codex_training_set_plan_2026-10-17.md) (Codex, 2026-10-17).
