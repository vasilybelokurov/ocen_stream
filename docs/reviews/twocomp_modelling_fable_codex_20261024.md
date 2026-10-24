# Two-component measurement set-up and the way forward for modelling: Fable and Codex (2026-10-24)

Briefing: main/branch spline fits (stream_spline_fit.py: 1-component PMs, 2-component extended sky fit dphi2_sky_final, per-band
PMs in +-1 deg windows), branch characterisation, branch sweep pilot; user proposal: use the two-component description as the
data vector (S5-style knot Gaussian likelihood with nuisance sigma).

## Both
- The knot vector is NOT a calibrated data vector yet; do not rescore the archive with it and read the ranking as evidence.
- Degenerate/boundary knots (verified by Claude): main width knots at phi1 0, 8.3 saturate the 1.5-deg cap (raw 16604, 1051);
  pmra_branch at 18.7 width 2.25 mas/yr, f = 1.0 (background absorbed); branch end knots unconstrained.
- Knots are correlated (Claude verified: m 8.3/16.7 r = -0.50, 16.7/25 -0.58, m2 18.7/23 -0.70): full covariance, not diagonal.
- Model side: band assignment by data windows is unsound (branch_sweep.py); prefer a likelihood on the full particle density,
  which needs no labels; a single-band model is then penalised by the missing density.
- Particle support must be uniform: chi > 0, age cut, enough particles (Fable: 8k particles -> lnL sd 7-9, biased low 8-12;
  179k -> sd 1.4-1.8); spray_grid4 (8k) and the ~1k-particle training/wave sets cannot be used for ranking.

## Fable (checked with short scripts on stored sprays)
- Profiled nuisance sigma (Koposov+2023 Sec 5.3 style) makes the knot likelihood non-discriminating: knot chi2 438-865 for 4
  models -> -2lnL 6-8 (main) with sigma_nuis 1.7-2.2 deg; the nuisance absorbs a wrong-band model.
- Stored sprays decompose into "ridge (1.0-1.5 deg) + broad fan", not main + branch; model ridge widths 1.0-2.5 vs data 0.6-1.5.
- Extended Poisson sky likelihood (data stars + fixed band background + histogram-KDE of particles, h 0.4 x 0.3 deg): < 0.1 s per
  model; dlnL vs background-only +164..+265 for 4 models (data spline +424); model differences stable to bandwidth (w3 - null
  78.5-79.1 for h 0.25-1.0).
- Recommends (b): extended sky term with a free low-order rate spline + score_conditional PM term with the empirical off-sky
  backgrounds; one completeness factor per band; mock-calibrated tempering. Rescore only >= 90k-particle sets (~1850).
- Against own choice: sprays' morphology (2x too wide, fans) may drive bar parameters through the fan, not the track.

## Codex (code reading, no runs)
- Recommends (a) only for screening; production (d): a joint binned sky x PM shape likelihood per phi1 slice, conditional on
  the slice total (no along-stream rates, no luminosity-function modelling), error-convolved particles + empirical field with
  off-region constraint; smooth relative-efficiency nuisance between bands.
- Background uncertainty must enter (area map is from star counts; band cells overlap the +-6 fit window); covariance is
  conditional on AIC knots and frozen masks; per-band PMs measure selected windows, scalar errors (no Gaia pmra-pmdec
  correlation).
- Sprays may be generated with one launch arm; differing age cuts across files -> require common support, regenerate.
- Saved branch dm +0.26 +- 0.16 is not bias-corrected (the ~0.36 figure is Claude's estimate, not the saved value).
- Gates: reproducible manifest; >= 200 measurement-recovery realisations (pull sd 0.9-1.1, coverage); label/morphology tests;
  Monte Carlo error < 0.2 sigma; null false-branch rate; parameter recovery at several truths (bias < 0.3 sigma); only then the
  archive.

## Synthesis (Claude)
Agreement: drop the knot vector as the likelihood; score the full model density against the data with fixed empirical
backgrounds, conditional on along-stream density (free rate spline or per-slice totals); uniform particle support; mock-based
calibration before trusting rankings. Difference: Fable's unbinned extended sky term + existing conditional PM term (fast,
already prototyped) vs Codex's joint binned sky x PM per slice (keeps sky-PM association). Spline fits remain diagnostics and the
figures of the paper.
