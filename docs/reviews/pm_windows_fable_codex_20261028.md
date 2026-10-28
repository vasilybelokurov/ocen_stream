# PM data vector: local 3-deg windows with data-only drop rules instead of spline knots? Fable and Codex (2026-10-28)

Proposal (user's idea): measure the PM tracks in a grid of phi1 windows with the same local estimator as the model, drop windows by
pre-set data-only rules. Motivation: misfit concentrates at the end PM knots (pmdec -2, 28; pmra 18, 28); data knots are global natural-
spline values (boundary values at -2 and 28), model knots local +-1.5 deg fits. Estimator test (bin/streams/knot_estimator_test.py, c0):
measuring the model with the spline estimator does not remove the misfit (pmra chi2 48.7 -> 49.3; pmdec 59.7 -> 103.5).

## Fable (computed on the real catalogue; reproduced: bin/streams/reviews/pm_windows_check_20261028.py)
Per 3-deg window -2..28, truncated Gaussian + linear trend + empirical background + per-star errors:
- pmdec: every window clean (stream stars 231, 155, 85, 81, 85, 117, 166, 183, 91, 34; track errors 0.027-0.14 mas/yr). Far end 25-28:
  +0.185 +- 0.054 (width 0.13) vs spline knot 28 +0.351 +- 0.093 (natural-boundary value). -2..1: +0.018 +- 0.033 (stream fraction 0.14).
- pmra: windows 7-10, 10-13, 13-16 degenerate with a free width (width 2.3-3.3 in |v| < 4, f up to 1.0, errors 0.28-0.83); the spline
  manages only because width and fraction are shared along phi1. Fix the width per window from the global spline (pre-registered),
  or use 6-deg windows over 4-16.
- Curvature over 3 deg <= 0.018 mas/yr except pmra -2..1 (0.13): linear trend per window adequate; constant not (pmra slopes to 0.45/deg).
- Drop rules (pre-registrable): mock recovery, N_stream >= 50, no parameter at a bound, |mean pull| < 0.3, pull sd 0.8-1.3, a null window
  off-stream < 3 sigma. By these rules both pmdec end windows PASS: the change will not remove the end misfit (25-28: +0.19 vs models ~0,
  3-4 sigma). Sky windows: keep (rules exist, width mocks at 18-24 calibrated); export the track-width covariance.
- Re-measure the ~10 valley minima with both vectors; redo the Asimov smoke test (N 150) before the main run. Against itself: ~1 day of
  work that will not change the qualitative result; fine to run the main SMC on the current vector instead.

## Codex (code reading; no experiments)
- Develop local measurements but not unchanged: the model estimator fits intrinsic particle PMs without background or error convolution,
  so centres/widths alone do not make the estimators identical (checked: true as code; both target the intrinsic mean track).
- Windows -2, 0, 3, ..., 27, 28 with explicit short terminal bins; linear residual trend with slope uncertainty propagated; compare 2/3/4
  deg designs on simulations, not real-data chi2. Background smoothed across phi1 -> windows not strictly independent: estimate joint
  covariance.
- Drop rules as proposed insufficient: >= 500 mocks per window and scenario (nulls, perturbed backgrounds, curvature, contamination,
  varying widths/fractions), |mean pull| < 0.2, pull sd 0.9-1.1, nominal 68/95% coverage, < 1% failures; >= 30 stream stars as a screen.
  Residuals are already known: call it prospective validation, freeze the rules.
- Reassess all sky windows by the same rules; preserve the track-width covariance.
- Risk: window averaging/selection may erase endpoint information; test with injected bar angles across the valley.
- Before the main run: noisy end-to-end recovery, background/error-convolved model measurement, CPU/GPU agreement, boundary/empty-window
  tests, versioned inputs and fresh caches (gpu_score_export overwrites a fixed filename).

## Agreed
Local windows are worth having (symmetry, local diagnostics); pmra interior needs care; rules frozen before use; re-measure models and
redo the Asimov test before the main run. The windows will not make the far-end pmdec misfit go away.

## Different
Effort: Fable - pragmatic (width fixed for pmra, simple mock rules, ~1 day; or skip and run now). Codex - full validation programme
(500 mocks x scenarios per window, joint covariance, error-convolved model measurement) before any main run.
