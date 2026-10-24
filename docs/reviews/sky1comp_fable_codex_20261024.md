# Single-component sky fit: Fable and Codex (2026-10-24)

Briefing: the single-component sky spline fit (key dphi2_W6ext_nohole: extended likelihood, fixed band background, +-6 deg,
phi1 0-28, no cap, no branch mask) as a knot data vector for spray models; proposed fix: phi1 0-25, width cap 1.5, +-0.7 deg
mask around the two-component branch track at phi1 14-22.

## Bugs (both verified by Claude in bin/streams/stream_spline_fit.py)
- Hole inconsistency (Codex, Fable): with the branch hole on, stars are removed but the extended stream integral uses Ag (area
  only, no hole). Fable: for the W6ext_nohole model 97 expected stream stars (18% of the phi1 14-23 stream) lie in the hole.
  No effect on --nohole runs.
- Mocks (Fable; Codex: "--test does not generate Poisson realisations"): simulate() uses curves()["f"] as a probability, but with
  --extended curves() returns the rate (exp of the spline, 20-70 /deg), so every mock star is a stream star.

## Checks (Fable, scratchpad runs, hole fixed in the integral)
- phi1 3-9: excess 122 +- 30 stars (4 sigma) on ~100-160 background per deg; stacked single Gaussian centre -1.51, width 0.97;
  local 6-deg windows width 2.0 +- 0.5 (phi1 4.5), 0.84 +- 0.32 (7.5). The global fit's 2.6-4.5 deg (saved: 4.52 at phi1 8) is a
  knot-placement artefact (width knots 0, 8.3, 12.5; the broad 9-12 overlap, width 1.9 +- 0.5, leaks into 3-9).
- phi1 14-18: two bands of near-equal weight (centres -0.28 / +2.39, widths 0.77 / 0.90); one Gaussian -> centroid +1.5, width
  1.8. Two-component vs one-component (cap 1.5) on the same 4069 stars: Delta lnL 37.5, Delta AIC 51.
- Variants: A (cap 1.5, phi1 0-25): width at the cap everywhere, track +1.66 at phi1 15. B (A + +-0.7 branch mask): still at the
  cap 0-16.5, track +1.33 (15): the strip removes only ~55% of a 0.9-deg branch. C (mask, no cap): width 3.9-4.9 at 3-9 again.
  Residual sd 0.98 does not discriminate (background-dominated pixels).

## Recommendations
- Both: describe the main band only (option a), not the centroid of main + branch; apply the same mask/aperture to particles.
- Fable: replace the global AIC spline by local fits at fixed knots (every 3 deg, +-3 deg windows, Gaussian + fixed band
  background; Koposov+2023 style), branch strip +-1.2 deg at phi1 13-22, knot 13.5 flagged as overlap; widths at 3-9 are limits.
  Acceptance: centres within 0.3 deg of the two-component main track at 15-22; -1.5 +- 0.3 at 3-9; no width at a bound; errors
  from 20 Poisson mocks within 20% of quoted. Against: overlapping windows correlate knots; non-overlapping 3-deg windows instead.
- Codex: fix selection/diagnostic consistency (one exclusion function for stars, integrals, mocks, plots; plot the fit's own
  background, not an independent one); freeze aperture and knots; caps only as sensitivity checks; >= 100 Poisson realisations
  (single band, branch, fan, perturbed background), track AND width bias and coverage; save full provenance (flags) with each
  result. Against: the branch/fan may carry the bar signal, and "main" may be arbitrary; if a core cannot be recovered across
  plausible models, use (b) with explicit aperture moments.
- Codex correction: the saved fit already widens at phi1 24 (1.66 deg), 13 deg at 28.
