# Mask track smoothing (spline clip) and orbit-start masks: Fable and Codex (2026-10-23)

Question: is clipping the per-iteration offset spline (smooth_offset) to the accepted offsets' range a correct fix, and should the
orbit-start (McMillan17 orbit, no anchor) masks be adopted as standard?

## Fable (scratch reruns of stream_masks.py)
- Clip = band-aid. Root defect: LSQ cubic spline (6-deg knots) evaluated across gaps between accepted bins. Default run: pmra bins
  at phi1 7-17 never accepted (1-3.9 sigma); spline bridges 5 -> 19 and oscillates (applied offset at phi1 11 per iteration
  -0.41, +1.14, -1.68, +0.01, -0.75, -0.49), hitting the +-1.5 anchor at 9-14. The current default pmra track at 7-17 is
  spline overshoot stopped by the anchor. Same in the sky: a lone accepted dphi2 bin at phi1 3-5 gives spline offsets up to
  -99/+11 deg (anchor-limited). Ill-conditioned even between adjacent accepted bins (+58.9 mas/yr at phi1 3).
- Clip fills gaps with the extreme accepted offset (non-local). Orbit-start clipped run: pmra at 9/13/15 = -5.83/-7.06/-7.82 vs
  orbit -7.25/-8.98/-9.88 (it did not stay on the orbit; it was pushed by the clip).
- Pooled unbinned pmra fit over phi1 7-17, ridge residual from the final track: current default +1.29 +- 0.19 (default centre
  1.3 mas/yr below the ridge); default+clip -0.12; default+linear -0.19; default+hold 0.00 (the marked start was on the ridge);
  orbit+clip +0.16 (slope +0.31/deg); orbit+linear/hold +0.44/+1.15.
- Recommended: (1) multi-scale acceptance: failed bins refitted on +-3 then +-5 deg windows (slope term); (2) bridge gaps > 4 deg
  linearly, drop the clip; (3) pooled windows only inside the single-bin-accepted range (otherwise phi1 29 drifts); (4) sky fits
  at phi1 >= 13 only; (5) guard: |spline offset| <= 2 x max accepted |mu|. Default + pooled + linear reproduces the current
  default outside 7-17 to <= 0.2 (0.33 at 27) and puts 7-17 on the ridge (+0.03 +- 0.21).
- Do not adopt orbit-start clipped masks (pmra 7-17 set by the clip; FREE settings change the fits, e.g. pmdec widths 0.66-0.74
  vs 0.33-0.46 at 5-13). It does establish: pmdec and pmra at -1..5 and 19-27 agree with the default (data-driven).
- Tests: regression vs current default; ridge residual < 0.3 at 7-17; start test with identical settings (< 0.5); null with a
  +3 mas/yr displaced track; end-bin stability; guard never fires.
- Against: the 7-17 ridge is only 4-7 sigma in 6-10 deg windows; starts still differ by 1.0 at phi1 9; a wide mask with no
  claimed centre may be the honest product there.

## Codex (read code + JSON)
- Clip = numerical guard, not a fix; plateaus across unsupported intervals; default movement can be iteration feedback.
- Do not adopt yet: not converged (pmra phi1 27 moved 0.28 in the last iteration); stopping rule uses per-bin offsets, not track
  changes; fit_pm truncation/normalisation, post-hoc floor, conditional errors; borders computed against pre-update offsets.
- Some "strong" near-cluster pmra bins fail cuts (phi1 -1 width 1.73 > cap; phi1 3 at 3.9 sigma, floor width).
- Recommended: gap-aware linear interpolation (bridge <= 6 deg, taper at segment ends), same for widths; then a normalised,
  penalised, pooled unbinned spline likelihood across phi1 including weak bins. Convergence: two iterations < 0.05 mas/yr,
  0.1 deg, 5% width. Version outputs with code/input hashes.
