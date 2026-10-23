# Mask borders for the Gaia cross-selection: Fable and Codex (2026-10-23)

Question: a robust, reproducible, minimally circular way to set the borders (centre, half-width vs phi1) of the dphi2, pmra*, pmdec
masks (bin/streams/gaia_xsel_marked.py), and whether the user's two candidate pmdec borders (RED narrow; GREEN broad, fanning
envelope) are supported. Claude re-ran Fable's check script (numbers reproduced).

## Fable (with quick measurements on the cache)
- pmdec (sky-lower + pmra masks; 1 Gaussian deconvolved with per-star errors + empirical off-sky background): intrinsic width
  <= 0.1 mas/yr at phi1 14-18, 0.41 +- 0.06 at 18-22, 0.50 +- 0.10 at 20-24, 0.8 +- 0.3 at 24-28, unconstrained at 26-30; same for
  G < 18.5 and 18.5-20 (error model fine). Growth exceeds the distance gradient -> kinematic fanning.
- The by-eye tracks are biased: pmdec centre +0.3..+0.6 mas/yr above the marked track at 16-22 (fit -9.26 +- 0.08 at 18-22 vs
  track -9.63); pmra ~1 mas/yr too negative at 22-26 (+1.25 +- 0.21 at 22-24); sky centre +0.7 -> -0.6 deg across 14-24 (s 0.6-1.0 deg).
- RED upper border cuts the stream at 18-24 (band between red and green upper borders: +44, 3.5 s; +54, 4.4 s). No second, lower-pmdec
  component: dBIC(1 vs 2) -24..-13 (1 preferred everywhere); green-minus-red below holds -1, -1, -4 stars at 14-24 and +17, +12 at
  24-30 = tail of one broadening component.
- Fixed +-1 mas/yr is ~5 sigma at 14-18 and ~1 sigma at 24-28. Use centre +- k sigma_tot(phi1, G), sigma_tot^2 = s_int^2 + e_i^2;
  S/N-optimal k = 1.4 (k 1.2-1.8 within 2.5 per cent); figure k = 1.5-2; counts from mixture amplitudes, not masks.
- Procedure: alternating 1-D unbinned fits per adaptive phi1 bin (>= 60 excess stars), empirical background from the unselected
  dimension, local slope, error deconvolution, wide masks (k = 2.5) while iterating, penalised-spline smoothing of centre and
  log-width (GCV), width floor, no extrapolation, ~5 iterations; init from the 18-22 VPD peak, step outwards. 3-D mixture in the
  cleanest bins as a cross-check. Null test already shows spurious N up to 44 +- 26 with zero-width spikes -> floor + regularisation
  mandatory; 24-30 detections only ~2 sigma above the spurious level.
- Validation: injection-recovery into off-stream bands (30 realisations), null bands, stability to bands, bin widths, k, init.
- Against: the 1-D scheme inherits the other masks (pmra off by 1 at 22-26 may bias pmdec there); a joint 3-D mixture avoids that
  at the cost of a parametric field model.

## Codex (code review + method)
- Verified flaws in the current scripts: (1) PM backgrounds in the cross-selection scale off-sky histograms by the ratio of ALL
  stars on/off (not area) -- assumes the same field PM distribution on/off; (2) sky-panel background fits ignore footprint exposure
  (edges, masks); (3) upper branch not excluded from the sky-panel background bands; (4) marked masks have constant widths, ad hoc
  spline weights/padding and ignore per-star errors; (5) cross-selection does not remove feedback (masks drawn on an earlier
  cross-selected figure); (6) displayed significance ignores background uncertainty; (7) the frame/footprint is member-informed
  (chord frame from the members' ridge).
- Method: joint, error-convolved Poisson point-process mixture on a fixed parent sample, integrated over the accessible region,
  with smooth (penalised-spline) tracks, log-widths, amplitudes; EM with soft responsibilities for all stars (hard alternating masks
  cannot recover excluded wings); compare one broadening component vs two tracks vs non-Gaussian wings by held-out prediction and
  simulations repeating the whole search (not BIC alone); borders by a declared objective (completeness mask for science, S/N mask
  for display); 3 marginal 95 per cent masks give ~86 per cent joint completeness; for two tracks use the union of component-consistent
  selections, not a broad envelope. Freeze the experiment and withhold member labels/CMD until masks are frozen.
- Against: at these S/N a flexible field model can absorb a broad component and a rigid one can invent one; if defensible
  backgrounds give different borders, report an uncertain border.
