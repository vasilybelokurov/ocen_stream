# Readiness of the sky+PM <-> CMD (dm) iteration: Fable and Codex (2026-10-23)

## Fable (scratch tests on the cached footprint)
- Template vs colour-centring difference is marginal with x2 inflated errors: per-bin (centring - template)/sigma = +1.9, -0.2,
  -0.6, -1.0, +0.8, -0.1 at phi1 4..24 (chi2 5.7/6), and not one-signed.
- LF tilt: not the cause (alpha unconstrained, dm moves <= 0.12). Common colour/reddening shift: not the cause (moves both
  estimators the same way; full-CMD fit pins the stream-template colour offset at 0.00 +- 0.02). Annulus E(B-V)_SFD median 0.132,
  stream 0.065-0.08.
- Background variants swing centring by 0.2-0.25 and the template by 0.1 at phi1 2-18; default 5-14 quadratic least biased.
- Driver of the split: faint stream excess (observed G0 > 18.8) redder than the shifted template by +0.04..+0.07 in every bin;
  centring dominated by G0 19.0-19.6 (14-18: +0.31 / 0.00 / -0.12 / -0.25 for faint limits 18.6 / 19.0 / 19.3 / 19.6); template
  never sees those stars at 10-18 and is itself unstable to its faint edge (0.2 swing). Cause (faint BP/RP bias annulus vs field,
  or background at the faint red end) not disentangled.
- Recommended driver: template fit on the calibrated domain, errors x2, anchored dm(0) = 0, 1-2 parameters, held constant beyond
  the last accepted bin: linear dm = -0.0266 phi1 (chi2 4.8/5) or quadratic 0.0062 phi1 - 0.0016 phi1^2 (2.9/4). Centring line
  gives dm(0) = +0.41 (unphysical); anchored centring chi2 18.4/5.
- Mask tolerance: a 0.3 mag dm error = 0.4-0.8 of the k = 2 MS half-width, 1.3 sigma on the TO/SGB -> one pass ~ fixed point.
- Freeze: template, PM-control scaling, per-bin faint domain edge and dm grid centre (at iteration 0), bands, bins, thresholds,
  nulls, k, g_bright, errors, reddening, anchor/form; fix the phi1 13 sky bin before the loop.
- Tests: stop rule + CMD-pass change < 2% + accepted set unchanged; two starting relations -> same end; nulls through the loop;
  end-to-end injection (bias < 0.1, coverage >= 60% with x2 errors); MS vs TO split within 1.5 sigma; faint-colour check (G-RP
  ridge, null-field slices).
- Against: if the faint red offset is a template artefact, faint-MS values (-0.25 at 12-16) are right and the anchored template
  line is too shallow by 0.1-0.2 at 8-16; mask insensitive at that level, published dm should wait for the faint-colour check.

## Codex (read code + JSON)
- Not ready. Estimators use different magnitude ranges (template faint limit ~18.4 at phi1 20; centring to 19.6) -> compare on
  common support. Centring: first zero crossing, no detection requirement, fixed background in errors. Injection reuses the fitted
  background (does not test background errors). Background has no footprint-area correction for partial bands. Sky upper-branch
  exclusion uses the bin-centre position for a 4-deg bin; sky retains a +-2 deg anchor to the marked track.
- Provisional: centring straight line uncapped (+0.41 - 0.046 phi1) as a selection coordinate only; no quadratic/spline.
- Protocol: freeze the measurement definition (domain and MS/TO split must not move with the incoming line); full-loop
  injections/nulls; stability of selection (two successive iterations within tolerance, widths/borders/accepted set stable,
  multiple starts, detect two-cycles); reproducible outputs with hashes; driver must reject < 3 accepted bins (currently silently
  returns old coefficients).
- Against: centring may be the biased one (at phi1 4 template MS and TO agree at -0.04/-0.12, centring +0.21).
