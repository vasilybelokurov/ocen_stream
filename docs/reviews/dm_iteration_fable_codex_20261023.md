# dm(phi1) refit by iteration: Fable and Codex (2026-10-23)

Briefing (Claude): masks from stream_masks.py (CMD-filtered, cross-selected PM/sky fits); dm from MS colour centring in 3 bins
(-0.32/-0.53/-0.75 at phi1 16/20/24) with the old hand-marked windows; proposed plan: refit dm with converged masks, 2-D CMD
likelihood on a fixed observed-G domain, more bins, iterate dm <-> masks to convergence, injection and null tests.

## Codex (read code + JSON; no runs)
- Iteration is circular in principle: PM/sky masks are trained on cmd_pass, which depends on dm; dropping the CMD cut from the dm
  selection does not break that. Numerical convergence does not establish unbiased dm.
- Current dm estimator: colour window and template-G limit move with trial dm; first zero crossing returned; bootstrap keeps the
  red-control normalisation fixed and drops failed roots (nanstd).
- Bug (confirmed): displayed PM background integrated over +-3 deg while the on-window uses SKYH(u) (2.0-2.7 at phi1 13-23).
  Display only.
- Not converged (confirmed): pmra track at phi1 9 moved 0.32 mas/yr in the last of 6 iterations; no non-convergence flag.
- dm off-sky control (4-12 deg) does not exclude the upper branch.
- fit_pm: truncated data, untruncated Gaussian; width floored after fit; 1-parameter errors; sky significance excess/sqrt(B).
  Thresholds are heuristic scores.
- "95%" borders are conditional on the other masks, not population widths.
- Reproducibility: CMD-pass cache keyed on JSON mtime; parameter JSON overwritten before the dm step.
- Recommended: freeze the analysis; normalised on/off CMD mixture in (c0, G0) on a fixed domain; MS + TO/SGB jointly with
  separate diagnostics; non-overlapping bins, keep weak bins; dm near the cluster unconstrained (cap = assumption; spray prior
  compromises independence); conservative faint-BP limit, compare G-RP; joint model later. Counter-argument: freezing/holding out
  loses signal; end-to-end injections should decide.

## Fable (quick checks on cached data, scratch scripts)
- dm is insensitive to the masks: old vs converged masks 14-18 -0.32 -> -0.29, 18-22 -0.53 -> -0.51, 22-26 -0.75 -> -0.79.
  Sky-centre jumps irrelevant (99% of on-sample unchanged). CMD-pass sample changes 1-5% for dm +-0.1. One extra pass suffices.
- Background normalisation (red-control ratio) is the main systematic: PM-shifted nulls give CMD-pass excesses of -30% to +50% of
  B (e.g. +76 on 160 at 18-22, "6 sigma"). Detections at phi1 4-12 and 24-32 not secure; near-cluster excess (50-80% of B)
  probably real.
- MS centring has no detection test: on nulls it returns dm -0.6..-1.4 +- 0.2-0.4; 26-30 (-1.14) is noise, 24-28 marginal.
- Colour lever arm 1/(dc/dG) = 6-14: 0.01 mag colour systematic = 0.06-0.14 mag in dm. Faint split: G 19-19.6 gives more
  negative dm than G < 19 in all bins; restricting to the calibrated template range moves dm by +0.07..+0.11.
- Injection bias of MS centring <= 0.09 (sky-shifted null), up to 0.17 (PM-shifted null). Hess-template / moving-domain
  methods biased.
- Near cluster: dm at phi1 2-6 = +0.03 [-0.12, +0.10] (null gives -1.2), so dm ~ 0 there is measured; 6-14 unconstrained.
- pmra at phi1 7-17 is the anchored hand track (bins 1-3.9 sigma, not accepted).
- Recommended: 2-D unbinned CMD likelihood (CMDMask template + LF weight, per-star errors, sigma_dm), field = off-sky KDE with free
  amplitude, fixed observed domain (G < 19.6, 0.3 < c0 < 1.3, G0 > 16.3); full / MS-only / TO-SGB-only per bin; 4-deg bins
  step 2 at 10-28 + one bin 0-8; gate on amplitude >= 3 sigma and null < 2 sigma; systematics budget (reddening scale, R, template
  range, [Fe/H] -1.9/-1.5); one mask re-run + one dm refit; then the joint mixture. Counter-argument: TO/SGB-only gives +-0.1-0.2,
  so pragmatic path = MS centring + gate + nulls + calibrated range, then go straight to the joint mixture.
