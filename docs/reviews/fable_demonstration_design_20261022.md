# Fable: how to demonstrate (1) no-bar fails, (2) the bar must rotate, (3) the bar is non-standard (2026-10-22)

Read-only review. Numbers spot-checked by Claude against results/streams/fixedbar_20261021/result.json
(from25 -23859 at Omega_b 33.4; from25_om27 -27138 at 30; from25_om42 -23225 at 44.8) and results/plot_data/paper_models.json
(old-frame axisymmetric variants -25450 to -26455).

## Exists vs missing
- Current setup: inflated 9-D profile 12.5 deg -20598 (se 17), 25 deg -21669; published bar at 27 deg, Omega_b free: -23225 (44.8), -23859 (33.4).
- No-bar: only old frame/score (S2, d free to 5.95, 32k): Hunter axi refit -3541 +- 196 vs 16-deg bar; McMillan17 and q/disc/v_c
  variants 3300-4300 below. Not quotable in the current setup.
- Static bar: never optimised; fixed-parameter excess test only (static ph90 recovers 0.63 at 25 deg, 0.25 at 12.5).
- Published bar at low angle: never refit (unrefit -3808 at the 12.5 best).
- Mock T2c0 (truth = published bar, 27 deg): free 9-D fit returns A 0.90, S 1.16 (A S^3 = 1.4) and lands 303 below the truth check.

## A. Logic
Common rule: same catalogue, box, score, priors, frame and optimiser (profile_nm.NM, CRN, 40/chain, 64k search, 4 x 256k finals),
>= 4 chains incl. starts at the barred best's nuisances. Quote Delta lnL + prior (256k, 4 seeds), star-bootstrap sd, T ~ 2,
segment split, and nuisance values side by side.
1. Axisymmetric: best no-bar Hunter host (d, PMs, v_c, q, disc_scale free) and McMillan17 (d, PMs free) vs inflated 12.5.
   Visual: data vs model tracks; cumulative Delta lnL vs phi1; flat L_z(t). Referee: optimised as hard; other potentials; LMC
   (force ratio ~0.01 vs 0.10 demanded).
2. Rotating: static bar (Omega_b = 0; orientation, A, S, nuisances free) vs rotating best; plus published-bar Omega_b sensitivity
   and the excess-pattern test. Risk: if an optimised static pattern recovers ~90%, the claim becomes "a specific phase history".
3. Non-standard: published bar (A = S = 1), Omega_b and nuisances free: gap 2627 at 27 deg (exists), 12.5/17.5 deg (new);
   force-ratio profile; mu_alpha* residual at phi1 7-19; mock T2c0 shows the fit does not inflate a standard bar.

## B. Figures
- F1 data vs three best models (no-bar [new], published 27 deg Omega_b 44.8, inflated 12.5): Delta phi2, mu_alpha*, mu_delta, v_los.
- F2 residuals + cumulative Delta lnL vs phi1.
- F3 likelihood ladder (all variants, error = star-bootstrap sd, segment-stacked; reference bars: optimisation floor ~300, T2c0 gap).
- F4 what the bar does: (a) force ratio vs R; (b) L_z(t): no-bar, published, fitted; (c) lnL vs Omega_b (0 [new], 33-35, 44.8).
- F5 (optional) mock calibration.

## C. New runs (8 workers)
- P1 no-bar fit: fixedbar_fit-style with amp 0 (barred branch so axi_variant is honoured), 7-D incl. disc_scale; 4 chains ~5.4 h;
  McMillan17 3-D in 2 extra chains.
- P2 published bar at 12.5 and 17.5 deg, 2 chains each, ~5.4 h (P1 + P2 together ~8-9 h).
- P3 static bar 8-D, 4 chains, ~5.4 h (optional Omega_b 10, 20: +5 h).
- P4 mock calibration on T2c0 with the fixedbar fit, ~3 h (+ ~5 h no-bar-truth mock, lower priority).
- Cheap: per-star re-scores for bootstrap sd and segments; LMC on/off pair at 12.5 best.

## D. Wording (pending numbers in brackets)
1. Best axisymmetric MW (d, PMs, v_c, q, disc mass free) fits worse than the best barred model by [P1], against sd ~80 and floor ~300;
   deficit in the PM turn at phi1 > 17.
2. Static bar (orientation, strength free) worse by [P3]; static copy of the required force recovers <= 63%, co-rotating 100%;
   published-bar fit changes by 3900 between Omega_b 30 and 45.
3. Published bar, Omega_b 20-50 and nuisances free: worse by 2627 at 27 deg [and P2 at 12.5-17.5], needs Omega_b 44.8; required
   force 2.5-3.5x published at 4-7 kpc; on a published-bar mock the same fit returns A S^3 = 1.4.

## Against the plan
~16 h on null fits whose sign is predictable. Fable still recommends P1 and P2 (matched no-bar fit was demanded by the 10-16 Codex
review; old-frame number not quotable); P3 droppable with claim 2 softened to "co-rotating (fixed-parameter test)".
