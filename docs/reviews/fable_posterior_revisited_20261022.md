# Fable review: path to a global posterior, revisited after Stage 0 (2026-10-22)

Claude verified: 256k final seed sd per angle 35/25/24/14/30/30 (profile_step2); a 9-D quadratic fitted to the 165 step-2
128k evaluations within Delta lnL < 1000 (55 coefficients) leaves residual sd 48 (~ the 128k noise) -> locally quadratic.
## Verdict
At the calibrated T ~ 2 the posterior's 68% region spans Delta lnL ~ T chi2_9(0.68)/2 ~ 10, below the 256k simulation noise
(sd 14-35): no spray-verified T~2 posterior can exist, and its widths (angle ~0.3-1 deg, Omega_b ~0.1, A ~0.02, q ~0.01; d and
v_c truncated at band edges) are 50-500x smaller than the shifts between model variants (bar model: angle 12.5 -> 27, Omega_b
33.5 -> 44.8, gap 2627; frame change 6000 at fixed params). Full surrogate + PMC at T ~ 2 is infeasible (ESS/N =
exp(-(30/2)^2) ~ 0; a surrogate accurate to ~2 in lnL would need ~22,000 sprays).
## Ranking
1. Laplace covariance from a designed local quadratic regression (~200 points around the best + replicates; 128k: ~1.6 h,
   256k: ~3.4 h on 14 cores), labelled "statistical, within-model" -- for the FINAL model only.
2. Profiles (exist for angle; an Omega_b profile at 12.5 deg ~2.7 h is optional).
3. Generalised posterior with a chosen T_sys ~ 300-1000: computable with the 10-20 design (~200 core-h) but T_sys is a free
   choice and misspecification shifts rather than broadens the optimum: presentation device only.
4. Full surrogate + PMC at T ~ 2: infeasible.
## Which model, when
Neither the inflated-bar 9-D model nor the published-bar model is final. Build the model first (bar-locked / dark-bar component,
Hercules check), then compute the covariance once for the surviving model. The honest "corner" now is a scatter of best-fit
points across model variants (systematic error), with ellipses too small to draw.
## Against
A referee may want a marginal posterior; a tempered T_sys ~ 300-500 posterior would show 17.5 deg as a ~1-sigma neighbour;
rejected because T_sys is arbitrary and the dominant error is a shift between discrete model variants.
