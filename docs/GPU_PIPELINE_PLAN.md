# GPU MCMC pipeline: stage plan (2026-10-25, for approval)

Goal: sample the 9-D posterior of the bar + halo + cluster parameters (bar angle, Omega_b, amplitude A, size S, halo q, v_c,
distance, cluster PMs) against the 25-point knot data vector, with every spray and score on the GPU (Colab A100 or CSD3).
Starting point (tested 2026-10-25): JAX Yoshida-4 spray in the tabulated host, dt 0.2 Myr, float64: 3.7 s per 179k-particle spray
on an A100; paired dchi2 +0.66 vs AGAMA (noise floor ~1). Fixed parameters, inputs exported from AGAMA, score still on CPU (~5 s).

## Stage 1 -- Host for any parameter point (no table rebuilds)
- Tables (exported once from AGAMA): axisymmetric baryons Phi_b(R, z); bar term B_m(R, z) = modes of [baryon_full - baryon_axi] at
  A = S = 1; halo Phi_h(R, z; q) on a q stack (0.65-1.25, step 0.05; cubic in q) with its v_c^2(R0; q).
- Host at parameters: Phi = Phi_b + c(q, v_c) Phi_h(q) + A * sum_m B_m(R/S, z/S) cos(m (phi - theta(t))), theta(t) = angle +
  Omega_b (t - T); c(q, v_c) = (v_c^2 - v_b^2(R0)) / v_h^2(R0; q) as mw_variants.axisymmetric_variant.
- Tests: force vs AGAMA host_potential at 10 random parameter points across the box (incl. q between stack nodes, S at its
  ceiling): 99% of points < 1e-5 relative, max < 1e-4.

## Stage 2 -- Whole spray in JAX (release included)
- Present-day 6-D point from (distance, PMs) in the gravity21 frame (port frames.to_model: fixed rotation + solar motion).
- Centre orbit back and forward (Yoshida-4, small step, or Dopri8 1e-12 for this single orbit); tidal radius r_J from the
  Hessian of the host (jax.hessian) and the frozen M(<r); Fardal release offsets from a FROZEN random bank (spray.release_ic draws,
  exported once); particles of the trailing arm < 700 Myr; Yoshida-4 dt 0.2.
- Chi (Gibbons+2014 ordering, used for the selection): accumulated during the integration (sum of (|r| - |r_prog|) dt).
- Tests: at the final_c0_2 point, initial conditions vs AGAMA's (same draws) < 1e-6 kpc / 1e-6 km/s; chi sign agreement > 99.9%;
  paired dchi2 vs final_c0_2 within 1 (seed noise 1.4-4.8); repeat at 3 other stored points (start points of the fit chains).

## Stage 3 -- Score on the GPU
- Port stream_model_compare.measure: projection to the stream frame (project_gc), Gaia PM error draws (frozen bank), frozen PM/sky
  masks, Asimov sky fits per window and truncated-Gaussian PM fits per knot as fixed-count Newton iterations from the data values
  (no Nelder-Mead), 8 particle bootstraps -> model errors; chi2 (+ log|C| if adopted).
- Tests: on >= 20 stored sprays spanning chi2 165-1000: |chi2_gpu - chi2_cpu| < 0.5 and every summary within 0.05 sigma_data;
  missing-window handling identical (1e3 penalty cases reproduced). Target < 0.5 s per model.

## Stage 4 -- Throughput
- Age bucketing for Yoshida (particles grouped by age so young particles do not ride along with the oldest): expected ~2x.
- Batch several models per call where the GPU is not saturated (it saturates at ~50k particles, so gains are expected mainly
  for smaller N); measure s/model at N = 90k and 179k.
- Particle number: fixed N for every model (chi2 converged from ~170k at the best point: 164 +- 3 at 173k vs 168 at 694k).
- Test: s/model and chi2 unchanged (paired) with bucketing.

## Stage 5 -- Sampler
- Ensemble sampler (affine-invariant stretch move, emcee algorithm) with a JAX likelihood; one frozen random bank shared by all
  walkers (common random numbers); explicit priors (box, uniform in physical parameters unless decided otherwise); checkpoint
  walkers + sampler RNG + bank hash + input hashes to Drive every few minutes; restart reproduces the chain bit-for-bit.
- Tests: restart equivalence; sampler recovers a known 9-D Gaussian (analytic likelihood) before using the stream likelihood.

## Stage 6 -- Validation before interpreting a posterior
- Mock recovery: >= 10 mock data vectors from sprays at known parameters (independent bank, data covariances), full MCMC on each;
  68 / 95% coverage per parameter.
- Repeat the real-data chain with 2 independent random banks: posterior medians within 0.3 sigma.
- Posterior predictive: data | model | residual maps at posterior draws; chain diagnostics (integrated autocorrelation time,
  > 50 tau per walker).

## Decisions needed from the user
1. Box ranges: the free fit sat at edges (angle ~12-14, lowest sampled; v_c 220 floor; q 0.75-0.8). Proposed: angle 0-50,
   Omega_b 20-50, A 0.6-2.0, S 0.8-ceiling (corotation), q 0.65-1.25, v_c 200-250, distance 5.28-5.68 (+-3 sigma), PMs +-4 sigma.
2. Likelihood: chi2 as now, or chi2 + log|C| (makes no difference at fixed large N in today's test; formally correct).
3. Compute: 1e5 evaluations x ~2-4 s = 2.5-5 A100-days. Colab Pro (100 units/month) covers ~7-18 A100-hours; the full run needs
   Colab Pro+ / extra units (~500-1500 units) or Cambridge CSD3 (Wilkes-3 A100s).

## Estimated effort (to tested code)
Stage 1: 0.5 day; 2: 1 day; 3: 1-1.5 days; 4: 0.5 day; 5: 0.5-1 day; 6: depends on compute (mocks = 10 MCMC runs).
