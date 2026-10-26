# Sampler design for the GPU fit: Fable and Codex (2026-10-26)

Brief: 9 parameters, deterministic (CRN) but jagged chi2 (+-2 near the best fit, up to +-16 at poor fits), best chi2 ~166 for 25
points, 0.54 s/model on an A100 in batches of 10-20. Literature: S5 (Koposov+2023, arXiv:2211.04495) emcee 250 walkers x 7500 steps;
Chemaly+2026 (arXiv:2601.15373) dynesty dynamic nested sampling, 2000 live points, rslice, ~10 h on 44 cores; HMC failed (multimodal).

## Both
- Frozen jitter changes the target slightly; it costs acceptance / autocorrelation time, not (much) bias, for MCMC. Fable measured on a
  toy 9-D Gaussian with deterministic jitter: emcee stretch acceptance 0.42 -> 0.28, tau 94 -> 112; DE moves 0.23 -> 0.17, tau 32 -> 49;
  means within 0.07 sd, sds within 4%; no stuck walkers.
- dynesty is the poor fit here: live points sit on lucky spikes (Fable toy: 'unif' 76k -> 837k calls, rslice 749k -> 1.1M), and rslice
  does not batch without custom code (Codex: queue_size alone does not batch). nlive 2000 would cost ~20-28 A100-days (Fable estimate).
- Look for separated modes before the main run; a known second basin exists (angle 33, Omega 44, chi2 351; JOURNAL 2026-10-25), weight
  e^-92 at T = 1 (Fable): an optimiser trap, not a posterior mode.
- emcee with DE moves, vectorised half-ensembles of 20 = one batch; checkpoint every step (state + RNG + hashes); restart bit-identical.
- S: sample a unit coordinate for S between 0.8 and S_max(Omega_b, v_c); decide whether the prior is uniform in S (add the Jacobian).
- The sampler does not fix the misfit (chi2/dof ~10): error bars are within-model.

## Disagree (user to decide)
1. Main sampler. Fable: adaptive-tempering SMC (own ~150 lines on gpupipe; 500 particles, resample at ESS N/2, 10 moves per stage in
   B = 20 batches), then a 500-1000-step DE emcee polish; ~0.5-1 A100-day (estimate). Codex: emcee (80% DE + 20% stretch), two
   independent ensembles of 40 walkers x 2.5-7.5k steps (1.25-3.75 days); SMC can lose modes in resampling.
2. Mode test. Fable: the first SMC stages are the test (~8k evaluations, ~1.2 A100-h). Codex: a separate capped pilot of 20k evaluations
   (~3 h): 4k Sobol points, 12k in dispersed local searches, 4k for connecting paths and independent-bank rescoring.
3. Target. Codex: gpupipe uses chi2 without log|C| although the model variances depend on the parameters: include the determinant or
   call it a chi2-based pseudo-posterior. Fable (2026-10-25): chi2 with C_data + C_model as now (not re-addressed today).
4. Temperature. Fable: report both T ~ 10 (error rescaling by reduced chi2; admits angle 15.6-17.5, i.e. "14 vs 18 deg undecided") and
   T = 1 populations, both from one SMC run. Codex: tempered contours are sensitivity contours, not calibrated; neither chi2/25 nor
   sqrt(7) rescaling establishes coverage; first optimisation/profiles + residual plots.

## Flags
- Distance (Fable): the best fit sits at d = 5.67 kpc, 4.8 sigma from 5.43 +- 0.05 (outside the agreed +-4 sigma box); a flat box lets
  it sit there for free, a Gaussian prior costs ~23 in chi2 and moves the optimum. Same question for the PMs. Prior choice = user.
- Codex: the B = 10 identity checks in gpupipe.main record booleans without asserting (they passed in the A100 run).
- Fable: check the jitter's correlation length first (~100 evaluations along a line through the best fit at 1e-4..1 posterior widths).

Sources: Codex (code path, effort high) scratchpad codex_sampler.md; Fable report (toy experiments in scratchpad toy_samplers.py).
