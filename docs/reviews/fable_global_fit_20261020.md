# Fable review: global fitting design for stream 54 (2026-10-20)

Read-only review of the pipeline (profile_nm, search_wave, emulator, score) and the journal 2026-10-17..20.
Claude verified: stream_batch.py calls make_spray without a frame (baumgardt default) and records frame="baumgardt";
emulator_design.load rejects runs with nrel != 256000; ~2729 stored run records. Journal seed scatter at 256k is sd 45-240
(2026-10-16/17), wider than the 45-65 quoted below.

## Diagnosis
- profile_nm gives a profile of local optima (start-dependent: 27.5 deg -29970 local vs -23028 from the 12.5 family; T2c0
  best 303 short of the truth); no posterior, no marginalisation, finals are lower bounds of the profile.
- Families (A~1.8/S~0.9 vs A~1.2/S~1.4/q~0.8), Omega-vc and angle-d degeneracies; edge pile-ups (d, vc, q) are
  misspecification absorbed by nuisances, not a sampler problem.
- Noise: sd ~300 at 32k, 60-90 at 128k, 45-65 (journal: 45-240) at 256k; correlated across stars; CRN reliable >= 0.05 unit.
- Emulator: constant-mean Matern GP; random K-fold CV cannot detect unsampled ridges; T2 missed by 16 deg; maxima optimistic
  by 430-1900. load() is 256k-only; stream_batch is baumgardt-only.
- Overconfidence: block/star bootstrap ratio 400-700/65-92; block bootstrap confounds signal heterogeneity along the stream
  with dependence, so 30-70 is not yet a calibrated T.
## Design
- Target: generalised (Gibbs) posterior pi_T ~ prior x exp(lnL_jk/T); T = 1 is not computable (sim noise, GP error);
  T calibrated (sandwich T = J/H from block bootstrap of seed-averaged per-star vectors minus sim noise; Fable's rough
  estimate T ~ 100-200), checked on mocks; report at T_cal/3, T_cal, 3T_cal.
- GP surrogate used only as a proposal; posterior verified on sprays by importance weights exp[(lnL_sim - mu_GP)/T] at 256k,
  iterated as population Monte Carlo until ESS/N >= 0.3 (needs sigma_GP <~ T in the posterior region).
- Multi-fidelity as heteroskedastic noise (32k/128k/256k observe the same jackknife-corrected function; learned 32k offset);
  history-matching waves at 32k (implausibility cut I < 3 + cT; 70% NROY, 20% global Sobol never excluded, 10% held out);
  independent seeds for surrogate data, CRN only for NM polishing.
- Multimodality: global 32k Sobol wave; NM polishing per stratum + transferred family; parallel tempering on the surrogate.
- Sampler in logit(unit) coordinates (edges). Validation: frozen rules, 4 well-specified + 2 misspecified mocks, coverage.
## Budget (14 cores)
Stage 0 (no sprays): T calibration on T2c0 mock catalogues, 32k->256k jackknife offset; ~3 h re-scoring.
Stage 1 (method on T2c0): ~1230 stored evaluations + 2 x 100 x 256k = 50 core-h (~4 h wall).
Stage 2 (real, step2): 32k wave 256 + 3 NROY waves + PMC 4 x 150 x 256k ~ 200 core-h (~15 h wall).
Stage 3 (coverage): 450-650 core-h (~1.5-2 days). Reuse: ~765 step2/gravity21 runs enter the real-data GP directly.
## Decision points / risks
Stage 0: if the sandwich T on T2c0 is not ~1, block bootstrap cannot calibrate T. Stage 1: if ESS/N < 0.1 after 3 rounds,
switch to per-stratum GPs or a 128k noisy-MH run (~3.5 days). Stage 2: edge pile-up expected -> report conditional on the
band; then inner-potential variants as extra dimensions. Risks: surrogate accuracy vs T; T is a calibration choice;
undiscovered ridges.
## Against its own recommendation
The posterior will probably say what the profile says (angle broad at T ~ 100; Omega 31.5-34, A >~ 1.1). A profile with
sandwich-corrected intervals plus 2-3 more angles and multi-start polishing costs ~10% and is defensible; the posterior is
worth it only for joint (Omega, A, S, q) constraints or to remove local-optimum fragility by design.
