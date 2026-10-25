# Speeding up one GPU model evaluation: Fable and Codex (2026-10-25)

Measured A100 baseline (code 3bf58f7): release 1.43 s + DOP853 spray ~10 s + score 0.86 s ~ 12 s/model.

## Agree
- Keep GPU-only DOP853 (CPU hand-off measured slower).
- Persistent JAX compilation cache: Fable measured 10.4 s cold -> 0.41 s cached per shape (~9 min/session -> tens of s).
- Score: batch the bootstraps (Fable: vmap gives bitwise-identical chi2; est. 0.86 -> 0.3-0.5 s).
- Larger n_att alone does not help (Codex est. 0-15%; Fable replay: no gain).
- Fewer kernels per attempt / a custom per-particle kernel is the only lever on the single-model tail (high effort).
- Tolerance, controller, precision and particle dropping are model changes (user decision).

## Disagree (user to decide)
1. Host synchronisation / copies per call: Codex rank 1 (est. 0-20%); Fable: not a factor -- a cost model t_attempt = 0.43 ms +
   m x 32.5 ns replayed on the measured per-particle finish counts reproduces the measured time (Claude verified: 9.31 s vs 9.5 s).
2. Analytic B-spline gradient: Codex rank 3 (est. 1.2-2x); Fable measured no gain on CPU (autodiff already fused; 102 vs 88 ns per
   particle for the full force, identical results to 1e-12).
3. Batching MCMC walkers: Fable rank 1 (replay: 9.3 -> 5.8 / 4.3 / 3.6 s per model at K = 2 / 4 / 8; Claude's replay 6.1 / 4.9 /
   3.9 s); Codex: improves throughput, not one evaluation's latency, and the current driver cannot simply be vmapped.

## Fable extra (measured on CPU)
- 28% of attempts are rejections (AGAMA's controller); 70 M particle-attempts per model = 2.3 s pure compute; the rest is
  latency (~476 kernels per attempt x ~1 us) in the tail.
- Fused force prototype: 476 -> 364 ops per attempt; GPU effect unmeasured.

## Codex bugs (fixed, 3c53ce4)
spray-check cache shared between methods; silent centre-orbit buffer exhaustion; no attempt limit; _spline2 docstring.
