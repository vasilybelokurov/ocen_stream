# Cross-machine reproducibility of the GPU chi2 (Colab A100 vs CSD3 A100): Fable and Codex (2026-10-29)

Measured: within one machine chi2 is bit-identical (batch vs single, repeat, slot permutation, launch order). The same 20 unit points
(Colab mock-run log, same inputs, bank 7) rescored on CSD3 (job 39589787, x64): 0 of 20 bit-identical; CSD3 - Colab sd ~1.0, max +3.2.
Same size as the known jitter (1e-6 unit offsets: 0.3-2; independent banks: sd 2.5).

## Agreed
- Same seed fixes the random numbers, not the rounding. Any last-bit (ULP) difference entering or inside the particle integration grows
  to O(1) in chi2 (chaotic close passages). Fable, computed and reproduced (bin/streams/reviews/roundoff_amplification_20261029.py):
  1 ULP in the centre-orbit input -> 5672 vs 5697 steps; fp-contract off vs on (1e-16 arithmetic) -> 84% of 3000 particles move
  > 1e-6 kpc, 6% > 1 pc, 0.1% > 0.1 kpc, chi-sign flips, 75% different attempt counts.
- Diagnostic: dump one point's intermediate products at every stage on both machines (parameters, host tables, centre orbit, resampled
  orbit, release ICs, kernel y/chi/natt, observables, summaries, chi2), compare bytes, then cross-feed saved inputs stage by stage; the
  first stage with a ULP difference is the source. Bypass the evaluation cache.
- Production: one frozen environment per run and for all resumes; record an environment fingerprint and refuse cross-environment
  resume (today the run manifest has no environment, and the C build cache key omits the compiler: checked).
- Bitwise portability is possible but costly and fragile: own transcendental functions in spray.c, h*h*h instead of h**3, pinned
  jaxlib/CUDA/NVRTC, XLA autotune off + deterministic ops; Fable ~1 week, slower force evaluation.

## Different
- Likely source: Fable ranks the kernel's libdevice maths (NVRTC version from the cupy pip wheel vs Colab's CUDA) first, then glibc
  (2.28 vs 2.35) in orbit_back, numpy pow dispatch in hermite, XLA autotuning; rules out compiler (-ffp-contract=off, no fast-math),
  driver/ptxas (fmad off: IEEE ops), score reductions and solves (cannot give sd ~1; sky histogram weights are integers). Codex ranks CPU
  libm/compiler in the backward orbit first, then JAX/XLA before integration, then NVRTC/driver.
- Mock recovery: Codex - the Colab mock is evidence for Colab, not CSD3; validate recovery and posterior sensitivity on CSD3 before the
  main run, measuring bias and tails, not only scatter. Fable - still valid (CSD3 is another realisation of the same noise class, which the
  sampler handled); cheap check: rescore ~50 logged points around the mock truth on CSD3 and confirm the minimum moves << posterior width.
