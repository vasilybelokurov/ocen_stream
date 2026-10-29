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

## Outcome (2026-10-29, after the review)
- Cheap check (CSD3 job 39590991): the Colab beta = 1 mock population (300 particles) rescored on CSD3: d chi2 mean 0.05, sd 0.49;
  reweighting by the CSD3 chi2 moves every posterior mean by <= 0.03 sd and every width by <= 1.1% (ESS 283 of 300).
- Stage diagnostic (ocen_stream.xdump; self-feed on CSD3 bit-identical at every stage): Colab (glibc 2.39, numpy 2.1.3, gcc 13.3,
  NVRTC 13.0, driver 13.0) vs CSD3 (glibc 2.28, numpy 2.5.4, gcc 8.5, NVRTC 12.9, driver 13.2). Each stage re-run on Colab from CSD3's
  saved inputs differs: parameter conversion (1 element, 5e-15); C centre orbit (5630 vs 5590 steps); Hermite resampling (1e-14);
  release ICs (JAX on GPU, 4.6e-11); the CUDA kernel on identical ICs and tables (particle positions differ by up to 370 kpc, 77% of
  attempt counts); observables (1e-14); score (1e-7 relative). Host tables and today's centre state bit-identical. Both reviewers' top
  candidates are sources; the kernel alone (NVRTC/libdevice version) suffices for O(1) chi2 changes. Colab reproduced its own chi2 for
  the point after two days (34.47018015502194).
