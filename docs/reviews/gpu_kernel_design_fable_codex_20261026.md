# Custom GPU kernel for the DOP853 spray: designs by Fable and Codex (2026-10-26)

Baseline (A100, single model, 179,200 particles, tol 1e-6): release 1.42 s + integration 5.82 s + score 0.66 s = 7.91 s, deterministic.

## Both
- CUDA C++ through CuPy RawModule (NVRTC), sm_80, no fast math, fixed compile flags (pinned binary); JAX interop by DLPack, zero copy;
  JAX FFI custom call only if it becomes permanent. Triton / Pallas rejected; Numba second choice.
- One particle per thread, float64 state in registers, stages unrolled with constant indices; read-only global tables (no float64
  textures); sort particles by predicted cost.
- No floating-point atomics; each lane's arithmetic independent of the batch -> per-model results bit-identical for any batch.
- Analytic spline derivatives (Fable verified on CPU: |da|/|a| max 5.3e-12, median 3.3e-14 vs jax.grad); port AGAMA's controller exactly,
  incl. sk == 0 skip and three square roots (JAX version aligned, e42c19d).
- Bit-identity with the JAX version is impossible (different rounding; core passages chaotic): validate by chi2 + bitwise
  batch/repeat tests.
- Estimates (unmeasured): integration 5.8 s -> 0.4-2.0 s (Codex) / 0.7-1.0 s (Fable); then release + score (~2.1 s) dominate.

## Fable measurements (C prototype, dev/kernel/kern.c + check.py; Claude reproduced)
- 1.9 us per attempt, 148 ns per force evaluation, one CPU core (JAX on CPU ~0.25 ms per attempt).
- 256 particles incl. the 32 slowest vs JAX dop853_chi: |dx| median 1.5e-4 kpc, chi sign agreement 98.8-99.2% (JAX GPU vs JAX CPU: 98.8%).
- Sorting by cost gives 99.4% warp efficiency even without lane refill (file order 63%).

## Disagree (user to decide)
1. Path: Codex -- go straight to the CUDA kernel. Fable -- first a C library on the CPU for the centre orbit (1.42 s -> ~10 ms) and the
   straggler tail (~0.1 s), ~2 days, integration ~2 s; it is ~80% of the kernel's source and all of its validation; then decide on CUDA
   with measured numbers.
2. Scheduling: Fable -- persistent kernel with lane refill (warp-aggregated atomic work queue) from the start; Codex -- start with one
   launch, one particle per thread, cost-sorted; add a scheduler only after profiling.
3. Validation: Codex -- particle-level gates for regular orbits (1e-5 kpc, 1e-3 km/s) and every outlier inspected; Fable -- statistical
   (attempt counts, |dchi| median, chi-sign agreement within the platform noise) plus chi2 at the 5 points.
