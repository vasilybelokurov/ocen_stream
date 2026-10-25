# GPU-ready MCMC on a JAX spray with the rotating bar: Fable and Codex (2026-10-25)

Briefing: user request to build a GPU MCMC pipeline on StreaMAX (Chemaly+2026, [arXiv:2601.15373](https://arxiv.org/abs/2601.15373);
github.com/David-Chemaly/StreaMAX v0.4.1, commit 9fd520e) with the Hunter+2024 rotating bar, run on Colab GPUs. Draft: JAX host
(port AGAMA expansions or tabulate), patched StreaMAX integrator, our release model, jit-able 25-point score, CRN ensemble MCMC.

## Corrections to the draft (verified by Claude)
- Bar scaling: AGAMA scale=[A S, S] gives Phi = A Phi0(x/S), force (A/S) F0(x/S) (Plummer check: ratios 2 and 2/3 for A 2, S 3).
  The draft's "A as a multiplier" on the force was wrong.
- Progenitor integrated over 1955.6 Myr (grid4_common.py), releases over the last 1 Gyr, kept debris < 700 Myr.
- Particle sensitivity to orbit accuracy is not chaos (JOURNAL 2026-10-11 correction).
- chi2 in stream_model_compare.py has model-dependent covariance and no log|C| term.
- StreaMAX __init__.py sets jax_enable_x64 = False globally.

## Both agree
- Do not build on StreaMAX: no t in host calls, float32 forced, O(n_steps^2) progenitor arrays, no composite host. Write a small
  JAX module (~300 lines) with our release model; StreaMAX as reference only.
- Host: tabulate the bar POTENTIAL as azimuthal modes Phi_m(R, z) (Fable verified: full - axi = m 2, 4, 6, 8 cosine terms plus a
  0.5% m = 0 residual; keep it), smooth spline, forces by autodiff; rotation and A, S in closed form. Halo: tables over (R, z, q),
  cubic in q, with the existing c(q, vc) normalisation; nothing rebuilt per step. AGAMA-port (option A) only as fallback.
- float64 first; fixed step set by convergence tests.
- Score: port the existing Asimov sky fits and truncated-Gaussian PM fits unchanged (no moments), fixed-count Newton steps,
  validated against the CPU score.
- Sampler: derivative-free ensemble MCMC with a frozen random bank (CRN); no NUTS (hard masks), no pseudo-marginal on a noisy chi2;
  repeat with independent banks.
- Particle number: chi2(N) (1100 at 8k vs 166 at 179k) must be measured first.
- Budget 1e5-1e6 evaluations; Colab needs checkpointing.

## Fable measurements (agama + numpy, scratch scripts; JAX not installed locally)
- agama full host force: 0.94 us/point single thread; agama.orbit 1e-8 ~6 ms/particle (180k ~ 18 CPU-min).
- Bar tables 256 x 128 x 5 modes (1.3 MB): force error 1.2e-6 (99%) of total force; axi 512 x 256: 2e-7; q-stack cubic: 5e-8.
- Integrators vs agama 1e-10 (400 particles, |dv| 90% / 99% km/s): agama 1e-8 2e-3 / 1.3; KDK dt 0.02 Myr 3e-3 / 2.4; Yoshida-4
  dt 0.2 Myr 2.8e-3 / 58; float32 floor 0.03-0.06 / 20-40. The 1-2% tail exists between two agama tolerances.
- Estimate (unmeasured) 1-3 s/model float64 on an A100.

## Disagreements (for the user)
1. Likelihood normalisation: Codex -- with model-dependent C use r^T C^-1 r + log|C|, calibrate the covariance (sky c-s
   correlation, full model covariance), define the prior explicitly; Fable -- chi2 with C_data + C_model is right, as now.
2. Integrator: Fable -- Yoshida-4, dt 0.2 Myr (measured at the agama floor for 90% at 1/3 of leapfrog's cost); Codex -- KDK
   leapfrog, step by halving.
3. Validation standard: Fable -- particle-level agreement is unattainable, validate at the chi2 level against seed noise;
   Codex -- matched-variates particle tests, then summaries within 0.1 sigma_data and |dlnL| < 0.1.

## Raised by one only
- Fable: chi2 165 / 25 means the model is rejected; the posterior will be driven by the width misfits and too narrow (~sqrt 7);
  a modelling decision, not solved by GPUs.
- Codex: model PM fits use intrinsic PMs while masks use Gaia-noised PMs (verified in stream_model_compare.py pm()); evaluating
  local model fits at data spline knots is not the same estimator as the global spline (needs injection tests); fit_knot_nm.py
  resume policy can mix incompatible evaluations (reuses run files by ID, no input hashes).
