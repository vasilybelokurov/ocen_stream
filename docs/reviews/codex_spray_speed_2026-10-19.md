# Codex review: speeding up the AGAMA spray (2026-10-19)

Codex CLI, effort high, read-only; full source of spray.py, restricted.py, mw_variants.py and make_spray supplied.


The rotating-frame premise is partly wrong: the local AGAMA implementation of Cartesian `Omega=...` still integrates in inertial coordinates and rotates the potential during force evaluation. Do not expect a major gain from changing frames alone. Combining potential components is a separate optimization. [AGAMA orbit.cpp:183](AGAMA: src/orbit.cpp:183), [upstream implementation](https://raw.githubusercontent.com/GalacticDynamics-Oxford/Agama/master/src/orbit.cpp).

**My recommendation: validate `accuracy=1e-6` first, benchmark `dprkn8` second, then investigate combining the host components. Defer satellite truncation and particle reduction.**

1. **Particle accuracy `1e-6`: highest-confidence speed opportunity, approximately 1.9×.** That is `108/57` from the supplied measurements, not an independently reproduced benchmark. The parameter reaches particle integration while progenitor integration remains at `1e-12`, which makes this a relatively isolated experiment. [packet:2018], [spray.py:121](src/ocen_dm/streams/spray.py:121), [spray.py:156](src/ocen_dm/streams/spray.py:156).

   **Risk/test:** numerical errors can systematically favour particular models despite remaining below seed scatter in one example. Compare identical seeds and release epochs at both tolerances across central, boundary, strong-bar and flattened-halo models. Measure paired score changes, changes in model-to-model score differences, and observable distributions. This requires statistical equivalence, not bit identity.

2. **Benchmark `method="dprkn8"`: plausible additional 1.2–1.3×, unmeasured here.** AGAMA documents that approximate improvement; the spray currently leaves `method` unspecified. Its lower-order dense output makes trajectory-derived quantities and final interpolated velocities explicit validation targets. [AGAMA reference.tex:1154](AGAMA: doc/reference.tex:1154), [AGAMA reference.tex:2180](AGAMA: doc/reference.tex:2180), [spray.py:156](src/ocen_dm/streams/spray.py:156).

   **Risk/test:** benchmark it separately at both tolerances against a tighter `dop853` reference, retaining the existing trajectory sampling. Check `chi`, final observables and scores. Do not assume its gain multiplies independently with the tolerance gain.

3. **Combine host components before attempting a frame rewrite: potentially substantial, but no defensible speed multiplier yet.** The production path evaluates an axisymmetric component plus separately scaled positive and negative baryonic components. For constant bar speed and size, a static combined representation wrapped in one rotation can be tested while retaining inertial particle integration and the existing satellite trajectory. Slowing bars and additional LMC/spiral terms need separate treatment. [restricted.py:83](src/ocen_dm/streams/restricted.py:83), [restricted.py:96](src/ocen_dm/streams/restricted.py:96), [restricted.py:106](src/ocen_dm/streams/restricted.py:106), [restricted.py:40](src/ocen_dm/streams/restricted.py:40).

   **Risk/test:** re-expansion introduces force and Hessian errors; Hessians also determine release radii. Compare potential, force and Hessian over the occupied spatial/time domain, including pericentres, the plane and expansion boundaries; then run matched sprays. Preserve the original release calculation initially to isolate integration error. [spray.py:32](src/ocen_dm/streams/spray.py:32), [spray.py:128](src/ocen_dm/streams/spray.py:128).

   If using `Omega`, rotate both positions and inertial velocity components consistently; **do not subtract** `Ω × r` from AGAMA’s velocity convention. Satellite-centre interpolation needs separate care because a centre-position derivative in rotating coordinates differs from those velocity components. [AGAMA orbit.cpp:104](AGAMA: src/orbit.cpp:104), [AGAMA reference.tex:510](AGAMA: doc/reference.tex:510).

4. **Thread/process tuning: expect limited throughput improvement if eight workers already occupy the available compute capacity.** AGAMA parallelizes across orbits; this redistributes the same work rather than reducing force evaluations. [AGAMA interface_python.cpp:7871](AGAMA: src/interface_python.cpp:7871).

   **Risk/test:** compare eight identical sprays using worker/thread combinations `(8,1)`, `(4,2)`, `(2,4)`, `(1,8)`. Measure completed sprays/hour, peak memory and output identity. Single-spray latency is insufficient. Hardware saturation and scaling remain assumptions.

5. **Cheaper satellite treatment: investigate only after a matched ablation measures its cost.** Timing inside `agama.orbit` does not distinguish host force, satellite force, centre interpolation and integration overhead: the particle call receives their combined potential. [spray.py:144](src/ocen_dm/streams/spray.py:144), [spray.py:156](src/ocen_dm/streams/spray.py:156).

   If satellite evaluation occupies fraction `f` of runtime, eliminating that cost has an optimistic ceiling of `1/(1−f)`. **Risk/test:** a distance threshold alone does not establish negligible cumulative deflection or prevent later encounters. Compare several switch radii, retain/re-enable gravity for returning particles, and check scores, stream length, widths and tails. The current calculation retains satellite gravity throughout. [spray.py:144](src/ocen_dm/streams/spray.py:144).

6. **Fewer particles: approximately proportional runtime savings, with a direct statistical cost.** Under ordinary Monte Carlo scaling, halving particles increases scatter by roughly `√2`; bias correction does not remove that variance. The existing jackknife is a bias correction, not a replacement for seed replication. [score_bias.py:75](src/ocen_dm/streams/score_bias.py:75).

   **Risk/test:** establish both bias and variance using independent replicates at 128k, 256k and a larger reference. Preserve the score definition: the wrapper defaults to pooled scoring, while training records also use fixed and legacy chi bins. [grid4_common.py:95](bin/streams/grid4_common.py:95), [training_set.py:189](bin/streams/training_set.py:189). Use fewer particles for screening only until those comparisons justify final inference.

Other proposed changes have little demonstrated upside:

- **Lower `trajsize` is not result-equivalent.** `chi` is a trapezoidal integral over trajectory samples and controls particle selection/binning. Two samples can preserve endpoints while changing the score. [spray.py:160](src/ocen_dm/streams/spray.py:160), [score.py:299](src/ocen_dm/streams/score.py:299).
- **Python vectorization, release-radius tuning and construction-only caching:** approximately **1% maximum benefit in the supplied 108-second benchmark**, assuming its timing attribution is complete. Those operations sit outside the measured 107 seconds in orbit integration. [packet:2018].
- **`dtype`/`der`:** the local AGAMA default already stores trajectories as float32; `der=True` requests additional deviation-vector calculations. The spray supplies neither option. Casting saved arrays to float64 does not recover discarded precision. [AGAMA interface_python.cpp:7524](AGAMA: src/interface_python.cpp:7524), [spray.py:156](src/ocen_dm/streams/spray.py:156), [training_set.py:186](bin/streams/training_set.py:186).

**1. Verified versus assumed**

Verified by reading: parameter propagation, combined host/satellite integration, trajectory-dependent `chi`, AGAMA’s local frame/storage conventions, and existing regression assertions. Subset and chunk tests demand exact array equality; stored-reference tests can skip when their files are absent. [tests/test_spray_subset.py:35](tests/test_spray_subset.py:35), [tests/test_spray_subset.py:66](tests/test_spray_subset.py:66), [tests/test_spray_subset.py:86](tests/test_spray_subset.py:86).

Assumed: the supplied timings and 30–45 score scatter are representative; production uses the inspected AGAMA implementation; eight workers substantially occupy the machine. I did not reproduce benchmarks or run the regression suite.

**2. Strongest argument against my answer**

Relaxed tolerance could introduce parameter-dependent bias that survives seed averaging and shifts inferred parameters. A more expensive implementation effort that preserves forces could therefore be preferable even with a smaller speed gain. The release calculation’s dependence on host Hessians makes careless re-expansion particularly vulnerable too. [spray.py:32](src/ocen_dm/streams/spray.py:32).

**3. What I would check to be sure**

- Freeze initial conditions, seed, release epochs, score settings and production AGAMA build for paired comparisons.
- For purported exact changes, require exact retained arrays and scores. Preserve RNG draw order: initial conditions are generated before subsetting. [spray.py:134](src/ocen_dm/streams/spray.py:134).
- For approximate changes, predeclare a systematic score-error budget substantially below 30–45—for example 5—and bound the paired mean error with uncertainty. Also inspect per-star errors, model rankings, track-bin shifts, widths and tails.
- Establish convergence against tighter integration and denser trajectory sampling; agreement with the current baseline alone does not establish accuracy.
- Keep optimized outputs separate from reference results, with parameter-complete cache keys and source/input hashes. Existing training resumption explicitly rejects incompatible manifests. [training_set.py:155](bin/streams/training_set.py:155), [training_set.py:239](bin/streams/training_set.py:239).