# Codex review: stochastic-kriging emulator and refinement design (2026-10-18)

Codex CLI, effort high, read-only. All six findings confirmed and fixed in emulator_design.py (stream_batch.py fixes after the replicate batch ends).


The packet understates the replicate campaign: the current default is **64 points × 2 additional runs**, and the existing job list contains 128 jobs. Check the remaining workload before relying on the five-hour estimate. [stream_batch.py:45](bin/streams/stream_batch.py:45), [jobs.json:1](results/streams/training_9d_reps_20261018/jobs.json:1)

**I would fix candidate uniqueness before launching. I would use this GP to guide further simulations, but would not yet use it as a posterior likelihood.**

**Ranked findings**

1. **High: the batch can contain unintended duplicate points.** Selected candidates remain eligible for every subsequent `argmax`. Positive fantasy noise leaves residual uncertainty at selected points; reducing their uncertainty does not guarantee they stop winning. Duplicate indices also propagate into the “eight highest-mu points,” potentially concentrating their extra replicates further. Mask selected candidates explicitly and assert the intended number of distinct coordinates before writing jobs. [emulator_design.py:91–103](bin/streams/emulator_design.py:91)

2. **High: loading results bypasses provenance and duplicate protection.** The loader accepts matching JSON files without checking manifests, score settings, particle counts, job identities, or artifact hashes. Supplying the same batch twice—or explicitly supplying the automatically included replicate directory—counts identical simulations as independent replicates. Grouping then trusts the key and silently takes its first coordinate. This can corrupt both `S2` and `s2/r`. Validate compatibility, deduplicate simulation identities, and reject inconsistent coordinates within a key. [emulator_design.py:41–55](bin/streams/emulator_design.py:41), [emulator.py:25–34](src/ocen_dm/streams/emulator.py:25)

3. **High for inference: validation does not hold out the complete fitted pipeline.** Once replicate fitting activates, the noise surface uses all points before the mean-GP folds are constructed. Held-out replicate scores therefore influence training weights and validation denominators. Refit the noise model inside each fold. This issue does **not** apply to the reported initial jackknife-only validation. Also, top-region reporting lacks coverage and signed bias; global coverage cannot establish accuracy near the maximum. [emulator_design.py:58–78](bin/streams/emulator_design.py:58), [emulator.py:82–93](src/ocen_dm/streams/emulator.py:82)

4. **Medium: seeds are unique within a generated refinement batch, but reused across batches.** Every design invocation starts simulation seeds at `300000`, regardless of `--seed`; epoch offsets depend on those simulation seeds. Later batches therefore reuse random streams, and identical point/job combinations can reproduce identical simulations. Include a batch identifier in seed generation and detect duplicate `(parameters, seed, epoch_offset, nrel)` identities when loading. [emulator_design.py:105–108](bin/streams/emulator_design.py:105)

5. **Medium: the noise model has unquantified uncertainty and a non-identical-replicate assumption.**
   - Original runs use endpoint-inclusive deterministic epoch grids; new runs use randomly shifted grids with different spacing. Their equal means and variances are an assumption, not an established consequence of different seeds. [training_set.py:182–183](bin/streams/training_set.py:182), [grid4_common.py:58–63](bin/streams/grid4_common.py:58)
   - `predict()` exponentiates the fitted log-variance mean and treats the result as known. That is a plug-in estimate, not an uncertainty-integrated variance. The regression also lacks protection against invalid `vjk` or a singular intercept/slope fit. These are secondary while interpolation dominates, but matter when interpreting calibrated intervals. [emulator.py:40–59](src/ocen_dm/streams/emulator.py:40)

6. **Medium: unattended failures can look like successful completion.** Workers catch failures and return them for printing; the runner does not exit unsuccessfully when jobs fail. Resume also crashes if a completed record’s spray exists but its per-star file is missing, because that file is hashed without an existence check. Require an explicit completion gate before fitting the next round. [stream_batch.py:128–130](bin/streams/stream_batch.py:128), [stream_batch.py:147–158](bin/streams/stream_batch.py:147), [stream_batch.py:178–186](bin/streams/stream_batch.py:178)

**Acquisition and model choice**

UCB with κ = 2 is defensible for discovering better fits. It does not directly target posterior accuracy: EI also targets improvement of the maximum, whereas posterior inference needs accurate relative likelihoods across regions carrying probability mass. Posterior-focused experimental design is the closer objective. [Järvenpää et al.](https://arxiv.org/abs/1905.01252)

The raw numerical range is not itself the problem. Positive affine rescaling—including constant tempering—leaves UCB rankings unchanged when both mean and standard deviation are scaled consistently.

My pragmatic choice would be:

- Run **32–48 distinct points**, then refit before spending the remaining budget.
- If all 150 must run unattended, allocate roughly **100 to conservative plausible-region refinement, 30 to global discovery, and 20 to independent validation**.
- Define plausibility using an uncertainty-aware upper bound, not predicted mean alone; otherwise an inaccurate GP can exclude an undiscovered mode.
- Keep the global GP as a scout and compare a local GP on **untransformed log-likelihood differences** around several promising regions. Do not introduce hard target clipping immediately before this run: it creates a plateau and changes the regression target.

The reported top-region RMSE of 1606 is insufficient evidence for posterior inference. At temperature \(T\), the relevant requirement is that errors in log-likelihood **differences**, divided by \(T\), are small. The temperature and those local difference errors have not been established here. The numerical validation results are supplied by the packet, not independently reproduced. [review packet:2017–2025]

**1. Verified versus assumed**

Verified by reading:

- I found no sign or degrees-of-freedom error in the digamma/trigamma correction, nor a missing square in normalized GP noise. The mean fit receives `s2/r` and divides by `ysd²`, consistent with scikit-learn’s variance-valued `alpha`. [emulator.py:40–50](src/ocen_dm/streams/emulator.py:40), [emulator.py:68–74](src/ocen_dm/streams/emulator.py:68), [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
- Keeping `mu` unchanged during fixed-kernel believer updates is mathematically appropriate when fantasy observations equal their conditional means. The questionable approximation is assigning every fantasy the median training-mean noise rather than anticipated single-run noise. [emulator_design.py:88–95](bin/streams/emulator_design.py:88)
- The coordinate formulas match the training design, including the corotation-dependent size map. However, `checkmap` only prints discrepancies for 40 points; it neither asserts agreement nor runs automatically before design generation. [emulator_design.py:31–38](bin/streams/emulator_design.py:31), [training_set.py:86–99](bin/streams/training_set.py:86), [emulator_design.py:127–130](bin/streams/emulator_design.py:127)
- Emulator tests cover a smooth toy’s average noise recovery and broad coverage bounds. They do not exercise batch selection or ingestion. The jackknife bias test uses i.i.d. toy particles, rather than randomized systematic epoch grids. [test_emulator.py:8–41](tests/test_emulator.py:8), [test_score_bias.py:17–44](tests/test_score_bias.py:17)

Assumed: approximately Gaussian, independent simulation errors; negligible residual jackknife bias near the optimum; interchangeable old/new epoch-grid distributions; and the packet’s numerical diagnostics. I could not run tests because the available Python environments lack the required scientific packages and `pytest`. Nothing was modified.

**2. Strongest argument against my answer**

Given interpolation errors much larger than simulation noise, further global exploration may deliver more benefit than changing acquisition or fitting a local model now. Restricting attention prematurely could miss a separate mode. If the actual 150-point selection contains no duplicates and tempering is sufficiently strong, the proposed UCB round could be a productive discovery experiment. My objections are strongest against treating its output as a validated posterior surrogate.

**3. What I would check to be sure**

- Actual selected coordinates: uniqueness, separation, and distribution across plausible regions.
- Strictly held-out top-region bias, coverage, ranking, and log-likelihood-difference errors at the intended temperature.
- Randomized-grid replicate scatter and residual particle-number bias at the best observed and predicted points.
- Loader rejection of duplicate runs, incompatible manifests, and inconsistent grouping keys.
- Failure/resume tests, plus a real eight-worker smoke test for wall time, peak memory, disk use, and numerical thread counts.
- Complete running replicates before editing their hashed sources: workers reject source changes, and resume rejects a changed manifest. [stream_batch.py:86–88](bin/streams/stream_batch.py:86), [stream_batch.py:133–141](bin/streams/stream_batch.py:133)