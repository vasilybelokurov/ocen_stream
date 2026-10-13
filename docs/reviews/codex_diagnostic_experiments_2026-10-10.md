The premise is broadly consistent with the code. One qualification: the grid’s \(16^\circ\) result is a **raw likelihood maximum**, not a fit with the stated \(N(27^\circ,2^\circ)\) prior; `grid4.py` scores each grid point without adding a prior, while the separate fit script defines that prior and bounds angle to \(21–33^\circ\) ([grid4.py:20–41](bin/streams/grid4.py); [fit_bar_ocen.py:26–35](bin/streams/fit_bar_ocen.py)).

## Q1. Ranked candidate causes

### 1. The knee is too shallow and the model thins out

1. **Spray release and selection mismatch — high plausibility.** The spray uses a Fardal-style release prescription, but the fit selects model particles by arm, age, and \(\chi\), not by a STREAMFINDER detection or membership process ([spray.py:42–64](src/ocen_dm/streams/spray.py); [score.py:250–253](src/ocen_dm/streams/score.py)). The observed catalogue is filtered by stream label and sky cuts ([spray_bar_grid2.py:24–31](bin/streams/spray_bar_grid2.py)). This could explain the apparent thinning without explaining a coherent proper-motion offset.

2. **The barred host or its history is too simple — high plausibility.** The host supports a constant pattern speed, present angle, and scaled barred-baryon component ([restricted.py:40–61](src/ocen_dm/streams/restricted.py)). A mismatch in bar history or shape can affect the outer-tail track and knee together.

3. **Centre-orbit state or frame conversion — moderate plausibility.** The grid fixes the cluster PMs and uses discrete distances; the centre state is built from those values ([grid4_common.py:14–30](bin/streams/grid4_common.py)). A small present-day state error can accumulate along a long integrated orbit.

4. **Release window or age cut — moderate-to-low plausibility.** Releases cover the last 1000 Myr, and scoring drops particles at age \(700\) Myr ([grid4_common.py:25–30](bin/streams/grid4_common.py); [score.py:250–253](src/ocen_dm/streams/score.py)). This can remove outer-tail support if the relevant debris is older, but it cannot by itself explain a systematic PM offset in debris that remains.

### 2. Bar angle is implausibly low and near the grid edge

1. **The model is compensating for an omitted ingredient — high plausibility.** The angle, amplitude, pattern speed, and distance vary together in the grid, while PMs and cluster radial velocity are held fixed ([grid4.py:20–40](bin/streams/grid4.py)). The code does not establish that the angle itself is identifiable when those assumptions are fixed.
2. **Likelihood or frame construction favors the knee over the inner stream — high plausibility.** The score conditions on observed along-stream coordinate and gives equal prior weight to valid \(\chi\) bins ([score.py:233–245](src/ocen_dm/streams/score.py)). The adopted frame is a fixed chord frame with a fitted polynomial track ([path.py:133–144](src/ocen_dm/streams/path.py)); its stored maximum ridge deviation is \(1.071^\circ\) ([stream54_gc_frame_chord.json:10–26](results/plot_data/stream54_gc_frame_chord.json)).
3. **The constant-speed bar prescription is incomplete — moderate plausibility.** `host_potential` encodes constant pattern speed and a specific scaled difference between barred and axisymmetrised baryons ([restricted.py:40–61](src/ocen_dm/streams/restricted.py)).
4. **A true preference for \(16^\circ\) — possible, but not established.** The supplied block-bootstrap and seed results already indicate that the raw difference is unstable; the code’s grid itself does not impose the literature-angle prior ([grid4.py:20–41](bin/streams/grid4.py); [seed_test.py:11–30](bin/streams/seed_test.py); [bootstrap_grid4.py:19–48](bin/streams/bootstrap_grid4.py)).

### 3. Statistics are overconfident or unstable

1. **Sparse local particle support — high plausibility.** Each \(\chi\) bin is a KDE over spray particles, and only bins with at least 20 particles are retained ([score.py:250–272](src/ocen_dm/streams/score.py)). The diagnostic computes effective particle count from the same kernel weights ([kernel_check.py:25–40](bin/streams/kernel_check.py)).
2. **Correlated members treated as independent — high plausibility.** The likelihood sums a per-member log likelihood ([score.py:273–275](src/ocen_dm/streams/score.py)); the bootstrap code explicitly contrasts iid resampling with contiguous \(\phi_1\) blocks ([bootstrap_grid4.py:2–7](bin/streams/bootstrap_grid4.py)).
3. **A fixed bandwidth and background share alter model ranking — moderate-to-high plausibility.** The conditional KDE uses fixed bandwidths and a fixed `eps=0.05` background mixture ([score.py:231–245](src/ocen_dm/streams/score.py)). The existing bandwidth diagnostic rescales widths, but that is a sensitivity check, not a calibrated uncertainty model ([kernel_check.py:49–57](bin/streams/kernel_check.py)).
4. **The grid maximum is a noisy boundary result — high plausibility.** Angle is sampled only at \(16,20,24,28^\circ\) in the reported grid ([grid4.py:20](bin/streams/grid4.py)); selecting the maximum among many noisy points adds a winner’s-curse effect.

### 4. Dark-matter effects are missed by the adopted score

1. **Along-stream counts are deliberately discarded — certain from code.** The score forms a conditional density by dividing the joint KDE by its along-stream marginal and uses equal weights across valid \(\chi\) bins ([score.py:233–245, 261–275](src/ocen_dm/streams/score.py)). It is therefore not a test of DM-driven escape counts.
2. **The tested grid fixes the progenitor profile to `A_nodm` — high plausibility.** The grid loads that profile for all models ([grid4_common.py:19](bin/streams/grid4_common.py); [grid4.py:26](bin/streams/grid4.py)).
3. **Selection confounds count comparisons — high plausibility.** The observed sample is catalogue-selected and sky-filtered ([spray_bar_grid2.py:24–31](bin/streams/spray_bar_grid2.py)); a count likelihood needs a defensible model of detection and membership completeness before count differences can be attributed to DM.

## Q2. Prioritised experiments

Costs below use the supplied timings: about 11 seconds for an 8000-epoch spray, 45 seconds for 32000 epochs, and 11.5 minutes for a 200k tracer run, each at the stated 8-thread setup. They are approximate and exclude plotting and setup. The 4D grid has \(7×4×4×4=448\) points, or about **82 minutes** of spray time at 11 seconds each, before overhead.

1. **Audit coordinates, bar clock, and release-time Jacobi radii.**  
   **Hypothesis:** a sign, clock, or numerical error is manufacturing some of the bar-angle preference or knee mismatch.  
   **Design:** no grid. For a representative orbit and release times, compare the bar angle evaluated at \(t=0,T\) with the requested present angle; compare `rj_vj_R`’s directional second derivative against a finite-difference radial-force derivative at several orbit phases; forward-integrate the reconstructed centre orbit and report present-day position and velocity residuals. Also project data and representative sprays through the stored chord frame and a pole-fit frame.  
   **Diagnostic / decision:** plot angle versus time, \(r_J\) residuals versus orbital phase, orbit round-trip residuals, and \(\Delta\phi_2\) between frames. A systematic sign or phase error, or release-radius discrepancy large enough to move the stream track, confirms an implementation concern.  
   **Cost:** negligible to minutes; no new spray required.  
   The relevant code evaluates the host at each release time and reconstructs the centre orbit backward then forward ([spray.py:22–39, 107–119](src/ocen_dm/streams/spray.py); [restricted.py:40–61](src/ocen_dm/streams/restricted.py)).

2. **Rescore existing sprays by segment and observable.**  
   **Hypothesis:** the low angle is driven mainly by the knee, while inner-stream and other-observable scores prefer a different model.  
   **Design:** reuse cached sprays at the best \(16^\circ\) point and at \(20,24,28^\circ\), including the best available amplitude, pattern speed, and distance for each. Report per-star \(\Delta\ln L\) and median/ridge residuals separately over \(\phi_1=-2..4\), \(4..17\), and \(17..27^\circ\), and separately for sky offset, PMs, and radial velocity. Do not change bandwidths or data cuts.  
   **Diagnostic / decision:** segment score table plus residual-versus-\(\phi_1\) plots. If angle preference reverses by segment or observable, prioritize model inadequacy over further global optimization.  
   **Cost:** scoring existing files; likely minutes, no new spray.

3. **Mock injection and recovery of the score and grid.**  
   **Hypothesis:** the score or finite particle support systematically recovers low angles even when the generating model has a literature-like angle.  
   **Design:** take cached sprays at \(28^\circ\) and \(20^\circ\) as separate truths. Generate 100 catalogues of 3681 mock members from each truth, with the observed \(\phi_1\) distribution held fixed, Gaia PM covariances and the observed radial-velocity missingness/error pattern applied. Score each mock against cached candidate models. Repeat with 8000- and 32000-epoch truth sprays if available; hold the grid and score widths fixed.  
   **Diagnostic / decision:** recovery histograms for angle and \(\Delta\ln L(16-20)\), plus calibration of nominal score separation. If mocks generated at \(28^\circ\) often recover \(16^\circ\), the current ranking is not trustworthy under its own assumptions. If recovery is accurate, the explanation likely lies in model mismatch or data selection.  
   **Cost:** no new sprays if cached; scoring 200 mock catalogues may take from minutes to hours depending on implementation. This estimate needs a short timing run before committing.

4. **Free the cluster state locally at fixed literature-like bar angle.**  
   **Hypothesis:** fixed cluster PMs, rather than bar angle, are forcing the knee and inner stream into conflict.  
   **Design:** at angle \(28^\circ\), pattern speed \(34.5\), amplitude \(1.2\), distance \(5.5\) kpc, compare catalogue PMs with the grid PMs \((-3.2223,-6.7517)\), plus each PM shifted by \(\pm0.027\) mas/yr individually. Hold \(v_{\rm los}\) fixed as the existing fit does; use common release epochs and seed. Use the conditional score, and report the PM prior penalty separately rather than hiding it in the likelihood.  
   **Diagnostic / decision:** a 2D PM score map and segment residuals. If plausible PM shifts materially improve both inner and knee segments while keeping angle near \(28^\circ\), the fixed-state assumption was consequential.  
   **Cost:** 5 sprays, about 55 seconds plus scoring. The separate fit script already has a PM-free path and Gaussian PM priors ([fit_bar_ocen.py:26–35, 51–72](bin/streams/fit_bar_ocen.py)).

5. **Calibrate spray against prescribed tracers in the barred host.**  
   **Hypothesis:** the release prescription, not the host orbit, produces the missing tail support or PM curvature.  
   **Design:** run 200k tracers at a \(28^\circ\) host and the same centre state and progenitor profile as the spray, without spin. Compare age-resolved ridge, counts per 2-degree \(\phi_1\) bin, PM medians, and radial velocity. If time permits, repeat at \(16^\circ\) using identical tracer ICs.  
   **Diagnostic / decision:** if tracer debris reaches the observed knee while the spray does not at the same host, the release model is implicated. If both miss similarly, the host, orbit, selection, or frame is more likely.  
   **Cost:** about 11.5 minutes per run by the supplied timing; two runs about 23 minutes. Note that the tracer setup integrates sampled equilibrium stars in a fixed moving satellite potential ([run_prescribed.py:73–109](bin/streams/run_prescribed.py)), whereas the spray injects particles at release points ([spray.py:42–64](src/ocen_dm/streams/spray.py)).

6. **Test release window and age selection.**  
   **Hypothesis:** the knee requires older debris omitted by the 700 Myr age cut or 1000 Myr release window.  
   **Design:** for the \(16^\circ\) and \(28^\circ\) reference models, keep 8000 epochs and seed 1, extend the release interval to 1500 Myr, and rescore with age limits 500, 700, 1000, and 1500 Myr. Plot the age distribution of particles contributing to each member’s likelihood.  
   **Diagnostic / decision:** if the knee residual improves only when older debris is admitted, this supports the age/window hypothesis; if not, deprioritize it.  
   **Cost:** two longer sprays; the supplied 11-second baseline may understate their cost because particles have longer integration times.

7. **Compare constant-speed and decelerating-bar histories.**  
   **Hypothesis:** a different bar history can fit the knee and inner stream without pushing the present angle below the literature range.  
   **Design:** after the clock/sign audit, compare constant speed to one explicitly parameterized slowing history, matched to the same present-day \(\Omega_b\), present angle \(28^\circ\), and amplitude \(1.2\). Use the same epochs, seed, distance, and cluster state. First run one model per history, then only a small local variation if track residuals improve.  
   **Diagnostic / decision:** compare segment PM/sky residuals and radial-velocity residuals, not only total conditional score. Improvement confined to one segment is insufficient.  
   **Cost:** roughly 22 seconds for two baseline sprays, plus any slower-history overhead. Implementing a new history is additional work.

8. **Only then test bar shape and halo/LMC alternatives.**  
   **Hypothesis:** the specific Hunter+2024 bar or fixed axisymmetric halo is inadequate.  
   **Design:** one alternative bar shape at \(28^\circ\) and one justified halo/LMC variant, matched to the same present centre state and broadly comparable circular-speed curve. Compare with the same observables and segment diagnostics.  
   **Diagnostic / decision:** proceed to a small follow-up only if an alternative reduces both the knee and inner-stream residuals without relying on the conditional score alone.  
   **Cost:** about 22 seconds for two baseline sprays, plus model setup; uncertainty is larger because the alternative potentials are not specified in the current code paths.

## Q3. Before another grid, and when to change the likelihood

Before any new grid, I would run experiments 1–4: verify time/frame mechanics, determine which segments drive the angle result, check recovery bias, and test whether the fixed cluster state matters. These are cheap and separate numerical or statistical effects from physical model error.

I would change the likelihood only if mock recovery is biased, the selected bandwidth/background materially changes the ranking, or the current conditional objective answers the wrong scientific question. In particular, if the question is about DM-driven escape counts, add a count term only after modeling the catalogue selection and completeness; the current score explicitly conditions out density along the stream ([score.py:233–245](src/ocen_dm/streams/score.py)). I would not treat broader bandwidths or an extra error floor as a fix: the current score has fixed widths and a 5% background component ([score.py:231–245](src/ocen_dm/streams/score.py)), and any new uncertainty model should be calibrated by mock recovery first.

## Q4. Code-path weaknesses and checks

**Verified in code:**

- `host_potential` sets the angle to \(\theta_{\rm today}+\Omega_b(t-t_{\rm today})\), with \(\Omega_b\) supplied in km/s/kpc and the angle in radians ([restricted.py:40–55](src/ocen_dm/streams/restricted.py)). The units are consistent if the time coordinate is AGAMA’s kpc/(km/s) unit, as the module states ([restricted.py:14–15](src/ocen_dm/streams/restricted.py)); I have not independently verified AGAMA’s rotation convention against the potential assets.
- `rj_vj_R` passes release times to `host.eval`, computes a radial second derivative from the returned derivative components, clips the Jacobi denominator to \(10^{-6}\), and runs 30 fixed-point updates without a convergence check ([spray.py:22–39](src/ocen_dm/streams/spray.py)). This deserves the finite-difference audit: a derivative layout, sign, or near-zero denominator error would alter release radii and thus stream length.
- The centre orbit is integrated backward from today and then forward; the second integration samples the centre track every 0.05 Myr for the moving satellite potential ([spray.py:107–119](src/ocen_dm/streams/spray.py)). This is a reasonable construction, but the stream code does not itself assert that the forward endpoint returns to the specified present state; the prescribed-tracer script records a centre offset ([run_prescribed.py:100–104, 120–126](bin/streams/run_prescribed.py)).
- The chord-frame projection uses `atan2` stream longitude and subtracts a polynomial \(\phi_2(\phi_1)\) ([path.py:127–130](src/ocen_dm/streams/path.py)). The stored frame spans \(\phi_1\approx-1.8..26.9^\circ\) and has `max_dev` \(1.071^\circ\) ([stream54_gc_frame_chord.json:10–26](results/plot_data/stream54_gc_frame_chord.json)). That does not prove it is a bad frame, but it is enough to justify rechecking near-cluster and knee residuals in an independent frame.
- The focused conditional-score tests cover background-only behavior, cancellation of along-stream density, duplicated bins, and a deliberately wrong PM model ([test_score_conditional.py:24–65](tests/test_score_conditional.py)). The focused spray-spin tests check the sign and geometry of the added spin velocity ([test_spray_spin.py:14–30](tests/test_spray_spin.py)); these tests do not establish bar-clock, Jacobi-radius, or end-to-end barred-orbit correctness.
- `score_conditional` uses a fixed mixture fraction, fixed widths, Gaussian PM covariance, and a finite uniform background box; it does not validate positive covariance determinants or adapt the model bandwidth to local particle support ([score.py:231–275](src/ocen_dm/streams/score.py)). Sparse support can therefore affect the score before the background component dominates.

**Assumptions:** I take the packet’s stated bar-angle literature range, timing measurements, tail residuals, and prior bootstrap/seed results as inputs; I did not reproduce them. I also assume the catalogue selection and the external Hunter potential assets are as described in the packet.

**Strongest argument against my top recommendation:** The mock-injection/recovery experiment can be circular. If mocks are generated from the same spray prescription, frame, and score being tested, successful recovery only shows internal self-consistency; it cannot show that the prescription resembles the real stream. That is why I would pair it with the barred-host tracer comparison before drawing a physical conclusion.

**What I would still need to check to be sure:** the AGAMA `eval(..., der=True)` component ordering and time semantics against a finite-difference oracle; forward/backward orbit residuals for the actual best-fit states; frame sensitivity of the segment results; mock-recovery behavior with realistic catalogue selection; and whether tracer-versus-spray differences persist across at least one seed and the two relevant bar angles.