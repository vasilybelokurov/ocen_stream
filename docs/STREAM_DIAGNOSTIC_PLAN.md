# Stream 54 fit: diagnostic and model-extension plan (2026-10-10; Phase 3 adopted the same day)

Built from: [STREAM_FIT_CAMPAIGN_PLAN.md](STREAM_FIT_CAMPAIGN_PLAN.md), [STREAM_FIT_METHODS_REVIEW.md](STREAM_FIT_METHODS_REVIEW.md),
Codex reviews [codex_likelihood_calibration_2026-10-09.md](codex_likelihood_calibration_2026-10-09.md),
[codex_diagnostic_experiments_2026-10-10.md](codex_diagnostic_experiments_2026-10-10.md),
[codex_code_review_2026-10-10.md](codex_code_review_2026-10-10.md), and the journal (2026-10-09/10).

## Where we are

- A rotating bar is the only ingredient found so far that makes the knee. Omega_b = 34-35 km/s/kpc is robust to every variation tried.
- Two problems remain:
  - **P1. The knee is not fully reproduced.** Beyond phi1 ~ 20 deg, model pmdec stays 0.5-1 mas/yr too high, and the model debris thins out.
  - **P2. The fit wants an implausible bar.** It prefers angle 16 deg (literature ~27) with amplitude 1.4. The 20-deg model fits better near the cluster, the 16-deg model better in the knee.
- **Ruled out:**
  - code errors: bar clock, r_J derivatives, centre orbit, frame round trip, arm/chi labels, data integrity, bar near end and rotation sense;
  - cluster internal rotation, even x10;
  - the debris age cut (500-1000 Myr);
  - kernel width;
  - Sgr;
  - axisymmetric hosts.
- **Statistics:** lnL is overconfident. Spray noise is +-120 at 1x and +-36 at 4x particles, and misfit is correlated over many degrees. Calibration has to come from mocks.

## Phase 1 -- diagnostics (running now; ~1 h of CPU in total)

| | Question | Design | Decision rule | Cost |
|---|---|---|---|---|
| D1 | What drives the angle preference: which stretch of the stream, which observable? | Best model at each angle (16, 20, 24, 28), 4x particles, seed 1. Per-star lnL split into phi1 segments (< 4, 4-17, > 17) and into per-observable conditional scores (dphi2; PM pair; v_los). | If the preference comes from one segment AND one observable, that localises the missing physics. If it is spread out, suspect the statistic. | 1 spray (24 deg at 4x) + minutes of scoring |
| D2 | Does the method itself pull towards low angles? | Mock members drawn from truth sprays at 28 and 20 deg (seed 7, 4x; independent of the candidate sprays). Same phi1 values, Gaia covariances and v_los pattern as the data. 100 mocks per truth, scored against 56 candidates (d = 5.6, amp 1.2/1.4, all Omega_b and angles; 1x, seed 1). | If 28-deg truths recover 16 deg often, the ranking is untrustworthy. If recovery is good, P2 is physical (model misfit). | 2 sprays + ~10 min with 8 processes (timing run first) |
| D3 | Is the fixed omega Cen orbit forcing the low angle? | At 28 deg (Omega_b 35.5, amp 1.2, d 5.6), a 5x5 grid of pmra, pmdec at catalogue +- 1, 2 sigma (sigma = 0.027 mas/yr), 4x particles, seed 1. | A PM shift within 2 sigma that improves the near-cluster and knee segments together, and closes much of the gap to 16 deg, means the state was the problem. | 25 sprays x 45 s ~ 19 min (8 threads) |

Context for D3: the earlier free-PM optimiser (old chi-KDE likelihood) already moved pmra by about -1 sigma and still drove the angle to the prior edge. A large effect is therefore not expected, but this is the first test with the conditional likelihood.

**Gate:**
- If D2 shows a biased method, fix the statistic before any physics.
- If D3 closes the gap, refit with the PMs free.
- Otherwise go to Phase 2.

## Phase 2 -- what the model may be missing (one ingredient at a time, at the literature angle 28 deg)

Each test uses the D1 segment/observable diagnostics against the 28-deg baseline. Success means improving the knee pmdec without losing the near-cluster fit.

| | Candidate | Why plausible | Test | Cost / status |
|---|---|---|---|---|
| M1 | Release prescription (Fardal spray) | Codex rated this high. But the 20-deg tracer run misses the knee the same way the spray does. | 200k-tracer run vs spray at 28 deg: same host and state; compare footprint medians and counts. | ~12 min (8 threads) |
| M2 | Slowing bar | A decelerating bar was faster in the past; over the ~700 Myr of observed debris the resonances sweep through the orbit, changing the knee and the near-cluster debris differently. | Dillamore+2024 prescription (eta = -dOmega/dt / Omega^2), present Omega_b and angle fixed; bar size scales with 1/Omega (existing `make_slowing_bar_potential`, ../oCen_bar/scripts/ocen_bar_common.py; ../chevron_bar_subhalo/src/barchevrons/bar.py). Scan eta = 0 + 3 values. | wiring ~1 h; ~1 min per eta at 1x, 4 min at 4x |
| M3 | Bar model / length / mass | Hunter+2024 is one model. A longer or more massive bar acts like a "strong bar at small angle". | (a) Portail+2017 bar (Sormani+2022 approximation; ../oCen_bar/agama_potentials/Portail17.ini) at 28 deg; (b) bar size scaling S (as in make_potential) at fixed angle. | wiring ~1 h; minutes of sprays |
| M4 | Release history | Release is uniform in time, while escape is pericentre-driven. Older debris (> 1 Gyr) may populate the knee. | Release window 1500 Myr; pericentre-weighted release (Erkal+2019 style). Compare knee counts and pmdec. | minutes |
| M5 | Selection (STREAMFINDER) | Template-orbit selection could shape the knee PMs. This was planned as Stage 0(b) and never done. | Independent Gaia DR3 selection in the knee window (WSDB: sky + CMD + broad PM window, no orbit templates); compare PM ridges with the members. | ~1 h of work, seconds of queries |
| M6 | Halo shape / LMC | Affects the orbit over Gyr. Less likely to produce a sharp knee. | Only if M1-M5 fail: one flattened halo and one LMC+reflex variant (no LMC in the code yet). | days (LMC) |

Literature for M2/M3: Dillamore, Belokurov & Evans 2024, MNRAS 532, 4389 (arXiv:2402.14907, https://arxiv.org/abs/2402.14907);
Sormani+2022, MNRAS 514, L5 (reference given in ../oCen_bar/agama_potentials/example_mw_bar_potential.py); Portail+2017 (MNRAS 465,
1621; UNVERIFIED citation details). Present-day bar deceleration rates from the literature are to be looked up before choosing eta (UNVERIFIED).

## Status after Phases 1-2 (2026-10-10; numbers in JOURNAL.md)

- **Not the cause:**
  - code: audits and end-to-end checks pass;
  - method: D2 mocks recover the true angle (28 -> 28 in 100/100; 20 -> 20/24);
  - STREAMFINDER selection: M5 recovers the knee PMs independently to phi1 ~ 24;
  - these ingredients: cluster rotation (even x10), spray release (M1 tracer run is further from the knee), Portail bar,
    older debris, slowing bar at today's Omega_b.
- **The misfit is two separate problems in the knee:**
  1. pmra at phi1 15-21: the 28-deg model is ~1 mas/yr too negative. This is all that the low angle fixes; it is the whole
     angle preference.
  2. pmdec at phi1 > 19: every model, at any angle, is ~1 mas/yr too shallow in pmdec. (An earlier claim that the models also
     put the knee too close rested on our CMD distance track; it is WITHDRAWN: the literature puts the knee at ~0.70-0.75 of
     omega Cen's distance, and the track's normalisation is flawed -- see codex_distances_and_misfit_2026-10-10.md.)
- **Partial fixes at 28 deg:** longer bar (x1.15-1.3: +622/+698); more negative omega Cen PMs (-2 sigma: +518, optimum beyond 2 sigma).

## Phase 3 -- next (adopted 2026-10-10)

**Strategy: avoid unrealistic bar angles.**
- **S1. Give the model the freedom it lacks.** The angle is currently the only knob that bends the knee, so the fit uses it.
  Test whether a longer bar + omega Cen PMs + re-scanned Omega_b reach the 16-deg fit level at 28 deg (Phase 2b grid, running).
- **S2. DROPPED (2026-10-10).** The fit stays distance-free (user decision; CMD track not a calibration, member parallaxes
  biased high). Original text kept for the record. **Add the distance track to the likelihood.** At present models are not penalised for putting the knee too close, and
  16 deg is barely better there. Including the CMD distance track constrains the model where the angle does not help.
  Design (to agree):
  - (a) binned term: model median d/d_ocen of footprint debris per CMD b-bin vs the CMD ratio, with the CMD error plus the
    model error of the mean; or
  - (b) per-star photometric distances.
  Note: next to 3681 per-star terms, a 6-bin term is inert unless the likelihood is calibrated (S3).
- **S3. Make the angle prior count.** With the overconfident likelihood the N(27, 2) prior has no effect. Defensible options:
  - (a) fix the angle at the literature value and report the remaining misfit as model systematics;
  - (b) temper the likelihood with an effective sample size calibrated on mocks (D2 machinery). **Needs the user's approval.**

**Physics: improve the stream fit (priority order).**
1. **Potential shape and mass where the knee lies (was M6; moved up).** A 3D mismatch -- too close and too slow in the same
   region -- points at the force field there, not at the bar's present orientation.
   - Variants: disc-to-halo mass ratio at fixed v_c(R0), and halo flattening q.
   - Build them with ../oCen_bar/agama_potentials/example_mw_potential_hunter24.py; its densities are exposed, and the
     halo is bunched with the central components in one Multipole, so variants need re-construction.
   - Cost: ~1 h of wiring, then minutes of sprays per variant. Diagnostics: D1 segments, knee PM medians, distance track.
2. **Combine the partial fixes** (longer bar, omega Cen PMs, Omega_b; slowing bar with Omega_b re-scanned) and measure how
   much of the gap to 16 deg remains. This is the Phase 2b grid; the pericentre-concentrated release (M4b) is queued after it.
3. ~~Add the distance track to the likelihood (S2)~~ -- dropped.
4. **Solar frame x omega Cen distance (Codex, 2026-10-10; next).** baumgardt (R0 8.178, V_sun,y 252.24) vs ibata19 (R0 8.122,
   V_sun,y 232.24) x d_ocen 5.43/5.6/5.8 at the working model (28 deg, bar x1.15, Omega_b 36, amp 1.2) and the 16-deg model;
   4x particles; compare knee PM medians and D1 segment lnL. ~12 sprays, ~10 min at 8 threads.
5. **Coupled bar angle x longer bar grid** (Codex): does the 16-deg preference survive once the longer bar is free?
6. LMC + reflex; alternative progenitor profiles -- expensive, only if 4-5 fail.

## Phase 4 -- calibration and statistic (after Phase 3)

- Calibrate parameter errors and model comparisons with mocks (D2 machinery). Use seed-averaged densities; use 4x particles in all grids.
- DM test (on hold by user decision): choose between track-only (conditional) and track+density along the stream (needs a selection model).

## Rules

Announce CPU before each launch; one ingredient at a time; no ad hoc error inflation; the bar angle stays near the literature
(no grid below 16 deg); journal and commit after each result.
