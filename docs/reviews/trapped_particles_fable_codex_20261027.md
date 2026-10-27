# Failing particles in box models: cap, hand-off or domain exit? Fable + Codex (2026-10-27)

Pilot run (A100): box points 2.1 s/model (benchmark 0.54), 1-4 of 20 models chi2 = inf; cause: 1-3 particles per model with natt = -1
(200k-attempt cap) and a slowest "successful" particle at ~180k attempts. Brief: scratchpad trapped_brief.md.

## Codex
- "Unfinished after N attempts" does not mean bound (attempts include rejections; -1 conflates cap, invalid step, non-finite error;
  bar and moving progenitor can free a particle). Proposed (c): cap the GPU at ~30k attempts as a scheduling threshold and finish
  the unfinished states in C (cspray.resume); drop particles only after validating score insensitivity.

## Fable (measured, C library, full 179,200 particles; scratchpad trapped_check.py / trapped_trace.py)
- The failing particles are NOT trapped: they are ejected at a pericentre of a cluster orbit that plunges through the Galactic centre
  (u1: cluster R_min 0.020 kpc, r_J down to 0.010 kpc), leave the host-table domain (R > 30 or |z| > 10 kpc; gpu_host_export.py RMAX,
  ZMAX), and the B-spline extrapolation (cell clamped, weights not; jaxspray._spline2, spray.c cell()) then accelerates them to
  2,400 -> 12,000 -> 30,000 km/s. All 3 at u1 hit the cap with a finite state; at u2 the step collapsed to 1e-11 Myr (stalled).
- The 181k-attempt "successful" particle at u1 is one of these: it ends at 21 Mpc with chi > 0, age 600 Myr, i.e. inside the score
  selection. Finished particles at u1 today: 2 with |z| > 10 kpc, 11 with |z| > 8 kpc.
- Codex's hand-off does not work here: resuming in C for 400,000 attempts leaves them 33-56 Myr from today.
- Legitimate slowest particles: 8.5-8.6k (u1, u2), 7.9-11.3k at the 5 reference points (bound core particles, chi = 0); no natt < 0
  at the reference points.
- Recommendation: stop a particle when it leaves the table domain (return code -2 "escaped"), exclude it from the selection, log the
  count per model; keep max_att = 200k as the safety net. Checks: per-particle max R, |z| at the 5 points (expect 0 exits -> chi2
  bit-identical), u1/u2 finite chi2 and batch time ~ best-fit time, escaped counts for 20 random box points.

## Disagreement / decision (user)
Fable's domain-exit rule changes the model for escapers (dropped); the alternative is to extend the host tables (larger R, |z|; the
asinh grids make this cheap) so escapers are integrated in the true potential, as AGAMA does. Codex's (c) is refuted by Fable's
measurement (resuming does not finish them).
