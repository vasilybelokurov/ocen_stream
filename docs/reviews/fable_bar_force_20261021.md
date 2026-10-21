# Fable review: is the outer bar-force demand robust, and what next (2026-10-21)

Claude re-ran bin/streams/diagnostics/orbit_force_check.py and lz_and_fourier.py (run with OCEN_ROOT=<repo>, PYTHONPATH=src);
numbers reproduce.
## Findings
- Along omega Cen's orbit the fitted bar's non-axisymmetric force averages to ~0 (|mean| <= 0.003 |F_axi| over 300-700 Myr;
  rms 0.12-0.13 vs 0.07 published); today at the cluster 0.057 vs 0.020. So the fit is NOT using the bar as a proxy for a static
  (axisymmetric) change (disc, halo q, v_c, frame, distance, PMs): it buys a TORQUE HISTORY. 12.5-deg best: Lz -609 (300 Myr ago)
  -> -512 today vs -561 -> -512 published; apocentre walks 7.1 -> 6.2 kpc over 1 Gyr (published 7.3 -> 6.95). The stream
  (300-700 Myr debris) records that history.
- Robust against the axisymmetric alternatives; NOT established that it is m=2 co-rotating with the bar (the Hunter bar was the
  only form offered). m=1 lopsided halo, tilted/triaxial halo, bar-induced halo wake not excluded; spiral (pre-screen) and LMC
  (~0.01 |F_axi|) too weak.
- Plausibility: m=2 radial amplitude at R0 (z 0) 0.062 (fitted) vs 0.017 (published); Hercules/OLR models use 0.01-0.02 (Dehnen
  2000, 2000AJ....119..800D; Hunt & Bovy 2018, arXiv:1803.02358). A long bar x2 mass x1.3 length has ~3.4x the published far-field
  quadrupole; dynamical bar lengths are if anything shorter (Lucey+2023 arXiv:2206.01798; Vislosky+2024 2024MNRAS.528.3576V)
  than Wegg+2015's 5.0 kpc. Nothing known supplies ~2x the published quadrupole at 5-7 kpc.
## Recommendation (b): one discriminating batch at fixed 12.5-deg parameters (32k, CRN, ~14 sprays, ~20 min on 2 cores + ~1 h code)
Host = published bar (A=S=1) + 'excess' (fitted minus published bar term) as a separate component with its own pattern speed and
phase: control (Omega_2 = Omega_b, phase 0; must reproduce the fitted score within 100); Omega_2 = Omega_b at phase +45, +90;
static (Omega_2 = 0) at 3 phases (triaxial-halo-like); Omega_2 = 22.5 (spiral-like) at 2 phases; Omega_2 = -33 (counter-rotating);
optional static triaxial halo p = 0.85 at 2 PAs; published bar + disc_scale 0.8 / 1.2 at fixed v_c.
Criteria relative to the 3808 gap (published -> fitted): any non-co-rotating configuration recovering > 70% -> generic outer
non-axisymmetric force, pattern speed unidentified -> stop the bar-refit line, write it up as a constraint; only Omega_2 ~ Omega_b
recovers (others < 30%) -> bar-locked -> long-bar refit with literature priors (2 angles, ~3 h on 8 cores) + Hercules tension;
disc_scale recovering > 1000 -> add it to the design first.
Against: fixed-parameter tests are unfair to alternatives (the published bar loses 3808 at fixed params); the honest alternative
(d) is to stop now and report "the stream requires a non-axisymmetric force history ~2x the published bar's rms along the orbit".
