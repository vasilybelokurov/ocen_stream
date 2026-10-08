# Fitting campaign: omega Cen's stream in a barred Milky Way, then the DM test (plan, 2026-10-08)

Status: **plan, not started.** Inputs: [STREAM_KNEE_INVESTIGATION.md](STREAM_KNEE_INVESTIGATION.md) (all tests so far) and the
Codex literature pass [codex_research_fit_campaign_2026-10-08.md](codex_research_fit_campaign_2026-10-08.md) (fallback web
research; arXiv IDs checked against the arXiv API: two errors in that report -- Koposov+2019 Orphan is arXiv:1812.08172, not
1812.08192 (= Erkal+2019 LMC mass); the 2026 omega Cen spectroscopy paper is Kuzma et al., arXiv:2605.23474).

## Goal

1. Find bar (and minimal host) parameters for which a model of omega Cen's tails reproduces the observed track of
   Ibata+2024 stream 54, including the knee: sky path, both PMs, relative distance and v_los vs position along the arm.
2. At that fit, test whether the three DM cases (A none, B moderate, C 5x DM inside 200 pc) are distinguishable, with nuisance
   parameters refitted under each and calibrated on mocks.

Starting point (tested): a rotating Hunter+2024 bar steepens the pmdec gradient to the observed one (Omega_b = 33 follows the
data to b = 37) and turns the sky track by about half of the observed ~10 deg; axisymmetric hosts fail.

## Design choices (my draft vs Codex; adopted)

| Item | Adopted | Why / source |
|---|---|---|
| Path coordinate | Galactic b along the northern arm (monotonic from the cluster, b = 15 -> 41) with a fixed omega-Cen-anchored stream frame for reporting | Simple, curved-path-safe for this segment; Codex prefers a general curved-path estimator (TrackStream, Starkman+2023 arXiv:2212.00949) -- not needed while b is monotonic |
| Data track | Per b bin (2 deg): stream + broad-background mixture in l, pmra, pmdec with Gaia per-star covariances; block (spatial) bootstrap errors; CMD-shift distance-modulus track with a zero-point nuisance; v_los kept per star | Erkal+2017 Pal 5 (arXiv:1609.01282), Koposov+2023 OC (arXiv:2211.04495) |
| Density / counts | Not used in the likelihood | Unknown STREAMFINDER completeness (Codex; agreed) |
| Selection bias | (a) magnitude-cut variants (G < 19 / < 20); (b) independent Gaia selection along the arm (WSDB: sky window + broad PM/CMD window, no orbit templates) and compare ridges; (c) systematic floor if they differ | STREAMFINDER template selection (Malhan & Ibata 2018) |
| Fast generator | AGAMA particle spray (Fardal+2015 arXiv:1410.1861 as in AGAMA's tutorial; Chen+2024 arXiv:2408.01496 release model) in the time-dependent barred host; Jacobi radius/frequency from the host evaluated at each release time | seconds per model (to be benchmarked) |
| Validation generator | Our prescribed-potential tracer runs (3.4 min axi / 6.5 min barred, 8 threads); already exist for Hunter axi and bars 33/37.5/41 | Codex: spray accuracy (~10% in Chen+24 tests) not established for a massive, rotating cluster in a bar |
| DM-aware generator | Tracer runs only (sprays know only total mass and r_J) | DM enters via the mass profile and release distribution |
| Objective | Gaussian track likelihood per bin: (y - m)^T (C_obs + C_model + C_sys)^-1 (y - m), model summaries built with identical frame, cuts, binning and error convolution | Erkal+2019 arXiv:1812.08192; Koposov+2023 |
| Optimiser | coarse grid -> Nelder-Mead / CMA-ES with common random numbers -> small emcee only if needed | Codex; brute-force MCMC on tracer runs is unaffordable |

## Stages

**Stage 0 -- freeze the observables (data only, ~1 day of work, negligible CPU).**
`bin/streams/measure_track.py`: ridge + width + errors per b bin for l, pmra, pmdec; distance-modulus track (from
`stream54_cmd_distance.py`); v_los per star. Variants: all members, G < 19, independent Gaia selection.
Gate: the knee (l turn and PM steepening at b > 30) persists across variants. Stream 55 measured separately, not fitted yet.

**Stage 1 -- spray generator and its validation (~0.5 day, minutes of CPU).**
`src/ocen_dm/streams/spray.py` (AGAMA, time-dependent host, Chen+24/Fardal release, release-time window 0-1 Gyr).
Benchmark runtime. Validate against the existing tracer runs (Hunter axi, bars 33/37.5/41; model A): same summaries through
the Stage-0 pipeline. Gate: spray ridges within the data errors of the tracer ridges, else calibrate the release parameters
(mean offset, dispersions) against the tracer runs, or fall back to tracer runs on a coarse grid.

**Stage 2 -- bar fit with sprays (hours of CPU, to be fixed after the benchmark).**
Free: Omega_b (25-50 km/s/kpc), bar angle (20-35 deg), bar amplitude (0.7-1.3 x Hunter), omega Cen 6D present-day point
(Gaussian priors: gc_catalog_full.fits values and errors). Host axisymmetric part fixed (Hunter+2024). Constant pattern speed.
Search: grid (~10 x 4 x 4 = 160 sprays) -> local optimisation -> uncertainty by emcee only if a basin is clear.
Gate: chi^2/dof acceptable, residuals unstructured along b; knee reproduced in l AND PMs (reject PM-only fits).

**Stage 3 -- robustness (hours).**
Host variants (halo flattening / disc mass, one parameter at a time), bar angle/strength degeneracies, selection variants from
Stage 0; slowing bar only if constant-speed residuals are structured.

**Stage 4 -- high-fidelity check (1-3 h, 8 threads).**
Tracer runs at the best fit and 2-4 posterior extremes (model A); compare with sprays and data.

**Stage 5 -- DM test (tracer runs; ~3 x (1 + nuisance refits) runs, plus mocks).**
At the Stage-2/3 host, run A, B, C (fitted rotation), refit the omega Cen nuisance point per case (sprays to locate, tracer runs
to evaluate), compare likelihoods; mocks from A, B and C with the data's sampling and errors to measure false-positive rate
and power. Detection criterion (campaign choice): Delta ln L (or ln Z) > 4.6 AND correct recovery in >= 90% of mocks.
C's DM profile is explicit (results/nbody/C_dm5x_shell/model_profiles.json); it is a fixed potential (no DM stripping), so it
is an upper limit on the DM effect for that profile.

## Risks and arguments against

- One ~25-deg segment may not separate Omega_b, bar angle, bar strength and the omega Cen point: priors on the latter from
  Gaia are tight, but bar parameters may stay degenerate. Report the degeneracy; it does not block the DM test if the DM
  signal is orthogonal to it -- check with mocks.
- The STREAMFINDER selection can shape the track; the independent Gaia selection (Stage 0b) is the check.
- Particle spray may misplace debris for a massive cluster with internal rotation; Stage 1/4 cross-checks.
- The DM signal so far appears only in counts (not used); if it does not appear in track/PM/width under the bar, the answer
  is "not detectable with this stream", which is a result.

## Compute and process rules

All launches announced with expected %CPU, threads and wall time, with options (OMP_NUM_THREADS). No live N-body. Results in
the journal; this file updated at each gate.
