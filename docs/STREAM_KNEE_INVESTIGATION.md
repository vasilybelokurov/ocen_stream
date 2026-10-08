# omega Cen tails vs Fimbulthul: DM cases, rotation, host, and the stream-54 "knee" (2026-10-08)

Consolidated record of the stream experiments of 2026-10-08. Day-by-day detail and all numbers are in `JOURNAL.md`
(entries from "Scoping: restricted N-body tails" to "No published model shows the PM knee"). Plan and user decisions:
[`STREAM_EXPERIMENTS_PLAN.md`](STREAM_EXPERIMENTS_PLAN.md).

## Summary

- **No model reproduces the northern arm of Ibata+2024 stream 54 beyond b ~ 25-30 deg.** In all 15 model variants
  (3 DM cases x {McMillan17, + fitted rotation, + maximal rotation, DB98 + rotation, DB98 + rotation in Ibata's frame})
  the dense model arm lies on the less-negative pmra edge of the data already at b = 15-30, keeps shallow gradients
  beyond b = 30 (pmra -6..-10 vs -10..-20, pmdec -7..-8.5 vs -9..-12.5 at b = 35-40), and continues straight to
  b ~ 45 at l = -50..-60 instead of bending to l ~ -30..-45. v_los agrees (200-230 vs 190-225 km/s at b = 30-40).
- **DM content changes debris counts only** (A no DM > C 5x DM inside 200 pc ~ B moderate DM), not the track or PM
  gradients. **Rotation** fitted to the observed rotation changes counts by 1-2 %; maximal rotation does not help.
  **DB98** raises counts by 22-36 % and broadens the arm, no knee. **Ibata's solar frame** (R0 8.122, Vsun 232.24,
  d = 5.63 kpc) moves the DB98 pericentre from 1.62 to 2.24 kpc and lowers counts; no knee.
- **No published model shows the PM knee either.** Ibata+2019 Fig. 1 plots PMs vs l only for the DR2 Fimbulthul arc
  (b ~ 35, l = -52..-32); their "knee" (l ~ -57, b ~ 20, Fig. 4b) is a sky feature near the cluster. Ibata+2024
  excludes stream 54 from its fit; Zheng+2026 report stronger observed bending than in their and Ibata's simulations.
  The premise "Ibata reproduces the knee" (adopted mid-day) was wrong.
- **Distance is not the explanation (tested).** CMD-shift distances along stream 54 (`bin/streams/stream54_cmd_distance.py`,
  `plots/stream54_cmd_distance.png`) fall by 8-12% from b = 15-20 to b = 33-42, matching the model arm (13%); there is no
  ~30% distance step. The PM knee is therefore a tangential-velocity mismatch at the same distance (~90 km/s at b = 35-40).
  A perturber or extra cluster DM cannot supply this (cluster: <~13 km/s even with 10x DM at 1 kpc; a flyby needs ~5e8 Msun
  within ~0.3 kpc).

- **Chemistry supports membership.** Gaia XP (Andrae+2023) [M/H] of bright members (G < 15.5, giants) in the knee
  (N 19, median -1.36), next to the cluster (N 16, -1.60) and stream 55 (N 18, -1.31) are consistent with omega Cen members at
  the same G (N 1163, -1.47; KS p 0.17-0.80). Fainter XP members are dominated by metal-rich foreground dwarfs everywhere.
  So the knee is most likely omega Cen debris on a track our models do not reproduce.

## Data

| Item | Source | Local |
|---|---|---|
| STREAMFINDER DR3 streams 54 (Fimbulthul, 3724 stars, 29 v_los) and 55 (Fimbulthul-S, 1734, 25) | Ibata+2024, ApJ 967, 89, [arXiv:2311.17202](https://arxiv.org/abs/2311.17202), MRT table 1 | `$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits` |
| Gaia DR3 PM rotation of omega Cen (0.6 deg, ruwe < 1.3, PM err < 0.15) | WSDB `gaia_dr3.gaia_source` | `results/plot_data/gaia_dr3_ocen_rotation{,_pull}.*` |
| oMEGACat VI LOS rotation (per annulus) and per-star v_los | Haberle+ (oMEGACat VI) | `data/processed/kinematics/omegacat_vi_los_rotation.ecsv` |

Ridge of stream 54 (mode-based, `bin/streams/arm_profile_b.py`): pmra -3.4 (b 15-17.5), -4.3 (20-22.5), -5.5 (25-27.5),
-6.0 (27.5-30), -7.0 (32.5-35), -10.2 (35-37.5), -11.8 (37.5-40).

## Models

All runs: massless star tracers (200k from each model's DF-sampled ICs, `results/nbody/<model>/ics.npz`) in a static host
plus the satellite's **prescribed (fixed) spherical potential** moving on the point-mass orbit integrated back from today
(`bin/streams/run_prescribed.py`). Stars with r_max(E) < 30 pc cannot escape and are counted only. ~3-4 min per run on
8-10 threads.

| Model | Mass profile |
|---|---|
| A `A_nodm` | no DM (stars + remnants) |
| B `B_dm_phot` | moderate DM (photometric stellar mass, M/N19 = 9 halo) |
| C `C_dm5x_shell` | B inside 63 pc; DM density 0.159 Msun/pc^3 at 63-200 pc (M_DM(<200 pc) 1.15e6 -> 5.73e6); B's star tracers |

Validation (earlier the same day): the refit restricted runner (`bin/streams/run_restricted.py`, satellite potential refitted
every 2 Myr from bound tracers, frozen core) agrees with the live pyfalcon run of A over 1.96 Gyr to 10 % in escape
fraction and 3 % in debris dE/dLz (`plots/streams_validation_A.png`); the prescribed potential loses 12 % fewer stars.

### Rotation (`src/ocen_dm/streams/rotation.py`, `bin/streams/fit_spin.py`)

Lynden-Bell flips: counter-rotating tracers have v_phi (about the spin axis) reversed with probability
q(r_max) = q0 x^2/(1+x^2)/(1+(r_max/r_q)^2), x = r_max/r_1. E and |v| unchanged (max change 1e-13 km/s), so the spherical
equilibrium is exact. Axis: Ibata+2019 (PA 12 deg E of N, pole 45 deg towards us); sense from our data: oMEGACat v_los
maximal at PA 103 deg (east receding) -> pole sky PA 192; Gaia DR3 rotation N -> E (-0.285, -0.245, -0.193, -0.125, -0.039
mas/yr at 3-6, 6-10, 10-15, 15-25, 25-36 arcmin; `bin/streams/measure_gaia_rotation.py`) -> pole towards us. Joint fit to
Gaia PM + oMEGACat LOS amplitudes: tilt 50 deg, q0 = 1, r_1 = 4 pc, r_q = 63 pc (all three models); chi2 PM 7-13 / 5 bins,
LOS 60-63 / 20 bins (model 1.5-2 km/s high inside 2'). Rotation in the escape region (36-60'): ~1 km/s. Maximal rotation:
q = 1 at all radii (`results/streams/spin_max/`).

### Frames (`src/ocen_dm/streams/frames.py`)

`baumgardt`: R0 8.178, z_sun 0, Vsun (11.1, 252.24, 7.25), d 5.43 (reproduces the legacy OCEN_TODAY to 0.6 km/s).
`ibata19`: R0 8.122, z_sun 17 pc, Vsun (11.1, 232.24, 7.25), d 5.63 (Ibata+2019 Methods); omega Cen's Galactocentric v_y
changes by 24 km/s.

## Runs and results

Unbound tracers within 70 deg of omega Cen today (A / B / C):

| Run dir (`results/streams/`) | Host, time, rotation, frame | Counts | Figures (`plots/`) |
|---|---|---|---|
| `prescribed` | McMillan17, 1.96 Gyr, none, baumgardt | 7890 / 5306 / 4949 | `streams_debris_vs_ibata2024_r70_xb.png`, `streams_arm_overlay_pm_b.png`, `streams_dm_models.png` |
| `prescribed_rot` | McMillan17, 1.96, fitted | 8073 / 5375 / 4990 | `..._r70_xb_prescribed_rot.png`, `streams_arm_overlay_pm_b_rot.png` |
| `prescribed_maxrot` | McMillan17, 1.96, maximal | 8004 / 5377 / 5134 | `..._r70_xb_prescribed_maxrot.png`, `streams_arm_overlay_pm_b_maxrot.png` |
| `db98_rot` | DB98 Model 1, 1.96, fitted, baumgardt | 9866 / 7321 / 6742 | `..._r70_xb_db98_rot.png`, `streams_arm_overlay_pm_b_db98rot.png`, `streams_knee_release_db98_rot_A_nodm.png` |
| `db98_rot_5g` | DB98, 5 Gyr, fitted, baumgardt | A only (queue stopped) | none |
| `db98_ibata` (deleted) | DB98, 1.96, fitted, ibata19 | 7057 / 4652 / 4264 | `..._r70_xb_db98_ibata.png`, `streams_arm_overlay_pm_b_db98ibata.png` (vertical streak at b = 15 is an artefact: centre ended 7.3 pc off, see below) |

Present-day model tracks (all variants): today's prescribed centre within 0.25-0.74 pc of omega Cen (except `db98_ibata`).

### Release times (`bin/streams/knee_release_time.py`, `bin/streams/plot_knee_release.py`)

Model debris at b > 30 in the northern arm was released 300-600 Myr ago (4-6 pericentres; period ~90 Myr); the b = 20-30
arm ~150 Myr ago; none > 1 Gyr. Ibata+2019: the Fimbulthul material left ~0.4 Gyr ago.

### Backtracking the observed members (`bin/streams/backtrack_fimbulthul.py`)

29 stream-54 stars with v_los, distance scanned 2.5-6.5 kpc, integrated back 1.5 Gyr in static McMillan17 and DB98: b < 30
stars return to omega Cen's orbit (dx < 0.33 kpc, dv < 35 km/s); b > 32 stars do not (dv 37-150 km/s). Conditional on a
static axisymmetric host and test-particle orbits; it does not exclude the knee stars as omega Cen debris.

## Corrections and pitfalls recorded

1. **Bin/box medians misled twice.** (a) Model arm medians in 5-deg bins were dragged by a broad low-pmra spray and
   wrongly "agreed" with the data at b < 30; the overlay shows the offset. (b) Medians of the few model tracers inside a
   hand-picked knee box (47-61 per model) were wrongly presented as a "model knee"; `streams_knee_release_db98_rot_A_nodm.png`
   shows only a sparse spray there. Rule: make the overlay before any comparison claim.
2. **Parallax-distance argument withdrawn**: it assumed agreement at b < 30. Stream-54 absolute parallax at the cluster
   (0.25 mas) exceeds omega Cen's 0.184 by 0.065 mas; only relative trends are usable.
3. **Centre-orbit accuracy**: AGAMA's default accuracy (1e-8) loses 5-12 pc in a 2-Gyr backward-forward round trip;
   `run_prescribed.py`, `run_restricted.py` and `restricted.centre_orbit` now use 1e-12. Irrelevant for arm morphology,
   but it broke the bound/unbound split in `db98_ibata` plots.
4. **Frame consistency**: DB98 runs before the `ibata19` frame used McMillan17's solar velocity (252 km/s) with a host of
   v_c(R0) = 222 km/s.
5. **Distance 5.43 vs 5.63 kpc** is a 3.7 % effect on PMs: not a candidate explanation.

## Literature checked (arXiv full text)

- Ibata+2019, Nat. Astron. 3, 667, [arXiv:1902.09544](https://arxiv.org/abs/1902.09544): live gyrfalcON, 1e5 particles,
  eps 1 pc, DB98 Model 1, 5 Gyr, Varri-Bertin rotating model (M 2.47e6, C 1.27), axis PA 12, pole 45 towards us; start
  (3.3, -5.6, 2.0) kpc, (98.3, 20.0, -47.2) km/s (does not land on today's omega Cen in our DB98 after 4.5-5.5 Gyr);
  non-rotating King gave a "substantially wider" stream; self-gravity of earlier debris "required".
- Ibata+2024, [arXiv:2311.17202](https://arxiv.org/abs/2311.17202), Sec. IX.5, Fig. 23: rotation needed for the
  "knee-shaped structure"; stream 54 excluded from the potential fit; stream 55 = older (0.5-1 Gyr) trailing debris at ~3 kpc.
- Zheng+2026, [arXiv:2603.02904](https://arxiv.org/abs/2603.02904): PeTar, MWPotential2014, 0.8 Gyr; observed bending
  stronger than simulated, also in Ibata's.
- Ferrone+2023 (e-TidalGCs), [arXiv:2301.05166](https://arxiv.org/abs/2301.05166): test-particle tails.
- 2026 spectroscopic study of the periphery and tails, [arXiv:2605.23474](https://arxiv.org/abs/2605.23474).
- Codex second opinion (ask-codex, repo-aware, effort high): no frame/sign/time bug; suggested present-day phase-space
  grid, projection test, catalogue cross-checks.

## User decisions recorded

No live N-body for now (user does not consider stream self-gravity the explanation); no 30-h runs; warn about CPU and offer
thread options before every launch; refit 5-Gyr run stopped as redundant; `db98_ibata` rerun stopped (accuracy fix
irrelevant for the knee).

## Next steps (proposed, not started)

1. Done: distance step excluded (see Summary).
2. Particle-spray streams (AGAMA, Fardal+15 / Chen+24) for fast scans of omega Cen's present-day PM/distance and the host's
   disc/halo parameters.
3. Cross-check the DR3 knee against DR2 candidates and RV/chemistry members (STREAMFINDER template influence).

## Code index

`bin/streams/`: `run_prescribed.py` (`--spin`, `--frame`, `--mw <ini>`), `run_restricted.py` (`--spin`), `fit_spin.py`,
`measure_gaia_rotation.py`, `plot_debris_vs_ibata.py` (`--radius --xlower --runs`), `plot_arm_overlay.py`,
`arm_profile_b.py`, `knee_region_check.py`, `knee_release_time.py`, `plot_knee_release.py`, `backtrack_fimbulthul.py`,
`run_rotation_tests.sh`, `run_db98_ibata_frame.sh`, `make_shell_model.py`, `compare_models.py`.
`src/ocen_dm/streams/`: `restricted.py`, `analysis.py`, `rotation.py`, `frames.py`.
