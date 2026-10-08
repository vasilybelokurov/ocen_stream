# Restricted N-body tails with and without DM vs Fimbulthul — scoping (2026-10-08)

Status: **proposal, not implemented.** Decisions marked D1–D5 are the user's.

## Question

How do no DM and different amounts of DM around omega Cen change the structure of its tidal
tails? Compare the results with Fimbulthul (Ibata+2019). Tails are made by constrained
(restricted) N-body: prescribed host and satellite potentials, with stars as massless tracers
(mLCS, Gibbons+2014; streams of Erkal+2019 and Vasiliev+2021).

## Findings from scoping

1. **Within the kinematically allowed models the DM barely changes the escape scale.**
   `bin/streams/scope_jacobi.py` (results/plot_data/streams_scope_jacobi.json) gives the
   instantaneous r_J in spherically averaged McMillan17:

   | model | rho20 | M_halo (taper 1000 pc) | M(<r_J) peri | r_J peri / apo [pc] |
   |---|---:|---:|---:|---|
   | A no DM | 0 | 0 | 3.24e6 | 78.7 / 195 |
   | rho20scan | 0.25 | 1.7e5 | 3.28e6 | 79.0 / 197 |
   | | 0.5 | 3.5e5 | 3.32e6 | 79.3 / 198 |
   | | 1 | 6.9e5 | 3.39e6 | 79.9 / 201 |
   | | 2 | 1.39e6 | 3.56e6 | 81.2 / 207 |
   | | 4 | 2.78e6 | 3.91e6 | 83.8 / 219 |
   | B (M/N19 = 9) | 2.6 | 1.81e6 | 3.63e6 | 81.8 / 210 |

   The halo is compact (r_s at 5 pc). Its mass inside r_J is offset by a lower M_star, so
   M(<r_J) rises by at most 21% and r_J by at most 6.5% at pericentre. Stream width and
   dispersion scale as M^{1/3}, so the expected differences are <~7%. This matches the
   live runs, where the tail kinematics agree to within 10%. **The tails respond to DM
   beyond ~80–200 pc, which the kinematics do not constrain.** The DM axis therefore has to
   be defined explicitly (D1).
2. **Orbit, host potential and integration time dominate.** The live 1.96 Gyr McMillan17
   debris in the Fimbulthul box has mu_dec = -6 to -9 mas/yr, against -9.7 to -11.5
   observed. Ibata+2019 used a static Dehnen & Binney (1998) Model 1, needed >= ~5 Gyr
   with a rotating progenitor, and found that a non-rotating King progenitor gives a
   substantially wider stream. Progenitor rotation is therefore a non-DM systematic at
   least as large as the expected DM signal.
3. **The tooling exists in AGAMA 1.0.152; it is not yet in the repo.**
   `agama/py/tutorial_streams.ipynb` and `example_tidal_stream.py` provide particle spray
   (Fardal+2015 as in Gala, and Chen+2024), streaklines with linear perturbation, and
   restricted N-body (Vasiliev+2021). In the restricted scheme, tracers move in host plus
   moving satellite potential, and the satellite potential and mass are refitted from the
   bound tracers at intervals, with Chandrasekhar friction optional. Potentials accept
   time-dependent `center=` and `scale=`.
4. **Data.** Local: I19 Table 1 (309 DR2 candidates, data/processed/tails/fimbulthul_members.ecsv)
   and 5 CFHT RVs (mean 199.7, rms 5.4 km/s). Also available: the Ibata+2024 DR3
   STREAMFINDER atlas (Fimbulthul and the separate Fimbulthul-S), Simpson+2020 (2 GALAH
   stars) and Kuzma+2026. No intrinsic width, RV dispersion or density has been
   published. Width and track have to be measured from the candidates. Absolute density is
   excluded without a STREAMFINDER selection model.

## Proposed experiment set

- **S0 — track (no DM).** Use fast spray/mLCS with satellite gravity to find a host,
  integration time and present-day 6D point that reproduce the Fimbulthul sky, PM, distance
  and RV track. Candidate hosts: McMillan17 and I19's DB98 Model 1 (D3). Then freeze them.
  The DM grid must not retune the orbit.
- **S1 — validation.** Run restricted N-body for A and B over 1.956 Gyr in static
  McMillan17, from the same DF samples as the live runs. Compare with the live runs:
  unbound fraction history, dE/dLz spreads, tail width, near-tail sigma_los and sky track.
  Tolerances are agreed in advance and tested, not assumed.
- **S2 — DM grid** on the frozen S0 orbit: no DM, plus the axis chosen in D1, with tracers
  of all species. Satellite potential per D2.
- **S3 — comparison** in Fimbulthul stream coordinates: phi2(phi1), PM tracks, distance, RV
  track, width, and (from model tracers only) along-stream density and epicyclic spacing.
  Apply an approximate STREAMFINDER-like selection (G limit, PM window) to the models.

Cost (estimate, to be benchmarked): spray tracks take seconds to minutes each. Restricted
runs with 1e5–1e6 tracers over 2–5 Gyr take of order 10–60 min each with AGAMA's
multithreaded integrator, against ~30 h for each live run. No run is launched without a go.

## Decisions for the user

- **D1 DM axis.** (a) The rho20 grid at the fitted taper. This is kinematically consistent
  and the expected result is null at <~7%. (b) Extended halos (taper / M_halo 1e6–1e8)
  that keep the fitted inner profile, the regime of Vitral & Boldrini 2022, Errani+2022 and
  Moore 1996. (c) Both: (a) as the control and (b) as the test.
- **D2 Satellite potential.** (i) Self-consistent restricted N-body (Vasiliev+2021): halo
  stripping follows from the tracers, so no hand-tuned halo-loss law is needed. (ii)
  Strictly prescribed potential (`scale=` in t, or r_J-truncated halo): closer to
  "parameterised", but the halo-loss law has to be assumed. Codex: calibrating it on live
  run B only is circular for the other halos.
- **D3 Host and time.** McMillan17 vs DB98 Model 1, 2 Gyr vs ~5 Gyr. A 5 Gyr run needs a
  progenitor heavier than today's model, and that mass has to be chosen.
- **D4 Rotation.** Keep non-rotating now, or add a rotating DF before S2 (I19: rotation
  matters for width).
- **D5 Data.** Fetch the Ibata+2024 DR3 members (VizieR) in addition to the local I19 table.

## References (arXiv IDs verified 2026-10-08)

Gibbons, Belokurov & Evans 2014, arXiv:1406.2243; Fardal, Huang & Weinberg 2015, arXiv:1410.1861;
Küpper, Lane & Heggie 2012, arXiv:1111.5013; Erkal+2019, arXiv:1812.08192; Vasiliev, Belokurov &
Erkal 2021, arXiv:2009.10726; Ibata+2019, arXiv:1902.09544 (VizieR J/other/NatAs/3.667);
Ibata+2021, arXiv:2012.05245; Ibata+2024, arXiv:2311.17202; Malhan, Valluri & Freese 2021,
arXiv:2005.12919; Vitral & Boldrini 2022, arXiv:2112.01265; Errani+2022, arXiv:2203.02513;
Evans, Strigari & Zivick 2022, arXiv:2109.10998; Kuzma+2026, arXiv:2605.23474.
Codex research pass (ordinary web-research fallback, not Deep Research) supplied the
bibliography. The I19 method details (DB98 Model 1, >= 5 Gyr, 1e5 particles, rotating
progenitor) are as reported by Codex from the Methods section and are consistent with the
journal entry of 2026-09-26. The Ibata+2024 Fimbulthul membership count (22) is UNVERIFIED.
