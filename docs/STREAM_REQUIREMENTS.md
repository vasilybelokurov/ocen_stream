# What the omega Cen stream (stream 54) requires to be fitted well

Status: 2026-10-22. Summary figure: `plots/stream_requirements.png` (`bin/streams/requirements_figure.py`).

## Summary

- **A good fit needs a non-axisymmetric force at R = 4-7 kpc about 2.5-3.5 times that of the published Milky Way bar**,
  locked to the bar (co-rotating, in phase). With the bar's strength and size free, every good fit inflates the
  Hunter+2024 / Sormani+2022 bar to amplitude A ~ 1.4 and size S ~ 1.25-1.33. Its quadrupole is then ~2.6-3.2 times the
  published one; inside 3 kpc the increase is only 1.2-1.4 times (panel a).
- **With the bar fixed at its published strength, size and angle (27 deg), the best fit is 2,627 worse in ln L** than the best
  inflated-bar fit, and even that needs an ultrafast bar (Omega_b = 44.8 km/s/kpc; panels b, c). The failure is mainly
  mu_alpha* in the middle of the stream, where the model is 1-3 mas/yr too negative at phi1 7-19 deg: the PM turn starts
  too early.
- **The stream measures a torque history, not a static force.** Along omega Cen's orbit the extra force averages to zero
  (<= 0.003 |F_axi| over 300-700 Myr) but its rms is about 2x the published bar's. omega Cen's L_z swings more widely
  (panel d), and the 300-700-Myr-old debris records that history. No axisymmetric change (disc, halo flattening, v_c,
  solar frame, distance, PMs) can replace it.
- **Within the inflated-bar model the stream prefers a low bar angle** (12.5-17.5 deg; 25 deg is 1,071 worse, 27.5 deg
  1,885 worse). The preference survives every test below. It is conditional on the inflated bar.
- **The demanded bar conflicts with local constraints:** its m=2 strength at the solar radius is 0.062 |F_axi|, against
  0.01-0.02 in Hercules/OLR models. Equivalently, the long bar would need to be ~2x as massive as Sormani+2022's (total bar
  ~2.4e10 vs 1.83e10 Msun) and ~1.3x longer, reaching corotation (~6.4 kpc, against 5.0 +- 0.2 kpc measured).

## Model and data

- **Members:** STREAMFINDER (Ibata+2024) stream 54 within the stream's footprint; Gaia DR3 PMs with covariances; 29 v_los.
- **Model:** Fardal-type particle spray from omega Cen in the Hunter+2024 barred Milky Way. The bar is A x
  [baryon_full(x/S) - baryon_axi(x/S)] rotating at Omega_b; the axisymmetric part uses an Einasto halo with flattening q,
  normalised to v_c(R0).
- **Frame and priors:** gravity21 solar frame (R0 8.275 kpc, Sgr A* PM, z_sun 20.8 pc). omega Cen's distance is held in
  5.33-5.53 kpc, with priors on the distance, systemic PMs and v_c (233 +- 7 km/s).
- **Score:** conditional likelihood p(Delta phi2, PMs, v_los | phi1) with fixed chi bins 0-200, jackknife-corrected; debris
  younger than 700 Myr.
- **Parameters (9):** bar angle, Omega_b, A, S, d, mu_alpha*, mu_delta, v_c, q.
- **Statistics:** the score is overconfident by T ~ 2 (calibrated on 20 mock catalogues, `bin/streams/tempering_calibration.py`).
  The star bootstrap gives the right sampling errors; the block bootstrap overstates them 2-6x. Statistical sd of
  angle-to-angle differences is ~30-80 in ln L.

## Evidence

| Test | Result | Source |
|---|---|---|
| Per-angle profile (inflated bar free) | 12.5 deg best (-20598); 17.5 -150, 22.5 -644, 25 -1071, 27.5 -1885, 32.5 -2125; one parameter family; segments agree | `results/streams/profile_step2_20261020`, `plots/segment_profile_step2.png` |
| Published bar fixed at 27 deg, Omega_b 20-50 free | best Omega_b 44.8: -23225 (gap 2627); Omega_b 33.4: -23859 | `bin/streams/fixedbar_fit.py`, `plots/bestfit_real_fixedbar_*.png` |
| Spiral arms (Hunter, amplitude 1) | only 0.5-2% of F_axi at 4-8 kpc, 6-10x too weak; at most ~600 recovered | `bin/streams/bar_prescreen.py` |
| Compact Ferrers bar | force concentrated inside 3 kpc, <= 0.06 beyond 6 kpc | `bin/streams/diagnostics/bar_options.py` |
| Long-bar scaling (Sormani components) | 2x mass and 1.3x length reproduces the demanded force profile | `plots/bar_options.png` |
| Extra force as a separate pattern | only co-rotating, in-phase recovers the fit (1.0); static at 90 deg 0.63 at 25 deg (0.25 at 12.5) with worse PMs and far end; all others negative | `bin/streams/excess_pattern_test.py`, `plots/excess_pattern_*.png` |
| Orbit-averaged force, L_z history | mean ~0, rms 2x published; L_z -609 -> -512 (fitted) vs -561 -> -512 (published) | `bin/streams/diagnostics/orbit_force_check.py` |
| Debris age cut (fixed params) | low angles (<= 17.5) beat >= 22.5 by 400-1400 at every cut; 12.5 vs 17.5 depends on the cut | `results/plot_data/agecut_rescore_step2*.json` |
| Pericentric vs uniform release | angle gap 840-1110 in both; branch prediction unchanged | `bin/streams/peri_release_test.py` |
| Gaia DR3 search for the 25-deg model's off-track branch | branch amplitude A = 1 disfavoured at 2.6-4.3 sigma (phi1 14-22); STREAMFINDER did not miss it | `bin/streams/branch_search_gaia.py`, `plots/branch_search_profiles.png` |
| Mechanism of the 25-deg branch | set by the bar phase at the release pericentre; Omega_b + 1.5 removes it (but costs 1144 at fixed params) | `bin/streams/bar_prescreen.py` |
| Per-observable split | sky track and PMs contribute comparably (12.5 vs 25: +771 and +729); v_los negligible | `bin/streams/excess_pattern_plots.py` (obs_split) |

## What the stream does not require (tested and ruled out as substitutes)

- An axisymmetric change: these cannot mimic a torque history. Halo flattening and disc mass were tested on 10-10 and all
  made the fit worse; disc_scale was not refitted in the current setup.
- Spiral arms at the published amplitude.
- A slowing bar (tested on 10-10 in the old frame: worse at every pattern speed).
- A release-history change (pericentric bursts).
- A selection effect: the Gaia search finds no off-track debris.

## Remaining misfits, common to every model tried

- Delta phi2 offsets: -1.2 to -2 deg at phi1 3-15 and +1.2 to +2.3 deg at 25-27 deg.
- Knee mu_delta 0.5-1 mas/yr too shallow (phi1 > 19); knee debris 7-10% too close against the CMD distance track.
- Fit pulled to range edges: distance at the band top (5.52-5.53 kpc), v_c 223-226 km/s.
- Along-stream density: the members' ~10x gap at phi1 6-13 is not reproduced (the score does not use the density).

## Interpretation

The stream constrains a **bar-locked, non-axisymmetric force history** along omega Cen's retrograde orbit (pericentre
1.5-2 kpc, apocentre 6.4-7.2 kpc) more firmly than it constrains the bar angle. Within the Hunter bar family this force
comes out ~2x the published bar's outer force; with it, the angle is low (~12-18 deg). The published bar cannot fit the
stream at any pattern speed in 20-50. The most physical candidate not yet tested is a co-rotating "dark bar", the halo's
response to the stellar bar (Petersen+2016), which extends beyond the stellar bar. Whether it can supply the force without
violating the solar-neighbourhood constraints is open.

## Statistical vs systematic uncertainty

With T ~ 2 the formal (statistical) errors are tiny: angle ~0.3-1 deg, Omega_b ~0.1, A ~0.02. The posterior's 68% region spans
Delta lnL ~ 10, below the simulation noise (sd 14-35 at 256k). Model changes move the best fit 50-500 times further, so the
uncertainty is systematic and is represented by the spread between model variants. A Laplace covariance for the final model
is planned once the model is settled (docs/reviews/fable_posterior_revisited_20261022.md).

## Next steps

1. A co-rotating dark-bar (halo-wake) component as the physical source of the extra force.
2. A Hercules / solar-neighbourhood check of the demanded bar (U-V distribution with test particles, against Gaia).
3. With the final model: a Laplace covariance, and per-segment residuals.

## References

- Hunter et al. 2024, A&A 692, A216 (Milky Way potential with bar and spiral) -- ADS 2024A&A...692A.216H.
- Sormani et al. 2022, MNRAS 514, L1 (analytic bar model) -- ADS 2022MNRAS.514L...1S.
- Portail et al. 2017, MNRAS 465, 1621 (bar mass, pattern speed) -- arXiv:1608.07954.
- Wegg, Gerhard & Portail 2015, MNRAS 450, 4050 (long-bar length) -- arXiv:1504.01401.
- Clarke & Gerhard 2022, MNRAS 512, 2171 (pattern speed 33.3 +- 1.8) -- arXiv:2107.10875.
- Lucey et al. 2023 (dynamical bar length) -- arXiv:2206.01798; Vislosky et al. 2024 -- ADS 2024MNRAS.528.3576V.
- Dehnen 2000, AJ 119, 800 (bar and Hercules) -- ADS 2000AJ....119..800D; Hunt & Bovy 2018 -- arXiv:1803.02358.
- Petersen, Weinberg & Katz 2016 (dark-matter "shadow bar") -- arXiv:1602.04826 (UNVERIFIED citation details).
- Ibata et al. 2024 (STREAMFINDER catalogue) -- ApJ, table apjad382dt1 (bibcode not re-checked).

Reviews behind this note: docs/reviews/fable_bar_model_20261021.md, fable_bar_force_20261021.md,
fable_selection_tests_20261021.md, fable_posterior_revisited_20261022.md, codex_solar_frame_20261020.md.
