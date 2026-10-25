# Potentials and auxiliary inputs

Shorthands (`OB` = oCen_bar clone, `CAT`) are defined on the [Data products](index.md) page.

## P1–P7. Potentials and auxiliary model inputs

**P1. Hunter+2024 MW potentials.**
- **Location.** `OB/agama_potentials/MWPotentialHunter24_*.ini`, from the oCen_bar clone (github.com/adllmr/oCen_bar per the project journal).
- **Path convention.** Both `streams/restricted.py:37 HUNTER24_DIR` and `tails/bar_migration.py:25` build the path from the `OCEN_BAR_DIR` env variable (`$OCEN_BAR_DIR/agama_potentials`).
- **Files used:**
  - `axi` (73,032 B): `restricted.py`, `lmc.py:63`, `mw_variants.py:45`, `bar_migration.py:100` and others.
  - `full` (227,828 B): `restricted.py:92`.
  - `baryon_full` and `baryon_axi`: bar = A × (baryon_full − baryon_axi).
  - `spiral` (735,071 B): `streams/spiral.py:82`.
- **Unused:** `nonaxi`, `rotating`, `rotspiral`, `pot_bar_mod`, `PriceWhelan19/22`, `MWPotential2014.ini`.
- **Citation.** Hunter et al. 2024, A&A 692, A216, arXiv:2403.18000: [2024A&A...692A.216H](https://ui.adsabs.harvard.edu/abs/2024A%26A...692A.216H/abstract) (`docs/kb/literature.json`).

**P2. Portail17.ini and Portail17_axi.ini** (241,764 B and 47,001 B).
- Used through opt-in `bar_model='portail17'` (`restricted.py:72,98`).
- Paper: Portail+2017, arXiv:1608.07954 (project journal). The provenance of the .ini files is **UNVERIFIED**.

**P3. MWPotential2014_mod.ini** (469 B).
- Used by `tails/bar_migration.py:155`. Source **UNVERIFIED**.

**P4. McMillan17.ini.**
- The code uses the AGAMA package copy (`site-packages/agama/data/McMillan17.ini`, AGAMA 1.0.152) via glob: `tails/progenitor_orbits.py:222`, `restricted.py:114`, `bin/streams/scope_jacobi.py:39`, `bin/nbody/run_nbody.py:52`.
- A second copy exists at `OB/agama_potentials/McMillan17.ini` (797 B).

**P5. DB98_Model1.ini.**
- File: `configs/potentials/DB98_Model1.ini` (1,284 B). This is the only file in `configs/potentials`.
- Model: Dehnen & Binney 1998, MNRAS 294, 429, arXiv:astro-ph/9612059, Model 1 (the Ibata+2019 host), built from DB98 Tables 1 and 3.
- Checks in the project journal: v_c(R0) = 222.0 km/s; Σ(|z| < 1.1 kpc) = 67.7 M☉ pc⁻².
- Consumers: `restricted.py`, `backtrack_fimbulthul.py:34`, `sgr_*`, `run_db98_ibata_frame.sh`, `run_rotation_tests.sh`.

**P6. Vasiliev, Belokurov & Erkal 2021 "Tango for three" archive.**
- **Location.** `CAT/vasiliev21_tango_zenodo4300977/`.
- **PROVENANCE.txt records:**
  - MNRAS 501, 2279; arXiv:2009.10726
  - https://doi.org/10.5281/zenodo.4300977 (v2; the paper cites v1, 4038141)
  - md5 `151950f0…`; downloaded 2026-10-12
  - units 232,500 M☉, G = 1, time unit 0.9778 Gyr
- **Files used:**
  - `potentials_{axisym,triax}/trajlmc.txt`
  - `accel.txt`
  - `potentials_axisym/potential_evolving.ini`
  - `mw00.pot`
- **Consumers.** `bin/streams/lmc_tango_check.py`, `lmc_vs_tango.py`. The LMC parameters in `streams/lmc.py` follow Tango Section 3.1.
- **Pitfall** (the project journal). AGAMA reorders the components of `potential_evolving.ini`, so select them by type.
- **Registration.** Not registered anywhere in `provenance/` or `configs/`.

**P7. GSE E–Lz contours.**
- **File.** `OB/artifacts/gaiadr3_gse_elz.fits` (17,280 B), labelled "Belokurov+2023" in the project journal with no reference: **UNVERIFIED**.
- **Consumer.** `tails/bar_migration.py:27,157`.

**Internal tails outputs, not external data.**
- `results/tails/{progenitor_orbits_summary.ecsv, progenitor_orbit_tracks.npz}`, from `tails/products.py:48`.
- `bar_migration_*`, `friction_test.ecsv`, `class2_barred.ecsv`, `resonances.ecsv`, `dm_density_grid.ecsv`, `po_*.ecsv`.
- The N-body scripts read only `results/nbody/<model>/{ics.npz, orbit/snap_today.npz}`.

## Frames and host variants (stream models)

These are fixed inputs of every stream spray, set in code rather than read from data files.

- **Solar frames** (`src/ocen_dm/streams/frames.py`, `FRAMES`):

  | Frame | R<sub>0</sub> (kpc) | z<sub>⊙</sub> (pc) | V<sub>⊙</sub> (km/s) | ω Cen distance (kpc) |
  |---|---|---|---|---|
  | `baumgardt` (default) | 8.178 | 0 | (11.1, 252.24, 7.25) | 5.43 (`cluster.OCEN_DISTANCE_KPC`) |
  | `ibata19` (Ibata+2019 host) | 8.122 | 17 | (11.1, 232.24, 7.25) | 5.63 |

  V<sub>⊙</sub> is astropy's `galcen_v_sun`. The model (AGAMA) frame flips the sign of x and v<sub>x</sub> relative to astropy's Galactocentric frame.
- **Cluster reference state** (`src/ocen_dm/cluster.py`, since 2026-10-16): centre (201.69683333, −47.47956944) deg; systemic PM (−3.250, −6.746) ± 0.022 mas/yr; v<sub>los</sub> 232.7 ± 0.2 km/s; distance 5.43 ± 0.05 kpc. Sources are on the [cluster data](cluster.md) page (L3). Runs before 2026-10-16 started from the Baumgardt orbit-table state, which differs by 0.04″ in position, 10⁻⁴ mas/yr in PM and 0.08 km/s in v<sub>los</sub> (`frames.py` docstring).
- **Axisymmetric host variants** (`src/ocen_dm/streams/mw_variants.py`, `axisymmetric_variant`). The Hunter+2024 axisymmetric baryons (`MWPotentialHunter24_baryon_axi.ini`) plus its dark halo, with:
  - halo flattening q (`halo_q`, `axisRatioZ`);
  - disc scale (`disc_scale`), which adds (scale − 1) × the thin and thick stellar discs;
  - v<sub>c</sub>(R<sub>0</sub> = 8.178 kpc) set through the halo normalisation: kept at the original model's value by default, or set to `vc_target`.

  The bar's non-axisymmetric part, A × (`baryon_full` − `baryon_axi`), is added unchanged.
- **LMC** (`src/ocen_dm/streams/lmc.py`). Rigid MW and rigid LMC on their mutual orbit, following Vasiliev, Belokurov & Erkal 2021 (Sect. 3.1), without halo deformation. Particles are integrated in the MW-centred frame with the reflex acceleration, so only the tidal LMC field acts. The LMC is a truncated NFW with r<sub>s</sub> = 8.5 (M/10¹¹ M<sub>⊙</sub>)<sup>0.6</sup> kpc, cut off at 10 r<sub>s</sub>; default mass 1.5×10¹¹ M<sub>⊙</sub> (`with_lmc`). Dynamical friction acts on the LMC only. Present-day LMC: ICRS (81.0, −69.75) deg, PM (1.8, 0.35) mas/yr, d = 50 kpc, v<sub>los</sub> = 260 km/s.
- **Bar size limit** (`bin/streams/training_set.py`). Size S scales the Hunter+2024 bar; at S = 1 its length is R<sub>bar</sub> = 4.25 kpc (the A2 half-peak length; `RBAR_KPC`). In the emulator design S runs from 0.8 to min(1.5, R<sub>CR</sub>/4.25 kpc), so the bar never extends beyond corotation R<sub>CR</sub> of that point's axisymmetric host.
- **Stream background box.** The uniform background of the stream likelihood is normalised over an explicit box (Δφ<sub>2</sub> −10 to 23 deg) since score S3; see [stream likelihood versions](../models/stream_likelihood_versions.md).
