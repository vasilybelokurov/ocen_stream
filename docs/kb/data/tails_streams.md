# Periphery, tails and streams

Shorthands are defined on the [Data products](index.md) page.

## T1/T2. Kuzma & Ishigaki 2025 Pristine periphery and its Gaia covariance

- **Source.**
  - arXiv: https://arxiv.org/abs/2502.01135
  - Journal: https://academic.oup.com/mnras/article/537/3/2752/8002861
  - Data: https://doi.org/10.5281/zenodo.14791430
- **Raw file.** `raw/kuzma2025_pristine/wCen_table.fits` (26,470,080 B). The published md5 `3dcd58e6…` was verified.
- **T1 builder.** `data/kuzma2025.py:117`.
- **T1 product.** `proc/tails/kuzma2025_periphery.ecsv` (57,434,008 B; 157,481 rows; to 5.1°).
  - Columns: `source_id`, `ra`, `dec`, `pmra`, `pmdec`(`_error`), `g0`/`bp0`/`rp0`/`cahk0_mag`, `membership_prob`, `feh(_lo/_hi)`, `src_*`.
  - Metadata rule: "membership_prob is diagnostic only; do not use it as a likelihood weight alongside the same PM/photometric information".
- **T2 builder.** `selection/gaia_covariance.py:50`: WSDB `sqlutilpy.local_join` on `gaia_dr3.gaia_source`, 2026-09-17T01:00Z, matched fraction 1.0.
- **T2 product.** `proc/tails/kuzma2025_periphery_gaia_covariance.ecsv` (37,850,464 B).
  - Metadata rule: "use the full 2x2 PM covariance… parallax terms for a future 3D treatment".
- **Consumers.**
  - `kinematics/periphery.py:64` (fixed-window PM dispersion; documented as circular because membership uses PMs).
  - `kinematics/pristine_profile.py:41` (mixture without a PM cut).
  - `selection/periphery_density.py:49`.
  - `plotting/constraints.py:1187–1466`.
  - `plotting/data_overview.py`.
- **Status.** d. Not in any fit.

## T3. Kuzma et al. 2026 spectroscopy

- **Source.**
  - arXiv: https://arxiv.org/abs/2605.23474
  - DOI: https://doi.org/10.1093/mnras/stag1147
  - ESO programme 108.22MM.001
- **Raw file.** `raw/kuzma2026_spectroscopy/kuzma2026_members.txt` (66,462 B). The user supplied it on 2026-09-16 as the extended Table 3 (592 rows, 157 members). There is no publisher checksum.
- **Reader.** `configs/data.yaml` uses `ascii.commented_header`, `header_start=-1`.
- **Builder.** `data/kuzma2026.py:118`.
- **Product.** `proc/tails/kuzma2026_spectroscopy.ecsv` (66,938 B). Columns `star_id`, `source_id`, `ra`, `dec`, `vlos(_error)` [km/s], `feh(_error)`, `g_mag`, `membership_prob`, `member_flag`.
- **Metadata rule.** "targeted sample: condition on target positions; do not treat detection counts as an unbiased tail surface density". This is also stated in `docs/MANUAL_DOWNLOADS.md`.
- **Consumers.** `kinematics/periphery.py:89`, `plotting/constraints.py:1277`, `plotting/data_overview.py:183,349`.
- **Status.** d. The N-body comparison hard-codes literature numbers instead (`bin/nbody/plot_observed_comparison.py:172–178`).

## T4. Fimbulthul candidates (Ibata et al. 2019)

- **Source.**
  - DOI: https://doi.org/10.1038/s41550-019-0751-x
  - arXiv:1902.09544, per `streams/frames.py:7`
  - VizieR `J/other/NatAs/3.667/tables1`; this is a CDS-curated representation of the published table.
- **Raw file.** `raw/fimbulthul_ibata2019/fimbulthul_supp_table1.vot` (71,578 B; sha256 `9d06aca4…`). The bytes are not reproducible because the VizieR export embeds a timestamp.
- **Builder.** `data/fimbulthul.py:92`.
- **Product.** `proc/tails/fimbulthul_members.ecsv` (37,618 B; 309 rows). Columns `name`, `ra`, `dec`, `pmra/pmdec(_error)`, `parallax(_error)`, `g0_mag`, `bp_rp0`.
- **Metadata rule.** "stream-track constraint only; no absolute density likelihood without a STREAMFINDER selection model".
- **Consumers.** `plotting/data_overview.py:224` only.
- **Status.** d. Superseded for stream modelling by S1 (all 309 are in Ibata+2021 stream 16; 264 are in Ibata+2024 stream 54, per `configs/data.yaml` and the project journal).

## S1. STREAMFINDER Gaia DR3 atlas (Ibata et al. 2024): the stream-54 sample

- **Source** (`configs/data.yaml:50–61`). ApJ 967, 89; arXiv:2311.17202; ADS 2024ApJ...967...89I. Table 1 is the MRT `apjad382dt1_mrt.txt` from content.cld.iop.org (not on VizieR). Fetched 2026-10-08; the MRT sha256 is `76742daa…`.
- **Files.**
  - `CAT/streamfinder_ibata2024_dr3_mrt.txt` (2,432,198 B): the registered `path`.
  - `CAT/streamfinder_ibata2024_dr3.fits` (2,364,480 B): what the code reads. No conversion script and no checksum exist for it.
- **Contents.** 24,540 rows.
  - Stream 54 (Fimbulthul): 3,724 stars, 29 with VHel.
  - Stream 55 (Fimbulthul-S): 1,734 stars, 25 with VHel.
  - `e_VHel = 300` means null.
- **Loader.** `bin/streams/spray_bar_grid2.py:24–33 load_data`.
  - Keeps Stream == 54 with b > 15° and −75° < l < −20°.
  - Sets VHel to NaN unless `e_VHel < 300`.
  - Attaches S2 errors and correlations.
  - About 20 scripts import it (`fit_bar_ocen.py`, `grid4_common.py` → grid4, `fit_lmc_ocen`, `rescore_*` and others). About 15 more read the FITS directly with the same cut.
- **Scoring.** `src/ocen_dm/streams/score.py`.
- **Restrictions.**
  - Track-only (STREAMFINDER selection, as for T4).
  - Ibata+2024 exclude stream 54 from their own potential fit (the project journal, 6618).
  - Member parallaxes are biased by +0.063 mas relative to the cluster (the project journal).
- **Status.** c.

## S2. Gaia DR3 errors and correlations for stream-54/55 members

- **Product.** `results/plot_data/stream5455_gaia_errors.npz` (168,341 B; 5,458 rows). Keys `source_id`, `pmra_error`, `pmdec_error`, `pmra_pmdec_corr`, `parallax_error`, `ruwe`, `g`.
- **Builder.** **None in the repo**: it was an ad hoc WSDB `source_id` join, recorded only at the project journal.
- **Consumers.** Every `load_data` user; `measure_track.py:21`.
- **Status.** c. This is a reproducibility gap.

## S3. STREAMFINDER EDR3 (Ibata et al. 2021)

- **Source.** ApJ 914, 123; arXiv:2012.05245; VizieR `J/ApJ/914/123/table1`. FITS sha256 `d5cdffa6…` (`configs/data.yaml:40–49`).
- **File.** `CAT/streamfinder_ibata2021_edr3.fits` (457,920 B; 5,960 rows; stream 16 = Fimbulthul, 1,006 stars).
- **Consumers.** None.
- **Status.** e. It is superseded by S1 (`data.yaml:58`).

## S4. Derived stream-54 geometry

All files are in `results/plot_data`.

| File | Builder | Consumers |
|---|---|---|
| `stream54_track.json` | `bin/streams/measure_track.py:39`, using `streams/track.py` | spray-grid scripts, `build_stream_path.py`, `validate_spray.py`, untracked `check_stream_coordinate.py` |
| `stream54_path.json` | `bin/streams/build_stream_path.py:16` | `build_gc_frame.py`, `knee_spread_and_frame.py`, `rescore_conditional.py` |
| `stream54_gc_frame.json` and `stream54_gc_frame_chord.json` | `bin/streams/build_gc_frame.py:18` (`--chord`) | the chord frame is current: grid4, `fit_bar_ocen`, `parallax_check`, `m5_independent_selection`, `rescore_conditional` and others |
| `stream54_cmd_distance.json` | `bin/streams/stream54_cmd_distance.py` | — |

The cluster mass profile used by the spray is `results/nbody/A_nodm/model_profiles.json`, an internal product. Status: c.

## S5–S7. WSDB caches in the stream project

- **S5, Gaia XP [M/H].**
  - Products: `results/plot_data/stream_xp_metallicity_members.npz` (74,969 B; 5,481 rows) and `ocen_members_andrae.npz` (34,416 B; 3,993 rows).
  - Built by q3c joins to `koposov.andrae2023_v21_table1` on WSDB (the project journal). **There is no builder script.**
  - Consumer: `bin/streams/plot_stream_xp_metallicity.py`.
  - Status: d. The project journal calls the result inconclusive.
  - The Andrae+2023 paper reference (arXiv or DOI) is not recorded: **UNVERIFIED**.
  - Large local XP/Andrae files in `CAT` (`gaia_xp_andrae_parallax01.fits` 1.89 GB, `gc_members_vasiliev_XP.fits` 280 MB, `xp_rvs_andrae_*`) are not referenced by repo code.
- **S6, Gaia DR3 rotation pull.**
  - Builder: `bin/streams/measure_gaia_rotation.py:18–25`. Cuts: within 0.6°, RUWE < 1.3, PM errors < 0.15 mas/yr, PM window ±2 mas/yr.
  - Products: `results/plot_data/gaia_dr3_ocen_rotation_pull.npz` (526,559 B; 19,062 rows) and `gaia_dr3_ocen_rotation.json`.
  - Consumer: `bin/streams/fit_spin.py:43` (spin-tracer fit).
  - Status: c.
- **S7, knee box and stream-54 parallax.**
  - `results/streams/m5_gaia_knee.npz` (241,961,873 B), built by `bin/streams/m5_independent_selection.py:33–47`. Status: d. It is a candidate for clean-up.
  - `results/streams/stream54_parallax.npz` (75,847 B), built by `bin/streams/parallax_check.py:22`. Status: d.

## S8. Full Baumgardt GC compilation

- **File.** `CAT/gc_catalog_full.fits` (60,480 B), described as a "Baumgardt & Vasiliev 2021 compilation" in the `sgr_orbit.py:5` docstring.
- **Consumers.** `bin/streams/sgr_orbit.py:29`, `sgr_recent_passages.py:21` (Sgr-passage side study).
- **Not registered.** Only `gc_catalog_updated.fits` is in `datasets.yaml`.
- **Status.** c.

## S9. ω Cen HST Large Programme e-material

- **Location.** `CAT/omegacen_hst_lp/` (P04–P07). `README_provenance.txt` holds the sha256 values; the files were fetched on 2026-10-08 from https://web.oapd.inaf.it/bedin/files/PAPERs_eMATERIALs/wCen_HST_LargeProgram/.
- **Papers:**
  - Scalco+2021, arXiv:2107.08726
  - Gerasimov+2022, arXiv:2111.10063
  - Scalco+2024, A&A 688, A180
  - Scalco+2024, A&A 691, A96, arXiv:2409.04533
- **Registered** in `configs/data.yaml:63–70` only.
- **Consumer.** `bin/plot_hst_lp_cmds.py:38`, writing `results/plot_data/cmd_hst_lp_fields.json`.
- **Status.** d.
