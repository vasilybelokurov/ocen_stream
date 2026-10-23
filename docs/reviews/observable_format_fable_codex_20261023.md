# Format of the reanalysed observables for model comparison: Fable and Codex (2026-10-23)

## Both
- Final likelihood: binned JOINT counts (format B in 3-D, not three marginal maps, which double-count stars and lose the
  sky-PM association) with Poisson statistics: N_cell ~ Poisson(A m_cell + B_cell), model particles passed through the SAME frozen
  selection (CMD pass via particle distance + stellar population, G < 20, footprint, GC masks), background from off-sky bands with
  its uncertainty as nuisance. Keep the stars (for a later star-level mixture C); per-bin summaries (A) for diagnostics/fast
  searches; no membership-weighted member list (D) for constraints (circular, loses density/wings).
- The tight k = 2 masks must not be the likelihood domain; keep the upper branch inside.
- End-to-end injection of model streams into the real field (rerunning masks/dm) is the required validation; nulls; convergence.

## Fable (checks on saved sprays and the data)
- Tight masks keep only 54-72% of the 12.5-deg model's debris at phi1 -2..22 (8-28% beyond 22); 25-deg 33-67% / 3-23%: model sky
  MAD 1.7-2.5 deg vs data 1.0-1.2; model pmra 1.3-2.1 vs 1.1-1.2 mas/yr; 12.5-deg model sky 1.3-3.8 deg below the data beyond 22.
- S/B per 4-deg bin in the tight masks 0.1-2.3; in a loose box B/S 10-25 -> background accuracy sets the ceiling.
- Product: obs_cube (phi1 2-deg x dphi2 0.25 x pmra 0.25 x pmdec 0.2-0.25 over the full box, absolute coordinates, ~13M cells,
  OFF bands inside; exclude cube for the b = 8 artefact, GC masks, footprint); selection.json (tracks/widths for marked/free/orbit
  starts, include rule k ~ 4 PMs and |dphi2 - c| < 8, OFF bands 5-14, CMD mask + dm line, footprint cuts); cmd_pass_curve per
  column (expected observable stars per template star vs particle dm offset, with A_G and G < 20); dm_bins (with template Hess +
  background so the same dm fit runs on the model's predicted Hess); vlos_stars; null include rules (+-6 mas/yr).
- Model side: particles -> frame -> masks -> weight n_obs(dm_p - dm_line) -> histogram -> smooth (h + error class) -> m_cell;
  ~0.5-1 s per evaluation (spray itself 90-4600 s). Per-column amplitude profiled by default (density-free), global optional.
- dm term: Gaussian per bin on fit-to-fit difference (x2), plus a global offset nuisance (0.1 mag). Model dm vs line within 0.1 at
  0-22, 0.1-0.26 fainter beyond 22; discriminates at 0.1-0.2 mag.
- No carry-over of T ~ 2; calibrate overdispersion from seed scatter and injection coverage.
- Tests: nulls (A_k ~ 0, flat residuals); injection + model discrimination (12.5 vs 25 injected); mask-start robustness of the
  ranking; seed convergence vs N; continuity with the STREAMFINDER conditional score.
- Against: result as good as the background; 3-5% error in the dphi2 interpolation of the field PM distribution makes fake wing
  signal of the size of model differences.

## Codex (read code)
- Current score is not fully density-free (mixing weight depends on f_u).
- Exports not usable as-is: one-parameter errors, overlapping bins; PM display counts vs background windows differ (SKYH vs
  SKY_HALF); dm background fixed in profile; quadratic relation uses raw errors; upper branch excluded from the sky fit.
- Deliverables: stars.parquet (all columns incl. errors, flags, region flags), cells.npz (fixed joint cells in phi1, dphi2, pmra,
  pmdec, c0, G0; zero cells kept), background.npz (control counts, model, covariance), summaries.npz (bootstrap covariances, full
  dm likelihood grids), manifest.json (hashes, frame, selection inequalities, CMD template, completeness). Boolean CMD mask is not
  the Gaia completeness function (GaiaUnlimited: catalogue inclusion x sample selection).
- Model: integrate a luminosity function per particle (not one random star), G < 20, extinction, errors, parallax cut,
  completeness; frozen data selection; correlated PM errors; both branches; global amplitude keeps density, per-bin discards it;
  release history as nuisance. Do not multiply by dm summaries from the same stars (CMD in the cells instead).
- Against: a full count likelihood may turn uncertain background/completeness/population into falsely precise bar constraints;
  if injections show that, use A as primary.
