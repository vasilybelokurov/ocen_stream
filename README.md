# oCen_stream — the Omega Centauri stream and the Milky Way bar

Constrain the Milky Way bar (pattern speed, angle, amplitude, size), the halo flattening, v_c and the cluster distance and
proper motion by fitting particle-spray models of the Omega Centauri tidal stream (STREAMFINDER stream 54, Ibata et al. 2024)
in Gaia DR3.

Split from [oCen_dm](https://github.com/vasilybelokurov/ocen_dm) on 2026-10-25 with the full history of the stream files.
The package `ocen_dm.streams` became `ocen_stream`; the cluster constants (`cluster.py`), `paths.py` and `provenance.py` were
copied with it. Documents written before the split (`docs/reviews/`, `docs/STREAM_*.md`, `docs/kb/`) cite the old paths
(`src/ocen_dm/streams/X.py` is now `src/ocen_stream/X.py`).

## Layout

- `src/ocen_stream/` — frames, sprays (AGAMA), Milky Way variants, scores, emulator.
- `bin/streams/` — data extraction (masks, spline and local fits), model comparison, fit drivers, plots; `bin/streams/paper/` figures.
- `docs/` — plans, reviews (Codex, Fable) and status pages.
- `results/` (not in git) — `results/streams` and `results/nbody` link to the simulation store; `results/plot_data` holds plotted data.

## Set-up

```bash
source ~/Work/venvs/.venv/bin/activate
pip install -e .            # agama separately
# .env (gitignored) sets OCEN_CATALOGUES, OCEN_ISOCHRONES, OCEN_BAR_DIR, OCEN_SIMS
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q
```
