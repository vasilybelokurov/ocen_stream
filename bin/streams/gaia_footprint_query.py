#!/usr/bin/env python3
"""Gaia DR3 stars in the omega Cen stream footprint with minimal selection, for visualisation (2026-10-23).

Footprint: chord frame (results/plot_data/stream54_gc_frame_chord.json) phi1 -8..32, |dphi2| < 15, b > 8 deg (~1060 deg2).
q3c_poly_query cannot take polygons wider than ~23 deg, so the footprint is queried in 10 tiles (phi1 strips of 8 deg x
dphi2 halves; convex hull + 0.2 deg pad), then trimmed to the exact footprint and de-duplicated by source_id.
Cuts (SQL): G < 20; parallax - 3 parallax_error < 0.5 mas (removes only stars clearly nearer than 2 kpc; the model debris in
this footprint is at d > 2.4 kpc for 99 per cent of particles). Nothing else (no PM, CMD, ruwe cut).
Output: results/streams/gaia_footprint_dr3.npz (float32 columns + int64 source_id; per-tile parts in results/streams/gaia_footprint_tiles/,
deleted once merged).
Usage: PYTHONPATH=src python bin/streams/gaia_footprint_query.py
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/gaia_footprint_dr3.npz"; TILES = ROOT/"results/streams/gaia_footprint_tiles"
COLS = "source_id, ra, dec, l, b, parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, phot_g_mean_mag AS g, bp_rp, ruwe"


def main():
    import astropy.coordinates as coord, astropy.units as u
    import sqlutilpy as sq
    from ocen_stream.path import project_gc
    from branch_search_gaia import hull_polygon, GC
    TILES.mkdir(parents=True, exist_ok=True)
    L, B = np.meshgrid(np.arange(-110., 10.01, 0.2), np.arange(-20., 75.01, 0.2)); L, B = L.ravel(), B.ravel()
    uu, xx = project_gc(L, B, GC)
    for a in range(-8, 32, 8):
        for x0, x1 in ((-15, 0), (0, 15)):
            f = TILES/f"tile_{a}_{x0}.npz"
            if f.exists():
                continue
            k = (uu >= a) & (uu <= a+8) & (xx >= x0) & (xx <= x1) & (B > 8)
            c = coord.SkyCoord(L[k]*u.deg, B[k]*u.deg, frame="galactic").icrs; poly = hull_polygon(c.ra.deg, c.dec.deg, pad=0.2)
            q = sq.get(f"""SELECT {COLS} FROM gaia_dr3.gaia_source WHERE q3c_poly_query(ra, dec, ARRAY[{','.join(f'{v:.4f}' for v in poly)}])
                          AND phot_g_mean_mag < 20 AND parallax - 3*parallax_error < 0.5""", asDict=True)
            q = {k_: (np.asarray(v, np.int64) if k_ == "source_id" else np.asarray(v, np.float32)) for k_, v in q.items()}
            np.savez(f, **q); print(f"tile {a} {x0}: {len(q['source_id'])} rows", flush=True)
    parts = [dict(np.load(f)) for f in sorted(TILES.glob("tile_*.npz"))]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    _, i = np.unique(D["source_id"], return_index=True); D = {k: v[i] for k, v in D.items()}
    l = np.where(D["l"] > 180, D["l"]-360, D["l"]).astype(np.float64)
    phi1, dphi2 = project_gc(l, D["b"].astype(np.float64), GC)
    keep = (phi1 >= -8) & (phi1 <= 32) & (np.abs(dphi2) <= 15) & (D["b"] > 8)
    D = {k: v[keep] for k, v in D.items()}; D["phi1"] = phi1[keep].astype(np.float32); D["dphi2"] = dphi2[keep].astype(np.float32)
    np.savez(OUT, **D); print(f"{OUT}: {keep.sum()} stars")
    for f in TILES.glob("tile_*.npz"):
        f.unlink()
    TILES.rmdir()


if __name__ == "__main__":
    main()
