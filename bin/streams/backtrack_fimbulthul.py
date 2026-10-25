#!/usr/bin/env python3
"""Integrate Ibata+2024 stream-54 stars with measured v_los back in time and find, per star, the heliocentric distance
(scanned 2.5-6.5 kpc) at which its orbit passes closest to omega Cen's orbit in phase space within the last 1.5 Gyr.
Metric: min over t of sqrt((dx/0.2 kpc)^2 + (dv/20 km/s)^2). Hosts: McMillan17 and DB98 Model 1. Single-threaded, seconds.
Usage: python bin/streams/backtrack_fimbulthul.py   -> prints a table; results/plot_data/fimbulthul_backtrack.json
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, astropy.units as u
from astropy.table import Table
from ocen_stream.restricted import AGAMA_T_MYR, OCEN_TODAY, agama_kpc, host_potential
from ocen_stream.rotation import _galcen

T_BACK, NT = 1500., 3001
D_GRID = np.arange(2.5, 6.51, 0.1)


def to_model(ra, de, d, pmra, pmde, vlos):
    c = coord.SkyCoord(ra=ra*u.deg, dec=de*u.deg, distance=d*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmde*u.mas/u.yr,
                       radial_velocity=vlos*u.km/u.s).transform_to(_galcen())
    return np.column_stack((-c.x.to_value(u.kpc), c.y.to_value(u.kpc), c.z.to_value(u.kpc),
                            -c.v_x.to_value(u.km/u.s), c.v_y.to_value(u.km/u.s), c.v_z.to_value(u.km/u.s)))


def main():
    ag = agama_kpc()
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits"))
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float)
    k = (np.asarray(t["Stream"]) == 54) & np.isfinite(v) & (ev < 300)
    t = t[k]; v = v[k]
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
    out = {}
    for host_name in ("McMillan17", "configs/potentials/DB98_Model1.ini"):
        P = host_potential(host_name)
        tc, oc = ag.orbit(potential=P, ic=OCEN_TODAY, time=-T_BACK/AGAMA_T_MYR, trajsize=NT)
        rows = []
        for i in range(len(t)):
            ics = to_model(np.full(len(D_GRID), float(t["RAdeg"][i])), np.full(len(D_GRID), float(t["DEdeg"][i])), D_GRID,
                           np.full(len(D_GRID), float(t["pmRA"][i])), np.full(len(D_GRID), float(t["pmDE"][i])), np.full(len(D_GRID), v[i]))
            res = ag.orbit(potential=P, ic=ics, time=-T_BACK/AGAMA_T_MYR, trajsize=NT)
            best = (np.inf, None, None, None, None)
            for j, (_, o) in enumerate(res):
                dx = np.linalg.norm(o[:, :3]-oc[:, :3], axis=1); dv = np.linalg.norm(o[:, 3:]-oc[:, 3:], axis=1)
                m = np.sqrt((dx/0.2)**2+(dv/20.)**2); q = m.argmin()
                if m[q] < best[0]:
                    best = (m[q], D_GRID[j], dx[q], dv[q], -tc[q]*AGAMA_T_MYR)
            rows.append(dict(b=float(g.b.deg[i]), l=float(((g.l.deg[i]+180) % 360)-180), plx=float(t["plx"][i]), vlos=float(v[i]),
                             metric=float(best[0]), d_best=float(best[1]), dx_kpc=float(best[2]), dv_kms=float(best[3]), t_ago_myr=float(best[4])))
        rows.sort(key=lambda r: r["b"])
        print(f"\n== {Path(host_name).stem}:  b     l    plx   1/plx  v_los | d_best  dx[kpc] dv[km/s] t_ago[Myr] metric")
        for r in rows:
            print(f"   {r['b']:5.1f} {r['l']:6.1f} {r['plx']:5.2f} {1/max(r['plx'],0.05):5.1f} {r['vlos']:6.1f} | {r['d_best']:5.1f}  {r['dx_kpc']:6.2f}  {r['dv_kms']:6.1f}  {r['t_ago_myr']:7.0f}  {r['metric']:5.2f}")
        out[Path(host_name).stem] = rows
    (ROOT/"results/plot_data/fimbulthul_backtrack.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
