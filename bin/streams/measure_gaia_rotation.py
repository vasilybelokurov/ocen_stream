#!/usr/bin/env python3
"""Plane-of-sky rotation of omega Cen from Gaia DR3 (WSDB): mean tangential PM in annuli, sign + = N -> W.

Selection: within 0.6 deg of (201.69683, -47.47958), ruwe < 1.3, PM errors < 0.15 mas/yr, |PM - cluster| < 2 mas/yr per
component. PMs relative to the sample median. Pull cached in results/plot_data/gaia_dr3_ocen_rotation_pull.npz.
Usage: python bin/streams/measure_gaia_rotation.py  -> results/plot_data/gaia_dr3_ocen_rotation.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RA0, DE0 = 201.69683, -47.47958
EDGES = [3, 6, 10, 15, 25, 36]


def main():
    cache = ROOT/"results/plot_data/gaia_dr3_ocen_rotation_pull.npz"
    if cache.exists():
        d = np.load(cache); ra, de, pa, pd = d["ra"], d["dec"], d["pmra"], d["pmdec"]
    else:
        import sqlutilpy as sq
        ra, de, pa, pd = sq.get(f"""select ra, dec, pmra, pmdec from gaia_dr3.gaia_source
            where q3c_radial_query(ra, dec, {RA0}, {DE0}, 0.6) and pmra is not null and ruwe < 1.3
            and pmra_error < 0.15 and pmdec_error < 0.15 and abs(pmra+3.25) < 2 and abs(pmdec+6.75) < 2""")
        np.savez_compressed(cache, ra=ra, dec=de, pmra=pa, pmdec=pd)
    x = (ra-RA0)*np.cos(np.radians(DE0))*60; y = (de-DE0)*60; r = np.hypot(x, y)
    vt = (-(pa-np.median(pa))*y+(pd-np.median(pd))*x)/r
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (r > lo) & (r < hi)
        out.append(dict(lo=lo, hi=hi, n=int(k.sum()), vtan=float(vt[k].mean()), vtan_err=float(vt[k].std()/np.sqrt(k.sum()))))
        print(f"{lo:2d}-{hi:2d}': N = {k.sum():6d}  <v_tan> = {out[-1]['vtan']:+.3f} +- {out[-1]['vtan_err']:.3f} mas/yr")
    (ROOT/"results/plot_data/gaia_dr3_ocen_rotation.json").write_text(json.dumps(dict(sign="+ = N->W", bins=out), indent=1))


if __name__ == "__main__":
    main()
