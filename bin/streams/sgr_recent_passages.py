#!/usr/bin/env python3
"""Recent Sgr (M54) passages near omega Cen and its tidal debris, last --tback Myr (default 600, the release epoch of the knee
debris), no dynamical friction (negligible over this span). Both present-day points from $OCEN_CATALOGUES/gc_catalog_full.fits
(Baumgardt & Vasiliev 2021 compilation) with Monte Carlo over their errors; frame 'baumgardt' (frames.py); AGAMA orbit
integration (accuracy 1e-10), 0.25 Myr output.
Reports per host: all local minima of the Sgr-omega Cen separation, and the closest approach of Sgr to the model debris of
run results/streams/<run>/A_nodm (tracer positions at the 50-Myr snapshots, interpolated Sgr track).
Usage: python bin/streams/sgr_recent_passages.py --nmc 500
"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from astropy.table import Table
from ocen_stream.restricted import AGAMA_T_MYR, agama_kpc, host_potential
from ocen_stream.frames import to_model
from ocen_stream.analysis import load


def catalogue(name):
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/gc_catalog_full.fits"))
    r = t[[str(n).strip() == name for n in t["NAME"]]][0]
    return {k: float(r[k]) for k in ("RA", "DEC", "DIST", "PMRA", "PMDEC", "RV", "DIST_ERR", "PMRA_ERR", "PMDEC_ERR", "RV_ERR")}


def draw(c, n, rng):
    z = lambda: rng.standard_normal(n)
    s = to_model(np.full(n, c["RA"]), np.full(n, c["DEC"]), c["DIST"]+c["DIST_ERR"]*z(), c["PMRA"]+c["PMRA_ERR"]*z(),
                 c["PMDEC"]+c["PMDEC_ERR"]*z(), c["RV"]+c["RV_ERR"]*z())
    s[0] = to_model(c["RA"], c["DEC"], c["DIST"], c["PMRA"], c["PMDEC"], c["RV"])[0]
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tback", type=float, default=600.); ap.add_argument("--nmc", type=int, default=500)
    a = ap.parse_args(); ag = agama_kpc(); rng = np.random.default_rng(3)
    cs, co = catalogue("NGC 6715"), catalogue("NGC 5139"); n = a.nmc+1
    S, O = draw(cs, n, rng), draw(co, n, rng)
    out = {}
    for mw, run in (("McMillan17", "prescribed_rot"), ("configs/potentials/DB98_Model1.ini", "db98_rot")):
        P = host_potential(mw); T = a.tback/AGAMA_T_MYR; ns = int(a.tback*4)+1
        rs = ag.orbit(potential=P, ic=S, time=-T, trajsize=ns, accuracy=1e-10, verbose=False)
        ro = ag.orbit(potential=P, ic=O, time=-T, trajsize=ns, accuracy=1e-10, verbose=False)
        t = -np.asarray(rs[0, 0])*AGAMA_T_MYR
        mins, closest = [], []
        for i in range(n):
            d = np.linalg.norm(np.asarray(rs[i, 1])[:, :3]-np.asarray(ro[i, 1])[:, :3], axis=1)
            k = np.where((d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]))[0]+1
            if i == 0:
                mins = [(float(t[j]), float(d[j]), float(np.linalg.norm(np.asarray(rs[0, 1])[j, 3:]-np.asarray(ro[0, 1])[j, 3:]))) for j in k]
            closest.append(d.min())
        closest = np.array(closest)
        print(f"\n{Path(mw).stem}: Sgr-omega Cen separation minima, nominal (t ago [Myr], d [kpc], dv [km/s]):")
        for m in mins:
            print(f"   {m[0]:6.1f}  {m[1]:6.2f}  {m[2]:5.0f}")
        print(f"   closest over {a.tback:.0f} Myr, MC 2.5/50/97.5%: {np.round(np.percentile(closest, [2.5, 50, 97.5]), 2)} kpc")
        # Sgr vs model debris (nominal Sgr track), using the run's snapshots
        R = ROOT/"results/streams"/run/"A_nodm"; rj = json.loads((R/"run.json").read_text())
        T_run = rj["snap_times_myr"][-1]; sg = np.asarray(rs[0, 1])
        dmin, tmin = np.inf, None
        for ts in rj["snap_times_myr"]:
            ago = T_run-ts
            if ago > a.tback:
                continue
            _, xv = load(R, ts) if ts < T_run else load(R, name="snap_today.npz")
            xs = np.array([np.interp(ago, t, sg[:, c]) for c in range(3)])
            dd = np.linalg.norm(xv[:, :3]-xs, axis=1).min()          # load() returns kpc
            if dd < dmin:
                dmin, tmin = dd, ago
        print(f"   closest approach of nominal Sgr to any model debris tracer ({run}/A_nodm, 50-Myr snapshots): {dmin:.2f} kpc at {tmin:.0f} Myr ago")
        out[Path(mw).stem] = dict(minima_nominal=mins, closest_mc=closest.tolist(), sgr_to_debris_kpc=float(dmin), sgr_to_debris_t_ago=tmin)
    (ROOT/"results/plot_data/sgr_recent_passages.json").write_text(json.dumps(dict(sgr=cs, ocen=co, tback=a.tback, hosts=out), indent=1))


if __name__ == "__main__":
    main()


def knee_members(tback=600., dists=(4.4, 4.8, 5.2)):
    """Closest approach of Sgr (nominal + MC as above) to the observed stream-54 knee members with v_los (b > 32), each integrated
    back in the same host at fixed distances bracketing the CMD-shift distance (~4.8 kpc at b = 33-42)."""
    import astropy.coordinates as coord
    ag = agama_kpc(); rng = np.random.default_rng(3)
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float)
    b = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic.b.deg
    k = np.isfinite(v) & (ev < 300) & (b > 32); t, v = t[k], v[k]
    S = draw(catalogue("NGC 6715"), 201, rng)
    for mw in ("McMillan17", "configs/potentials/DB98_Model1.ini"):
        P = host_potential(mw); T = tback/AGAMA_T_MYR; ns = int(tback*4)+1
        rs = ag.orbit(potential=P, ic=S, time=-T, trajsize=ns, accuracy=1e-10, verbose=False)
        for d in dists:
            ic = to_model(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), np.full(len(t), d),
                          np.asarray(t["pmRA"], float), np.asarray(t["pmDE"], float), v)
            rk = ag.orbit(potential=P, ic=ic, time=-T, trajsize=ns, accuracy=1e-10, verbose=False)
            dm = np.array([[np.linalg.norm(np.asarray(rk[j, 1])[:, :3]-np.asarray(rs[i, 1])[:, :3], axis=1).min() for j in range(len(t))]
                           for i in range(len(S))])
            star_min = dm.min(axis=1)                       # closest knee star, per Sgr MC draw
            print(f"{Path(mw).stem}, knee members at d = {d} kpc (N {len(t)}): closest Sgr approach to any knee star in the last "
                  f"{tback:.0f} Myr: nominal {star_min[0]:.2f} kpc, MC 2.5/50/97.5% {np.round(np.percentile(star_min, [2.5, 50, 97.5]), 2)} kpc")


if __name__ == "__main__" and os.environ.get("KNEE"):
    knee_members()
