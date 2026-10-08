#!/usr/bin/env python3
"""Closest approach of Sgr (M54; catalogue values + 200 MC draws, as sgr_recent_passages.py) over the last 600 Myr to
(1) all Ibata+2024 stream 54 (N 29) and 55 (N 25) members with v_los, integrated back at fixed heliocentric distances:
    stream 54 at the CMD-shift distance scale (5.4 kpc x d/d_ref of stream54_cmd_distance.json, interpolated in b) and +-0.4 kpc;
    stream 55 at 3.0 kpc (Ibata+2024) and +-0.4 kpc;
(2) every unbound model tracer of the northern arm today (b > 5, -80 < l < -15) of results/streams/<run>/A_nodm, integrated back
    in the host only (exact after release; before release they move with omega Cen, whose own approach is in sgr_recent_passages).
Outputs every 0.25 Myr; AGAMA accuracy 1e-10.
Usage: python bin/streams/sgr_stream_approach.py
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, astropy.coordinates as coord
from astropy.table import Table
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, agama_kpc, bound_set, host_potential
from ocen_dm.streams.frames import to_model, observables
from ocen_dm.streams.analysis import load, run_core, run_particles
from sgr_recent_passages import catalogue, draw

TB = 600.; NS = int(TB*4)+1
w = lambda x: np.where(x > 180, x-360, x)


def min_dist(track_sgr, trajs):
    """min over time and objects of |x_obj(t) - x_sgr(t)|; trajs: (Nobj, NS, 3)."""
    return np.min(np.linalg.norm(trajs-track_sgr[None, :, :], axis=2), axis=1)


def main():
    ag = agama_kpc(); rng = np.random.default_rng(3); S = draw(catalogue("NGC 6715"), 201, rng)
    t = Table.read(os.path.expanduser("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits"))
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); hv = np.isfinite(v) & (ev < 300)
    gb = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic.b.deg
    cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
    bc = np.array([15+2.5] + [0.5*(x["b"][0]+x["b"][1]) for x in cmd]); dr = np.array([1.0] + [x["d_ratio"] for x in cmd])
    out = {}
    for mw, run in (("McMillan17", "prescribed_rot"), ("configs/potentials/DB98_Model1.ini", "db98_rot")):
        P = host_potential(mw); T = TB/AGAMA_T_MYR
        rs = ag.orbit(potential=P, ic=S, time=-T, trajsize=NS, accuracy=1e-10, verbose=False)
        sg = np.stack([np.asarray(r)[:, :3] for r in rs[:, 1]])                   # (201, NS, 3)
        host = Path(mw).stem; out[host] = {}
        for sid, base in ((54, None), (55, 3.0)):
            k = hv & (np.asarray(t["Stream"]) == sid)
            d0 = 5.4*np.interp(gb[k], bc, dr) if base is None else np.full(k.sum(), base)
            for off in (-0.4, 0., 0.4):
                ic = to_model(np.asarray(t["RAdeg"], float)[k], np.asarray(t["DEdeg"], float)[k], d0+off,
                              np.asarray(t["pmRA"], float)[k], np.asarray(t["pmDE"], float)[k], v[k])
                rk = ag.orbit(potential=P, ic=ic, time=-T, trajsize=NS, accuracy=1e-10, verbose=False)
                tr = np.stack([np.asarray(r)[:, :3] for r in rk[:, 1]])
                m = np.array([min_dist(sg[i], tr).min() for i in range(len(S))])
                print(f"{host}: stream {sid} v_los members (N {k.sum()}), d offset {off:+.1f} kpc: closest Sgr approach "
                      f"nominal {m[0]:.2f} kpc, MC 2.5/50/97.5% {np.round(np.percentile(m, [2.5, 50, 97.5]), 2)}")
                out[host][f"stream{sid}_off{off:+.1f}"] = m.tolist()
        R = ROOT/"results/streams"/run/"A_nodm"
        ms, sp = run_particles(R); _, xv = load(R, name="snap_today.npz")
        bnd, _ = bound_set(xv, OCEN_TODAY.copy(), ms, start=np.linalg.norm(xv[:, :3]-OCEN_TODAY[:3], axis=1) < 0.5, core=run_core(R))
        o = observables(xv)
        k = ~bnd & (o["b"] > 5) & (w(o["l"]) < -15) & (w(o["l"]) > -80)
        rk = ag.orbit(potential=P, ic=xv[k], time=-T, trajsize=NS, accuracy=1e-10, verbose=False)
        tr = np.stack([np.asarray(r)[:, :3] for r in rk[:, 1]])
        m = np.array([min_dist(sg[i], tr).min() for i in range(len(S))])
        print(f"{host}: model northern-arm debris ({run}/A_nodm, N {k.sum()}): closest Sgr approach nominal {m[0]:.2f} kpc, "
              f"MC 2.5/50/97.5% {np.round(np.percentile(m, [2.5, 50, 97.5]), 2)}")
        out[host]["model_arm"] = m.tolist()
    (ROOT/"results/plot_data/sgr_stream_approach.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
