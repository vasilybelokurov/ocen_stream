#!/usr/bin/env python3
"""4D grid: Omega_b {33, 34, 34.5, 35, 35.5, 36, 37} x bar angle {16, 20, 24, 28} x amplitude {1.0, 1.2, 1.4, 1.6} x omega Cen distance
{5.45, 5.50, 5.55, 5.60} kpc; PMs fixed at the free-angle best fit (pmra -3.2223, pmdec -6.7517), v_los catalogue. Model A spray
(8000 release epochs over the last 1000 Myr, seed 1), Hunter+2024 bar. Score: score_conditional, chord great-circle frame (variant A).
Particles saved to results/streams/spray_grid4/<tag>.npz (trailing arm) for later re-scoring; summary results/plot_data/grid4.json.
Resumable (skips existing npz, re-scores them).
Usage: python bin/streams/grid4.py
"""
import itertools, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_conditional
from ocen_dm.streams.path import project_gc

OMEGA = [33., 34., 34.5, 35., 35.5, 36., 37.]; ANGLE = [16., 20., 24., 28.]; AMP = [1.0, 1.2, 1.4, 1.6]; DIST = [5.45, 5.50, 5.55, 5.60]
PMRA, PMDEC = -3.2223, -6.7517
d = load_data(); nd = len(d["l"])
cov = np.zeros((nd, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2; cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
du, dx = project_gc(d["l"], d["b"], GC); dW = np.column_stack((dx, d["pmra"], d["pmdec"])); win = (du.min(), du.max())
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-1000/AGAMA_T_MYR, T, 8000)
outd = ROOT/"results/streams/spray_grid4"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/grid4.json"; res = []
for om, an, am, dist in itertools.product(OMEGA, ANGLE, AMP, DIST):
    t0 = time.time(); tag = f"om{om:g}_an{an:g}_am{am:g}_d{dist:g}"; f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am)
        today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], dist, PMRA, PMDEC, OCEN_OBS["vlos"])[0]
        sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=1); tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
        m = dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"], chi=sp["chi"][tr], age=sp["t_release_myr_ago"][tr])
        np.savez_compressed(f, **m)
    mu, mx = project_gc(m["l"], m["b"], GC)
    s = score_conditional(du, dW, d["v"], d["e_v"], cov, mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"], u_window=win)
    res.append(dict(omega=om, angle=an, amp=am, d=dist, lnL=s["total"], frac_bg=s["frac_bg"], K=s["K"], seconds=time.time()-t0))
    print(f"{tag:28s} lnL {s['total']:10.1f}  bg {s['frac_bg']:.2f}  [{time.time()-t0:.0f} s]", flush=True)
    outp.write_text(json.dumps(dict(pm=[PMRA, PMDEC], grid=res)))
