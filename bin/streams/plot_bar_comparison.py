#!/usr/bin/env python3
"""Northern-arm debris of model A in the Hunter+2024 axisymmetric control and the rotating-bar runs, over Ibata+2024 stream 54
(grey): rows l-b, pmra-b, pmdec-b, v_los-b; one column per run (results/streams/<run>/A_nodm).
Usage: python bin/streams/plot_bar_comparison.py hunter_axi hunter_bar33 hunter_bar37.5 hunter_bar41 -> plots/streams_bar_comparison.png
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from ocen_stream.analysis import load, run_core, run_particles
from ocen_stream.restricted import OCEN_TODAY, bound_set
from ocen_stream.frames import observables

runs = sys.argv[1:]
w = lambda x: np.where(x > 180, x-360, x)
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
gl, gb = w(g.l.deg), g.b.deg
v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); hv = np.isfinite(v) & (ev < 300)
fig, ax = plt.subplots(4, len(runs), figsize=(4.6*len(runs), 15), squeeze=False)
for j, run in enumerate(runs):
    R = ROOT/"results/streams"/run/"A_nodm"
    rj = json.loads((R/"run.json").read_text())
    m, s = run_particles(R); _, xv = load(R, name="snap_today.npz"); c = np.array(rj["ocen_today"])
    bnd, _ = bound_set(xv, c, m, start=np.linalg.norm(xv[:, :3]-c[:3], axis=1) < 0.5, core=run_core(R))
    o = observables(xv[~bnd], rj.get("frame", "baumgardt"))
    k = (o["b"] > 5) & (w(o["l"]) < -15) & (w(o["l"]) > -80)
    lab = "Hunter+24 axisymmetric" if rj.get("bar_omega") is None else f"Hunter+24 bar, Omega_b = {rj['bar_omega']:g}"
    a = ax[0, j]; a.plot(gl, gb, ".", color="0.7", ms=2); a.plot(w(o["l"][k]), o["b"][k], ".", color="C3", ms=2.5)
    a.set_xlim(-15, -80); a.set_ylim(5, 50); a.set_xlabel("l [deg]"); a.set_ylabel("b [deg]")
    a.set_title(f"{lab}\nA no DM, N arm = {k.sum()}, centre off {rj['centre_offset_today_pc']:.2f} pc", fontsize=9)
    for i, (q, dq, yl) in enumerate((("pmra", np.asarray(t["pmRA"], float), (-22, 0)), ("pmdec", np.asarray(t["pmDE"], float), (-16, 0)))):
        a = ax[i+1, j]; a.plot(gb, dq, ".", color="0.7", ms=2); a.plot(o["b"][k], o[q][k], ".", color="C3", ms=2.5)
        a.set_xlim(10, 45); a.set_ylim(*yl); a.set_xlabel("b [deg]"); a.set_ylabel(q + " [mas/yr]")
    a = ax[3, j]; a.plot(o["b"][k], o["vlos"][k], ".", color="C3", ms=2.5); a.plot(gb[hv], v[hv], "o", color="k", mfc="w", ms=6)
    a.set_xlim(10, 45); a.set_ylim(120, 300); a.set_xlabel("b [deg]"); a.set_ylabel("v_los [km/s]")
for a in ax.ravel():
    a.grid(alpha=0.3)
fig.suptitle("Rotating bar test (model A, fitted rotation, 1.96 Gyr): red = model northern-arm debris, grey = Ibata+2024 stream 54", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/streams_bar_comparison.png", dpi=75); print("plots/streams_bar_comparison.png")
