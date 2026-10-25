#!/usr/bin/env python3
"""Northern arm of one run coloured by release time (last snapshot within 0.2 kpc of the cluster centre) over Ibata+2024
stream 54 (grey): l-b, pmra-b, pmdec-b, v_los-b, distance-b (data: 1/plx median per 2.5-deg bin). Knee and arm boxes of
bin/streams/knee_region_check.py drawn in the l-b panel.
Usage: python bin/streams/plot_knee_release.py db98_rot A_nodm  -> plots/streams_knee_release_<run>_<model>.png
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from plot_debris_vs_ibata import observables
from ocen_stream.analysis import load, run_core, run_particles
from ocen_stream.restricted import AGAMA_T_MYR, OCEN_TODAY, agama_kpc, bound_set, host_potential

run, key = sys.argv[1], sys.argv[2]
KNEE = dict(l=(-45, -28), b=(31, 41)); ARM = dict(l=(-62, -50), b=(20, 30))
w = lambda x: np.where(x > 180, x-360, x)
R = ROOT/"results/streams"/run/key; rj = json.loads((R/"run.json").read_text())
ag = agama_kpc(); host = host_potential(rj["mw"]); times = np.asarray(rj["snap_times_myr"]); T = times[-1]
tc, orb = ag.orbit(potential=host, ic=np.array(rj["start"]), timestart=0., time=T/AGAMA_T_MYR, trajsize=int(T/0.5)+1)
cen = np.array([np.interp(times, tc*AGAMA_T_MYR, orb[:, i]) for i in range(3)]).T
m, s = run_particles(R); _, xv = load(R, name="snap_today.npz")
bnd, _ = bound_set(xv, OCEN_TODAY.copy(), m, start=np.linalg.norm(xv[:, :3]-OCEN_TODAY[:3], axis=1) < 0.5, core=run_core(R))
o = observables(xv)
sel = np.where(~bnd & (o["b"] > 5) & (w(o["l"]) < -15) & (w(o["l"]) > -80))[0]
last = np.zeros(len(sel))
for k, tk in enumerate(times):
    _, x = load(R, tk) if k < len(times)-1 else load(R, name="snap_today.npz")
    last = np.where(np.linalg.norm(x[sel, :3]-cen[k], axis=1) < 0.2, tk, last)
age = T-last
t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); t = t[np.asarray(t["Stream"]) == 54]
g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
gl, gb = w(g.l.deg), g.b.deg
v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); hv = np.isfinite(v) & (ev < 300)
plx = np.asarray(t["plx"], float)+0.017
fig, ax = plt.subplots(1, 5, figsize=(26, 5.6))
kw = dict(c=age, cmap="viridis_r", vmin=0, vmax=1000, s=6, zorder=3)
ax[0].plot(gl, gb, ".", color="0.7", ms=2, zorder=1); sc = ax[0].scatter(w(o["l"][sel]), o["b"][sel], **kw)
for B, col in ((KNEE, "crimson"), (ARM, "k")):
    ax[0].add_patch(plt.Rectangle((B["l"][0], B["b"][0]), B["l"][1]-B["l"][0], B["b"][1]-B["b"][0], fill=False, ec=col, lw=1.5, zorder=4))
ax[0].set_xlim(-15, -80); ax[0].set_ylim(5, 50); ax[0].set_xlabel("l [deg]"); ax[0].set_ylabel("b [deg]")
for a, q, dq, lab in ((ax[1], "pmra", np.asarray(t["pmRA"], float), "pmra* [mas/yr]"), (ax[2], "pmdec", np.asarray(t["pmDE"], float), "pmdec [mas/yr]")):
    a.plot(gb, dq, ".", color="0.7", ms=2, zorder=1); a.scatter(o["b"][sel], o[q][sel], **kw); a.set_ylabel(lab)
    a.set_ylim(-22, 0)
ax[3].scatter(o["b"][sel], o["vlos"][sel], **kw); ax[3].plot(gb[hv], v[hv], "o", color="k", mfc="w", ms=6, zorder=5, label="Ibata+2024 v_los")
ax[3].set_ylim(120, 300); ax[3].set_ylabel("v_los [km/s]"); ax[3].legend(loc="lower left")
ax[4].scatter(o["b"][sel], o["dist"][sel], **kw)
eb = np.arange(15, 42.6, 2.5); bc = 0.5*(eb[1:]+eb[:-1])
for lo, hi, c in zip(eb[:-1], eb[1:], bc):
    k = (gb >= lo) & (gb < hi)
    if k.sum() > 20:
        p = plx[k]; e = 1.2533*1.4826*np.median(np.abs(p-np.median(p)))/np.sqrt(k.sum())
        ax[4].errorbar(c, 1/np.median(p), yerr=e/np.median(p)**2, fmt="s", color="k", mfc="w", ms=7, zorder=5)
ax[4].errorbar([], [], fmt="s", color="k", mfc="w", label="Ibata 54: 1/median(plx+0.017)"); ax[4].legend(loc="lower left")
ax[4].set_ylim(2.5, 7); ax[4].set_ylabel("heliocentric distance [kpc]")
for a in ax[1:]:
    a.set_xlim(10, 45); a.set_xlabel("b [deg]")
for a in ax:
    a.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.015); cb.set_label("release time [Myr ago]")
fig.suptitle(f"Northern arm of {key} ({run}: {Path(rj['mw']).stem}, {T/1e3:.2f} Gyr) coloured by release time; grey: Ibata+2024 stream 54. "
             "Red box: knee; black box: arm b = 20-30", fontsize=11)
out = ROOT/f"plots/streams_knee_release_{run}_{key}.png"; fig.savefig(out, dpi=85, bbox_inches="tight"); print(out)
