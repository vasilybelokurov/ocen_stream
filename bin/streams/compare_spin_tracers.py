#!/usr/bin/env python3
"""Rotation test B: tracer runs (run_prescribed.py, model A_nodm, 200k tracers) at the same barred host and omega Cen state,
without and with internal rotation (Lynden-Bell flips, fitted spin). Unbound debris (> 0.3 kpc from the centre today) in the
observable space and along phi1 of the chord great-circle frame.
Figures: plots/rotB_observables.png (log-density contours of both runs, members as points; same panels as the KDE plots);
plots/rotB_along_phi1.png (per 1-deg phi1 bin: tracer counts, medians of dphi2, pmra, pmdec, v_los, d; members' medians).
Numbers: results/plot_data/rotB_compare.json.
Usage: python bin/streams/compare_spin_tracers.py [run_a run_b]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.frames import observables
from ocen_dm.streams.path import project_gc

runs = [a for a in sys.argv[1:]] or ["rotB_nospin", "rotB_spin"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
d = load_data(); du, dx = project_gc(d["l"], d["b"], GC); d["x"], d["u"] = dx, du
cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
M = {}
for r in runs:
    s = np.load(ROOT/f"results/streams/prescribed/{r}/snap_today.npz"); meta = json.loads((ROOT/f"results/streams/prescribed/{r}/run.json").read_text())
    xv = np.hstack((s["pos"]/1e3, s["vel"])).astype(float); c = np.array(meta["ocen_today"])
    k = np.linalg.norm(xv[:, :3]-c[:3], axis=1) > 0.3; o = observables(xv[k])
    m = dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], d=o["dist"]); m["u"], m["x"] = project_gc(m["l"], m["b"], GC)
    M[r] = m; print(r, "unbound tracers", k.sum(), "of", len(k))
PANELS = (("l", "b", (-75, -10), (5, 45)), ("b", "l", (10, 45), (-70, -30)), ("b", "pmra", (10, 45), (-22, 0)),
          ("b", "pmdec", (10, 45), (-15, -3)), ("b", "vlos", (10, 45), (120, 300)), ("b", "d", (10, 45), (2.5, 6.5)))
fig, ax = plt.subplots(1, 6, figsize=(30, 5))
for a, (xq, yq, xr, yr) in zip(ax, PANELS):
    if yq == "vlos":
        hv = np.isfinite(d["v"]); a.errorbar(d["b"][hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="k", mfc="w", ms=5, lw=0.8, zorder=5)
    elif yq == "d":
        a.errorbar([17.5]+[0.5*(c_["b"][0]+c_["b"][1]) for c_ in cmd], [5.43]+[5.43*c_["d_ratio"] for c_ in cmd],
                   yerr=[0]+[5.43*c_["d_ratio"]*np.log(10)/5*c_["dm_err"] for c_ in cmd], fmt="s", color="k", mfc="w", ms=6, zorder=5)
    else:
        a.plot(d[xq], d[yq], ".", color="0.5", ms=1.2, alpha=0.5, zorder=1)
    for r, col in zip(runs, ("C0", "C3")):
        H, xe, ye = np.histogram2d(M[r][xq], M[r][yq], bins=(120, 120), range=(sorted(xr), yr))
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, 1.2).T; lv = np.log10(H+1e-9); v = lv.max()
        a.contour(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), lv, levels=v-np.array([2.5, 2., 1.5, 1., 0.5]), colors=col, linewidths=1.)
    a.set_xlim(*(xr[::-1] if xq == "l" else xr)); a.set_ylim(*yr); a.set_xlabel(xq); a.set_ylabel(yq)
ax[0].plot([], [], "C0", label=runs[0]); ax[0].plot([], [], "C3", label=runs[1]); ax[0].legend()
fig.suptitle("Rotation test B: tracer debris (> 0.3 kpc from the centre) without (blue) and with (red) internal rotation; contours 10^-0.5..10^-2.5 of peak; members grey/black")
fig.tight_layout(); fig.savefig(ROOT/"plots/rotB_observables.png", dpi=62)
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+1, 1.); cen = 0.5*(edges[1:]+edges[:-1]); Q = ["x", "pmra", "pmdec", "vlos", "d"]
fig, ax = plt.subplots(1, 6, figsize=(30, 4.5)); out = dict(phi1=cen.tolist())
for r, col in zip(runs, ("C0", "C3")):
    m = M[r]; ib = np.digitize(m["u"], edges)-1; n = np.bincount(ib[(ib >= 0) & (ib < len(cen))], minlength=len(cen))
    ax[0].plot(cen, n, col, label=r); out[r] = dict(n=n.tolist())
    for a, q in zip(ax[1:], Q):
        med = [np.median(m[q][ib == j]) if (ib == j).sum() >= 10 else np.nan for j in range(len(cen))]
        a.plot(cen, med, col); out[r][q] = [float(v) for v in med]
dib = np.digitize(du, edges)-1
for a, q in zip(ax[1:], Q):
    if q == "d": continue
    v = d["v"] if q == "vlos" else d[q]
    med = [np.nanmedian(v[dib == j]) if np.isfinite(v[dib == j]).sum() >= (3 if q == "vlos" else 10) else np.nan for j in range(len(cen))]
    a.plot(cen, med, "ko", ms=4, label="members (median)")
for a, q in zip(ax, ["tracers per 1-deg bin"]+Q):
    a.set_xlabel(r"$\phi_1$ [deg]"); a.set_ylabel(q)
ax[0].set_yscale("log"); ax[0].legend(); ax[1].legend()
fig.suptitle("Rotation test B along phi1 (chord frame): blue no spin, red spin; black = members' medians"); fig.tight_layout()
fig.savefig(ROOT/"plots/rotB_along_phi1.png", dpi=70)
(ROOT/"results/plot_data/rotB_compare.json").write_text(json.dumps(out))
a0, a1 = out[runs[0]], out[runs[1]]
for j in range(len(cen)):
    if a0["n"][j] >= 10 and a1["n"][j] >= 10:
        print(f"phi1 {cen[j]:5.1f}: n {a0['n'][j]:5d} -> {a1['n'][j]:5d}; d(dphi2) {a1['x'][j]-a0['x'][j]:+.2f}; d(pmra) {a1['pmra'][j]-a0['pmra'][j]:+.3f}; "
              f"d(pmdec) {a1['pmdec'][j]-a0['pmdec'][j]:+.3f}; d(vlos) {a1['vlos'][j]-a0['vlos'][j]:+.1f}")
