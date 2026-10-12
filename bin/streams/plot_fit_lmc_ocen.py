#!/usr/bin/env python3
"""LMC refit (bin/streams/fit_lmc_ocen.py) vs data: stream-54 members and model footprint medians (pmra, pmdec, v_los per 2-deg
phi1 bin, trailing debris < 700 Myr, |phi2| < 6 deg) for four sprays at the fixed bar: no LMC and M_LMC = 1.5e11, each at the
start point (d 5.6 kpc, catalogue PMs) and at its best fit. Last panel: cumulative per-star lnL vs phi1 relative to the no-LMC
best fit. Re-runs the four sprays (seed 1, 8000 epochs; ~20 s each, 1 core). Output plots/fit_lmc_ocen.png.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/plot_fit_lmc_ocen.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, d, OCEN_OBS

AN, OM, SIZE, AMP = 16., 34., 1.0, 1.2
runs = []
for mass, col in ((0., "k"), (1.5e11, "C3")):
    log = json.loads((ROOT/f"results/plot_data/fit_lmc_ocen_M{mass:.1e}_an{AN:g}_om{OM:g}_s1.json").read_text())["log"]
    b = max(log, key=lambda e: e["lnpost"])
    for tag, p, ls in (("start", (5.6, OCEN_OBS["pmra"], OCEN_OBS["pmdec"]), ":"), ("best", (b["d"], b["pmra"], b["pmdec"]), "-")):
        hk = dict(bar_size=SIZE, **(dict(lmc=dict(mass=mass)) if mass else {}))
        m = make_spray(OM, AN, AMP, p[0], nrel=8000, seed=1, pm=p[1:], host_kw=hk, max_age_myr=700.)
        s = score(m)
        runs.append(dict(mass=mass, tag=tag, p=p, m=m, s=s, col=col, ls=ls))
        print(f"M {mass:.1e} {tag}: d {p[0]:.3f} pmra {p[1]:.4f} pmdec {p[2]:.4f} lnL {s['total']:.1f}", flush=True)

edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
for j, (q, dq, lab) in enumerate((("pmra", d["pmra"], "pmra [mas/yr]"), ("pmdec", d["pmdec"], "pmdec [mas/yr]"), ("vlos", d["v"], "v_los [km/s]"))):
    ax[j].scatter(du, dq, s=4, c="0.6", label="stream-54 members")
    ib = np.digitize(du, edges)-1
    dm = [np.nanmedian(dq[ib == i]) if np.isfinite(dq[ib == i]).sum() >= 3 else np.nan for i in range(len(cen))]
    ax[j].plot(cen, dm, "s", ms=6, mfc="none", mec="k", label="data median")
    for r in runs:
        mu, mx = project_gc(r["m"]["l"], r["m"]["b"], GC); k = (np.abs(mx) < 6) & (r["m"]["chi"] > 0)
        jb = np.digitize(mu[k], edges)-1; y = r["m"][q][k]
        med = [np.median(y[jb == i]) if (jb == i).sum() >= 10 else np.nan for i in range(len(cen))]
        ax[j].plot(cen, med, ls=r["ls"], color=r["col"], lw=1.6,
                   label=f"{'LMC 1.5e11' if r['mass'] else 'no LMC'} {r['tag']}: lnL {r['s']['total']:.0f}")
    lo, hi = np.nanpercentile(dq, [2, 98]); ax[j].set(xlabel="phi1 (chord frame) [deg]", ylabel=lab, ylim=(lo-0.1*(hi-lo), hi+0.1*(hi-lo)))
ax[0].legend(fontsize=7)
o = np.argsort(du); ref = next(r for r in runs if r["mass"] == 0 and r["tag"] == "best")["s"]["per_star"][o]
for r in runs:
    ax[3].plot(du[o], np.cumsum(r["s"]["per_star"][o]-ref), ls=r["ls"], color=r["col"])
ax[3].axhline(0, c="0.5", lw=0.5)
ax[3].set(xlabel="phi1 (chord frame) [deg]", ylabel="cumulative lnL - (no-LMC best)", title="where the likelihood is gained / lost")
fig.suptitle(f"Bar fixed ({AN:g} deg, Omega {OM:g}, size {SIZE:g}, amp {AMP:g}); omega Cen d and PMs refitted with / without the LMC "
             f"(dotted: start, solid: best)")
fig.savefig(ROOT/"plots/fit_lmc_ocen.png", dpi=110); print("plots/fit_lmc_ocen.png")
