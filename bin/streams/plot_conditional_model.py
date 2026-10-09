#!/usr/bin/env python3
"""The model exactly as score_conditional sees it: for each observable q in (dphi2, pmra, pmdec, v_los), the conditional density
p(q | u) of the (model + 5% background) mixture, u = phi1 of the chord great-circle frame. Same cuts and constants as the
likelihood: trailing arm, chi > 0, age < 700 Myr, chi bins of 5 kpc Myr with >= 20 particles, weight 1/(K n_k) per particle,
fixed bandwidths h_u = 0.5 deg, h_x = 0.5 deg, h_pm = 0.2 mas/yr, h_v = 5 km/s; background uniform in u over the data window and in
q over its box (x 12 deg, PMs 30 mas/yr, v 400 km/s). Shown: log10 p(q|u) (image, absolute, same scale per panel) and contours
at 10^-0.5 ... 10^-2.5 of the panel maximum; members as points. Not included: the per-star Gaia PM / v_los errors (added to the
kernel per star in the likelihood). Columns where the background is > 50% of p(u) are shaded grey (the model barely reaches them).
Usage: python bin/streams/plot_conditional_model.py om34.5_an16_am1.4_d5.6 [more tags] [--dir=spray_grid4]
  -> plots/condmodel_<tag>.png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
from ocen_dm.streams.path import project_gc

H_U, EPS, DCHI, NMIN, AGE_MAX = 0.5, 0.05, 5., 20, 700.
Q = (("x", r"$\Delta\phi_2$ [deg]", 0.5, 12., (-6, 6)), ("pmra", "pmra [mas/yr]", 0.2, 30., (-22, 0)),
     ("pmdec", "pmdec [mas/yr]", 0.2, 30., (-15, -3)), ("vlos", r"$v_{\rm los}$ [km/s]", 5., 400., (120, 300)))
SDIR = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--dir=")), "spray_grid4")
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
d = load_data(); du, dx = project_gc(d["l"], d["b"], GC); D = dict(x=dx, pmra=d["pmra"], pmdec=d["pmdec"], vlos=d["v"])
lo, hi = du.min(), du.max(); g_u = 1./(hi-lo)
ug = np.linspace(lo-2, hi+2, 300)
gauss = lambda a, b, h: np.exp(-0.5*((a[:, None]-b[None, :])/h)**2)/(np.sqrt(2*np.pi)*h)

for tag in [x for x in sys.argv[1:] if not x.startswith("--")]:
    m = dict(np.load(ROOT/f"results/streams/{SDIR}/{tag}.npz")); mu, mx = project_gc(m["l"], m["b"], GC); m["x"] = mx
    k = (m["chi"] > 0) & (m["age"] < AGE_MAX)
    edges = np.arange(0., m["chi"][k].max()+DCHI, DCHI)
    bins = [np.where(k & (m["chi"] >= e) & (m["chi"] < e+DCHI))[0] for e in edges]; bins = [b for b in bins if len(b) >= NMIN]; K = len(bins)
    idx = np.concatenate(bins); wt = np.concatenate([np.full(len(b), 1./(K*len(b))) for b in bins])
    Au = gauss(ug, mu[idx], H_U)*wt[None, :]                       # (Nu, M) weighted kernels in u
    fu = Au.sum(1); den = (1-EPS)*fu+EPS*g_u; bgfrac = EPS*g_u/den
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.5))
    for a, (q, lab, h, box, yr) in zip(ax, Q):
        qg = np.linspace(*yr, 300)
        f = Au @ gauss(qg, m[q][idx], h).T                          # (Nu, Nq) joint f(u, q)
        p = ((1-EPS)*f+EPS*g_u/box)/den[:, None]                    # conditional p(q | u) of the mixture
        lp = np.log10(p.T); vmax = lp.max()
        a.imshow(lp, origin="lower", aspect="auto", extent=(ug[0], ug[-1], *yr), cmap="Blues", vmin=vmax-3.5, vmax=vmax)
        a.contour(ug, qg, lp, levels=vmax+np.array([-2.5, -2., -1.5, -1., -0.5]), cmap="viridis", linewidths=1.)
        for j in np.where(bgfrac > 0.5)[0]:
            a.axvspan(ug[j]-0.5*(ug[1]-ug[0]), ug[j]+0.5*(ug[1]-ug[0]), color="0.6", alpha=0.5, lw=0)
        if q == "vlos":
            hv = np.isfinite(d["v"]); a.errorbar(du[hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="C3", mfc="w", ms=5, lw=0.8, zorder=5)
        else:
            a.scatter(du, D[q], s=3, c="C1", alpha=0.6, lw=0, zorder=5)
        a.set_xlim(ug[0], ug[-1]); a.set_ylim(*yr); a.set_xlabel(r"$\phi_1$ (u) [deg]"); a.set_ylabel(lab)
    fig.suptitle(f"{tag}: conditional model p(q | u) as used by score_conditional (K = {K} chi bins, eps = {EPS}); "
                 "contours 10^-0.5...10^-2.5 of peak; grey = background > 50% of p(u); stream-54 members as points")
    fig.tight_layout(); out = ROOT/f"plots/condmodel_{tag}.png"; fig.savefig(out, dpi=100); print(out)
