#!/usr/bin/env python3
"""The conditional-likelihood model (score_conditional) recast into the original observable space (l, b, PMs, v_los, d vs b).
Model particles: trailing arm, chi > 0, age < 700 Myr, chi bins of 5 kpc Myr with >= 20 particles, base weight 1/(K n_k).
  Row 1 (what the likelihood compares with the data): weights multiplied by n_data(u) / [(1-eps) f_u(u) + eps g_u], i.e. the model's
        density along the chord great-circle coordinate u (phi1) replaced by the members' (KDE, h_u = 0.5 deg), with the same
        background-regularised denominator as the likelihood. Particles outside the data's u range get ~0 weight.
  Row 2: the same particles and kernels with the base weights only (the model's own density along the stream).
Kernels are the likelihood's fixed bandwidths: 0.5 deg on the sky (isotropic; sigma_l = 0.5/cos b), 0.2 mas/yr, 5 km/s;
distance (not in the likelihood) 0.1 kpc for display. Per-star Gaia errors and the 5% uniform background are not drawn.
Contours at 10^-0.5 ... 10^-2.5 of each panel's peak; members as points (v_los stars, CMD distance track).
Usage: python bin/streams/plot_conditional_orig.py om34.5_an16_am1.4_d5.6 [more tags] [--dir=spray_grid4]
  -> plots/condorig_<tag>.png
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data
from ocen_dm.streams.path import project_gc

H_U, EPS, DCHI, NMIN, AGE_MAX = 0.5, 0.05, 5., 20, 700.
H = dict(b=0.5, pmra=0.2, pmdec=0.2, vlos=5., d=0.1)            # sigma_l per particle = 0.5/cos b
PANELS = (("l", "b", (-75, -10), (5, 45)), ("b", "l", (10, 45), (-70, -30)), ("b", "pmra", (10, 45), (-22, 0)),
          ("b", "pmdec", (10, 45), (-15, -3)), ("b", "vlos", (10, 45), (120, 300)), ("b", "d", (10, 45), (2.5, 6.5)))
SDIR = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--dir=")), "spray_grid4")
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
cmd = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
d = load_data(); du, _ = project_gc(d["l"], d["b"], GC); lo, hi = du.min(), du.max(); g_u = 1./(hi-lo)
gauss = lambda a, b, h: np.exp(-0.5*((a[:, None]-b[None, :])/h)**2)/(np.sqrt(2*np.pi)*h)


def image(m, wt, xq, yq, xr, yr, n=300):
    xg, yg = np.linspace(*xr, n), np.linspace(*yr, n)
    hx = 0.5/np.cos(np.radians(m["b"])) if xq == "l" else np.full(len(wt), H[xq])
    hy = 0.5/np.cos(np.radians(m["b"])) if yq == "l" else np.full(len(wt), H[yq])
    Gx = np.exp(-0.5*((xg[:, None]-m[xq][None, :])/hx)**2)/(np.sqrt(2*np.pi)*hx)
    Gy = np.exp(-0.5*((yg[:, None]-m[yq][None, :])/hy)**2)/(np.sqrt(2*np.pi)*hy)
    return xg, yg, (Gy*wt[None, :]) @ Gx.T                        # (ny, nx)


for tag in [x for x in sys.argv[1:] if not x.startswith("--")]:
    m0 = dict(np.load(ROOT/f"results/streams/{SDIR}/{tag}.npz"))
    k = (m0["chi"] > 0) & (m0["age"] < AGE_MAX)
    edges = np.arange(0., m0["chi"][k].max()+DCHI, DCHI)
    bins = [np.where(k & (m0["chi"] >= e) & (m0["chi"] < e+DCHI))[0] for e in edges]; bins = [b for b in bins if len(b) >= NMIN]; K = len(bins)
    idx = np.concatenate(bins); w0 = np.concatenate([np.full(len(b), 1./(K*len(b))) for b in bins])
    m = {q: m0[q][idx] for q in ("l", "b", "pmra", "pmdec", "vlos", "d")}; mu, _ = project_gc(m["l"], m["b"], GC)
    ug = np.linspace(min(lo, mu.min())-2, max(hi, mu.max())+2, 2000)
    fu = (gauss(ug, mu, H_U)*w0[None, :]).sum(1); nd = gauss(ug, du, H_U).mean(1)
    w1 = w0*np.interp(mu, ug, nd)/np.interp(mu, ug, (1-EPS)*fu+EPS*g_u); w1 /= w1.sum()
    fig, ax = plt.subplots(2, 6, figsize=(30, 9.5))
    for r, wt in enumerate((w1, w0)):
        for c, (xq, yq, xr, yr) in enumerate(PANELS):
            a = ax[r, c]; xg, yg, img = image(m, wt, xq, yq, xr, yr); lv = np.log10(img+1e-30); vmax = lv.max()
            if yq == "vlos":
                hv = np.isfinite(d["v"]); a.errorbar(d["b"][hv], d["v"][hv], yerr=d["e_v"][hv], fmt="o", color="C3", mfc="w", ms=5, lw=0.8)
            elif yq == "d":
                a.errorbar([17.5]+[0.5*(c_["b"][0]+c_["b"][1]) for c_ in cmd], [5.43]+[5.43*c_["d_ratio"] for c_ in cmd],
                           yerr=[0]+[5.43*c_["d_ratio"]*np.log(10)/5*c_["dm_err"] for c_ in cmd], fmt="s", color="C3", mfc="w", ms=6)
            else:
                a.plot(d[xq], d[yq], ".", color="0.35", ms=1.5, alpha=0.6, zorder=1)
            a.contour(xg, yg, lv, levels=vmax-np.array([2.5, 2.0, 1.5, 1.0, 0.5]), cmap="viridis", linewidths=1.3, zorder=3)
            a.set_xlim(*(xr[::-1] if xq == "l" else xr)); a.set_ylim(*yr); a.set_xlabel(xq); a.set_ylabel(yq)
            if c == 0:
                a.set_title(("reweighted to members' density along phi1 (as in likelihood)" if r == 0 else
                             "model's own density along stream") + f", K = {K}", fontsize=9)
    fig.suptitle(f"score_conditional model of {tag} in observable space: fixed kernels (0.5 deg, 0.2 mas/yr, 5 km/s; d 0.1 kpc display only); "
                 "contours 10^-0.5...10^-2.5 of peak; stream-54 members as points", fontsize=12)
    fig.tight_layout(); out = ROOT/f"plots/condorig_{tag}.png"; fig.savefig(out, dpi=62); print(out)
