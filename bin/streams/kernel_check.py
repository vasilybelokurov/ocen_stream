#!/usr/bin/env python3
"""Kernel diagnostics for score_conditional on grid4 sprays (no new sprays).
(1) n_eff per member: effective number of model particles contributing to f(u_i, w_i), n_eff = (sum t_j)^2 / sum t_j^2 with
    t_j = particle weight x kernel value (same weights, cuts and fixed bandwidths as the likelihood; PM kernel includes the
    star's Gaia covariance; v_los term for stars that have it).
(2) bandwidth sensitivity: lnL with h = (h_u, h_x, h_pmra, h_pmdec) and hv scaled by 1, 1.5, 2 for a set of grid models.
Usage: python bin/streams/kernel_check.py -> results/plot_data/kernel_check.json
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data
from ocen_dm.streams.score import score_conditional
from ocen_dm.streams.path import project_gc

H0, HV0 = np.array([0.5, 0.5, 0.2, 0.2]), 5.
d = load_data(); nd = len(d["l"])
cov = np.zeros((nd, 2, 2)); cov[:, 0, 0] = d["e_pmra"]**2; cov[:, 1, 1] = d["e_pmdec"]**2; cov[:, 0, 1] = cov[:, 1, 0] = d["rho"]*d["e_pmra"]*d["e_pmdec"]
GC = json.loads((ROOT/"results/plot_data/stream54_gc_frame_chord.json").read_text())
du, dx = project_gc(d["l"], d["b"], GC); dW = np.column_stack((dx, d["pmra"], d["pmdec"])); win = (du.min(), du.max())
load = lambda t: dict(np.load(ROOT/f"results/streams/spray_grid4/{t}.npz"))


def neff(m, h=H0, hv=HV0):
    mu, mx = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700)
    edges = np.arange(0., m["chi"][k].max()+5., 5.)
    bins = [np.where(k & (m["chi"] >= e) & (m["chi"] < e+5.))[0] for e in edges]; bins = [b for b in bins if len(b) >= 20]; K = len(bins)
    idx = np.concatenate(bins); w = np.concatenate([np.full(len(b), 1./(K*len(b))) for b in bins])
    U, X, PA, PD, V = mu[idx], mx[idx], m["pmra"][idx], m["pmdec"][idx], m["vlos"][idx]
    sx2 = h[2]**2+cov[:, 0, 0]; sy2 = h[3]**2+cov[:, 1, 1]; cxy = cov[:, 0, 1]; det = sx2*sy2-cxy**2
    hasv = np.isfinite(d["v"]); sv = np.sqrt(hv**2+np.nan_to_num(d["e_v"])**2); out = np.zeros(nd)
    for i0 in range(0, nd, 300):
        s = slice(i0, min(nd, i0+300))
        lt = -0.5*((du[s, None]-U)/h[0])**2-0.5*((dx[s, None]-X)/h[1])**2
        ax_, ay = d["pmra"][s, None]-PA, d["pmdec"][s, None]-PD
        lt += -0.5*(sy2[s, None]*ax_**2-2*cxy[s, None]*ax_*ay+sx2[s, None]*ay**2)/det[s, None]
        lt += np.where(hasv[s, None], -0.5*((np.nan_to_num(d["v"][s])[:, None]-V)/sv[s, None])**2, 0.)
        t = w*np.exp(lt-lt.max(1, keepdims=True)); out[s] = t.sum(1)**2/(t**2).sum(1)
    return out


tags = ["om34.5_an16_am1.4_d5.6", "om34.5_an20_am1.2_d5.6", "om34.5_an24_am1.2_d5.6", "om35_an28_am1.2_d5.6",
        "om36_an16_am1.4_d5.6", "om33_an20_am1.2_d5.5", "om34_an16_am1.2_d5.6", "om35_an20_am1.4_d5.45"]
res = dict(neff={}, lnL={})
for t in tags:
    n = neff(load(t)); res["neff"][t] = dict(p10=float(np.percentile(n, 10)), p50=float(np.median(n)), frac_lt5=float(np.mean(n < 5)), frac_lt20=float(np.mean(n < 20)))
    print(t, "n_eff p10 %.1f median %.1f  frac<5 %.2f  frac<20 %.2f" % tuple(res["neff"][t].values()), flush=True)
for f in (1., 1.5, 2.):
    for t in tags:
        m = load(t); mu, mx = project_gc(m["l"], m["b"], GC)
        s = score_conditional(du, dW, d["v"], d["e_v"], cov, mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"], m["chi"], m["age"],
                              h=tuple(H0*f), hv=HV0*f, u_window=win)
        res["lnL"].setdefault(str(f), {})[t] = s["total"]
    v = res["lnL"][str(f)]; best = max(v.values())
    print(f"h x{f}: " + "  ".join(f"{t}:{v[t]-best:.0f}" for t in tags), flush=True)
(ROOT/"results/plot_data/kernel_check.json").write_text(json.dumps(res, indent=1))
