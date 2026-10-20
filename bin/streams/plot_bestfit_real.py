#!/usr/bin/env python3
"""Real data vs the best model of the direct per-angle profile (2026-10-20).

Model: the final 256k run with the highest jackknife ln L (+ prior) at the best angle of results/streams/profile_real_20261019
(override with --run). Panels vs phi1 (chord great-circle frame): (a) Delta phi2, (b) mu_alpha*, (c) mu_delta, (d) v_los --
model debris (chi > 0, age < 700 Myr) as a 2-D histogram, members as points; (e) binned residuals (data - model, 2-deg phi1
bins): Delta phi2 medians, and PMs with the model PM of a member = median of particles within 0.5 deg of its
(phi1, Delta phi2) (paper fig3 convention); the paper's refined 16-deg model (32k, seed 4) for comparison.
Output: plots/bestfit_real_<tag>.png
Usage: PYTHONPATH=src python bin/streams/plot_bestfit_real.py [--run PATH.npz] [--tag TAG]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from grid4_common import d, du, dx, project_gc, GC


def load(npz):
    m = dict(np.load(npz)); u, x = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700.)
    return dict(u=u[k], x=x[k], pmra=m["pmra"][k], pmdec=m["pmdec"][k], v=m["vlos"][k])


def resid(m, edges):
    c = 0.5*(edges[1:]+edges[:-1]); km = np.abs(m["x"]) < 4
    dx_mod = np.array([np.median(m["x"][km][(m["u"][km] >= a) & (m["u"][km] < b)]) if ((m["u"][km] >= a) & (m["u"][km] < b)).sum() >= 20 else np.nan
                       for a, b in zip(edges[:-1], edges[1:])])
    dx_dat = np.array([np.median(dx[(du >= a) & (du < b)]) if ((du >= a) & (du < b)).sum() >= 15 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    tree = cKDTree(np.column_stack((m["u"], m["x"]))); nb = tree.query_ball_point(np.column_stack((du, dx)), 0.5)
    out = dict(c=c, dphi2=dx_dat-dx_mod)
    for q in ("pmra", "pmdec"):
        pred = np.array([np.median(m[q][i]) if len(i) >= 5 else np.nan for i in nb]); r = d[q]-pred
        out[q] = np.array([np.nanmedian(r[(du >= a) & (du < b)]) if np.isfinite(r[(du >= a) & (du < b)]).sum() >= 15 else np.nan
                           for a, b in zip(edges[:-1], edges[1:])])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--tag", default="profile_best"); a = ap.parse_args()
    P = ROOT/"results/streams/profile_real_20261019"
    if a.run:
        run = Path(a.run); rec = json.loads(run.with_suffix(".json").read_text())
    else:
        prof = json.loads((P/"profile.json").read_text())
        best_ang = max(prof, key=lambda k: prof[k]["mean"]+prof[k].get("ln_prior", 0.))
        recs = [json.loads(f.read_text()) for f in (P/"runs").glob(f"final_a{best_ang}_*.json")]
        rec = max(recs, key=lambda r: r["total_jk"]); run = P/"runs"/f"{rec['run_id']}.npz"
    m = load(run); p = rec["params"]; ref = load(ROOT/"results/streams/paper/bar16_seed4.npz")
    fig, ax = plt.subplots(5, 1, figsize=(9, 15), sharex=True, constrained_layout=True)
    ext = (-6, 30)
    for i, (q, dq, lab, yl) in enumerate((("x", dx, r"$\Delta\phi_2$ [deg]", (-12, 25)), ("pmra", d["pmra"], r"$\mu_{\alpha*}$ [mas/yr]", (-28, 2)),
                                          ("pmdec", d["pmdec"], r"$\mu_\delta$ [mas/yr]", (-17, -2)), ("v", d["v"], r"$v_{\rm los}$ [km/s]", (-50, 450)))):
        h = ax[i].hist2d(m["u"], m[q], bins=(180, 160), range=(ext, yl), cmap="Greys", cmin=1, norm=matplotlib.colors.LogNorm())
        ok = np.isfinite(dq); ax[i].plot(du[ok], dq[ok], ".", color="C1", ms=2.5 if q != "v" else 6, alpha=0.6, label="members" if i == 0 else None)
        ax[i].set(ylabel=lab, ylim=yl)
    ax[0].legend(loc="upper left", fontsize=8); ax[0].text(0.99, 0.95, "grey: model debris (age < 700 Myr)", transform=ax[0].transAxes, ha="right", va="top", fontsize=8)
    e = np.arange(-2., 28.1, 2.); r1, r0 = resid(m, e), resid(ref, e)
    for q, mk, col in (("dphi2", "o", "C0"), ("pmra", "s", "C2"), ("pmdec", "^", "C3")):
        ax[4].plot(r1["c"], r1[q], mk+"-", color=col, label=f"{q}: best model")
        ax[4].plot(r0["c"], r0[q], mk+":", color=col, alpha=0.6, label=f"{q}: paper 16 deg")
    ax[4].axhline(0, color="k", lw=0.6); ax[4].set(ylabel="data - model (deg; mas/yr)", xlabel=r"$\phi_1$ [deg]", xlim=ext); ax[4].legend(fontsize=7, ncol=3)
    fig.suptitle(f"real data vs best profile model: angle {p['angle']:.1f}, Omega {p['omega']:.1f}, A {p['amp']:.2f}, S {p['size']:.2f}, "
                 f"d {p['dist']:.2f} kpc, vc {p['vc']:.0f}, q {p['halo_q']:.2f}; ln L (jk) {rec['total_jk']:.0f}", fontsize=9)
    fig.savefig(ROOT/f"plots/bestfit_real_{a.tag}.png", dpi=110); print(ROOT/f"plots/bestfit_real_{a.tag}.png")


if __name__ == "__main__":
    main()
