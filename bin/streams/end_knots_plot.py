#!/usr/bin/env python3
"""Data vs model at the 25 knots for the best real-data models at three points along the valley (2026-10-28).

From the pilot logs in the private runs repository ($OCEN_RUNS_DIR): the lowest-chi2 model (bank 7) with bar angle in
5-10, 17.5-22.5 and 27.5-32.5 deg. Top row: data (black, 1 sigma) and the three models (summary +- bootstrap error) per observable:
sky track and width (one-Gaussian fits in 3-deg windows), pmra and pmdec (spline knots; values relative to the reference track, as in
score_inputs.npz). Bottom row: residual (data - model) / sqrt(sigma_data^2 + e^2) per knot; light band |residual| < 3.
Grey bands mark the knots discussed in docs/reviews/error_floors_fable_codex_20261028.md (pmdec first/last, width last two windows).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/end_knots_plot.py   -> plots/end_knots_20261028.png
"""
import glob, os
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocen_stream import pilots as PL, jaxspray as J, jaxscore as JS

ROOT = Path(__file__).resolve().parents[2]; RUNS = Path(os.environ["OCEN_RUNS_DIR"])
BINS = ((5, 10), (17.5, 22.5), (27.5, 32.5)); COL = ("#2a78d6", "#eb6834", "#1baf7a")    # categorical slots 1-3


def main():
    si = JS.load(ROOT/"results/gpu/score_inputs.npz"); D = {k: np.asarray(v) for k, v in si["D"].items()}; ht = J.load_host(ROOT/"results/gpu/host_tables.npz")
    L = [np.load(f) for d in ("pilots_20261027", "pilots_20261027_scan") for f in sorted(glob.glob(str(RUNS/d/"log"/"0*.npz")))]
    cat = lambda k: np.concatenate([q[k] for q in L])
    c, b, U, M, E = cat("chi2"), cat("bank"), cat("u"), cat("m"), cat("e"); g = (b == 7) & np.isfinite(c)
    ang = np.full(len(c), np.nan); ang[g] = [PL.to_params(u, ht)["angle"] for u in U[g]]
    k, na = len(D["pk"]), len(D["pmra_k"])
    blocks = (("sky track [deg]", D["pk"], D["sky_c"], D["sky_ce"], slice(0, k)),
              ("sky width [deg]", D["pk"], D["sky_s"], D["sky_se"], slice(k, 2*k)),
              ("pmra* offset [mas/yr]", D["pmra_k"], D["pmra"], np.sqrt(np.diag(D["pmra_C"])), slice(2*k, 2*k+na)),
              ("pmdec offset [mas/yr]", D["pmdec_k"], D["pmdec"], np.sqrt(np.diag(D["pmdec_C"])), slice(2*k+na, None)))
    flag = {1: (19.5, 22.5), 3: (-2., 28.)}
    best = []
    for lo, hi in BINS:
        s = np.where(g & (ang >= lo) & (ang < hi))[0]; i = s[np.argmin(c[s])]; best.append(i)
        print(f"angle {ang[i]:.1f}: chi2 {c[i]:.1f}")
    fig, ax = plt.subplots(2, 4, figsize=(18, 7.5), sharex="col", gridspec_kw=dict(height_ratios=(2, 1.2), hspace=0.05, wspace=0.3))
    for j, (name, x, d, sd, sl) in enumerate(blocks):
        A, R = ax[0, j], ax[1, j]
        for xf in flag.get(j, ()):
            for a in (A, R):
                a.axvspan(xf-0.9, xf+0.9, color="0.88", zorder=0)
        for n, (i, col) in enumerate(zip(best, COL)):
            dx = (n-1)*0.35; m, e = M[i][sl], E[i][sl]
            A.errorbar(x+dx, m, e, color=col, marker="o", ms=5, lw=2, capsize=0, label=f"model, angle {ang[i]:.0f}° (χ² {c[i]:.0f})")
            R.plot(x+dx, (d-m)/np.sqrt(sd**2+e**2), color=col, marker="o", ms=6, lw=0)
        A.errorbar(x, d, sd, color="k", marker="s", ms=6, lw=0, elinewidth=1.5, capsize=3, label="data", zorder=5)
        A.set_title(name, fontsize=11); R.axhline(0, color="0.5", lw=1)
        R.axhspan(-3, 3, color="0.95", zorder=0); R.set_ylim(-7, 7); R.set_xlabel("φ1 [deg]")
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04)); ax[1, 0].set_ylabel("(data − model) / σ")
    fig.suptitle("Best real-data models at three bar angles vs the 25-knot data vector; grey bands: the end knots that miss at every angle "
                 "(light band in residuals: |r| < 3)", fontsize=11)
    out = ROOT/"plots/end_knots_20261028.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); print("plot:", out)


if __name__ == "__main__":
    main()
