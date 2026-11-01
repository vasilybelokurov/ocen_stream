#!/usr/bin/env python3
"""Corner plot of a production SMC run (2026-11-01): the final population (beta = 1, equally weighted after resampling), all nine
parameters in physical units; red star/lines = best particle; dashed grey lines = prior box edges that fall inside a panel's range
(flat prior in the box). Axis ranges hold the central 98.5% of each parameter (the few particles of the low-angle clump
~5-10 deg fall outside). Contours at 1-2 sigma (39%/86% of the 2-D mass).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/main_corner.py --run smc_main_n1200  -> plots/main_corner_<run>.png
"""
import argparse, os
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from ocen_stream import pilots as PL, jaxspray as J

ROOT = Path(__file__).resolve().parents[2]; RUNS = Path(os.environ["OCEN_RUNS_DIR"])
LAB = dict(angle=r"$\phi_b$ [deg]", omega=r"$\Omega_b$ [km/s/kpc]", amp=r"$A$", size=r"$S$", dist=r"$d$ [kpc]",
           pmra=r"$\mu_{\alpha*}$ [mas/yr]", pmdec=r"$\mu_\delta$ [mas/yr]", vc=r"$v_c(R_0)$ [km/s]", halo_q=r"halo $q$")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    ht = J.load_host(ROOT/"results/gpu/host_tables.npz"); z = np.load(RUNS/a.run/"smc_final.npz"); x, c = z["x"], z["c"]
    X = np.array([[PL.to_params(u, ht)[k] for k in PL.NAMES] for u in x]); b = int(np.argmin(c))
    fig = corner.corner(X, labels=[LAB[k] for k in PL.NAMES], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                        levels=(1-np.exp(-0.5), 1-np.exp(-2.)), plot_datapoints=True, data_kwargs=dict(ms=1.5, alpha=0.4),
                        bins=30, smooth=1.0, range=[0.985]*len(PL.NAMES), color="tab:blue", label_kwargs=dict(fontsize=12), title_kwargs=dict(fontsize=10))
    corner.overplot_lines(fig, X[b], color="tab:red", lw=1); corner.overplot_points(fig, X[b][None], marker="*", ms=12, color="tab:red")
    ax = np.array(fig.axes).reshape(len(PL.NAMES), len(PL.NAMES))
    for j, k in enumerate(PL.NAMES):                                                 # box edges inside the plotted range
        for e in PL.LIM.get(k, ()):
            for i in range(j, len(PL.NAMES)):
                lo, hi = ax[i, j].get_xlim()
                if lo <= e <= hi:
                    ax[i, j].axvline(e, c="0.4", ls="--", lw=1)
            for jj in range(j):
                lo, hi = ax[j, jj].get_ylim()
                if lo <= e <= hi:
                    ax[j, jj].axhline(e, c="0.4", ls="--", lw=1)
    fig.suptitle(f"{a.run}: SMC posterior, N = {len(x)}, beta = 1 (red: best particle, chi2 {c[b]:.1f}; dashed: prior box edge)",
                 fontsize=14, y=1.01)
    out = ROOT/f"plots/main_corner_{a.run}.png"; fig.savefig(out, dpi=90, bbox_inches="tight"); print("plot:", out)


if __name__ == "__main__":
    main()
