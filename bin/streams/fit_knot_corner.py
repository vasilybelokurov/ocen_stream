#!/usr/bin/env python3
"""Corner-style plot of all fit_knot_nm.py evaluations, coloured by knot chi2 (2026-10-25). NOT a posterior: points are where
Nelder-Mead went, not draws from a probability; it shows which parameter pairs are constrained and which trade off.
Search evaluations (128k releases) as dots, coloured by chi2 (clipped at best + 300); stars: each chain's 256k final point.
Usage: python bin/streams/fit_knot_corner.py results/streams/fitknot_20261025 -> plots/fit_knot_corner_<dirname>.png
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
NAMES = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")
LAB = dict(angle="angle [deg]", omega=r"$\Omega_b$ [km/s/kpc]", amp="amp", size="size", halo_q="q", vc=r"$v_c$ [km/s]", dist="d [kpc]",
           pmra=r"$\mu_{\alpha*}$", pmdec=r"$\mu_\delta$")


def main():
    od = Path(sys.argv[1]); od = od if od.is_absolute() else ROOT/od
    R = json.loads((od/"result.json").read_text())
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(od/"runs/c*_e*.json"))]
    X = np.array([[r["params"][n] for n in NAMES] for r in recs]); y = np.array([r["chi2"]["total"] for r in recs])
    o = np.argsort(-y); X, y = X[o], y[o]                                            # best points drawn last (on top)
    lo = min(d["mean"] for d in R.values()); n = len(NAMES)
    fig, ax = plt.subplots(n, n, figsize=(17, 17), constrained_layout=True)
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                a.scatter(X[:, i], y, c=y, s=8, cmap="viridis_r", vmin=lo, vmax=lo+300); a.set_yscale("log")
                for k, (cid, d) in enumerate(sorted(R.items())):
                    a.plot(d["params"][NAMES[i]], d["mean"], "*", color=f"C{k}", ms=12, mec="k")
            else:
                sc = a.scatter(X[:, j], X[:, i], c=y, s=8, cmap="viridis_r", vmin=lo, vmax=lo+300)
                for k, (cid, d) in enumerate(sorted(R.items())):
                    a.plot(d["params"][NAMES[j]], d["params"][NAMES[i]], "*", color=f"C{k}", ms=12, mec="k", label=f"{cid} {d['mean']:.0f}")
            if i == n-1: a.set_xlabel(LAB[NAMES[j]])
            else: a.set_xticklabels([])
            if j == 0 and i > 0: a.set_ylabel(LAB[NAMES[i]])
            elif i != j: a.set_yticklabels([])
    ax[1, 0].legend(fontsize=8, loc="best"); ax[0, 0].set_ylabel(r"$\chi^2$")
    fig.colorbar(sc, ax=ax[:3, 5:], shrink=0.6, label=r"knot $\chi^2$ (128k search evaluations)")
    fig.suptitle(f"{od.name}: all {len(y)} Nelder-Mead evaluations (NOT a posterior); stars: chain best, 256k finals mean chi2", fontsize=14)
    out = ROOT/f"plots/fit_knot_corner_{od.name}.png"; fig.savefig(out, dpi=80); print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
