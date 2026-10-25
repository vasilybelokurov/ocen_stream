#!/usr/bin/env python3
"""Can any stored model reach the data at each of the 25 knot points? (2026-10-25)

For every data point (sky track c and width s per window, pmra / pmdec at the spline knots) shows the model values of all rescored
models with total chi2 < CUT (colour = total chi2) against the data +- 1 sigma; then, for the persistent misfits (sky width at
phi1 19.5 / 22.5, pmra at knot ~18), the model value vs each parameter, to see which parameters move it and whether any model
gets there. Reads results/plot_data/stream_model_rescore_vec.csv (stream_model_rescore.py --tag vec).
Usage: PYTHONPATH=src python bin/streams/misfit_reach.py [CUT] -> plots/misfit_reach.png, plots/misfit_reach_params.png
"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
CUT = float(sys.argv[1]) if len(sys.argv) > 1 else 600.
sys.argv = [sys.argv[0]]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import stream_model_compare as MC

PAR = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")
OBS = (("sky_c", "sky track c [deg]"), ("sky_s", "sky width s [deg]"), ("pmra", "pmra - frozen track [mas/yr]"), ("pmdec", "pmdec - frozen track [mas/yr]"))


def main():
    D = MC.data_vector(); rows = [r for r in csv.DictReader(open(ROOT/"results/plot_data/stream_model_rescore_vec.csv"))
                                  if r.get("chi2_total") and float(r["chi2_total"]) < CUT]
    V = {k: np.array([json.loads(r["v_"+k]) for r in rows], float) for k, _ in OBS}; C = np.array([float(r["chi2_total"]) for r in rows])
    X = {k: np.array([float(r[k]) for r in rows]) for k in PAR}; o = np.argsort(-C)                       # best drawn last
    fig, ax = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for a, (k, lab) in zip(ax.ravel(), OBS):
        x = D["pk"] if k.startswith("sky") else D[k+"_k"]; de = D[k+"e"] if k.startswith("sky") else np.sqrt(np.diag(D[k+"_C"]))
        xx = np.broadcast_to(x, V[k].shape)+np.random.default_rng(0).uniform(-0.4, 0.4, V[k].shape)
        s = a.scatter(xx[o].ravel(), V[k][o].ravel(), c=np.repeat(C[o], len(x)), s=4, cmap="viridis_r", vmax=CUT)
        a.errorbar(x, D[k], de, fmt="ro", ms=6, capsize=4, zorder=5, label="data +- 1 sigma")
        frac = [np.mean(np.abs(V[k][:, i]-D[k][i]) < de[i]) for i in range(len(x))]
        for xi, f in zip(x, frac):
            a.annotate(f"{100*f:.0f}%", (xi, a.get_ylim()[1]), fontsize=7, ha="center", va="top")
        a.set(xlabel=r"$\phi_1$ [deg]", ylabel=lab, title=f"{k}: models (chi2 < {CUT:.0f}); top: % of models within 1 sigma of the data"); a.legend(fontsize=8)
    fig.colorbar(s, ax=ax, label="total chi2"); fig.savefig(ROOT/"plots/misfit_reach.png", dpi=80); print("plots/misfit_reach.png")
    # the persistent misfits vs parameters
    picks = [("sky_s", int(np.argmin(np.abs(D["pk"]-19.5)))), ("sky_s", int(np.argmin(np.abs(D["pk"]-22.5)))), ("pmra", int(np.argmin(np.abs(D["pmra_k"]-18))))]
    fig, ax = plt.subplots(len(picks), len(PAR), figsize=(24, 9), constrained_layout=True, sharey="row")
    for i, (k, j) in enumerate(picks):
        de = D[k+"e"][j] if k.startswith("sky") else np.sqrt(D[k+"_C"][j, j]); xk = D["pk"][j] if k.startswith("sky") else D[k+"_k"][j]
        for a, p in zip(ax[i], PAR):
            a.scatter(X[p][o], V[k][o, j], c=C[o], s=4, cmap="viridis_r", vmax=CUT); a.axhspan(D[k][j]-de, D[k][j]+de, color="r", alpha=0.25)
            a.set(xlabel=p); r_ = np.corrcoef(X[p], V[k][:, j])[0, 1] if np.isfinite(V[k][:, j]).all() else np.nan; a.set_title(f"r = {r_:+.2f}", fontsize=9)
        ax[i, 0].set_ylabel(f"{k} at phi1 {xk:.1f}")
        print(f"{k} at phi1 {xk:.1f}: data {D[k][j]:.2f} +- {de:.2f}; models min/median/max {np.nanmin(V[k][:, j]):.2f} / {np.nanmedian(V[k][:, j]):.2f} / "
              f"{np.nanmax(V[k][:, j]):.2f}; within 1 sigma {np.mean(np.abs(V[k][:, j]-D[k][j]) < de)*100:.1f}%")
    fig.suptitle("Persistent misfits vs parameters (red band = data +- 1 sigma; colour = total chi2)")
    fig.savefig(ROOT/"plots/misfit_reach_params.png", dpi=70); print("plots/misfit_reach_params.png")


if __name__ == "__main__":
    main()
