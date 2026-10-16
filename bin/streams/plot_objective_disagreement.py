#!/usr/bin/env python3
"""Why the track objective and the star-by-star likelihood disagree: refined 28 deg vs the track-optimised 27B long bar.

Models (32000 release epochs, seed 4, trailing arm, chi > 0, age < 700 Myr): A = refined 28-deg solution (paper Table 1;
results/streams/paper/bar28_seed4.npz); B = best point of the 27B long-bar track fit
(results/plot_data/track_fits_20261016/27B_long.json), sprayed here.
Columns 1-2: log particle density in (phi1, Delta phi2 / mu_alpha* / mu_delta) with the members (orange) and the median
tracks (black: members; colour: model; as in bin/streams/track_sensitivity.py). Column 3: members coloured by the per-star
ln L difference B - A from grid4_common.score (the star-by-star likelihood); bottom: per-star Delta ln L summed in 2-deg
phi1 bins (bars) and cumulative (line), and the track chi2 difference per bin (open symbols).
Output: plots/objective_disagreement_20261016.png, results/plot_data/objective_disagreement_20261016.json.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/plot_objective_disagreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"bin/streams/paper"), str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

from fig3_residuals import predict
from fit_track_objective import track_chi2, weights
from grid4_common import GC, d, du, dx, project_gc, score
from paper_models import spray
from track_sensitivity import EDGES

NREL, SEED = 32000, 4


def prepare(m):
    k = (m["chi"] > 0) & (m["age"] < 700.)
    mu, mx = project_gc(m["l"][k], m["b"][k], GC)
    return dict(phi1=mu, dphi2=mx, pmra=m["pmra"][k], pmdec=m["pmdec"][k])


def tracks(mm):
    pred = predict(mm)
    out = dict(dphi2=[], pmra=[], pmdec=[], n_pred=[])
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        km = (mm["phi1"] >= a) & (mm["phi1"] < b) & (np.abs(mm["dphi2"]) < 4)
        out["dphi2"].append(float(np.median(mm["dphi2"][km])) if km.sum() >= 20 else np.nan)
        ks = (du >= a) & (du < b)
        for q in ("pmra", "pmdec"):
            v = pred[q][ks]
            out[q].append(float(np.nanmedian(v)) if np.isfinite(v).sum() >= 15 else np.nan)
        out["n_pred"].append(int(np.isfinite(pred["pmra"][ks]).sum()))
    return out


def main():
    pA = json.loads((ROOT/"results/plot_data/paper_models.json").read_text())["display"]["bar28"]["params"]
    pB = json.loads((ROOT/"results/plot_data/track_fits_20261016/27B_long.json").read_text())["best"]["params"]
    mA_raw = dict(np.load(ROOT/"results/streams/paper/bar28_seed4.npz"))
    mB_raw = spray(pB, NREL, SEED)
    sA, sB = score(mA_raw), score(mB_raw)
    mA, mB = prepare(mA_raw), prepare(mB_raw)
    tA, tB = tracks(mA), tracks(mB)
    m_obs, var = weights()
    cA, cB = track_chi2(tA, m_obs, var), track_chi2(tB, m_obs, var)
    dl = np.asarray(sB["per_star"])-np.asarray(sA["per_star"])
    c = 0.5*(EDGES[:-1]+EDGES[1:])
    nb = len(c)
    obs = dict(dphi2=m_obs[:nb], pmra=m_obs[nb:2*nb], pmdec=m_obs[2*nb:])
    rows = [("dphi2", r"$\Delta\phi_2$ [deg]", (-6, 6), dx), ("pmra", r"$\mu_{\alpha*}$ [mas/yr]", (-22, 0), d["pmra"]),
            ("pmdec", r"$\mu_\delta$ [mas/yr]", (-14, -4), d["pmdec"])]
    fig, ax = plt.subplots(4, 3, figsize=(15, 12), sharex=True, gridspec_kw=dict(hspace=0.06, wspace=0.18, height_ratios=[1, 1, 1, 0.9]))
    xe = np.linspace(-3, 28, 125)
    titles = [f"A: refined 28° (Ω {pA['om']:.2f}, A {pA['am']:.2f}, S {pA['size']:.2f}, d {pA['dist']:.2f})\n"
              f"star-by-star ln L {sA['total']:.0f}; track χ² {cA[0]:.0f} (missing {cA[1]})",
              f"B: track fit 27B long bar (Ω {pB['om']:.2f}, A {pB['am']:.2f}, S {pB['size']:.2f}, d {pB['dist']:.2f})\n"
              f"star-by-star ln L {sB['total']:.0f}; track χ² {cB[0]:.0f} (missing {cB[1]})"]
    for j, (mm, tr, col) in enumerate(((mA, tA, "C0"), (mB, tB, "C3"))):
        for i, (q, lab, lim, dv) in enumerate(rows):
            h, _, _ = np.histogram2d(mm["phi1"], mm[q], (xe, np.linspace(*lim, 90)))
            ax[i, j].pcolormesh(xe, np.linspace(*lim, 90), h.T, cmap="Greys", norm=LogNorm(vmin=1, vmax=max(h.max(), 2)), rasterized=True)
            ok = np.isfinite(dv)
            ax[i, j].scatter(du[ok], dv[ok], s=0.8, lw=0, color="#d95f02", alpha=0.5, rasterized=True)
            ax[i, j].plot(c, obs[q], "k-o", ms=3, lw=1.2, label="members (median)" if i == 0 else None)
            ax[i, j].plot(c, tr[q], "-s", color=col, ms=3, lw=1.2, label="model (median)" if i == 0 else None)
            ax[i, j].set_ylim(*lim)
            if j == 0:
                ax[i, j].set_ylabel(lab)
        ax[0, j].set_title(titles[j], fontsize=8.5); ax[0, j].legend(fontsize=7, loc="lower left")
    lim = np.nanpercentile(np.abs(dl), 98)
    for i, (q, lab, ylim, dv) in enumerate(rows):
        ok = np.isfinite(dv)
        sc = ax[i, 2].scatter(du[ok], dv[ok], c=dl[ok], s=3, lw=0, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), rasterized=True)
        ax[i, 2].plot(c, obs[q], "k-", lw=0.8); ax[i, 2].plot(c, tA[q], "-", color="C0", lw=1); ax[i, 2].plot(c, tB[q], "-", color="C3", lw=1)
        ax[i, 2].set_ylim(*ylim)
    ax[0, 2].set_title("members coloured by per-star ln L(B) − ln L(A)\n(red: B better; lines: medians, black data, blue A, red B)", fontsize=8.5)
    cb = fig.colorbar(sc, ax=ax[:3, 2], fraction=0.04, pad=0.01); cb.set_label(r"$\Delta\ln L$ per star (B − A)")
    # bottom row: binned per-star Delta lnL and track chi2 difference
    sums = np.array([dl[(du >= a) & (du < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])])
    nst = np.array([((du >= a) & (du < b)).sum() for a, b in zip(EDGES[:-1], EDGES[1:])])

    def per_bin_chi2(tr):
        from track_sensitivity import vec
        m = vec(tr); r2 = (m_obs-m)**2/var
        return np.nansum(r2.reshape(3, nb), axis=0)
    dchi = per_bin_chi2(tB)-per_bin_chi2(tA)
    for j in range(3):
        a = ax[3, j]
        if j < 2:
            a.axis("off"); continue
    a = ax[3, 0]; a.axis("on")
    a.bar(c, sums, width=1.7, color=np.where(sums > 0, "C3", "C0"), alpha=0.7, label="Σ per-star Δln L (B − A) in bin")
    a.plot(np.sort(du), np.cumsum(dl[np.argsort(du)]), "k-", lw=1, label="cumulative Δln L")
    a.axhline(0, color="k", lw=0.5); a.set_ylabel(r"$\Delta\ln L$ (B − A)"); a.set_xlabel(r"$\phi_1$ [deg]"); a.legend(fontsize=7)
    a2 = ax[3, 1]; a2.axis("on")
    a2.bar(c, -dchi/2, width=1.7, color=np.where(dchi < 0, "C3", "C0"), alpha=0.7, label=r"track: $-\frac{1}{2}\Delta\chi^2$ (B − A) in bin")
    a2.axhline(0, color="k", lw=0.5); a2.set_ylabel(r"$-\Delta\chi^2_{\rm track}/2$ (B − A)"); a2.set_xlabel(r"$\phi_1$ [deg]"); a2.legend(fontsize=7)
    ax[3, 2].axis("on")
    ax[3, 2].plot(c, nst, "k-o", ms=3); ax[3, 2].set_ylabel("members per 2° bin"); ax[3, 2].set_yscale("log"); ax[3, 2].set_xlabel(r"$\phi_1$ [deg]")
    ax[0, 0].set_xlim(-3, 28)
    fig.savefig(ROOT/"plots/objective_disagreement_20261016.png", dpi=130, bbox_inches="tight")
    out = dict(params_A=pA, params_B=pB, lnL_A=sA["total"], lnL_B=sB["total"], track_chi2_A=cA[0], track_chi2_B=cB[0],
               missing_A=cA[1], missing_B=cB[1], bins=c.tolist(), dlnL_per_bin=sums.tolist(), n_members=nst.tolist(),
               dchi2_track_per_bin=dchi.tolist(), segments_A=cA[2], segments_B=cB[2])
    (ROOT/"results/plot_data/objective_disagreement_20261016.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items() if k not in ("params_A", "params_B")}))


if __name__ == "__main__":
    main()
