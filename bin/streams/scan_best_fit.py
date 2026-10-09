#!/usr/bin/env python3
"""Conditional likelihood maps around the free-angle best fit (results/plot_data/fit_bar_ocen_freeangle.json, max lnpost):
1D slices (9 points; other parameters at the best fit) in angle 21-33, Omega_b 33.5-36.5, amplitude 0.8-1.4, d +-0.2 kpc,
pmra/pmdec +-0.06 mas/yr; 2D grid angle 21-31 (2 deg) x amplitude 0.9-1.3 (0.1). Score = robust chi-KDE lnL + the fit's
log-priors (amplitude N(1.0, 0.1) extended beyond the 1.2 cap; d N(5.43, 0.15); PMs N(cat, 0.027); angle N(27, 2)).
Same spray set-up as the fit (model A, 8000 epochs over the last 1000 Myr, seed 1). Not re-optimised: conditional slices.
Usage: python bin/streams/scan_best_fit.py -> results/plot_data/scan_best_fit.json, plots/scan_best_fit.png
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, host_potential
from ocen_dm.streams.frames import observables, to_model, OCEN_OBS
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_chi_kde

PRIOR = dict(amp=(1.0, 0.1), d=(5.43, 0.15), pmra=(OCEN_OBS["pmra"], 0.027), pmdec=(OCEN_OBS["pmdec"], 0.027), angle=(27., 2.))
d = load_data()
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR; trel = np.linspace(T-1000/AGAMA_T_MYR, T, 8000)
L = json.loads((ROOT/"results/plot_data/fit_bar_ocen_freeangle.json").read_text())["log"]
best = max(L, key=lambda x: x["lnpost"]); B = {k: best[k] for k in ("angle", "omega", "amp", "d", "pmra", "pmdec")}
print("best fit:", B, "lnpost", round(best["lnpost"], 1), flush=True)


def evaluate(p):
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=p["amp"])
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["d"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"])[0]
    sp = spray_unwrapped(host, today, T, r_kpc, M, trel, seed=1); tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
    lnl = score_chi_kde(d, dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], chi=sp["chi"][tr],
                                age=sp["t_release_myr_ago"][tr]), robust=True)["total"]
    return lnl, lnl+sum(-0.5*((p[k]-m)/s)**2 for k, (m, s) in PRIOR.items())


def main():
    RANGES = dict(angle=np.linspace(21, 33, 9), omega=np.linspace(33.5, 36.5, 9), amp=np.linspace(0.8, 1.4, 9),
                  d=B["d"]+np.linspace(-0.2, 0.2, 9), pmra=B["pmra"]+np.linspace(-0.06, 0.06, 9), pmdec=B["pmdec"]+np.linspace(-0.06, 0.06, 9))
    out = dict(best=B, best_lnpost=best["lnpost"], slices={}, grid=[]); outp = ROOT/"results/plot_data/scan_best_fit.json"
    for k, vals in RANGES.items():
        out["slices"][k] = []
        for v in vals:
            t0 = time.time(); lnl, lp = evaluate(dict(B, **{k: float(v)}))
            out["slices"][k].append(dict(value=float(v), lnL=lnl, lnpost=lp))
            print(f"slice {k:6s} = {v:8.4f}: lnL {lnl:10.1f} lnpost {lp:10.1f} [{time.time()-t0:.0f} s]", flush=True)
            outp.write_text(json.dumps(out, indent=1))
    for an in np.arange(21, 31.1, 2.):
        for am in np.arange(0.9, 1.31, 0.1):
            t0 = time.time(); lnl, lp = evaluate(dict(B, angle=float(an), amp=float(am)))
            out["grid"].append(dict(angle=float(an), amp=float(round(am, 2)), lnL=lnl, lnpost=lp))
            print(f"grid angle {an:4.0f} amp {am:4.2f}: lnL {lnl:10.1f} lnpost {lp:10.1f} [{time.time()-t0:.0f} s]", flush=True)
            outp.write_text(json.dumps(out, indent=1))
    # figure
    fig, ax = plt.subplots(2, 4, figsize=(22, 9)); ax = ax.ravel()
    ref = max(max(x["lnpost"] for x in v) for v in out["slices"].values())
    for a, (k, v) in zip(ax[:6], out["slices"].items()):
        x = [s["value"] for s in v]
        a.plot(x, [s["lnpost"]-ref for s in v], "o-", label="lnL + lnprior"); a.plot(x, [s["lnL"]-ref for s in v], "s--", ms=4, label="lnL only")
        a.axvline(B[k], color="r", lw=0.8); a.set_xlabel(k); a.set_ylabel("Delta (vs best of scans)"); a.grid(alpha=0.3); a.set_ylim(-3000, 100)
    ax[0].legend(fontsize=8)
    an = sorted({g["angle"] for g in out["grid"]}); am = sorted({g["amp"] for g in out["grid"]})
    for a, key in zip(ax[6:], ("lnpost", "lnL")):
        Z = np.full((len(am), len(an)), np.nan)
        for g in out["grid"]:
            Z[am.index(g["amp"]), an.index(g["angle"])] = g[key]
        Z -= np.nanmax(Z)
        im = a.imshow(Z, origin="lower", aspect="auto", extent=(an[0]-1, an[-1]+1, am[0]-0.05, am[-1]+0.05), cmap="viridis", vmin=-2000)
        for g in out["grid"]:
            a.text(g["angle"], g["amp"], f"{g[key]-np.nanmax([h[key] for h in out['grid']]):.0f}", ha="center", va="center", fontsize=7, color="w")
        a.axhline(1.2, color="r", ls=":", lw=1); a.set_xlabel("bar angle [deg]"); a.set_ylabel("amplitude"); a.set_title(f"angle x amplitude: Delta {key}", fontsize=9)
        fig.colorbar(im, ax=a)
    fig.suptitle("Conditional slices and angle-amplitude grid around the free-angle best fit (others fixed at the best fit)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/"plots/scan_best_fit.png", dpi=70); print("plots/scan_best_fit.png")


if __name__ == "__main__":
    main()
