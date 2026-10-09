#!/usr/bin/env python3
"""2D conditional likelihood maps with the bar pattern speed: Omega_b (33.75-36.25 by 0.5) x {bar angle 20-30 by 2; distance
5.40-5.65 by 0.05 kpc; pmdec -6.80..-6.70 by 0.02 mas/yr}, other parameters at the free-angle best fit; same score and
spray set-up as scan_best_fit.py (robust chi-KDE lnL + log-priors).
Usage: python bin/streams/scan_omega_pairs.py -> results/plot_data/scan_omega_pairs.json, plots/scan_omega_pairs.png
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scan_best_fit as S      # loads data, best fit, evaluate()

OM = np.arange(33.75, 36.26, 0.5)
PAIRS = dict(angle=np.arange(20., 30.1, 2.), d=np.round(np.arange(5.40, 5.651, 0.05), 3), pmdec=np.round(np.arange(-6.80, -6.699, 0.02), 3))
out = dict(best=S.B, best_lnpost=S.best["lnpost"], maps={k: [] for k in PAIRS}); outp = ROOT/"results/plot_data/scan_omega_pairs.json"
for k, vals in PAIRS.items():
    for om in OM:
        for v in vals:
            t0 = time.time(); lnl, lp = S.evaluate(dict(S.B, omega=float(om), **{k: float(v)}))
            out["maps"][k].append(dict(omega=float(om), value=float(v), lnL=lnl, lnpost=lp))
            print(f"{k}: Omega {om:5.2f} {k} {v:7.3f}: lnpost {lp:10.1f} [{time.time()-t0:.0f} s]", flush=True)
            outp.write_text(json.dumps(out, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
lab = dict(angle="bar angle [deg]", d="distance [kpc]", pmdec="pmdec [mas/yr]")
for a, (k, rows) in zip(ax, out["maps"].items()):
    vals = sorted({r["value"] for r in rows}); Z = np.full((len(vals), len(OM)), np.nan)
    for r in rows:
        Z[vals.index(r["value"]), list(OM).index(r["omega"])] = r["lnpost"]
    Z -= np.nanmax(Z); dv = vals[1]-vals[0]
    im = a.imshow(Z, origin="lower", aspect="auto", extent=(OM[0]-0.25, OM[-1]+0.25, vals[0]-dv/2, vals[-1]+dv/2), cmap="viridis", vmin=-3000)
    for r in rows:
        a.text(r["omega"], r["value"], f"{r['lnpost']-np.nanmax([q['lnpost'] for q in rows]):.0f}", ha="center", va="center", fontsize=7, color="w")
    a.plot(S.B["omega"], S.B[k], "r*", ms=12); a.set_xlabel("Omega_b [km/s/kpc]"); a.set_ylabel(lab[k]); fig.colorbar(im, ax=a, label="Delta lnpost")
fig.suptitle("Conditional 2D maps with the bar pattern speed (others at the free-angle best fit, red star)", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/scan_omega_pairs.png", dpi=75); print("plots/scan_omega_pairs.png")
