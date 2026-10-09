#!/usr/bin/env python3
"""Grid-3 maps: robust chi-KDE Delta lnL over (Omega_b, angle) for each amplitude, plus the overshoot fraction.
Usage: python bin/streams/plot_grid3_maps.py -> plots/grid3_maps.png (prints the top 10)
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g = json.loads((ROOT/"results/plot_data/spray_bar_grid3.json").read_text())["grid"]
om = sorted({x["omega"] for x in g}); an = sorted({x["angle"] for x in g}); am = sorted({x["amp"] for x in g}); best = max(x["chikde"] for x in g)
fig, ax = plt.subplots(2, len(am), figsize=(5*len(am), 8.5), squeeze=False)
for j, A in enumerate(am):
    for i, key in enumerate(("chikde", "overshoot")):
        Z = np.full((len(an), len(om)), np.nan)
        for x in g:
            if x["amp"] == A:
                Z[an.index(x["angle"]), om.index(x["omega"])] = x[key]-best if key == "chikde" else x[key]
        im = ax[i, j].imshow(Z, origin="lower", aspect="auto", cmap="viridis" if key == "chikde" else "magma_r",
                             extent=(om[0]-0.75, om[-1]+0.75, an[0]-2, an[-1]+2), vmin=-4000 if key == "chikde" else None)
        for x in g:
            if x["amp"] == A:
                v = x[key]-best if key == "chikde" else x[key]
                ax[i, j].text(x["omega"], x["angle"], f"{v:.0f}" if key == "chikde" else f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax[i, j].set_title(f"amp {A:g}: " + ("chi-KDE Delta lnL" if key == "chikde" else "overshoot fraction"), fontsize=9)
        ax[i, j].set_xlabel("Omega_b [km/s/kpc]"); ax[i, j].set_ylabel("bar angle [deg]"); fig.colorbar(im, ax=ax[i, j])
fig.suptitle("Grid 3 (model A, 1.96 Gyr): robust chi-KDE likelihood and overshoot", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT/"plots/grid3_maps.png", dpi=70)
for x in sorted(g, key=lambda x: -x["chikde"])[:10]:
    print(f"  Om {x['omega']:5.1f} ang {x['angle']:4.0f} amp {x['amp']:3.1f}: {x['chikde']-best:7.0f}  overshoot {x['overshoot']:.2f}")
