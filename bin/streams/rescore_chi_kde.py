#!/usr/bin/env python3
"""Re-score the saved grid-2 sprays (results/streams/spray_grid2/*.npz) with score_chi_kde (KDE per chi bin, latent chi) and
compare with the sky-conditional score. Output: results/plot_data/spray_bar_grid2_chikde.json
Usage: python bin/streams/rescore_chi_kde.py
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data
from ocen_dm.streams.score import score_chi_kde

d = load_data()
ROB = '--robust' in sys.argv
G = json.loads((ROOT/"results/plot_data/spray_bar_grid2.json").read_text())["grid"]
out = []
for x in G:
    tag = f"om{x['omega']:g}_an{x['angle']:g}_am{x['amp']:g}"; m = dict(np.load(ROOT/f"results/streams/spray_grid2/{tag}.npz"))
    t0 = time.time(); s = score_chi_kde(d, m, robust=ROB)
    out.append(dict(omega=x["omega"], angle=x["angle"], amp=x["amp"], chikde=s["total"], K=s["K"], n_bg=s["n_bg_dominated"],
                    share=s["share"], chi_centres=s["chi_centres"], sky_score=x["score"]["total"], overshoot=x["overshoot"]))
    print(f"{tag:22s} chiKDE lnL {s['total']:10.1f}  K {s['K']:3d}  bg-dominated {s['n_bg_dominated']:4d}   sky-score {x['score']['total']:9.1f}  [{time.time()-t0:.1f} s]", flush=True)
(ROOT/f"results/plot_data/spray_bar_grid2_chikde{'_robust' if ROB else ''}.json").write_text(json.dumps(out))
b1 = max(o["chikde"] for o in out); b2 = max(o["sky_score"] for o in out)
from scipy.stats import spearmanr
print("\nSpearman rank correlation chiKDE vs sky score:", round(spearmanr([o["chikde"] for o in out], [o["sky_score"] for o in out])[0], 3))
print("top 8 by chiKDE (dlnL, sky-score dlnL, overshoot):")
for o in sorted(out, key=lambda o: -o["chikde"])[:8]:
    print(f"  {o['omega']:5.1f} {o['angle']:4.0f} {o['amp']:3.1f}   {o['chikde']-b1:8.0f}   {o['sky_score']-b2:8.0f}   {o['overshoot']:.2f}")
