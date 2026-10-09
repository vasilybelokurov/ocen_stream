#!/usr/bin/env python3
"""Spray-noise test for score_conditional: the two best grid4 models (34.5,16,1.4,5.6) and (34.5,20,1.2,5.6), seeds 1-5 at 8000
release epochs (seed 1 = the grid4 spray) and seeds 1-3 at 32000 epochs (4x particles). Sprays in results/streams/seed_test/;
lnL and per-star lnL in results/plot_data/seed_test.json/.npz. Prints the scatter of lnL and of Delta lnL (16 deg - 20 deg).
Usage: OMP_NUM_THREADS=8 python bin/streams/seed_test.py
"""
import json, time
import numpy as np
from grid4_common import ROOT, make_spray, score

MODELS = [(34.5, 16., 1.4, 5.6), (34.5, 20., 1.2, 5.6)]; RUNS = [(8000, s) for s in (1, 2, 3, 4, 5)] + [(32000, s) for s in (1, 2, 3)]
outd = ROOT/"results/streams/seed_test"; outd.mkdir(parents=True, exist_ok=True); res = []; per = {}
for nrel, seed in RUNS:
    for om, an, am, dist in MODELS:
        t0 = time.time(); tag = f"om{om:g}_an{an:g}_am{am:g}_d{dist:g}"
        f = (ROOT/f"results/streams/spray_grid4/{tag}.npz") if (nrel, seed) == (8000, 1) else outd/f"{tag}_n{nrel}_s{seed}.npz"
        if f.exists():
            m = dict(np.load(f))
        else:
            m = make_spray(om, an, am, dist, nrel=nrel, seed=seed); np.savez_compressed(f, **m)
        s = score(m); per[f"{tag}_n{nrel}_s{seed}"] = s["per_star"]
        res.append(dict(tag=tag, nrel=nrel, seed=seed, lnL=s["total"], K=s["K"]))
        print(f"{tag} nrel {nrel} seed {seed}: lnL {s['total']:.1f} [{time.time()-t0:.0f} s]", flush=True)
(ROOT/"results/plot_data/seed_test.json").write_text(json.dumps(res, indent=1)); np.savez_compressed(ROOT/"results/plot_data/seed_test_perstar.npz", **per)
for nrel in (8000, 32000):
    a = {(r["tag"], r["seed"]): r["lnL"] for r in res if r["nrel"] == nrel}; seeds = sorted({k[1] for k in a})
    t16, t20 = [f"om{m[0]:g}_an{m[1]:g}_am{m[2]:g}_d{m[3]:g}" for m in MODELS]
    L16 = np.array([a[(t16, s)] for s in seeds]); L20 = np.array([a[(t20, s)] for s in seeds]); dL = L16-L20
    print(f"nrel {nrel}: lnL(16) mean {L16.mean():.1f} sd {L16.std(ddof=1):.1f}; lnL(20) mean {L20.mean():.1f} sd {L20.std(ddof=1):.1f}; "
          f"dlnL(16-20) {np.round(dL, 1)} mean {dL.mean():.1f} sd {dL.std(ddof=1):.1f}")
