#!/usr/bin/env python3
"""D3 (docs/STREAM_DIAGNOSTIC_PLAN.md): is the fixed omega Cen orbit forcing the low bar angle? At angle 28 deg (Omega_b 35.5, amp 1.2,
d 5.6 kpc; best grid4 point at 28 deg) a 5x5 grid of present-day PMs at catalogue +- 1, 2 sigma (sigma = 0.027 mas/yr; Vasiliev &
Baumgardt 2021 incl. systematics, as fit_bar_ocen.py), v_los catalogue. 32000 release epochs, seed 1 (common random numbers with the
seed-1 4x reference sprays). Per-star lnL saved for segment analysis. Output results/plot_data/d3_free_pm.json/.npz,
sprays results/streams/d3_free_pm/.
Usage: OMP_NUM_THREADS=8 python bin/streams/d3_free_pm.py
"""
import json, time, itertools
import numpy as np
import grid4_common as G
from grid4_common import ROOT, score, OCEN_OBS

SIG = 0.027; OM, AN, AM, D = 35.5, 28., 1.2, 5.6; outd = ROOT/"results/streams/d3_free_pm"; outd.mkdir(parents=True, exist_ok=True)
res, per = [], {}
for i, j in itertools.product(range(-2, 3), range(-2, 3)):
    pa, pd = OCEN_OBS["pmra"]+i*SIG, OCEN_OBS["pmdec"]+j*SIG; t0 = time.time(); f = outd/f"pm_{i:+d}_{j:+d}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = G.make_spray(OM, AN, AM, D, nrel=32000, seed=1, pm=(pa, pd)); np.savez_compressed(f, **m)
    s = score(m); per[f"{i:+d}_{j:+d}"] = s["per_star"]
    res.append(dict(dpmra_sig=i, dpmdec_sig=j, pmra=pa, pmdec=pd, lnL=s["total"]))
    print(f"pmra {pa:.4f} ({i:+d}s) pmdec {pd:.4f} ({j:+d}s): lnL {s['total']:.1f} [{time.time()-t0:.0f} s]", flush=True)
(ROOT/"results/plot_data/d3_free_pm.json").write_text(json.dumps(dict(setup=[OM, AN, AM, D], sigma=SIG, grid=res), indent=1))
np.savez_compressed(ROOT/"results/plot_data/d3_free_pm_perstar.npz", **per)
