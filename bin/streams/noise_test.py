#!/usr/bin/env python3
"""Monte Carlo noise of the robust chi-KDE likelihood: re-run sprays with different random seeds (release offsets) and with
2x particles, for given bar models; report lnL per run, scatter and the Delta lnL between models per seed.
Usage: python bin/streams/noise_test.py --models 34.5,16,1.4 34.5,24,1.2 --seeds 2 3 4 --double
       -> results/plot_data/noise_test.json
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"bin/streams")); sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from spray_bar_grid2 import load_data, w
from ocen_dm.streams.restricted import AGAMA_T_MYR, OCEN_TODAY, host_potential
from ocen_dm.streams.frames import observables
from ocen_dm.streams.spray import spray_unwrapped
from ocen_dm.streams.score import score_chi_kde

ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
ap.add_argument("--double", action="store_true"); a = ap.parse_args()
d = load_data()
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text()); r_kpc, M = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
T = 1955.58/AGAMA_T_MYR
runs = [(s, 8000) for s in a.seeds] + ([(2, 16000)] if a.double else [])
out = {}
for mstr in a.models:
    om, an, am = map(float, mstr.split(",")); host = host_potential("x", bar_omega=om, bar_angle_deg=an, t_today=T, bar_amp=am); out[mstr] = []
    for seed, nrel in runs:
        t0 = time.time()
        sp = spray_unwrapped(host, OCEN_TODAY.copy(), T, r_kpc, M, np.linspace(T-1000/AGAMA_T_MYR, T, nrel), seed=seed)
        tr = sp["arm"] == 1; o = observables(sp["xv"][tr])
        ck = score_chi_kde(d, dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"], chi=sp["chi"][tr],
                                   age=sp["t_release_myr_ago"][tr]), robust=True)
        out[mstr].append(dict(seed=seed, nrel=nrel, lnL=ck["total"]))
        print(f"{mstr}: seed {seed}, nrel {nrel}: lnL {ck['total']:.1f}  [{time.time()-t0:.0f} s]", flush=True)
        (ROOT/"results/plot_data/noise_test.json").write_text(json.dumps(out, indent=1))
for mstr, rs in out.items():
    v = np.array([r["lnL"] for r in rs if r["nrel"] == 8000])
    print(f"{mstr}: 8000-release seeds: mean {v.mean():.1f}, std {v.std(ddof=1):.1f}; 16000: {[r['lnL'] for r in rs if r['nrel'] == 16000]}")
if len(out) == 2:
    (m1, r1), (m2, r2) = out.items()
    print("Delta lnL per seed:", [round(x["lnL"]-y["lnL"], 1) for x, y in zip(r1, r2)])
