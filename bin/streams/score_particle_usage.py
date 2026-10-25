#!/usr/bin/env python3
"""Which spray particles does the knot score actually use, and how many particles do the summaries need? (2026-10-25)

(a) Usage: for a stored spray, the fraction of particles that enter the sky fits (chi > 0, age < 700, both PM masks, phi1 in the
    local windows 0-24 deg, |v| < 6 deg) and the PM knot fits (sky window, other-PM mask, |v_pm| < W, within 1.5 deg of a knot), as in
    stream_model_compare (Obs.particles / sky / pm).
(b) Particle number: random subsets of N particles of the 4 pooled final_c0 sprays (694k): summary values (b = 0, no bootstrap)
    compared with the full pool, mean and sd of (m_N - m_pool) / sigma_data per summary, 4 subsets per N (fixed bank).
Usage: PYTHONPATH=src python bin/streams/score_particle_usage.py -> results/plot_data/score_particle_usage.json
"""
import glob, json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
sys.argv = [sys.argv[0]]
import stream_model_compare as MC
from ocen_stream.jaxscore import make_bank

D = MC.data_vector(); O = MC.Obs(); sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]
vec = lambda M: np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]; res = {}
# (a) usage
f = ROOT/"results/streams/fitknot_20261025/runs/final_c0_2.npz"; z = np.load(f); n = len(z["chi"])
P = O.particles(str(f), None, make_bank(n, MC.NBOOT, 7)); sel = P["k"]
sky = P["m_pmra"] & P["m_pmdec"] & (P["u"] >= 0) & (P["u"] < 24) & (np.abs(P["v"]) < 6)
W = {q: D[q+"_W"] for q in ("pmra", "pmdec")}; pm = np.zeros(P["u"].size, bool)
for q, oth in (("pmra", "pmdec"), ("pmdec", "pmra")):
    vq = P[q]-O.ev(q, P["u"]); base = P["inwin"] & P["m_"+oth] & (np.abs(vq) < W[q])
    pm |= base & (np.min(np.abs(P["u"][:, None]-D[q+"_k"][None, :]), 1) < 1.5)
res["usage"] = dict(total=n, selected_chi_age=int(sel.sum()), sky=int(sky.sum()), pm=int(pm.sum()), any=int((sky | pm).sum()))
print("usage of", f.name, res["usage"], f"-> {(sky | pm).sum()/n*100:.1f}% of all particles enter the score", flush=True)
# (b) summaries vs N
K = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age"); Z = [np.load(x) for x in sorted(glob.glob(str(ROOT/"results/streams/fitknot_20261025/runs/final_c0_?.npz")))]
pool = {k: np.concatenate([q[k] for q in Z]) for k in K}; N = len(pool["l"]); rng = np.random.default_rng(1)
with tempfile.TemporaryDirectory() as td:
    def summ(idx, seed):
        g = Path(td)/"s.npz"; np.savez(g, age_cut_myr=700., **{k: pool[k][idx] for k in K}); return vec(MC.measure(O, D, str(g), None, bank=make_bank(len(idx), MC.NBOOT, seed)))
    ref = summ(np.arange(N), 7)
    for m in (22400, 44800, 89600, 179200):
        d = np.array([(summ(np.sort(rng.choice(N, m, replace=False)), 7+r)-ref)/sd for r in range(4)])
        res[str(m)] = dict(mean_abs_bias=float(np.nanmean(np.abs(np.nanmean(d, 0)))), max_abs_bias=float(np.nanmax(np.abs(np.nanmean(d, 0)))),
                           rms_scatter=float(np.sqrt(np.nanmean(np.nanvar(d, 0)))), max_abs=float(np.nanmax(np.abs(d))))
        print(f"N = {m} (pool {N}): summaries vs pool in sigma_data: mean |bias| {res[str(m)]['mean_abs_bias']:.3f}, max |bias| "
              f"{res[str(m)]['max_abs_bias']:.3f}, rms scatter {res[str(m)]['rms_scatter']:.3f}, max |dev| {res[str(m)]['max_abs']:.3f}", flush=True)
(ROOT/"results/plot_data/score_particle_usage.json").write_text(json.dumps(res, indent=1))
