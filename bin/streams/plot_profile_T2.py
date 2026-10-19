#!/usr/bin/env python3
"""Mock T2c0 bar-angle profile from the direct per-angle search (2026-10-19).
(a) profile: 256k final values (4 seeds, +-1 se) at each angle; best search value (128k) per chain; starting values from the
waves; emulator profile after wave 3 for comparison; truth angle and truth score.
(b) parameters of each angle's best point relative to the truth (Omega, amplitude, size, distance, v_c).
Output: plots/profile_T2c0.png
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/plot_profile_T2.py
"""
import sys, json, re, hashlib, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = ROOT/"results/streams/profile_T2c0_20261019"; TRUTH, TANG = -22174., 27.
T = dict(omega=35., amp=1.0, size=1.0, dist=5.43, vc=229.)
prof = json.loads((O/"profile.json").read_text()); ang = sorted(float(k) for k in prof)
m = np.array([prof[f"{a:g}"]["mean"] for a in ang]); se = np.array([prof[f"{a:g}"]["se"] for a in ang])
chains = {}
for f in (O/"runs").glob("a*_e*.json"):
    g = re.match(r"a([\d.]+)_(\d)_e", f.stem); k = (float(g.group(1)), int(g.group(2))); r = json.loads(f.read_text())
    chains[k] = max(chains.get(k, -np.inf), r["total_jk"])
starts = {12.5: -23606, 17.5: -23770, 22.5: -22962, 27: -23151, 27.5: -23151, 32.5: -23569}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
try:
    import search_wave as SW
    from scipy.optimize import minimize
    CAT = ROOT/"results/streams/mocks_20261018/T2/mock_wide_c0.npz"; csha = hashlib.sha256(CAT.read_bytes()).hexdigest()
    W = [ROOT/"results/streams/search_T2c0_wave1_20261018", ROOT/"results/streams/search_T2c0_wave2_20261019", ROOT/"results/streams/search_T2c0_wave3_20261019"]
    g, _, em = SW.fit(SW.observations(ROOT/"results/streams/rescore_mockwide_T2c0_20261018", W, csha))
    a_s = np.arange(10.5, 35, 1.25); pe = []
    for a in a_s:
        ua = (a-10)/25; bv = -np.inf
        for x0 in g["X"][np.argsort(-g["ybar"])[:8]]:
            r = minimize(lambda z: -em.predict(np.insert(z, 0, ua)[None], return_std=False)[0], x0[1:], method="L-BFGS-B", bounds=[(0, 1)]*8)
            bv = max(bv, -r.fun)
        pe.append(bv)
    ax[0].plot(a_s, pe, "-", color="C1", alpha=0.6, label="emulator after wave 3 (not trusted)")
except Exception as e:
    print("emulator curve skipped:", e)
ax[0].plot(sorted(starts), [starts[k] for k in sorted(starts)], "s:", color="0.6", label="start: best from the waves (256k)")
for (a, j), v in chains.items():
    ax[0].plot(a, v, "x", color="C2", ms=6, label="search best per chain (128k)" if (a, j) == min(chains) else None)
ax[0].errorbar(ang, m, se, fmt="o-", color="C0", ms=6, capsize=3, lw=1.8, label="profile: 4 independent 256k seeds")
ax[0].axhline(TRUTH, color="k", ls="--", label="truth score"); ax[0].axvline(TANG, color="C3", lw=1, label="truth angle")
ax[0].set(xlabel="bar angle [deg]", ylabel="best ln L at fixed angle", title="(a) mock T2: bar-angle profile (direct search)", ylim=(-24300, -21700))
ax[0].legend(fontsize=7, loc="lower center")
for k, mk in (("omega", "o"), ("amp", "s"), ("size", "D"), ("dist", "^"), ("vc", "v")):
    v = [prof[f"{a:g}"]["params"][k]/T[k] for a in ang]; ax[1].plot(ang, v, mk+"-", label=f"{k} / truth")
ax[1].axhline(1, color="k", lw=0.6); ax[1].axvline(TANG, color="C3", lw=1)
ax[1].set(xlabel="bar angle [deg]", ylabel="best-point parameter / truth", title="(b) best-fit parameters at each angle"); ax[1].legend(fontsize=7)
fig.savefig(ROOT/"plots/profile_T2c0.png", dpi=120); print(ROOT/"plots/profile_T2c0.png")
