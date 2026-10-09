#!/usr/bin/env python3
"""Exaggerated cluster rotation in the spray: v_rot(r) of the fitted spin model (results/streams/spin/A_nodm_vrot.json) scaled by
f = 0, 1, 3, 10, -10 (-10 = reversed sense), for two setups: best grid4 point (34.5, 16, 1.4, 5.6) and best at the literature
angle (35.5, 28, 1.2, 5.6). 32000 release epochs, seed 1 (common random numbers). lnL (score_conditional) and a figure of the
trailing-arm debris (age < 700 Myr) in the stream footprint (|dphi2| < 6 deg): medians per 2-deg phi1 bin vs members.
Outputs: results/streams/spin_exag/*.npz, results/plot_data/spin_exaggerate.json, plots/spin_exaggerate.png.
Usage: OMP_NUM_THREADS=8 python bin/streams/spin_exaggerate.py
"""
import json, time
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, make_spray, score, SPIN, GC, du, dx, d, project_gc

SETUPS = [(34.5, 16., 1.4, 5.6), (35.5, 28., 1.2, 5.6)]; F = [0., 1., 3., 10., -10.]
outd = ROOT/"results/streams/spin_exag"; outd.mkdir(parents=True, exist_ok=True); res = []; S = {}
for st in SETUPS:
    tag = "om{:g}_an{:g}_am{:g}_d{:g}".format(*st)
    for f_ in F:
        t0 = time.time(); fn = outd/f"{tag}_f{f_:g}.npz"
        if fn.exists():
            m = dict(np.load(fn))
        else:
            sp = None if f_ == 0 else dict(SPIN, vrot_kms=SPIN["vrot_kms"]*f_)
            m = make_spray(*st, nrel=32000, seed=1, spin=sp); np.savez_compressed(fn, **m)
        L = score(m)["total"]; S[(tag, f_)] = m; res.append(dict(setup=tag, factor=f_, lnL=L))
        print(f"{tag} x{f_:g}: lnL {L:.1f} [{time.time()-t0:.0f} s]", flush=True)
(ROOT/"results/plot_data/spin_exaggerate.json").write_text(json.dumps(res, indent=1))
for r in res:
    r0 = next(x for x in res if x["setup"] == r["setup"] and x["factor"] == 0.)
    print(f"{r['setup']} x{r['factor']:g}: dlnL vs no rotation {r['lnL']-r0['lnL']:+.1f}")
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1]); Q = ["x", "pmra", "pmdec", "vlos"]
D = dict(x=dx, pmra=d["pmra"], pmdec=d["pmdec"], vlos=d["v"]); dsel = np.abs(dx) < 6; dib = np.where(dsel, np.digitize(du, edges)-1, -1)
fig, ax = plt.subplots(2, 5, figsize=(27, 9)); cols = dict(zip(F, ("k", "C0", "C2", "C3", "C1")))
for row, st in enumerate(SETUPS):
    tag = "om{:g}_an{:g}_am{:g}_d{:g}".format(*st)
    for f_ in F:
        m = S[(tag, f_)]; u, x = project_gc(m["l"], m["b"], GC); k = (np.abs(x) < 6) & (m["age"] < 700) & (m["chi"] > 0)
        M = dict(x=x[k], pmra=m["pmra"][k], pmdec=m["pmdec"][k], vlos=m["vlos"][k]); ib = np.digitize(u[k], edges)-1
        n = np.bincount(ib[(ib >= 0) & (ib < len(cen))], minlength=len(cen)); ax[row, 0].plot(cen, n, color=cols[f_], label=f"v_rot x {f_:g}")
        for a, q in zip(ax[row, 1:], Q):
            a.plot(cen, [np.median(M[q][ib == j]) if (ib == j).sum() >= 10 else np.nan for j in range(len(cen))], "-", color=cols[f_], lw=1.3)
    for a, q in zip(ax[row, 1:], Q):
        v = D[q]; a.plot(cen, [np.nanmedian(v[dib == j]) if np.isfinite(v[dib == j]).sum() >= (3 if q == "vlos" else 10) else np.nan
                               for j in range(len(cen))], "o", color="0.3", mfc="w", ms=5, label="members (median)")
        a.set_ylabel(q); a.set_xlabel(r"$\phi_1$ [deg]")
    ax[row, 0].set_ylabel(f"{tag}\nparticles per 2-deg bin"); ax[row, 0].set_xlabel(r"$\phi_1$ [deg]"); ax[row, 0].legend(fontsize=8)
ax[0, 1].legend()
fig.suptitle("Exaggerated cluster rotation in the spray (trailing arm, age < 700 Myr, |dphi2| < 6 deg): medians per 2-deg phi1 bin; members' medians open circles")
fig.tight_layout(); fig.savefig(ROOT/"plots/spin_exaggerate.png", dpi=70); print("plots/spin_exaggerate.png")
