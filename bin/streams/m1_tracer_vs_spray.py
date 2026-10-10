#!/usr/bin/env python3
"""M1 (docs/STREAM_DIAGNOSTIC_PLAN.md): release prescription. Tracer run (run_prescribed.py, 200k A_nodm tracers, frozen moving
satellite potential) vs the Fardal spray at the same host and omega Cen state: angle 28 deg, Omega_b 35.5, amp 1.2, d 5.6, catalogue
PMs (tracer run results/streams/prescribed/m1_28; spray results/streams/d3_free_pm/pm_+0_+0.npz, 32000 epochs, age < 700 Myr).
Debris in the stream footprint (|dphi2| < 6 deg): medians of pmra, pmdec, v_los and counts per 2-deg phi1 bin (bootstrap errors for
the tracers), and members' medians. Also the 20-deg pair (tracer rotB_nospin vs grid4 spray om34.5_an20_am1.2_d5.6, grid PMs) if present.
Output plots/m1_tracer_vs_spray.png. Usage: python bin/streams/m1_tracer_vs_spray.py
"""
import json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, project_gc, GC, du, dx, d, w
from ocen_dm.streams.frames import observables

edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1]); Q = ("pmra", "pmdec", "vlos")


def tracer(run):
    s = np.load(ROOT/f"results/streams/prescribed/{run}/snap_today.npz"); meta = json.loads((ROOT/f"results/streams/prescribed/{run}/run.json").read_text())
    xv = np.hstack((s["pos"]/1e3, s["vel"])).astype(float); c = np.array(meta["ocen_today"])
    o = observables(xv[np.linalg.norm(xv[:, :3]-c[:3], axis=1) > 0.3]); m = dict(l=w(o["l"]), b=o["b"], pmra=o["pmra"], pmdec=o["pmdec"], vlos=o["vlos"])
    m["u"], m["x"] = project_gc(m["l"], m["b"], GC); return {k: v[np.abs(m["x"]) < 6] for k, v in m.items()}


def spray(path):
    m = dict(np.load(ROOT/"results/streams"/path)); m["u"], m["x"] = project_gc(m["l"], m["b"], GC)
    k = (np.abs(m["x"]) < 6) & (m["age"] < 700) & (m["chi"] > 0); return {q: m[q][k] for q in ("u", "x", "pmra", "pmdec", "vlos")}


def meds(m, boot=False):
    ib = np.digitize(m["u"], edges)-1; out = {}
    for q in Q:
        md, er = [], []
        for j in range(len(cen)):
            v = m[q][ib == j]
            if len(v) < 8:
                md.append(np.nan); er.append(np.nan); continue
            md.append(np.median(v)); er.append(np.std([np.median(np.random.default_rng(j).choice(v, len(v))) for _ in range(200)]) if boot else 0.)
        out[q] = (np.array(md), np.array(er))
    out["n"] = np.bincount(ib[(ib >= 0) & (ib < len(cen))], minlength=len(cen)); return out


pairs = [("28 deg, catalogue PMs", "m1_28", "d3_free_pm/pm_+0_+0.npz"), ("20 deg, grid PMs", "rotB_nospin", "spray_grid4/om34.5_an20_am1.2_d5.6.npz")]
dib = np.where(np.abs(dx) < 6, np.digitize(du, edges)-1, -1)
fig, ax = plt.subplots(len(pairs), 4, figsize=(24, 5*len(pairs)))
for r, (lab, trun, spath) in enumerate(pairs):
    T, S = meds(tracer(trun), boot=True), meds(spray(spath))
    for c, q in enumerate(Q):
        a = ax[r, c]; a.errorbar(cen-0.15, T[q][0], yerr=T[q][1], fmt="o-", color="C0", ms=3, label="tracer run (200k)")
        a.plot(cen+0.15, S[q][0], "s-", color="C3", ms=3, label="spray (Fardal, 4x / 1x)")
        v = d["v"] if q == "vlos" else d[q]
        a.plot(cen, [np.nanmedian(v[dib == j]) if np.isfinite(v[dib == j]).sum() >= (3 if q == "vlos" else 10) else np.nan for j in range(len(cen))],
               "ko", mfc="w", ms=6, label="members")
        a.set_xlabel(r"$\phi_1$ [deg]"); a.set_ylabel(q); a.set_title(lab, fontsize=9)
    ax[r, 3].plot(cen, T["n"]/T["n"].sum(), "o-", color="C0"); ax[r, 3].plot(cen, S["n"]/S["n"].sum(), "s-", color="C3")
    dn = np.histogram(du[np.abs(dx) < 6], edges)[0]; ax[r, 3].plot(cen, dn/dn.sum(), "ko", mfc="w")
    ax[r, 3].set_ylabel("fraction per 2-deg bin (footprint)"); ax[r, 3].set_xlabel(r"$\phi_1$ [deg]")
    print(lab, "tracer footprint debris:", int(T["n"].sum()))
    for j in range(len(cen)):
        if cen[j] > 14 and np.isfinite(T["pmdec"][0][j]):
            print(f"  phi1 {cen[j]:4.0f}: pmdec tracer {T['pmdec'][0][j]:6.2f}+-{T['pmdec'][1][j]:.2f} spray {S['pmdec'][0][j]:6.2f} | pmra tracer {T['pmra'][0][j]:6.2f}+-{T['pmra'][1][j]:.2f} spray {S['pmra'][0][j]:6.2f}")
ax[0, 0].legend(fontsize=8); fig.suptitle("M1: tracer run vs Fardal spray at the same host and omega Cen state (stream footprint |dphi2| < 6 deg)")
fig.tight_layout(); fig.savefig(ROOT/"plots/m1_tracer_vs_spray.png", dpi=70); print("plots/m1_tracer_vs_spray.png")
