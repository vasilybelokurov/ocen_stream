#!/usr/bin/env python3
"""Compare the two sampler pilots (2026-10-27; docs/reviews/sampler_fable_codex_20261026.md) on the agreed metrics.

Reads the run logs pulled from the private runs repository ($OCEN_RUNS_DIR, a clone of ocen_stream_runs): Fable's SMC
(pilots_20261027) and Codex's scan (pilots_20261027_scan). For each pilot:
  1. local minima (pilots.diagnose; bank 7) with their parameters; a minimum's best nearby points rescored with 8 independent
     score banks (mean, sd); minima of both pilots are grouped into basins when they lie within 0.15 of each other in the unit box
  2. best chi2 (bank 7) and its rescored mean
  3. separation: chi2 barrier along the straight path between each pair of minima (max along the path - higher endpoint)
  4. A100 seconds and models (from the per-model log, across restarts)
  5. spread of the log|C| term among models with chi2 < 200
Plot: plots/pilot_compare_20261027.png (both pilots' models and minima in angle-Omega_b, angle-q, distance-v_c; chi2 vs angle).
Usage: OCEN_RUNS_DIR=... PYTHONPATH=src python bin/streams/pilot_compare.py
"""
import glob, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocen_stream import pilots as PL, jaxspray as J

ROOT = Path(__file__).resolve().parents[2]; RUNS = Path(os.environ["OCEN_RUNS_DIR"])
PILOTS = dict(smc=("pilots_20261027", "smc"), scan=("pilots_20261027_scan", "scan"))


def load(d, prefix):
    z = [np.load(f) for f in sorted(glob.glob(str(RUNS/d/"log"/"0*.npz")))]
    cat = lambda k: np.concatenate([q[k] for q in z])
    t = cat("tag"); own = np.array([str(x).startswith(prefix) for x in t])
    return dict(u=cat("u")[own], c=cat("chi2")[own], l=cat("chi2ld")[own], s=cat("s")[own], bank=cat("bank")[own],
                diag=np.array(["_diag" in str(x) for x in t[own]]), D=np.load(RUNS/d/f"{prefix}_diag.npz"))


def main():
    ht = J.load_host(ROOT/"results/gpu/host_tables.npz"); res, mins = {}, []
    for name, (d, prefix) in PILOTS.items():
        L = load(d, prefix); D = L["D"]; nm = len(D["cmin"]); R = D["rescore"]; near, cn = D["unear"], D["cnear"]
        k7 = (L["bank"] == 7) & ~L["diag"]; g = k7 & np.isfinite(L["c"]) & (L["c"] < 200)
        B = np.full((nm, nm), np.nan)
        for i in range(nm):
            for j in range(i+1, nm):
                B[i, j] = B[j, i] = np.max(D[f"p{i}_{j}"])-max(D["cmin"][i], D["cmin"][j])
        rows = []
        for i, (u, c) in enumerate(zip(D["umin"], D["cmin"])):
            k = np.max(np.abs(near-u), 1) < 1e-12; r = R[:, k][:, 0] if k.any() else np.full(len(R), np.nan)
            p = PL.to_params(u, ht); rows.append(dict(chi2=float(c), rescored_mean=float(np.mean(r)), rescored_sd=float(np.std(r)), **{q: round(p[q], 3) for q in PL.NAMES}))
            mins.append((name, i, u, c, p))
        best = int(np.argmin(D["cmin"]))
        res[name] = dict(models=int(len(L["c"])), a100_hours=float(L["s"].sum()/3600), best_chi2=float(D["cmin"][best]),
                         best_rescored=f"{rows[best]['rescored_mean']:.1f} +- {rows[best]['rescored_sd']:.1f}", minima=rows,
                         barriers=np.round(B, 1).tolist(), logdet_spread_chi2_lt_200=dict(n=int(g.sum()), median=float(np.median((L["l"]-L["c"])[g])),
                         range=float(np.ptp((L["l"]-L["c"])[g]))), L=L)
    # basins: minima of both pilots within 0.15 (unit box, max norm) are linked
    lab = list(range(len(mins)))
    for a in range(len(mins)):
        for b in range(a+1, len(mins)):
            if np.max(np.abs(mins[a][2]-mins[b][2])) < 0.15:
                la, lb = lab[a], lab[b]; lab = [la if x == lb else x for x in lab]
    basins = {}
    for (name, i, u, c, p), l in zip(mins, lab):
        basins.setdefault(l, []).append((name, i, float(c), round(p["angle"], 1), round(p["omega"], 1), round(p["halo_q"], 2)))
    print(json.dumps({k: {q: v for q, v in r.items() if q != "L"} for k, r in res.items()}, indent=1, default=str))
    print("basins (pilot, minimum, chi2, angle, Omega, q):")
    for l, m in sorted(basins.items(), key=lambda kv: min(x[2] for x in kv[1])):
        print(" ", sorted({x[0] for x in m}), m)
    # plot
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8)); col = dict(smc="tab:blue", scan="tab:orange")
    nam = dict(angle="bar angle [deg]", omega="Omega_b [km/s/kpc]", halo_q="halo q", dist="distance [kpc]", vc="v_c(R0) [km/s]")
    for name, r in res.items():
        L = r["L"]; k = (L["bank"] == 7) & np.isfinite(L["c"]) & (L["c"] < 250); P = {q: np.array([PL.to_params(u, ht)[q] for u in L["u"][k]]) for q in ("angle", "omega", "halo_q", "dist", "vc")}
        Pm = [m[4] for m in mins if m[0] == name]; cm = [m[3] for m in mins if m[0] == name]
        ax[0].scatter(P["angle"], L["c"][k], s=2, c=col[name], alpha=0.3); ax[0].scatter([p["angle"] for p in Pm], cm, s=90, c=col[name], marker="*", edgecolor="k", label=f"{name} ({r['models']} models, {r['a100_hours']:.1f} A100-h)")
        for a, (x, y) in zip(ax[1:], (("angle", "omega"), ("angle", "halo_q"), ("dist", "vc"))):
            a.scatter(P[x], P[y], s=2, c=col[name], alpha=0.3); a.scatter([p[x] for p in Pm], [p[y] for p in Pm], s=90, c=col[name], marker="*", edgecolor="k")
            a.set_xlabel(nam[x]); a.set_ylabel(nam[y])
    ax[0].set_xlabel(nam["angle"]); ax[0].set_ylabel("chi2 (25 points, bank 7)"); ax[0].set_ylim(125, 250); ax[0].legend(fontsize=8)
    fig.suptitle("Sampler pilots: models with chi2 < 250 (dots) and local minima (stars); Fable's SMC (blue) vs Codex's scan (orange)")
    fig.savefig(ROOT/"plots/pilot_compare_20261027.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
