#!/usr/bin/env python3
"""Phase 2 of docs/STREAM_DIAGNOSTIC_PLAN.md: one host/release ingredient at a time at the literature bar angle.
Baseline: Hunter+2024 bar, angle 28 deg, Omega_b 35.5 (today), amplitude 1.2, d 5.6 kpc, omega Cen catalogue PMs (D3 centre point),
release uniform over the last 1000 Myr, 32000 epochs, seed 1, scored with age < 700 Myr.
Variants: M2 slowing bar eta = 0.002, 0.004, 0.008 (Omega_b and angle fixed today; bar size ~ 1/Omega);
M3a Portail+2017 bar (amp 1.0, 1.2); M3b Hunter bar size x1.15, x1.3; M4 release window 1500 Myr (48000 epochs) scored with
age < 1000 and < 1500 Myr; P1 (Phase 3 item 1) axisymmetric MW variants at fixed v_c(R0): halo flattening q 0.8, 0.9, 1.2;
stellar discs x0.8, x1.2; and two combinations; M4b release concentrated at pericentres (Gaussian sigma 10, 20 Myr; pericentres ~88 Myr apart),
also for the 16-deg reference. Reference: best 16-deg model (34.5, 16, 1.4, 5.6; grid PMs, 4x, seed 1).
Per variant: lnL (score_conditional) total and per phi1 segment (< 4, 4-17, > 17) relative to the baseline; trailing debris
(age cut, chi > 0) in the stream footprint (|dphi2| < 6): median pmra, pmdec per 2-deg phi1 bin vs the members' medians, and the
median d/d_ocen per CMD-distance b bin (dist_ratio; results/plot_data/stream54_cmd_distance.json).
Outputs results/plot_data/phase2_variants.json, plots/phase2_variants.png, sprays results/streams/phase2/.
Usage: OMP_NUM_THREADS=8 python bin/streams/phase2_variants.py [names...]
"""
import json, sys, time
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, dx, d, OCEN_OBS

CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); BASE = dict(setup=(35.5, 28., 1.2, 5.6), pm=CAT, host_kw=None, window=1000., nrel=32000, age=700.,
                                             release="uniform", psig=10.)
V = {"baseline": {},
     "M2 eta 0.002": dict(host_kw=dict(bar_eta=0.002)), "M2 eta 0.004": dict(host_kw=dict(bar_eta=0.004)), "M2 eta 0.008": dict(host_kw=dict(bar_eta=0.008)),
     "M3a Portail amp 1.0": dict(host_kw=dict(bar_model="portail17"), setup=(35.5, 28., 1.0, 5.6)),
     "M3a Portail amp 1.2": dict(host_kw=dict(bar_model="portail17")),
     "M3b size 1.15": dict(host_kw=dict(bar_size=1.15)), "M3b size 1.3": dict(host_kw=dict(bar_size=1.3)),
     "M4 window 1500, age<1000": dict(window=1500., nrel=48000, age=1000.), "M4 window 1500, age<1500": dict(window=1500., nrel=48000, age=1500.),
     "ref 16 deg (grid PMs)": dict(setup=(34.5, 16., 1.4, 5.6), pm=(-3.2223, -6.7517)),
     "M4b peri release s10": dict(release="peri", psig=10.), "M4b peri release s20": dict(release="peri", psig=20.),
     "M4b 16 deg peri s10": dict(setup=(34.5, 16., 1.4, 5.6), pm=(-3.2223, -6.7517), release="peri", psig=10.),
     "M4b 16 deg peri s20": dict(setup=(34.5, 16., 1.4, 5.6), pm=(-3.2223, -6.7517), release="peri", psig=20.),
     "P1 halo q 0.8": dict(host_kw=dict(axi_variant=dict(halo_q=0.8))), "P1 halo q 0.9": dict(host_kw=dict(axi_variant=dict(halo_q=0.9))),
     "P1 halo q 1.2": dict(host_kw=dict(axi_variant=dict(halo_q=1.2))),
     "P1 disc x0.8": dict(host_kw=dict(axi_variant=dict(disc_scale=0.8))), "P1 disc x1.2": dict(host_kw=dict(axi_variant=dict(disc_scale=1.2))),
     "P1 q 0.8 disc x1.2": dict(host_kw=dict(axi_variant=dict(halo_q=0.8, disc_scale=1.2))),
     "P1 q 1.2 disc x0.8": dict(host_kw=dict(axi_variant=dict(halo_q=1.2, disc_scale=0.8)))}
SEG = [(-99, 4), (4, 17), (17, 99)]; outd = ROOT/"results/streams/phase2"; outd.mkdir(parents=True, exist_ok=True)
names = [a for a in sys.argv[1:]] or list(V)
edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])
res = json.loads((ROOT/"results/plot_data/phase2_variants.json").read_text()) if (ROOT/"results/plot_data/phase2_variants.json").exists() else {}
per = {}
for nm in names:
    c = dict(BASE, **V[nm]); t0 = time.time()
    tag = nm.replace(" ", "_").replace(",", "").replace("<", "lt").replace("(", "").replace(")", "")
    sprtag = tag.split("_age")[0] if nm.startswith("M4") else tag
    f = outd/f"{sprtag}.npz"
    if nm == "baseline":
        f = ROOT/"results/streams/d3_free_pm/pm_+0_+0.npz"
    if nm.startswith("ref 16"):
        f = ROOT/"results/streams/seed_test/om34.5_an16_am1.4_d5.6_n32000_s1.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(*c["setup"], nrel=c["nrel"], seed=1, pm=c["pm"], host_kw=c["host_kw"], window_myr=c["window"],
                          release=c["release"], peri_sigma_myr=c["psig"]); np.savez_compressed(f, **m)
    s = score(m, age_max=c["age"]); per[nm] = s["per_star"]
    mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < c["age"]) & (m["chi"] > 0); ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else np.nan for j in range(len(cen))] for q in ("pmra", "pmdec")}
    cnt = np.bincount(ib[(ib >= 0) & (ib < len(cen))], minlength=len(cen)).tolist()
    cmdb = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())["bins"]
    dist = [float(np.median(m["d"][k & (m["b"] >= c_["b"][0]) & (m["b"] < c_["b"][1])])/c["setup"][3])
            if (k & (m["b"] >= c_["b"][0]) & (m["b"] < c_["b"][1])).sum() > 20 else None for c_ in cmdb]
    res[nm] = dict(dist_ratio=dist, lnL=s["total"], seg=[float(s["per_star"][(du >= a) & (du < b)].sum()) for a, b in SEG], med=med, n=cnt)
    print(f"{nm:28s} lnL {s['total']:9.1f} [{time.time()-t0:.0f} s]", flush=True)
(ROOT/"results/plot_data/phase2_variants.json").write_text(json.dumps(res, indent=1))
if "baseline" in res:
    b = res["baseline"]
    for nm, r in res.items():
        print(f"{nm:28s} dlnL vs baseline {r['lnL']-b['lnL']:+8.0f}   segments " + " ".join(f"{x-y:+7.0f}" for x, y in zip(r["seg"], b["seg"])))
dsel = np.abs(dx) < 6; dib = np.where(dsel, np.digitize(du, edges)-1, -1)
fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
for i, (nm, r) in enumerate(res.items()):
    st = "-" if nm != "baseline" else "k-"; lw = 2.5 if nm in ("baseline", "ref 16 deg (grid PMs)") else 1.2
    ax[0].plot(cen, r["med"]["pmra"], st, lw=lw, label=nm); ax[1].plot(cen, r["med"]["pmdec"], st, lw=lw); ax[2].plot(cen, r["n"], st, lw=lw)
for a, q in zip(ax[:2], ("pmra", "pmdec")):
    a.plot(cen, [np.median(d[q][dib == j]) if (dib == j).sum() >= 10 else np.nan for j in range(len(cen))], "ko", mfc="w", ms=6, label="members")
    a.set_xlabel(r"$\phi_1$ [deg]"); a.set_ylabel(q)
ax[2].set_xlabel(r"$\phi_1$ [deg]"); ax[2].set_ylabel("model particles per 2-deg bin (footprint)"); ax[0].legend(fontsize=7)
fig.suptitle("Phase 2 variants at 28 deg (catalogue PMs; 4x particles, seed 1): footprint medians vs members"); fig.tight_layout()
fig.savefig(ROOT/"plots/phase2_variants.png", dpi=75); print("plots/phase2_variants.png")
