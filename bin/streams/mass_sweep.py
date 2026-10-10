#!/usr/bin/env python3
"""Host-mass sweep at the W model (28 deg, bar size 1.15, Omega_b 36, amp 1.2, catalogue PMs, d 5.6, baumgardt frame), 4x
particles, seeds 1-3 (docs/codex_frame_decomp_next_2026-10-10.md).
  H  halo-only: axisymmetric part = mw_variants.axisymmetric_variant(vc_target) for v_c(R0) = 220, 225, 228.8 (reference), 235,
     240 km/s; baryons and bar term unchanged (so the bar's share of the force changes)           -- 15 sprays;
  S  whole host x mass_scale = (v_c/vc0)^2 for v_c = 220, 225, 235, 240 (bar share unchanged; reference = H at vc0) -- 12 sprays.
Per model also records v_c(R) at R = 2, 4, 6, 8.178 kpc, omega Cen's peri/apo over the last 1 Gyr, and the median ratio
|F_bar| / |F_axi| along that orbit (F_bar = full host force minus the axisymmetric part).
Output results/plot_data/mass_sweep.json; per-observable segment sums included. Usage: OMP_NUM_THREADS=8 python bin/streams/mass_sweep.py
"""
import itertools, json, os, time
import numpy as np
from grid4_common import ROOT, make_spray, score, project_gc, GC, du, OCEN_OBS, T
from obs_split import single_scores
from ocen_dm.streams.restricted import host_potential, agama_kpc, HUNTER24_DIR
from ocen_dm.streams.mw_variants import axisymmetric_variant, _vc2
from ocen_dm.streams.frames import to_model

agama = agama_kpc()
VC0 = float(np.sqrt(_vc2(agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini")))))
CAT = (OCEN_OBS["pmra"], OCEN_OBS["pmdec"]); W = dict(om=36., an=28., am=1.2, size=1.15)
CFG = [dict(kind="H", vc=vc, seed=s) for vc, s in itertools.product((220., 225., round(VC0, 1), 235., 240.), (1, 2, 3))]
CFG += [dict(kind="S", vc=vc, seed=s) for vc, s in itertools.product((220., 225., 235., 240.), (1, 2, 3))]
SEG = [(-99, 4), (4, 17), (17, 99)]; edges = np.arange(np.floor(du.min()), np.ceil(du.max())+2, 2.); cen = 0.5*(edges[1:]+edges[:-1])


def host_kw(c):
    if c["kind"] == "H":
        return dict(bar_size=W["size"], axi_variant=dict(vc_target=c["vc"]))
    return dict(bar_size=W["size"], mass_scale=(c["vc"]/VC0)**2)


def diagnostics(c):
    hk = host_kw(c); host = host_potential("x", bar_omega=W["om"], bar_angle_deg=W["an"], t_today=T, bar_amp=W["am"], **hk)
    axi = (axisymmetric_variant(HUNTER24_DIR, vc_target=c["vc"])[0] if c["kind"] == "H" else
           agama.Potential(potential=agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini")), scale=[hk["mass_scale"], 1.]))
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], 5.6, *CAT, OCEN_OBS["vlos"])[0]
    t, o = agama.orbit(potential=host, ic=today, timestart=T, time=-1000./977.79, trajsize=2001, accuracy=1e-10)
    r = np.linalg.norm(o[:, :3], axis=1); fa = axi.force(o[:, :3]); fh = np.array([host.force(o[i:i+1, :3], t=t[i])[0] for i in range(0, len(t), 10)])
    ratio = np.linalg.norm(fh-fa[::10], axis=1)/np.linalg.norm(fa[::10], axis=1)
    vc = {R: float(np.sqrt(-R*axi.force(np.array([[R, 0., 0.]]))[0, 0])) for R in (2., 4., 6., 8.178)}
    return dict(vcR=vc, peri=float(r.min()), apo=float(r.max()), bar_frac_med=float(np.median(ratio)))


outd = ROOT/"results/streams/mass_sweep"; outd.mkdir(parents=True, exist_ok=True); outp = ROOT/"results/plot_data/mass_sweep.json"
res = json.loads(outp.read_text()) if outp.exists() else []; done = {r["tag"] for r in res}
print(f"VC0 = {VC0:.2f} km/s", flush=True)
for c in CFG:
    tag = f"{c['kind']}_vc{c['vc']:g}_s{c['seed']}"
    if tag in done:
        continue
    t0 = time.time(); f = outd/f"{tag}.npz"
    if f.exists():
        m = dict(np.load(f))
    else:
        m = make_spray(W["om"], W["an"], W["am"], 5.6, nrel=32000, seed=c["seed"], pm=CAT, host_kw=host_kw(c)); np.savez_compressed(f, **m)
    s = score(m); ss = single_scores(m); mu, mx = project_gc(m["l"], m["b"], GC); k = (np.abs(mx) < 6) & (m["age"] < 700) & (m["chi"] > 0)
    ib = np.digitize(mu[k], edges)-1
    med = {q: [float(np.median(m[q][k][ib == j])) if (ib == j).sum() >= 10 else None for j in range(len(cen))] for q in ("pmra", "pmdec")}
    segs = lambda v: [float(np.nansum(v[(du >= a) & (du < b)])) for a, b in SEG]
    r = dict(c, tag=tag, lnL=s["total"], seg=segs(s["per_star"]), seg_x=segs(ss["x"]), seg_pm=segs(ss["pm"]), med=med, phi1=cen.tolist(), **diagnostics(c))
    res.append(r); tmp = outp.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, outp)
    print(f"{tag:16s} lnL {s['total']:9.1f} seg " + " ".join(f"{x:8.0f}" for x in r["seg"]) + f"  peri/apo {r['peri']:.2f}/{r['apo']:.2f}"
          f"  bar/axi {r['bar_frac_med']:.3f}  vc(2,4,6,R0) " + " ".join(f"{v:.0f}" for v in r["vcR"].values()) + f"  [{time.time()-t0:.0f} s]", flush=True)
