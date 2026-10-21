#!/usr/bin/env python3
"""Pre-screen of the alternative-bar plan (docs/reviews/fable_bar_model_20261021.md; 2026-10-21). No fitting.

Step 0 (mechanism): the 25-deg step-2 best at its Omega_b and at Omega_b -+1.5 km/s/kpc (moves the ~330-Myr release pericentre
  off the bar major axis): does the off-track branch fraction respond?
Step 1a (force budget): max over azimuth of the spiral's force (Hunter 2-arm pattern, amplitude 1, src/ocen_dm/streams/spiral.py)
  relative to the axisymmetric force at R 4-8 kpc, z 0 and 1.5, against the excess non-axisymmetric force of the fitted bars
  over the published one (bin/streams/diagnostics/bar_diag2.py: ~0.1-0.12 of F_axi at R 5-7, z 0).
Step 1b (sprays): the 12.5- and 25-deg best parameters with the bar at its published strength and size (A = S = 1): no spiral;
  + spiral amplitude 1 at the reference phase 25 deg and at 115 deg (pattern speed 22.5).
All sprays: 32k release epochs, seed 900011, epoch offset 0.1 (common random numbers), gravity21 frame, v_c normalised at its
R0, age < 700 Myr; scored like the profile (data box, jackknife) + step-2 priors.
Metrics: ln L + prior; branch fraction (debris with pmra < members' median - 3 among |dphi2| < 15, phi1 6-22; ages 300-350 and
all); Gaia-search branch/track ratio R_pred (windows of bin/streams/branch_search_gaia.py); knee pmdec residual (members' median
minus on-tube model median, phi1 19-27); far-end density (on-tube model / members, phi1 22-28, each normalised over phi1 14-22).
Output: results/plot_data/bar_prescreen.json; results/streams/bar_prescreen_20261021/runs/*.npz
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/bar_prescreen.py --workers 5
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/bar_prescreen_20261021"
BEST = {"12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.json", "25": "results/streams/profile_step2_20261020/runs/final_a25_1.json"}


def jobs():
    J = [("25 fitted", "25", {}, None), ("25 Omega-1.5", "25", dict(omega=-1.5), None), ("25 Omega+1.5", "25", dict(omega=+1.5), None),
         ("12.5 fitted", "12.5", {}, None)]
    for lab in ("12.5", "25"):
        J += [(f"{lab} A=S=1", lab, dict(amp=1.0, size=1.0), None),
              (f"{lab} A=S=1 + spiral ph25", lab, dict(amp=1.0, size=1.0), dict(amplitude=1.0, phase_deg=25.)),
              (f"{lab} A=S=1 + spiral ph115", lab, dict(amp=1.0, size=1.0), dict(amplitude=1.0, phase_deg=115.))]
    return J


def run(job):
    name, lab, over, spiral = job
    rid = name.replace(" ", "_").replace("=", "").replace("+", "p").replace("-", "m")
    jf = OUT/"runs"/f"{rid}.json"
    if jf.exists():
        return json.loads(jf.read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_dm.streams.frames import frame_r0
    from ocen_dm.streams.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    p = dict(json.loads((ROOT/BEST[lab]).read_text())["params"])
    for k, v in over.items():
        p[k] = p[k]+v if k == "omega" else v
    host_kw = dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21")))
    if spiral:
        host_kw["spiral"] = spiral
    t0 = time.time()
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=32000, seed=900011, pm=(p["pmra"], p["pmdec"]), host_kw=host_kw,
                   max_age_myr=700., epoch_offset=0.1, frame="gravity21")
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}
    np.savez_compressed(OUT/"runs"/f"{rid}.npz", **ms)
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(name=name, run_id=rid, params=p, spiral=spiral, total_jk=float(s["total_jk"]), ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    jf.write_text(json.dumps(rec, indent=1))
    return rec


def metrics(npz):
    from grid4_common import d, du, dx, project_gc, GC
    import branch_search_gaia as BS
    m = dict(np.load(npz)); u, x = project_gc(m["l"], m["b"], GC); k = (m["chi"] > 0) & (m["age"] < 700.)
    u, x, pa, pd, age = u[k], x[k], m["pmra"][k], m["pmdec"][k], m["age"][k]
    mem = BS.members(); win, _ = BS.model_windows(mem); out = {}
    br = {"all": [0, 0], "300-350": [0, 0]}; A = T = 0
    for (a0, a1), W in win.items():
        km = (du >= a0) & (du < a1); mp = np.median(d["pmra"][km])
        kk = (u >= a0) & (u < a1) & (np.abs(x) < 15)
        for key, ka in (("all", kk), ("300-350", kk & (age >= 300) & (age < 350))):
            br[key][0] += np.sum(ka & (pa < mp-3)); br[key][1] += ka.sum()
        k2 = (u >= a0) & (u < a1)
        A += np.sum(k2 & BS.sky_A(x, W["track"]) & BS.in_pm(pa, pd, W["bpm"])); T += np.sum(k2 & BS.sky_T(x, W["track"]) & BS.in_pm(pa, pd, W["tpm"]))
    out["branch_all"] = br["all"][0]/br["all"][1]; out["branch_300_350"] = br["300-350"][0]/max(br["300-350"][1], 1); out["R_pred"] = A/max(T, 1)
    res = []
    for a0 in (19, 21, 23, 25):
        km = (du >= a0) & (du < a0+2); tr = np.median(dx[km]); mp, md = np.median(d["pmra"][km]), np.median(d["pmdec"][km])
        kk = (u >= a0) & (u < a0+2) & (np.abs(x-tr) < 2) & (np.abs(pa-mp) < 4)
        res.append(float(md-np.median(pd[kk])) if kk.sum() > 20 else np.nan)
    out["knee_pmdec_resid"] = res
    e = np.arange(14, 28.01, 1.); trk = np.array([np.median(dx[(du >= a) & (du < a+1)]) for a in e[:-1]])
    on = np.abs(x-np.interp(u, 0.5*(e[1:]+e[:-1]), trk)) < 3
    hm = np.histogram(u[on], e)[0]; hd = np.histogram(du, e)[0]; ref = slice(0, 8); far = slice(8, 14)
    out["far_end_density_ratio"] = float((hm[far].sum()/hm[ref].sum())/(hd[far].sum()/hd[ref].sum()))
    return out


def force_budget():
    from ocen_dm.streams.restricted import host_potential, agama_kpc
    from ocen_dm.streams.frames import frame_r0
    from grid4_common import T
    agama_kpc(); p = json.loads((ROOT/BEST["25"]).read_text())["params"]
    kw = dict(bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=1.0, bar_size=1.0,
              axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21")))
    h0 = host_potential("x", **kw); h1 = host_potential("x", spiral=dict(amplitude=1.0, phase_deg=25.), **kw)
    from ocen_dm.streams.mw_variants import axisymmetric_variant
    from ocen_dm.streams.restricted import HUNTER24_DIR
    axi = axisymmetric_variant(HUNTER24_DIR, halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21"))[0]
    phi = np.linspace(0, 2*np.pi, 73)[:-1]; out = {}
    for z in (0., 1.5):
        row = []
        for R in (4., 5., 6., 7., 8.):
            P = np.column_stack((R*np.cos(phi), R*np.sin(phi), np.full_like(phi, z)))
            Fs = h1.force(P, t=T)-h0.force(P, t=T); Fa = axi.force(P)
            row.append(float(np.max(np.linalg.norm(Fs, axis=1))/np.mean(np.linalg.norm(Fa, axis=1))))
        out[f"z{z:g}"] = row
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); a = ap.parse_args()
    (OUT/"runs").mkdir(parents=True, exist_ok=True)
    fb = force_budget()
    print("Step 1a: spiral (amplitude 1) |F|max / |F_axi| at R = 4,5,6,7,8 kpc:", {k: [round(v, 3) for v in r] for k, r in fb.items()},
          "| fitted-minus-published bar excess at z 0, R 5-7: ~0.10-0.12 (bar_diag2)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, jobs()))
    out = dict(force_budget=fb, runs=[])
    ref = {r["name"].split()[0]: r["total_jk"]+r["ln_prior"] for r in recs if r["name"].endswith("fitted")}
    print("\nStep 0 / 1b (32k, CRN seed 900011): lnL+prior, change vs that angle's fitted model, branch fractions, R_pred, knee pmdec residual "
          "(phi1 19-27), far-end density ratio (model/members, 1 = match)")
    for r in recs:
        mt = metrics(OUT/"runs"/f"{r['run_id']}.npz"); r.update(mt); out["runs"].append(r)
        tot = r["total_jk"]+r["ln_prior"]; lab = r["name"].split()[0]
        print(f"  {r['name']:28s} {tot:9.0f} ({tot-ref[lab]:+6.0f})  branch {mt['branch_all']:.2f} (300-350 Myr {mt['branch_300_350']:.2f})  "
              f"R_pred {mt['R_pred']:.2f}  knee pmdec resid {np.round(mt['knee_pmdec_resid'], 2).tolist()}  far-end {mt['far_end_density_ratio']:.2f}", flush=True)
    (ROOT/"results/plot_data/bar_prescreen.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
