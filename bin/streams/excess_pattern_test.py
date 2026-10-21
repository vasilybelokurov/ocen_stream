#!/usr/bin/env python3
"""Discriminating test: must the extra non-axisymmetric force the stream demands rotate with the bar? (2026-10-21)

Design: docs/reviews/fable_bar_force_20261021.md. At the step-2 best parameters for 25 deg (primary: the literature angle) and
12.5 deg (comparison), the bar is set to its published strength and size (A = S = 1) and the fitted bar's excess,
A_fit [full(x/S_fit) - axi(x/S_fit)] - [full - axi] (restricted.host_potential excess=...), is added as a separate rigid pattern
with its own pattern speed and phase (phase = its major-axis angle today minus the bar's):
  co-rotating (Omega_b) at phase 0 (control: equals the fitted host), 45, 90 deg; static (0) at 0, 45, 90, 135 deg (a stand-in
  for a triaxial halo of the same shape); spiral-like (22.5) at 0, 90 deg; counter-rotating (-Omega_b) at 0 deg.
References: the fitted (inflated) bar; the published bar alone.
32k release epochs, seed 900011, epoch offset 0.1 (common random numbers), gravity21, age < 700, data box, step-2 priors.
Metrics as bin/streams/bar_prescreen.py (lnL + prior, branch fractions, Gaia R_pred, knee pmdec residual, far-end density) and
the recovered fraction (lnL - published) / (fitted - published).
Output: results/plot_data/excess_pattern_test.json; runs in results/streams/excess_pattern_20261021/runs
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/excess_pattern_test.py --workers 4
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/excess_pattern_20261021"
BEST = {"25": "results/streams/profile_step2_20261020/runs/final_a25_1.json", "12.5": "results/streams/profile_step2_20261020/runs/final_a12.5_3.json"}


def jobs():
    J = []
    for lab in ("25", "12.5"):
        J += [(lab, "fitted bar", None), (lab, "published bar", "none")]
        J += [(lab, f"co-rotating ph{ph}", ("co", ph)) for ph in (0, 45, 90)]
        J += [(lab, f"static ph{ph}", (0., ph)) for ph in (0, 45, 90, 135)]
        J += [(lab, f"Omega22.5 ph{ph}", (22.5, ph)) for ph in (0, 90)]
        J += [(lab, "counter-rotating ph0", ("counter", 0))]
    return J


def run(job):
    lab, name, ex = job
    rid = f"a{lab}_" + name.replace(" ", "_").replace(".", "p")
    jf = OUT/"runs"/f"{rid}.json"
    if jf.exists():
        return json.loads(jf.read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_dm.streams.frames import frame_r0
    from ocen_dm.streams.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    p = json.loads((ROOT/BEST[lab]).read_text())["params"]
    host_kw = dict(axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21")))
    if ex is None:
        amp = p["amp"]; host_kw["bar_size"] = p["size"]
    else:
        amp = 1.0; host_kw["bar_size"] = 1.0
        if ex != "none":
            om = {"co": p["omega"], "counter": -p["omega"]}.get(ex[0], ex[0])
            host_kw["excess"] = dict(amp=p["amp"], size=p["size"], omega=float(om), angle_deg=float(p["angle"]+ex[1]))
    t0 = time.time()
    m = make_spray(p["omega"], p["angle"], amp, p["dist"], nrel=32000, seed=900011, pm=(p["pmra"], p["pmdec"]), host_kw=host_kw,
                   max_age_myr=700., epoch_offset=0.1, frame="gravity21")
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}
    np.savez_compressed(OUT/"runs"/f"{rid}.npz", **ms)
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(angle_label=lab, name=name, run_id=rid, excess=host_kw.get("excess"), total_jk=float(s["total_jk"]),
               ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    jf.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    (OUT/"runs").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, jobs()))
    from bar_prescreen import metrics
    out = dict(runs=[])
    for lab in ("25", "12.5"):
        R = {r["name"]: r for r in recs if r["angle_label"] == lab}
        fit, pub = R["fitted bar"]["total_jk"]+R["fitted bar"]["ln_prior"], R["published bar"]["total_jk"]+R["published bar"]["ln_prior"]
        print(f"\n{lab} deg: fitted {fit:.0f}, published {pub:.0f} (gap {fit-pub:.0f}); recovered fraction = (lnL - published)/gap")
        for name, r in R.items():
            mt = metrics(OUT/"runs"/f"{r['run_id']}.npz"); r.update(mt); tot = r["total_jk"]+r["ln_prior"]; r["recovered"] = (tot-pub)/(fit-pub)
            out["runs"].append(r)
            print(f"  {name:22s} {tot:9.0f}  recovered {r['recovered']:+5.2f}  branch {mt['branch_all']:.2f} (300-350 {mt['branch_300_350']:.2f})  "
                  f"R_pred {mt['R_pred']:.2f}  knee pmdec {np.round(mt['knee_pmdec_resid'], 2).tolist()}  far-end {mt['far_end_density_ratio']:.2f}", flush=True)
    (ROOT/"results/plot_data/excess_pattern_test.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
