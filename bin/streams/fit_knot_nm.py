#!/usr/bin/env python3
"""Free 9-D fit of the bar + halo + cluster parameters to the single-component knot vector (2026-10-25, agreed with the user).

Objective: chi2 of stream_model_compare.py (sky track and width in 7 local windows, mu_ra*, mu_de at the PM spline knots; model
measured like the data; no nuisance term), minimised by the bounded ask/tell Nelder-Mead of profile_nm.py (NM, 9 free unit
coordinates, initial step 0.1, stop at simplex size 0.05 or BUDGET evaluations). No priors: box ranges only (user 2026-10-25):
    angle 0-50 deg, Omega_b 20-50 km/s/kpc, amplitude 0.6-2.0, size 0.8 .. min(1.5, R_cr/4.25 kpc), halo q (density flattening)
    0.7-1.2, v_c 220-250 km/s (gravity21 R0), distance 5.28-5.68 kpc (+-3 sigma of BV21 5.43 +- 0.05 and oMEGACat VI
    5.494 +- 0.061), cluster PM +-4 sigma of VB21 (-3.250, -6.746) +- 0.022 mas/yr.
Sprays: grid4_common.make_spray as in profile_nm (gravity21 frame, max age 700 Myr, trailing arm), NREL releases during the
search with common random numbers per chain (one seed and epoch offset), then NFINAL independent NREL_FINAL seeds at each chain's
best point. Starts: the parameters of the stored runs given with --start (JSON next to each npz), mapped into the unit cube.
Sprays run in spawned workers (agama only); the parent scores each finished spray (~5 s; it holds the data vector and imports
torch via stream_spline_fit -- two OpenMP runtimes in one process abort). Every evaluation is stored (OUT/runs/<id>.npz + .json)
and reused on restart.
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/fit_knot_nm.py --outdir OUT --start A.npz B.npz ... [--workers 8]
           [--budget 40] [--nrel 128000] [--nrel-final 256000] [--seed-base 3100000]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import multiprocessing as mp
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
NAMES = ("angle", "omega", "amp", "size", "dist", "pmra", "pmdec", "vc", "halo_q")
LIM = dict(angle=(0., 50.), omega=(20., 50.), amp=(0.6, 2.0), dist=(5.28, 5.68), vc=(220., 250.), halo_q=(0.7, 1.2))
MU_SYS, MU_ERR, RBAR, FRAME, NFINAL = (-3.250, -6.746), 0.022, 4.25, "gravity21", 4
G = {}


def _smax(p):
    from training_set import r_cr
    from ocen_dm.streams.frames import frame_r0
    return min(1.5, r_cr(p["omega"], p["vc"], p["halo_q"], frame_r0(FRAME))/RBAR)


def unit_to_params(u):
    lin = lambda k, x: LIM[k][0]+(LIM[k][1]-LIM[k][0])*x
    p = dict(angle=lin("angle", u[0]), omega=lin("omega", u[1]), amp=lin("amp", u[2]), dist=lin("dist", u[4]),
             pmra=MU_SYS[0]+MU_ERR*(8.*u[5]-4.), pmdec=MU_SYS[1]+MU_ERR*(8.*u[6]-4.), vc=lin("vc", u[7]), halo_q=lin("halo_q", u[8]))
    p["size"] = 0.8+u[3]*(_smax(p)-0.8)
    return {k: float(p[k]) for k in NAMES}


def params_to_unit(p):
    un = lambda k: (p[k]-LIM[k][0])/(LIM[k][1]-LIM[k][0])
    u = np.clip([un("angle"), un("omega"), un("amp"), 0., un("dist"), (p["pmra"]-MU_SYS[0])/MU_ERR/8.+0.5, (p["pmdec"]-MU_SYS[1])/MU_ERR/8.+0.5,
                 un("vc"), un("halo_q")], 0., 1.)
    q = unit_to_params(u); u[3] = np.clip((p["size"]-0.8)/(_smax(q)-0.8), 0., 1.); return u


def spray_job(args):
    """worker: spray at unit point u, saved to OUT/runs/<rid>.npz (skipped if present); returns the job record."""
    u, seed, off, nrel, outdir, rid = args
    f = Path(outdir)/"runs"/f"{rid}.npz"; rec = dict(run_id=rid, unit=list(map(float, u)), params=unit_to_params(np.asarray(u)), seed=seed,
                                                     epoch_offset=off, nrel=nrel, npz=str(f))
    if f.exists() or (Path(outdir)/"runs"/f"{rid}.json").exists():
        return rec
    from grid4_common import make_spray
    from ocen_dm.streams.frames import frame_r0
    p = unit_to_params(np.asarray(u)); t0 = time.time()
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0(FRAME))),
                   max_age_myr=700., epoch_offset=off, frame=FRAME)
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}; ms["age_cut_myr"] = np.float64(700.)
    tmp = f.with_name(f"{rid}_tmp.npz"); np.savez_compressed(tmp, **ms); os.replace(tmp, f); rec["seconds_spray"] = time.time()-t0
    return rec


def score(rec):
    """parent: knot chi2 of a finished spray; record written to OUT/runs/<rid>.json (reused on restart)."""
    rec_f = Path(rec["npz"]).with_suffix(".json")
    if rec_f.exists():
        return json.loads(rec_f.read_text())
    MC = G["MC"]; M = MC.measure(G["O"], G["D"], rec["npz"], np.random.default_rng([rec["seed"], 99])); rec = dict(rec, n_particles=M["n_particles"], chi2=MC.chi2(G["D"], M))
    tj = rec_f.with_suffix(".json.tmp"); tj.write_text(json.dumps(rec, indent=1)); os.replace(tj, rec_f)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True); ap.add_argument("--start", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--nrel", type=int, default=128000); ap.add_argument("--nrel-final", type=int, default=256000)
    ap.add_argument("--seed-base", type=int, default=3100000)
    a = ap.parse_args()
    sys.argv = [sys.argv[0]]                                                     # stream_model_compare sets its own argv
    import stream_model_compare as MC
    from profile_nm import NM
    P = lambda x: Path(x) if Path(x).is_absolute() else ROOT/x
    od = P(a.outdir); (od/"runs").mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = dict(limits=LIM, mu_sys=MU_SYS, mu_err=MU_ERR, frame=FRAME, budget=a.budget, nrel=a.nrel, nrel_final=a.nrel_final, seed_base=a.seed_base,
               start=[str(P(s).relative_to(ROOT)) for s in a.start], objective="stream_model_compare.chi2 (single-component knots)")
    mf = od/"manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text()); old.pop("git", None)
        if old != man:
            raise SystemExit("manifest differs from an earlier run in this directory; refusing to resume")
    mf.write_text(json.dumps(dict(man, git=git), indent=1))
    G["MC"] = MC; G["D"] = MC.data_vector(); G["O"] = MC.Obs()
    chains, queue, bestp = [], {}, {}

    def submit(ex, ch):
        for tag, z in ch.ask():
            k = ch.n_eval+len([1 for v in queue.values() if v[0] is ch]); rid = f"{ch.cid}_e{k:03d}"
            queue[ex.submit(spray_job, (ch.full(z).tolist(), ch.seed, ch.off, a.nrel, str(od), rid))] = (ch, tag, z)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for j, s in enumerate(a.start):                                           # unit maps need agama: done in a worker
            p0 = json.loads(P(s).with_suffix(".json").read_text())["params"]; z = ex.submit(params_to_unit, p0).result(); seed = a.seed_base+100*j
            chains.append(NM(f"c{j}", None, z, seed, float(np.random.default_rng([seed, 777]).uniform()), a.budget, ndim=9, full=lambda x: x))
            print(f"chain c{j}: start {Path(s).stem} -> " + ", ".join(f"{k} {v:.3g}" for k, v in ex.submit(unit_to_params, z).result().items()), flush=True)
        for ch in chains:
            submit(ex, ch)
        while queue:
            ds, _ = wait(list(queue), return_when=FIRST_COMPLETED)
            for f in ds:
                ch, tag, z = queue.pop(f); rec = score(f.result()); ch.tell(tag, np.asarray(rec["unit"]), rec["chi2"]["total"])
                if ch.cid not in bestp or rec["chi2"]["total"] < bestp[ch.cid]["chi2"]["total"]:
                    bestp[ch.cid] = rec
                print(f"{time.strftime('%H:%M:%S')} {rec['run_id']}: chi2 {rec['chi2']['total']:.1f} (spray {rec.get('seconds_spray', 0):.0f} s)", flush=True)
                busy = any(v[0] is ch for v in queue.values())
                if not busy and not ch.done():
                    submit(ex, ch)
                elif not busy:
                    print(f"{time.strftime('%H:%M:%S')} chain {ch.cid} done: {ch.n_eval} evals, best chi2 {np.nanmin(ch.F):.1f}", flush=True)
            st = {ch.cid: dict(n_eval=ch.n_eval, best=float(np.nanmin(ch.F)) if np.isfinite(ch.F).any() else None, state=ch.state,
                               best_run=bestp.get(ch.cid, {}).get("run_id"), best_params=bestp.get(ch.cid, {}).get("params")) for ch in chains}
            (od/"state.json").write_text(json.dumps(st, indent=1))
        futs = []
        for j, ch in enumerate(chains):
            u = ch.S[int(np.nanargmin(ch.F))]
            for k in range(NFINAL):
                seed = a.seed_base+50000+100*j+k
                futs.append((ch, ex.submit(spray_job, (u.tolist(), seed, float(np.random.default_rng([seed, 777]).uniform()), a.nrel_final, str(od), f"final_{ch.cid}_{k}"))))
        out = {}
        for ch, f in futs:
            r = score(f.result()); d = out.setdefault(ch.cid, dict(search_best=float(np.nanmin(ch.F)), params=r["params"], final=[])); d["final"].append(r["chi2"]["total"])
    for cid, d in out.items():
        y = np.array(d["final"]); d.update(mean=float(y.mean()), sd=float(y.std(ddof=1)))
    (od/"result.json").write_text(json.dumps(out, indent=1)); print(f"done in {(time.time()-t0)/3600:.1f} h")
    for cid, d in sorted(out.items(), key=lambda x: x[1]["mean"]):
        print(f"{cid}: final chi2 {d['mean']:.1f} +- {d['sd']:.1f} (search {d['search_best']:.1f}); " + ", ".join(f"{k} {v:.3g}" for k, v in d["params"].items()))


if __name__ == "__main__":
    main()
