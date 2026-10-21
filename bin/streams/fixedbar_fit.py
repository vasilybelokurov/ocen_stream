#!/usr/bin/env python3
"""Best stream fit with the bar fixed to its measured form (2026-10-21).

Bar: Hunter+2024 / Sormani+2022 as published (amplitude A = 1, size S = 1), major axis at 27 deg (Wegg+2015; Bland-Hawthorn &
Gerhard 2016: 27 +- 2). Free (6): pattern speed Omega_b 20-50 km/s/kpc, omega Cen distance 5.33-5.53 kpc, systemic PMs
(+-4 sigma), v_c(R0) 220-250, halo flattening q 0.8-1.2; step-2 priors (distance, PMs, v_c), gravity21 frame, data box, score as
the profiles (jackknife, cutoff 7, chi 0-200).
Optimiser: profile_nm.NM (bounded ask/tell Nelder-Mead, CRN per chain) in the 6-D unit cube; 4 chains from: the 25-deg and
27.5-deg step-2 bests (bar reset to A = S = 1, Omega_b kept), and the 25-deg best with Omega_b 27 and 42. Search at 64k release
epochs (budget 40 per chain), then 4 independent 256k seeds at each chain's best point.
Output: results/streams/fixedbar_20261021/{runs/, state.json, result.json}
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/fixedbar_fit.py --workers 8
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/fixedbar_20261021"
ANGLE = 27.0
MU_SYS, MU_ERR = (-3.250, -6.746), 0.022
LO = dict(omega=20., dist=5.33, vc=220., halo_q=0.8); HI = dict(omega=50., dist=5.53, vc=250., halo_q=1.2)


def to_params(z):
    z = np.clip(np.asarray(z, float), 0, 1)
    return dict(angle=ANGLE, omega=LO["omega"]+(HI["omega"]-LO["omega"])*z[0], amp=1.0, size=1.0, dist=LO["dist"]+(HI["dist"]-LO["dist"])*z[1],
                pmra=MU_SYS[0]+MU_ERR*(8*z[2]-4), pmdec=MU_SYS[1]+MU_ERR*(8*z[3]-4), vc=LO["vc"]+(HI["vc"]-LO["vc"])*z[4],
                halo_q=LO["halo_q"]+(HI["halo_q"]-LO["halo_q"])*z[5])


def to_unit(p):
    return np.clip([(p["omega"]-LO["omega"])/(HI["omega"]-LO["omega"]), (p["dist"]-LO["dist"])/(HI["dist"]-LO["dist"]),
                    (p["pmra"]-MU_SYS[0])/MU_ERR/8+0.5, (p["pmdec"]-MU_SYS[1])/MU_ERR/8+0.5,
                    (p["vc"]-LO["vc"])/(HI["vc"]-LO["vc"]), (p["halo_q"]-LO["halo_q"])/(HI["halo_q"]-LO["halo_q"])], 0, 1)


def evaluate(args):
    z, seed, off, nrel, rid = args
    jf = OUT/"runs"/f"{rid}.json"
    if jf.exists():
        return json.loads(jf.read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_dm.streams.frames import frame_r0
    from ocen_dm.streams.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    p = to_params(z); t0 = time.time()
    m = make_spray(p["omega"], p["angle"], 1.0, p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=1.0, axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21"))),
                   max_age_myr=700., epoch_offset=off, frame="gravity21")
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}; ms["age_cut_myr"] = np.float64(700.)
    np.savez_compressed(OUT/"runs"/f"{rid}_tmp.npz", **ms); (OUT/"runs"/f"{rid}_tmp.npz").rename(OUT/"runs"/f"{rid}.npz")
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(run_id=rid, unit=list(map(float, z)), params=p, seed=seed, epoch_offset=off, nrel=nrel, total_jk=float(s["total_jk"]),
               ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    jf.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--nrel", type=int, default=64000); a = ap.parse_args()
    from profile_nm import NM
    (OUT/"runs").mkdir(parents=True, exist_ok=True)
    B = {k: json.loads((ROOT/f"results/streams/profile_step2_20261020/runs/{f}.json").read_text())["params"]
         for k, f in (("25", "final_a25_1"), ("27.5", "final_a27.5_0"))}
    starts = [("from25", B["25"]), ("from27.5", B["27.5"]), ("from25_om27", dict(B["25"], omega=27.)), ("from25_om42", dict(B["25"], omega=42.))]
    chains = []
    for i, (cid, p) in enumerate(starts):
        seed = 2400000+100*i; chains.append(NM(cid, None, to_unit(p), seed, float(np.random.default_rng([seed, 777]).uniform()), a.budget, ndim=6, full=lambda z: z))
    queue = {}

    def submit(ex, ch):
        for tag, z in ch.ask():
            k = ch.n_eval+len([1 for v in queue.values() if v[0] is ch])
            queue[ex.submit(evaluate, (list(map(float, ch.full(z))), ch.seed, ch.off, a.nrel, f"{ch.cid}_e{k:03d}"))] = (ch, tag, z)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for ch in chains:
            submit(ex, ch)
        while queue:
            done, _ = wait(list(queue), return_when=FIRST_COMPLETED)
            for f in done:
                ch, tag, z = queue.pop(f); rec = f.result(); ch.tell(tag, np.asarray(rec["unit"]), -(rec["total_jk"]+rec["ln_prior"]))
                busy = any(v[0] is ch for v in queue.values())
                if not busy and not ch.done():
                    submit(ex, ch)
                elif not busy:
                    print(f"{time.strftime('%H:%M:%S')} chain {ch.cid} done: {ch.n_eval} evals, best {-np.nanmin(ch.F):.0f}", flush=True)
            (OUT/"state.json").write_text(json.dumps({ch.cid: dict(n_eval=ch.n_eval, best=float(-np.nanmin(ch.F)) if np.isfinite(ch.F).any() else None,
                                                                   best_params=to_params(ch.S[int(np.nanargmin(ch.F))]) if np.isfinite(ch.F).any() else None)
                                                      for ch in chains}, indent=1, default=float))
        futs = []
        for i, ch in enumerate(chains):
            zb = ch.S[int(np.nanargmin(ch.F))]
            for k in range(4):
                seed = 2450000+100*i+k
                futs.append((ch.cid, ex.submit(evaluate, (list(map(float, zb)), seed, float(np.random.default_rng([seed, 777]).uniform()), 256000, f"final_{ch.cid}_{k}"))))
        res = {}
        for cid, f in futs:
            r = f.result(); res.setdefault(cid, dict(params=r["params"], totals=[]))["totals"].append(r["total_jk"]+r["ln_prior"])
    for cid, d in res.items():
        y = np.array(d["totals"]); d.update(mean=float(y.mean()), se=float(y.std(ddof=1)/2))
        print(f"{cid:12s} 256k {d['mean']:.0f} +- {d['se']:.0f}  " + " ".join(f"{k} {v:.3f}" for k, v in d["params"].items()))
    (OUT/"result.json").write_text(json.dumps(res, indent=1, default=float)); print(f"done in {(time.time()-t0)/3600:.1f} h")


if __name__ == "__main__":
    main()
