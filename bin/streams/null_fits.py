#!/usr/bin/env python3
"""Optimised comparison fits for the three claims (docs/reviews/fable_demonstration_design_20261022.md; 2026-10-22).

Models (each fitted with the same machinery as bin/streams/fixedbar_fit.py: gravity21 frame, data box, step-2 priors on the
distance, PMs and v_c, score as the profiles, profile_nm.NM per chain with common random numbers, 64k search with budget 40,
then 4 independent 256k seeds at each chain's best point):
  nobar     (claim 1) Hunter+2024 axisymmetric MW, no bar (A = 0); free: d 5.33-5.53, PMs (+-4 sigma), v_c 220-250, halo q 0.8-1.2,
            stellar-disc mass factor 0.6-1.4 (mw_variants.axisymmetric_variant, halo renormalised to v_c); 4 chains.
  mcmillan  (claim 1) McMillan (2017); free: d, PMs; the v_c prior is evaluated at McMillan's own v_c(R0); 2 chains.
  pub12.5, pub17.5 (claim 3) published bar (A = S = 1) at 12.5 / 17.5 deg; free as fixedbar_fit (Omega_b 20-50, d, PMs, v_c, q);
            2 chains each (the step-2 best at that angle with A = S = 1, at its Omega_b and at 44).
  static    (claim 2) non-rotating bar (Omega_b = 0): free major-axis angle 0-180, A 0.6-2, S 0.8-1.5, d, PMs, v_c, q; 4 chains
            from the 25-deg step-2 best's A, S and nuisances at angles 25, 70, 115, 160.
Output: results/streams/nullfits_20261022/{runs/, state.json, result.json}
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/null_fits.py --workers 8 [--models nobar mcmillan pub12.5 pub17.5 static]
       --smoke: one 32k evaluation per model at its first start, nothing stored in runs/.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/nullfits_20261022"
MU_SYS, MU_ERR = (-3.250, -6.746), 0.022
NUIS = [("dist", 5.33, 5.53), ("pmra", MU_SYS[0]-4*MU_ERR, MU_SYS[0]+4*MU_ERR), ("pmdec", MU_SYS[1]-4*MU_ERR, MU_SYS[1]+4*MU_ERR),
        ("vc", 220., 250.), ("halo_q", 0.8, 1.2)]
MODELS = dict(
    nobar=dict(free=NUIS+[("disc", 0.6, 1.4)], fixed=dict(angle=0., omega=0., amp=0., size=1.)),
    mcmillan=dict(free=NUIS[:3], fixed=dict(angle=0., omega=0., amp=0., size=1., halo_q=1., disc=1.)),
    **{f"pub{a}": dict(free=[("omega", 20., 50.)]+NUIS, fixed=dict(angle=float(a), amp=1., size=1., disc=1.)) for a in ("12.5", "17.5")},
    static=dict(free=[("angle", 0., 180.), ("amp", 0.6, 2.0), ("size", 0.8, 1.5)]+NUIS, fixed=dict(omega=0., disc=1.)))


def to_params(model, z):
    z = np.clip(np.asarray(z, float), 0, 1); M = MODELS[model]
    return dict(M["fixed"], **{k: lo+(hi-lo)*zi for (k, lo, hi), zi in zip(M["free"], z)})


def to_unit(model, p):
    return np.clip([(p[k]-lo)/(hi-lo) for k, lo, hi in MODELS[model]["free"]], 0, 1)


_MCM = {}


def mcmillan():
    if not _MCM:
        from ocen_stream.restricted import host_potential
        from ocen_stream.frames import frame_r0
        h = host_potential("McMillan17"); r0 = frame_r0("gravity21")
        _MCM.update(host=h, vc=float(np.sqrt(-r0*h.force([r0, 0., 0.])[0])))
    return _MCM


def evaluate(args):
    model, z, seed, off, nrel, rid = args
    jf = OUT/"runs"/f"{rid}.json"
    if rid and jf.exists():
        return json.loads(jf.read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_stream.frames import frame_r0
    from ocen_stream.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    p = to_params(model, z); t0 = time.time()
    if model == "mcmillan":
        p["vc"] = mcmillan()["vc"]; kw = dict(host=mcmillan()["host"])
    else:
        kw = dict(host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], disc_scale=p["disc"], vc_target=p["vc"],
                                                                     r0_kpc=frame_r0("gravity21"))))
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]), max_age_myr=700.,
                   epoch_offset=off, frame="gravity21", **kw)
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}; ms["age_cut_myr"] = np.float64(700.)
    if rid:
        np.savez_compressed(OUT/"runs"/f"{rid}_tmp.npz", **ms); (OUT/"runs"/f"{rid}_tmp.npz").rename(OUT/"runs"/f"{rid}.npz")
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(model=model, run_id=rid, unit=list(map(float, z)), params=p, seed=seed, epoch_offset=off, nrel=nrel,
               total_jk=float(s["total_jk"]), ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    if rid:
        jf.write_text(json.dumps(rec, indent=1))
    return rec


def starts():
    """(chain id, model, start parameters)."""
    B = {a: json.loads((ROOT/f"results/streams/profile_step2_20261020/runs/final_a{a}_0.json").read_text())["params"] for a in ("12.5", "17.5", "25")}
    F = json.loads((ROOT/"results/streams/fixedbar_20261021/result.json").read_text())
    cat = dict(dist=5.43, pmra=MU_SYS[0], pmdec=MU_SYS[1], vc=233., halo_q=1.0)
    S = [("nobar_from12.5", "nobar", dict(B["12.5"], disc=1.)), ("nobar_fromfb44", "nobar", dict(F["from25_om42"]["params"], disc=1.)),
         ("nobar_fromfb33", "nobar", dict(F["from25"]["params"], disc=0.8)), ("nobar_cat", "nobar", dict(cat, disc=1.2)),
         ("mcmillan_cat", "mcmillan", cat), ("mcmillan_from12.5", "mcmillan", B["12.5"])]
    for a in ("12.5", "17.5"):
        S += [(f"pub{a}_om{B[a]['omega']:.0f}", f"pub{a}", B[a]), (f"pub{a}_om44", f"pub{a}", dict(B[a], omega=44.))]
    S += [(f"static_ang{g}", "static", dict(B["25"], angle=float(g))) for g in (25, 70, 115, 160)]
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--nrel", type=int, default=64000); ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    from profile_nm import NM
    (OUT/"runs").mkdir(parents=True, exist_ok=True)
    S = [s for s in starts() if s[1] in a.models]
    if a.smoke:
        first = {}
        for cid, model, p in S:
            first.setdefault(model, (cid, p))
        with ProcessPoolExecutor(a.workers) as ex:
            fs = {m: ex.submit(evaluate, (m, list(map(float, to_unit(m, p))), 1, 0.1, 32000, "")) for m, (cid, p) in first.items()}
            for m, f in fs.items():
                r = f.result(); print(f"smoke {m:9s} 32k lnL+prior {r['total_jk']+r['ln_prior']:.0f} ({r['seconds']:.0f} s) "
                                      + " ".join(f"{k} {v:.3f}" for k, v in r["params"].items()), flush=True)
        return
    chains = []
    for i, (cid, model, p) in enumerate(S):
        seed = 2600000+100*i; nd = len(MODELS[model]["free"])
        ch = NM(cid, None, to_unit(model, p), seed, float(np.random.default_rng([seed, 777]).uniform()), a.budget, ndim=nd, full=lambda z: z)
        ch.model = model; chains.append(ch)
    queue = {}

    def submit(ex, ch):
        for tag, z in ch.ask():
            k = ch.n_eval+len([1 for v in queue.values() if v[0] is ch])
            queue[ex.submit(evaluate, (ch.model, list(map(float, z)), ch.seed, ch.off, a.nrel, f"{ch.cid}_e{k:03d}"))] = (ch, tag, z)
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
            (OUT/"state.json").write_text(json.dumps({ch.cid: dict(model=ch.model, n_eval=ch.n_eval,
                                                                   best=float(-np.nanmin(ch.F)) if np.isfinite(ch.F).any() else None,
                                                                   best_params=to_params(ch.model, ch.S[int(np.nanargmin(ch.F))]) if np.isfinite(ch.F).any() else None)
                                                      for ch in chains}, indent=1, default=float))
        futs = []
        for i, ch in enumerate(chains):
            zb = ch.S[int(np.nanargmin(ch.F))]
            for k in range(4):
                seed = 2650000+100*i+k
                futs.append((ch, ex.submit(evaluate, (ch.model, list(map(float, zb)), seed, float(np.random.default_rng([seed, 777]).uniform()),
                                                      256000, f"final_{ch.cid}_{k}"))))
        res = {}
        for ch, f in futs:
            r = f.result(); res.setdefault(ch.cid, dict(model=ch.model, params=r["params"], totals=[]))["totals"].append(r["total_jk"]+r["ln_prior"])
    for cid, d in res.items():
        y = np.array(d["totals"]); d.update(mean=float(y.mean()), se=float(y.std(ddof=1)/2))
        print(f"{cid:20s} 256k {d['mean']:.0f} +- {d['se']:.0f}  " + " ".join(f"{k} {v:.3f}" for k, v in d["params"].items()))
    (OUT/"result.json").write_text(json.dumps(res, indent=1, default=float)); print(f"done in {(time.time()-t0)/3600:.1f} h")


if __name__ == "__main__":
    main()
