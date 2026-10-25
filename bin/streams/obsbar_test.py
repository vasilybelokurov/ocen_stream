#!/usr/bin/env python3
"""Can a bar consistent with the observations fit the stream, missed by the earlier fits? (user request 2026-10-22)

Step 1 (grid): the published bar (A = S = 1) at angles 22, 25, 27, 29, 32 deg and Omega_b = 20, 21, ..., 50 km/s/kpc; nuisances
  fixed at the best published-bar fit of the nearest fitted angle (22 -> 17.5-deg fit, others -> 27-deg fit; null_fits.py,
  fixedbar_fit.py). 32k release epochs, one seed for all points (common random numbers), score + priors as the profiles.
Step 2 (refit): the 4 best grid peaks (local maxima in Omega_b, at least 3 apart within an angle) refitted with the angle fixed
  and free: Omega_b (peak +-3), A 0.8-1.2, S 0.9-1.1 (observationally allowed range, agreed 2026-10-22), d, PMs, v_c, q (ranges
  of null_fits.NUIS). profile_nm.NM, 64k search with budget 40, then 4 x 256k at each chain's best point.
Decision rule (stated before running): no refitted peak within ~500 of the best strong-bar fit (-20598) -> the published-type
  bar cannot fit; otherwise the earlier fits missed it.
Output: results/streams/obsbar_20261022/{grid.json, peaks.json, runs/, state.json, result.json}
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/obsbar_test.py --workers 8 [--step grid|refit|all]
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/obsbar_20261022"
ANGLES = (22., 25., 27., 29., 32.)
OMEGAS = np.arange(20., 50.01, 1.)
REF = -20598.4
from null_fits import NUIS   # noqa: E402  (same nuisance ranges as the comparison fits)


def nuisances(angle):
    nf = json.loads((ROOT/"results/streams/nullfits_20261022/result.json").read_text())
    fb = json.loads((ROOT/"results/streams/fixedbar_20261021/result.json").read_text())
    p = max((v for v in nf.values() if v["model"] == "pub17.5"), key=lambda v: v["mean"])["params"] if angle < 23.5 else fb["from25_om42"]["params"]
    return {k: p[k] for k in ("dist", "pmra", "pmdec", "vc", "halo_q")}


def free_spec(peak):
    o = peak["omega"]
    return [("omega", max(20., o-3.), min(50., o+3.)), ("amp", 0.8, 1.2), ("size", 0.9, 1.1)]+NUIS


def to_params(peak, z):
    z = np.clip(np.asarray(z, float), 0, 1)
    return dict(angle=peak["angle"], **{k: lo+(hi-lo)*zi for (k, lo, hi), zi in zip(free_spec(peak), z)})


def to_unit(peak, p):
    return np.clip([(p[k]-lo)/(hi-lo) for k, lo, hi in free_spec(peak)], 0, 1)


def evaluate(args):
    p, seed, off, nrel, rid = args
    jf = OUT/"runs"/f"{rid}.json"
    if jf.exists():
        return json.loads(jf.read_text())
    from grid4_common import make_spray, win, project_gc, GC
    from ocen_stream.frames import frame_r0
    from ocen_stream.score import score_conditional
    import rescore_batch as RB
    from profile_nm import ln_prior
    t0 = time.time()
    m = make_spray(p["omega"], p["angle"], p["amp"], p["dist"], nrel=nrel, seed=seed, pm=(p["pmra"], p["pmdec"]),
                   host_kw=dict(bar_size=p["size"], axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=frame_r0("gravity21"))),
                   max_age_myr=700., epoch_offset=off, frame="gravity21")
    ms = {k: np.asarray(m[k], np.float64) for k in ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")}; ms["age_cut_myr"] = np.float64(700.)
    if nrel >= 64000:
        np.savez_compressed(OUT/"runs"/f"{rid}_tmp.npz", **ms); (OUT/"runs"/f"{rid}_tmp.npz").rename(OUT/"runs"/f"{rid}.npz")
    c, _, ob = RB.catalogue("real"); BG, VB = RB.bounds("data"); mu, mx = project_gc(ms["l"], ms["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, ms["pmra"], ms["pmdec"])), ms["vlos"],
                          ms["chi"], ms["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    rec = dict(run_id=rid, params=p, seed=seed, epoch_offset=off, nrel=nrel, total_jk=float(s["total_jk"]),
               ln_prior=float(ln_prior(p, True)), seconds=time.time()-t0)
    jf.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def grid(workers):
    jobs = [(dict(angle=a, omega=float(o), amp=1., size=1., **nuisances(a)), 3100001, 0.37, 32000, f"grid_a{a:g}_om{o:g}")
            for a in ANGLES for o in OMEGAS]
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(evaluate, jobs))
    G = {f"{a:g}": [next(r["total_jk"]+r["ln_prior"] for r in recs if r["params"]["angle"] == a and r["params"]["omega"] == o) for o in OMEGAS]
         for a in ANGLES}
    (OUT/"grid.json").write_text(json.dumps(dict(omega=OMEGAS.tolist(), lnL=G), indent=1))
    peaks = []
    for a, y in G.items():
        y = np.array(y)
        for i in np.argsort(y)[::-1]:
            if (i == 0 or y[i] >= y[i-1]) and (i == len(y)-1 or y[i] >= y[i+1]) and all(abs(OMEGAS[i]-p["omega"]) >= 3 for p in peaks if p["angle"] == float(a)):
                peaks.append(dict(angle=float(a), omega=float(OMEGAS[i]), lnL32k=float(y[i])))
    peaks = sorted(peaks, key=lambda p: -p["lnL32k"])
    print("grid (32k, nuisances fixed) best per angle: " + ", ".join(f"{a}: {max(y):.0f} at {OMEGAS[int(np.argmax(y))]:g}" for a, y in G.items()))
    print("top peaks: " + ", ".join(f"{p['angle']:g}/{p['omega']:g} {p['lnL32k']:.0f}" for p in peaks[:8]), flush=True)
    (OUT/"peaks.json").write_text(json.dumps(peaks[:4], indent=1))
    return peaks[:4]


def refit(workers, budget=40):
    from profile_nm import NM
    peaks = json.loads((OUT/"peaks.json").read_text()); chains = []
    for i, pk in enumerate(peaks):
        p0 = dict(angle=pk["angle"], omega=pk["omega"], amp=1., size=1., **nuisances(pk["angle"]))
        seed = 3200000+100*i; ch = NM(f"peak{i}_a{pk['angle']:g}_om{pk['omega']:g}", None, to_unit(pk, p0), seed,
                                     float(np.random.default_rng([seed, 777]).uniform()), budget, ndim=len(free_spec(pk)), full=lambda z: z)
        ch.peak = pk; chains.append(ch)
    queue = {}

    def submit(ex, ch):
        for tag, z in ch.ask():
            k = ch.n_eval+len([1 for v in queue.values() if v[0] is ch])
            queue[ex.submit(evaluate, (to_params(ch.peak, z), ch.seed, ch.off, 64000, f"{ch.cid}_e{k:03d}"))] = (ch, tag, z)
    with ProcessPoolExecutor(workers) as ex:
        for ch in chains:
            submit(ex, ch)
        while queue:
            done, _ = wait(list(queue), return_when=FIRST_COMPLETED)
            for f in done:
                ch, tag, z = queue.pop(f); rec = f.result(); ch.tell(tag, np.asarray(z), -(rec["total_jk"]+rec["ln_prior"]))
                busy = any(v[0] is ch for v in queue.values())
                if not busy and not ch.done():
                    submit(ex, ch)
                elif not busy:
                    print(f"{time.strftime('%H:%M:%S')} chain {ch.cid} done: {ch.n_eval} evals, best {-np.nanmin(ch.F):.0f}", flush=True)
            (OUT/"state.json").write_text(json.dumps({ch.cid: dict(n_eval=ch.n_eval, best=float(-np.nanmin(ch.F)) if np.isfinite(ch.F).any() else None,
                                                                   best_params=to_params(ch.peak, ch.S[int(np.nanargmin(ch.F))]) if np.isfinite(ch.F).any() else None)
                                                      for ch in chains}, indent=1, default=float))
        futs = []
        for i, ch in enumerate(chains):
            pb = to_params(ch.peak, ch.S[int(np.nanargmin(ch.F))])
            for k in range(4):
                seed = 3250000+100*i+k
                futs.append((ch.cid, ex.submit(evaluate, (pb, seed, float(np.random.default_rng([seed, 777]).uniform()), 256000, f"final_{ch.cid}_{k}"))))
        res = {}
        for cid, f in futs:
            r = f.result(); res.setdefault(cid, dict(params=r["params"], totals=[]))["totals"].append(r["total_jk"]+r["ln_prior"])
    for cid, d in res.items():
        y = np.array(d["totals"]); d.update(mean=float(y.mean()), se=float(y.std(ddof=1)/2), delta=float(y.mean()-REF))
        print(f"{cid:22s} 256k {d['mean']:.0f} +- {d['se']:.0f} (vs strong-bar best {d['delta']:+.0f})  "
              + " ".join(f"{k} {v:.3f}" for k, v in d["params"].items()))
    (OUT/"result.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--step", default="all")
    a = ap.parse_args(); (OUT/"runs").mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.step in ("grid", "all"):
        grid(a.workers)
    if a.step in ("refit", "all"):
        refit(a.workers)
    print(f"done in {(time.time()-t0)/3600:.1f} h")


if __name__ == "__main__":
    main()
