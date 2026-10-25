#!/usr/bin/env python3
"""Stage 0 of the profile route: calibrate the likelihood temperature T on the T2 mock (2026-10-20).

The score treats stars as more independent than they are, so ln L differences between bar angles are overconfident by a
factor T. Here T is measured where the truth is known:
  1. K new catalogues from the stored T2 generating spray (mock_data.draw, wide box, sampling seeds [700000, 2, k]);
     like the original mocks they keep the observed along-stream positions and redraw the other observables.
  2. The 24 stored T2c0 profile finals (6 angles x 4 seeds, results/streams/profile_T2c0_20261019) are re-scored against
     every catalogue with per-star jackknife ln L (score_conditional, cutoff 7, G 10; outside_bounds='zero', because the
     sampler keeps the few draws that fall outside the wide box, where only the signal term exists). Sprays are fixed, so the spread of
     Delta(angle) = lnL(angle) - lnL(27) across catalogues is pure catalogue-sampling noise.
  3. Per angle: true sampling variance Var_k(Delta); the block-bootstrap (4" blocks along u) and star-bootstrap variance
     estimates on each catalogue (as bin/streams/segment_profile.py); T_mock = Var_k(Delta) / (2 |mean_k Delta|) (= 1 for a
     correctly scaled likelihood in the quadratic regime); c_block = Var_k(Delta) / mean_k(block variance).
  4. Real data: T_real = c_block x block variance / (2 |Delta|) from results/plot_data/segment_profile_real.json.
Output: results/plot_data/tempering_calibration.json, plots/tempering_calibration.png
Usage: OMP_NUM_THREADS=1 PYTHONPATH=src python bin/streams/tempering_calibration.py --k 20 --workers 4
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
OUT = ROOT/"results/streams/tempering_cal_20261020"
PROF = ROOT/"results/streams/profile_T2c0_20261019"


def make_catalogue(k):
    import mock_data as MD
    f = OUT/f"cat_{k:02d}.npz"
    if not f.exists():
        m = dict(np.load(MD.OUT/"T2"/"truth_spray.npz"))
        cat = MD.draw(m, np.random.default_rng([700000, 2, k]), box="wide")
        np.savez_compressed(OUT/f"cat_{k:02d}_tmp.npz", **{a: v for a, v in cat.items() if isinstance(v, np.ndarray)})
        (OUT/f"cat_{k:02d}_tmp.npz").rename(f)
    return str(f)


def score(args):
    cat, spray = args
    import rescore_batch as RB
    from grid4_common import win, project_gc, GC
    from ocen_stream.score import score_conditional
    c, _, _ = RB.catalogue(cat); BG, VB = RB.bounds("wide")
    m = dict(np.load(spray)); mu, mx = project_gc(m["l"], m["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"],
                          m["chi"], m["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds="zero")   # draws may land outside the box: signal only there
    return cat, spray, np.asarray(s["per_star_jk"])


def block_var(D, u, blen, rng, n=1000):
    order = np.argsort(u); us, Ds = u[order], D[order]
    if blen == 0:
        return float(np.var([Ds[rng.integers(0, len(us), len(us))].sum() for _ in range(n)]))
    edges = np.arange(us.min(), us.max()+blen, blen); bid = np.digitize(us, edges); blocks = [np.flatnonzero(bid == b) for b in np.unique(bid)]
    return float(np.var([Ds[np.concatenate([blocks[p] for p in rng.integers(0, len(blocks), len(blocks))])].sum() for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=20); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    finals = sorted((PROF/"runs").glob("final_a*_*.json")); sprays = [str(f.with_suffix(".npz")) for f in finals]
    ang_of = {s: float(Path(s).stem.split("_")[1][1:]) for s in sprays}
    with ProcessPoolExecutor(a.workers) as ex:
        cats = list(ex.map(make_catalogue, range(a.k)))
        print(f"{len(cats)} catalogues; scoring {len(cats)*len(sprays)} (catalogue, spray) pairs", flush=True)
        cache = OUT/"per_star_jk.npz"
        if cache.exists():
            z = dict(np.load(cache, allow_pickle=True)); res = list(zip(z["cat"], z["spray"], z["v"]))
        else:
            res = list(ex.map(score, [(c, s) for c in cats for s in sprays], chunksize=4))
            np.savez_compressed(cache, cat=np.array([r[0] for r in res]), spray=np.array([r[1] for r in res]), v=np.array([r[2] for r in res]))
    per = {}
    for c, s, v in res:
        per.setdefault(str(c), {}).setdefault(ang_of[str(s)], []).append(np.asarray(v, float))
    cats = [str(c) for c in cats]; n_dropped = {}
    angles = sorted(set(ang_of.values())); ref = 27.0
    c0 = dict(np.load(cats[0])); u = c0["du"]; rng = np.random.default_rng(5)
    D = {x: [] for x in angles}; vb = {x: [] for x in angles}; vs = {x: [] for x in angles}
    for c in cats:
        L = {x: np.mean(per[c][x], axis=0) for x in angles}
        good = np.all([np.isfinite(L[x]) for x in angles], axis=0)        # same stars for every angle
        n_dropped[Path(c).name] = int((~good).sum())
        for x in angles:
            d = (L[x]-L[ref])[good]; D[x].append(d.sum())
            if x != ref:
                vb[x].append(block_var(d, u[good], 4., rng)); vs[x].append(block_var(d, u[good], 0, rng))
    out = dict(k=a.k, angles=angles, ref=ref, per_angle={}, stars_dropped_nonfinite=n_dropped)
    print("stars dropped (non-finite lnL for some angle):", {k_: v for k_, v in n_dropped.items() if v}, flush=True)
    print(f"T2 mock (truth 27): Delta = lnL(angle) - lnL(27) over {a.k} catalogues")
    for x in angles:
        if x == ref:
            continue
        d = np.array(D[x]); vt = float(d.var(ddof=1)); mb, ms = float(np.mean(vb[x])), float(np.mean(vs[x]))
        out["per_angle"][x] = dict(mean_delta=float(d.mean()), var_true=vt, var_block4=mb, var_star=ms, T_mock=vt/(2*abs(d.mean())),
                                   c_block=vt/mb, c_star=vt/ms)
        r = out["per_angle"][x]
        print(f"  {x:5.1f}: mean {r['mean_delta']:+8.0f}  sd_true {np.sqrt(vt):6.0f}  sd_block4 {np.sqrt(mb):6.0f}  sd_star {np.sqrt(ms):5.0f}  "
              f"T_mock {r['T_mock']:6.2f}  Var_true/Var_block {r['c_block']:.2f}  Var_true/Var_star {r['c_star']:.2f}", flush=True)
    # real data
    sp = json.loads((ROOT/"results/plot_data/segment_profile_real.json").read_text()); ia = sp["angles"]; ib = ia.index(sp["best"])
    cb = float(np.median([r["c_block"] for r in out["per_angle"].values()])); out["c_block_median"] = cb; out["real"] = {}
    print(f"real data (step-1 profile, best {sp['best']}): calibrated block variance = {cb:.2f} x block-4 variance")
    for i, x in enumerate(ia):
        if i == ib:
            continue
        dl = sp["total"][i]-sp["total"][ib]; sb = sp["bootstrap"]["block_4deg"][i]
        T = cb*sb**2/(2*abs(dl)); out["real"][x] = dict(delta=dl, sd_block4=sb, T_real=T, z_calibrated=abs(dl)/np.sqrt(cb*sb**2))
        print(f"  {x:5.1f}: Delta {dl:+6.0f}  sd_block4 {sb:5.0f}  T_real {T:6.1f}  significance |Delta|/sd_cal {out['real'][x]['z_calibrated']:.1f}")
    (ROOT/"results/plot_data/tempering_calibration.json").write_text(json.dumps(out, indent=1, default=float))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    xs = [x for x in angles if x != ref]
    for x in xs:
        ax[0].plot([x]*a.k, D[x], ".", color="0.6", ms=4)
    ax[0].errorbar(xs, [out["per_angle"][x]["mean_delta"] for x in xs], [np.sqrt(out["per_angle"][x]["var_true"]) for x in xs], fmt="ko", capsize=3,
                   label="mean +- sd over catalogues (true)")
    ax[0].errorbar(np.array(xs)+0.4, [out["per_angle"][x]["mean_delta"] for x in xs], [np.sqrt(out["per_angle"][x]["var_block4"]) for x in xs],
                   fmt="s", color="C1", capsize=3, label="block-bootstrap sd (mean)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set(xlabel="bar angle [deg]", ylabel="lnL(angle) - lnL(27)", title=f"T2 mock: {a.k} catalogues")
    ax[0].legend(fontsize=8)
    ax[1].plot(xs, [out["per_angle"][x]["T_mock"] for x in xs], "o-", label="T_mock (should be ~1)")
    ax[1].plot(xs, [out["per_angle"][x]["c_block"] for x in xs], "s-", label="Var_true / Var_block")
    ax[1].plot(xs, [out["per_angle"][x]["c_star"] for x in xs], "^-", label="Var_true / Var_star")
    ax[1].axhline(1, color="k", lw=0.5); ax[1].set(xlabel="bar angle [deg]", yscale="log", title="calibration ratios"); ax[1].legend(fontsize=8)
    fig.savefig(ROOT/"plots/tempering_calibration.png", dpi=110); print(ROOT/"plots/tempering_calibration.png")


if __name__ == "__main__":
    main()
