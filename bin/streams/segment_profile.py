#!/usr/bin/env python3
"""Where the bar-angle signal lives, and how firm it is (Fable step 1, 2026-10-20).

For a profile run (profile_nm.py output: final_a<angle>_<k> 256k sprays, 4 seeds per angle), re-score every final spray
against its catalogue keeping the per-star jackknife ln L (score_conditional cutoff 7, G 10), average over the seeds per
angle, and report:
  - the profile per stream segment (u < 4, 4-17, 17-23, > 23 deg, chord frame) and with u > 20 removed;
  - block-bootstrap uncertainty of the difference between each angle and the best angle: resample contiguous blocks of
    members along u (block lengths 2, 4, 6 deg; 2000 draws), compared with the star bootstrap (blocks of one star).
The measurement priors (as in the run) are added to the angle totals.
Output: results/plot_data/segment_profile_<tag>.json, plots/segment_profile_<tag>.png
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/segment_profile.py --profile DIR --tag real
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
SEG = [(-99, 4), (4, 17), (17, 23), (23, 99)]


def score_one(args):
    npz, spec = args
    import rescore_batch as RB
    from grid4_common import win, project_gc, GC
    from ocen_dm.streams.score import score_conditional
    c, _, ob = RB.catalogue("real" if spec["catalogue"] == "real" else str(ROOT/spec["catalogue"]))
    BG, VB = RB.bounds(spec["box"]); ob = "raise" if spec["box"] == "wide" else ob
    m = dict(np.load(npz)); mu, mx = project_gc(m["l"], m["b"], GC)
    s = score_conditional(c["du"], c["dw"], c["dv"], c["dev"], c["cov"], mu, np.column_stack((mx, m["pmra"], m["pmdec"])), m["vlos"],
                          m["chi"], m["age"], u_window=win, w_bounds=BG, v_bounds=VB, chi_edges=RB.E, cutoff=RB.CUTOFF,
                          jackknife_groups=RB.G, chunk=50, outside_bounds=ob)
    return str(npz), np.asarray(s["per_star_jk"]), float(s["total_jk"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    P = Path(a.profile) if Path(a.profile).is_absolute() else ROOT/a.profile
    man = json.loads((P/"manifest.json").read_text()); spec = man["spec"]
    from profile_nm import ln_prior
    from grid4_common import du
    finals = sorted((P/"runs").glob("final_a*_*.json"))
    jobs = [(str(f.with_suffix(".npz")), spec) for f in finals]
    with Pool(a.workers) as pool:
        res = dict((k, (v, t)) for k, v, t in pool.map(score_one, jobs))
    per, prior = {}, {}
    for f in finals:
        r = json.loads(f.read_text()); ang = float(f.stem.split("_")[1][1:])
        v, t = res[str(f.with_suffix(".npz"))]
        assert abs(t-r["total_jk"]) < 1e-2, (f, t, r["total_jk"])
        per.setdefault(ang, []).append(v); prior[ang] = ln_prior(r["params"], man.get("vc_prior", False)) if man.get("prior") else 0.
    ang = sorted(per); L = np.array([np.mean(per[x], axis=0) for x in ang])          # (n_angle, n_star)
    tot = L.sum(1)+np.array([prior[x] for x in ang]); ib = int(np.argmax(tot))
    out = dict(tag=a.tag, angles=ang, total=tot.tolist(), best=ang[ib], segments={}, without_u_gt_20=None, bootstrap={})
    for lo, hi in SEG:
        k = (du >= lo) & (du < hi); out["segments"][f"{lo}..{hi}"] = dict(n=int(k.sum()), lnL=L[:, k].sum(1).tolist())
    k = du <= 20; out["without_u_gt_20"] = (L[:, k].sum(1)+np.array([prior[x] for x in ang])).tolist()
    rng = np.random.default_rng(1); D = L-L[ib]                                  # per-star differences vs the best angle
    order = np.argsort(du); us = du[order]; Ds = D[:, order]
    for blen in (0, 2, 4, 6):
        if blen == 0:
            draws = [Ds[:, rng.integers(0, len(us), len(us))].sum(1) for _ in range(2000)]
        else:
            edges = np.arange(us.min(), us.max()+blen, blen); bid = np.digitize(us, edges); blocks = [np.flatnonzero(bid == b) for b in np.unique(bid)]
            draws = []
            for _ in range(2000):
                pick = rng.integers(0, len(blocks), len(blocks)); idx = np.concatenate([blocks[p] for p in pick]); draws.append(Ds[:, idx].sum(1))
        sd = np.std(draws, axis=0); out["bootstrap"][f"block_{blen}deg" if blen else "star"] = sd.tolist()
    (ROOT/f"results/plot_data/segment_profile_{a.tag}.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax[0].errorbar(ang, tot-tot[ib], out["bootstrap"]["block_4deg"], fmt="o-", capsize=3, label="all members (4-deg block bootstrap)")
    ax[0].errorbar(np.array(ang)+0.3, tot-tot[ib], out["bootstrap"]["star"], fmt="s", ms=3, capsize=2, label="star bootstrap")
    w = np.array(out["without_u_gt_20"]); ax[0].plot(ang, w-w.max(), "^--", label="without phi1 > 20")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set(xlabel="bar angle [deg]", ylabel="ln L - best", title=f"({a.tag}) profile and its uncertainty")
    ax[0].legend(fontsize=7)
    for lo, hi in SEG:
        s = np.array(out["segments"][f"{lo}..{hi}"]["lnL"]); ax[1].plot(ang, s-s.max(), "o-", label=f"phi1 {lo}..{hi} (n={out['segments'][f'{lo}..{hi}']['n']})")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set(xlabel="bar angle [deg]", ylabel="segment ln L - its max", title="profile by stream segment")
    ax[1].legend(fontsize=7)
    fig.savefig(ROOT/f"plots/segment_profile_{a.tag}.png", dpi=110)
    print(f"{a.tag}: best {ang[ib]} | total-best: " + ", ".join(f"{x:g}: {t-tot[ib]:+.0f}" for x, t in zip(ang, tot)))
    for k_, v in out["bootstrap"].items():
        print(f"  sd ({k_}): " + ", ".join(f"{x:g}: {s:.0f}" for x, s in zip(ang, v)))
    for k_, v in out["segments"].items():
        s = np.array(v["lnL"]); print(f"  segment {k_:9s} (n={v['n']}): " + ", ".join(f"{x:g}: {t-s[ib]:+.0f}" for x, t in zip(ang, s)))
    w = np.array(out["without_u_gt_20"]); print("  without phi1 > 20: " + ", ".join(f"{x:g}: {t-w[ib]:+.0f}" for x, t in zip(ang, w)))


if __name__ == "__main__":
    main()
