#!/usr/bin/env python3
"""Knot chi2 vs particle number at one parameter point (2026-10-25): is chi2 converged at the ~173k particles of the final sprays?

Pools the NFINAL independent sprays of a fit_knot_nm.py chain's best point (same parameters, different seeds / epoch offsets), so
N runs up to 4 x 173k with no new spray; random subsets of size N (REP draws each, disjoint where possible) are scored with
stream_model_compare.measure/chi2 (same data vector, masks, Asimov fits).
Usage: PYTHONPATH=src python bin/streams/chi2_vs_npart.py [RUNGLOB] -> results/plot_data/chi2_vs_npart.json, plots/chi2_vs_npart.png
"""
import glob, json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
PAT = sys.argv[1] if len(sys.argv) > 1 else "results/streams/fitknot_20261025/runs/final_c0_?.npz"
sys.argv = [sys.argv[0]]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import stream_model_compare as MC

KEYS = ("l", "b", "pmra", "pmdec", "vlos", "d", "chi", "age")
FR, REP = (1/32, 1/16, 1/8, 1/4, 1/2, 1.), (4, 4, 4, 4, 2, 1)


def main():
    files = sorted(glob.glob(str(ROOT/PAT))); Z = [np.load(f) for f in files]
    P = {k: np.concatenate([z[k] for z in Z]) for k in KEYS}; n = P["l"].size; age_cut = float(Z[0]["age_cut_myr"])
    D = MC.data_vector(); O = MC.Obs(); rng = np.random.default_rng(11); out = []
    with tempfile.TemporaryDirectory() as td:
        for fr, rep in zip(FR, REP):
            perm = rng.permutation(n); m = int(fr*n)
            for r in range(rep):
                idx = perm[r*m:(r+1)*m] if (r+1)*m <= n else rng.choice(n, m, replace=False)
                f = Path(td)/"sub.npz"; np.savez(f, age_cut_myr=age_cut, **{k: P[k][idx] for k in KEYS})
                M = MC.measure(O, D, str(f), np.random.default_rng([r, m])); c = MC.chi2(D, M)
                out.append(dict(frac=fr, n_released=int(m), n_particles=M["n_particles"], **c)); print(out[-1], flush=True)
    res = ROOT/"results/plot_data/chi2_vs_npart.json"; res.write_text(json.dumps(dict(files=[str(Path(f).relative_to(ROOT)) for f in files], rows=out), indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True); N = np.array([o["n_particles"] for o in out])
    for k, col in (("total", "k"), ("sky_c", "C0"), ("sky_s", "C1"), ("pmra", "C2"), ("pmdec", "C3")):
        y = np.array([o[k] for o in out]); ax[0 if k == "total" else 1].plot(N, y, "o", color=col, label=k, alpha=0.8)
        u = np.unique(N.round(-3)); ax[0 if k == "total" else 1].plot([N[N.round(-3) == v].mean() for v in u], [y[N.round(-3) == v].mean() for v in u], "-", color=col)
    for a in ax:
        a.set(xscale="log", yscale="log", xlabel="particles kept (chi > 0, age < 700 Myr; before masks)", ylabel="knot chi2"); a.legend()
    ax[0].set_title("total chi2 (25 points); points = independent subsets"); ax[1].set_title("per observable")
    fig.suptitle(f"chi2 vs particle number at the best free-fit point ({Path(files[0]).stem[:-2]}, {len(files)} pooled sprays)")
    pf = ROOT/"plots/chi2_vs_npart.png"; fig.savefig(pf, dpi=85); print(pf.relative_to(ROOT))


if __name__ == "__main__":
    main()
