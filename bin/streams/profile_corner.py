#!/usr/bin/env python3
"""Profile-likelihood corner of the 9-parameter stream model from stored evaluations (2026-10-22). No new sprays.

Points: every evaluation of the step-2 profile (results/streams/profile_step2_20261020/runs: 12 Nelder-Mead chains at 128k release
epochs + 4 x 256k finals per angle; gravity21 frame, distance band 5.33-5.53, step-2 priors), real data, data box.
Value: ln L (jackknife) + prior, relative to the best 128k point; the 256k finals are shifted by the mean 128k - 256k offset at the
same parameters so both sit on one scale (offset printed).
Lower triangle: points coloured by Delta lnL (clipped at -3000), the best point per 2-D cell plotted on top; diagonal: the upper
envelope (profile) of Delta lnL vs the parameter in 20 bins. Caveat: the points come from local optimisation chains, so the
surface is mapped only where the chains went; this is a profile over explored points, not a marginal posterior.
Output: plots/profile_corner_step2.png
Usage: PYTHONPATH=src python bin/streams/profile_corner.py [--dir results/streams/profile_step2_20261020]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
NAMES = ("angle", "omega", "amp", "size", "dist", "pmra", "pmdec", "vc", "halo_q")
LAB = dict(angle="bar angle [deg]", omega=r"$\Omega_b$", amp="A", size="S", dist="d [kpc]", pmra=r"$\mu_{\alpha*}$", pmdec=r"$\mu_\delta$",
           vc=r"$v_c$", halo_q="q")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/streams/profile_step2_20261020"); a = ap.parse_args()
    from profile_nm import ln_prior
    man = json.loads((ROOT/a.dir/"manifest.json").read_text()); vcp = man.get("vc_prior", False)
    R = [json.loads(f.read_text()) for f in (ROOT/a.dir/"runs").glob("*.json")]
    P = np.array([[r["params"][k] for k in NAMES] for r in R]); y = np.array([r["total_jk"]+ln_prior(r["params"], vcp) for r in R])
    n = np.array([r["nrel"] for r in R])
    fin = n == 256000
    # offset 128k -> 256k at the same parameters (each final point vs the chain's best 128k evaluation at that point)
    off = []
    for i in np.flatnonzero(fin):
        same = (~fin) & np.all(np.isclose(P, P[i], atol=1e-9), axis=1)
        if same.any():
            off.append(y[same].max()-y[i])
    o = float(np.mean(off)) if off else 0.
    y = np.where(fin, y+o, y); d = y-y.max()
    print(f"{len(R)} points ({fin.sum()} at 256k, shifted by {o:+.0f} to the 128k scale); best {y.max():.0f} at " +
          ", ".join(f"{k} {v:.2f}" for k, v in zip(NAMES, P[np.argmax(y)])))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(NAMES); fig, ax = plt.subplots(k, k, figsize=(18, 18), constrained_layout=True)
    order = np.argsort(d); dc = np.clip(d, -3000, 0)
    for i in range(k):
        for j in range(k):
            x_ = ax[i, j]
            if j > i:
                x_.axis("off"); continue
            if i == j:
                e = np.linspace(P[:, i].min(), P[:, i].max(), 21); c = 0.5*(e[1:]+e[:-1])
                env = [d[(P[:, i] >= e[b]) & (P[:, i] <= e[b+1])].max() if np.any((P[:, i] >= e[b]) & (P[:, i] <= e[b+1])) else np.nan for b in range(20)]
                x_.plot(c, env, "k.-"); x_.set(ylim=(-3000, 100)); x_.axhline(0, color="0.6", lw=0.5)
                x_.set_ylabel(r"max $\Delta$lnL" if j == 0 else None)
            else:
                sc = x_.scatter(P[order, j], P[order, i], c=dc[order], s=6, cmap="viridis", vmin=-3000, vmax=0)
                x_.plot(P[np.argmax(d), j], P[np.argmax(d), i], "r*", ms=10)
            if i == k-1:
                x_.set_xlabel(LAB[NAMES[j]])
            if j == 0 and i > 0:
                x_.set_ylabel(LAB[NAMES[i]])
    fig.colorbar(sc, ax=ax[1:, 0], shrink=0.4, label=r"$\Delta$lnL (ln L + prior, relative to the best)", location="left")
    fig.suptitle("Profile-likelihood corner (step-2 runs: 12 NM chains x 40 at 128k + 256k finals; gravity21, d band, priors). "
                 "Points = evaluated models; red star = best. Explored region only -- not a posterior.", fontsize=12)
    tag = Path(a.dir).name
    fig.savefig(ROOT/f"plots/profile_corner_{tag}.png", dpi=80); print(ROOT/f"plots/profile_corner_{tag}.png")


if __name__ == "__main__":
    main()
