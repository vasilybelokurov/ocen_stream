#!/usr/bin/env python3
"""Block bootstrap of the grid4 conditional likelihood over members (no new sprays).
Members are cut into contiguous blocks along phi1 (u, chord GC frame) of width B deg; each resample draws as many blocks as there
are, with replacement, and sums the stored per-star lnL (results/plot_data/grid4_perstar.npz) for every grid model. B = 0 means
the i.i.d. star bootstrap (what the raw likelihood assumes). For each B: distribution of the best grid point (argmax), fraction
of resamples won by each bar angle (profile over the rest), and sd of Delta lnL between the best and best-at-20-deg models,
compared with the i.i.d. sd (ratio^2 ~ N/N_eff inflation). Also: autocorrelation length of per-star Delta lnL along u.
Plots: plots/bootstrap_grid4.png (cumulative Delta lnL(16 - 20 deg) along phi1; argmax histograms). Output results/plot_data/bootstrap_grid4.json.
Usage: python bin/streams/bootstrap_grid4.py [--nboot 2000]
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
NB = int(sys.argv[sys.argv.index("--nboot")+1]) if "--nboot" in sys.argv else 2000
z = np.load(ROOT/"results/plot_data/grid4_perstar.npz"); P, L, u = z["params"], z["lnL"], z["u"]
o = np.argsort(u); u, L = u[o], L[:, o]; N = len(u); tot = L.sum(1); ib = int(np.argmax(tot))
i20 = int(np.argmax(np.where(P[:, 1] == 20, tot, -np.inf)))
names = ["omega", "angle", "amp", "d"]; rng = np.random.default_rng(42)
dl = L[ib]-L[i20]                                           # per-star Delta lnL, best - best at 20 deg
print(f"best {P[ib]} lnL {tot[ib]:.1f}; best at 20 deg {P[i20]} Delta {tot[ib]-tot[i20]:.1f}")
# autocorrelation of dl along u (members sorted by u), in 0.25-deg lags using binned sums
edges = np.arange(u.min(), u.max()+0.25, 0.25); s = np.histogram(u, edges, weights=dl-dl.mean())[0]
ac = [np.corrcoef(s[:-k], s[k:])[0, 1] for k in range(1, 25)]
print("autocorr of binned (0.25 deg) Delta lnL at lags 0.25..2 deg:", np.round(ac[:8], 2))
out = dict(best=P[ib].tolist(), best20=P[i20].tolist(), dlnL=float(tot[ib]-tot[i20]), autocorr_0p25deg=[float(a) for a in ac], B={})
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].plot(u, np.cumsum(dl), "k-"); ax[0].set_xlabel(r"$\phi_1$ [deg]"); ax[0].set_ylabel(r"cumulative $\Delta\ln L$ (best - best at 20 deg)")
ax[0].axhline(0, color="0.6", lw=0.8)
for B in (0., 0.5, 1., 2., 4.):
    if B == 0:
        blocks = [np.array([i]) for i in range(N)]
    else:
        bid = np.floor((u-u.min())/B).astype(int); blocks = [np.where(bid == k)[0] for k in np.unique(bid)]
    S = np.array([L[:, b].sum(1) for b in blocks]).T          # (models, blocks)
    nb = len(blocks); arg = np.empty(NB, int); dd = np.empty(NB); angwin = np.zeros(4)
    for r in range(NB):
        c = np.bincount(rng.integers(0, nb, nb), minlength=nb); t = S @ c
        arg[r] = np.argmax(t); dd[r] = t[ib]-t[i20]
        prof = [t[P[:, 1] == a].max() for a in (16., 20., 24., 28.)]; angwin[int(np.argmax(prof))] += 1
    sd_iid = out["B"]["0.0"]["sd_dlnL"] if B > 0 else dd.std()
    q = {n: np.percentile(P[arg, k], [16, 50, 84]).tolist() for k, n in enumerate(names)}
    out["B"][str(B)] = dict(nblocks=nb, sd_dlnL=float(dd.std()), frac_dlnL_le0=float(np.mean(dd <= 0)), inflation=float((dd.std()/sd_iid)**2),
                            angle_win_frac=dict(zip(["16", "20", "24", "28"], (angwin/NB).tolist())), argmax_16_50_84=q,
                            frac_best_is_global=float(np.mean(arg == ib)))
    e = out["B"][str(B)]
    print(f"B={B:3.1f} deg ({nb} blocks): sd(dlnL) {e['sd_dlnL']:.1f} (inflation vs iid {e['inflation']:.1f}); P(dlnL<=0) {e['frac_dlnL_le0']:.3f}; "
          f"angle wins {e['angle_win_frac']}; argmax 16/50/84%: " + "; ".join(f"{n} {q[n]}" for n in names), flush=True)
    if B in (0., 2.):
        for k, (n, a) in enumerate(zip(("omega", "angle"), (ax[1], ax[2]))):
            vals = np.unique(P[:, k if n == "omega" else 1]); h = [np.mean(P[arg, k if n == "omega" else 1] == v) for v in vals]
            a.bar(np.arange(len(vals))+(0.2 if B else -0.2), h, width=0.4, label=f"B = {B:g} deg" + (" (i.i.d. stars)" if B == 0 else ""))
            a.set_xticks(range(len(vals)), [f"{v:g}" for v in vals]); a.set_xlabel(n); a.set_ylabel("fraction of resamples (best grid point)")
ax[1].legend(); ax[2].legend()
fig.suptitle("Block bootstrap of the grid4 conditional likelihood (members resampled in phi1 blocks)"); fig.tight_layout()
fig.savefig(ROOT/"plots/bootstrap_grid4.png", dpi=90)
(ROOT/"results/plot_data/bootstrap_grid4.json").write_text(json.dumps(out, indent=1)); print("plots/bootstrap_grid4.png")
