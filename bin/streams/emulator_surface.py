#!/usr/bin/env python3
"""Likelihood surface of the 9-D stream emulator as corner plots of profiles (2026-10-18).

Target: emulator mean mu(theta) of the jackknife-corrected ln L (fixed chi bins 0-200), plus the measurement priors on omega
Cen's distance N(OCEN_DISTANCE_KPC, OCEN_DISTANCE_ERR_KPC) and PMs N(MU_SYS, MU_SYS_ERR) (src/ocen_stream/cluster.py). The
stream ln L is untempered (no calibration chosen yet), so it outweighs the priors by orders of magnitude; the distance prior
costs only 0.5 ((d - 5.43)/0.05)^2, e.g. 27 at 5.8 kpc.
Views: A) distance over the design range (5.25-5.95 kpc); B) distance restricted to the measured value +-1 sigma.
Each panel (i, j): maximum over the other parameters of mu + ln prior in 2-D bins, relative to the global maximum of the
view (profile, not marginal; finite candidate set: 2^17 scrambled-Sobol points + local points around the best measured
runs). Diagonal: 1-D profiles. A second figure shows the emulator sd at the maximising candidate (reliability).
Output: plots/emulator_corner_{A,B}.png, plots/emulator_corner_sd_{A,B}.png, results/plot_data/emulator_surface_20261018.json
Usage: OMP_NUM_THREADS=4 PYTHONPATH=src python bin/streams/emulator_surface.py --batch DIR ...
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
warnings.filterwarnings("ignore")
import emulator_design as ED      # noqa: E402
import emulator_report as ER      # noqa: E402
from ocen_stream.cluster import OCEN_DISTANCE_KPC, OCEN_DISTANCE_ERR_KPC, MU_SYS, MU_SYS_ERR   # noqa: E402

SHORT = dict(angle="angle [deg]", omega=r"$\Omega_b$", amp="amplitude", size="size", dist="d [kpc]", pmra=r"$\mu_{\alpha*}$",
             pmdec=r"$\mu_\delta$", vc=r"$v_c$ [km/s]", halo_q="halo q")


def ln_prior(P):
    return (-0.5*((P["dist"]-OCEN_DISTANCE_KPC)/OCEN_DISTANCE_ERR_KPC)**2
            - 0.5*((P["pmra"]-MU_SYS[0])/MU_SYS_ERR)**2 - 0.5*((P["pmdec"]-MU_SYS[1])/MU_SYS_ERR)**2)


def candidates(g, n_log2, dist_range, seed):
    from scipy.stats import qmc
    U = qmc.Sobol(9, scramble=True, seed=seed).random_base2(n_log2)
    rng = np.random.default_rng(seed)
    top = g["X"][np.argsort(-g["ybar"])[:20]]
    U = np.vstack((U, np.clip(np.repeat(top, 1000, axis=0)+rng.normal(0, 0.06, (20000, 9)), 0, 1)))
    lo, hi = [(x-5.25)/0.7 for x in dist_range]
    U[:, 4] = lo+U[:, 4]*(hi-lo) if (lo, hi) != (0., 1.) else U[:, 4]
    return U


def corner(P, val, sd, title, path, vmin=-3000.):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ED.NAMES; n = len(names); nb = 14
    rng_ = {k: (P[k].min(), P[k].max()) for k in names}
    fig, ax = plt.subplots(n, n, figsize=(15, 15)); out = {}
    fig2, ax2 = plt.subplots(n, n, figsize=(15, 15))
    for i in range(n):
        for j in range(n):
            a, a2 = ax[i, j], ax2[i, j]
            if j > i:
                a.axis("off"); a2.axis("off"); continue
            kx, ky = names[j], names[i]
            ex = np.linspace(*rng_[kx], nb+1)
            if i == j:
                ix = np.clip(np.digitize(P[kx], ex)-1, 0, nb-1); m = np.full(nb, np.nan); s = np.full(nb, np.nan)
                for b in range(nb):
                    k = ix == b
                    if k.any():
                        q = np.flatnonzero(k)[np.argmax(val[k])]; m[b], s[b] = val[q], sd[q]
                c = 0.5*(ex[1:]+ex[:-1])
                for aa, y in ((a, m), (a2, s)):
                    aa.plot(c, y, "k-"); aa.set_xlim(*rng_[kx])
                a.fill_between(c, m-2*s, m+2*s, color="0.85"); a.set_ylim(vmin, 300)
                out[kx] = dict(x=c.tolist(), dlnL=m.tolist(), sd=s.tolist())
            else:
                ey = np.linspace(*rng_[ky], nb+1)
                ix = np.clip(np.digitize(P[kx], ex)-1, 0, nb-1); iy = np.clip(np.digitize(P[ky], ey)-1, 0, nb-1)
                M = np.full((nb, nb), np.nan); S = np.full((nb, nb), np.nan)
                order = np.argsort(val)                       # last write per cell = maximum
                M[iy[order], ix[order]] = val[order]; S[iy[order], ix[order]] = sd[order]
                a.pcolormesh(ex, ey, M, vmin=vmin, vmax=0, cmap="viridis")
                a.contour(0.5*(ex[1:]+ex[:-1]), 0.5*(ey[1:]+ey[:-1]), M, levels=[-1000, -300], colors=["w", "r"], linewidths=0.7)
                a2.pcolormesh(ex, ey, S, vmin=0, vmax=1500, cmap="magma")
            for aa in (a, a2):
                if i == n-1:
                    aa.set_xlabel(SHORT[kx], fontsize=8)
                else:
                    aa.set_xticklabels([])
                if j == 0 and i != 0:
                    aa.set_ylabel(SHORT[ky], fontsize=8)
                elif i != j:
                    aa.set_yticklabels([])
                aa.tick_params(labelsize=6)
    fig.suptitle(title+"\ncolour: max ln L + ln prior relative to the best (viridis, -3000..0); contours -300 (red), -1000 (white); "
                 "diagonal: 1-D profile with 2 sd band", fontsize=10)
    fig2.suptitle(title+"\nemulator sd at the maximising candidate (0..1500); diagonal: sd along the 1-D profile", fontsize=10)
    fig.savefig(path, dpi=110); fig2.savefig(str(path).replace("corner_", "corner_sd_"), dpi=110)
    plt.close(fig); plt.close(fig2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", nargs="*", default=[]); a = ap.parse_args()
    g, s2, nm, em, info = ED.fit(a.batch)
    smax = ER.smax_interp(); res = dict(info=info, n_points=len(g["ybar"]), n_runs=int(g["r"].sum()), views={})
    sig = OCEN_DISTANCE_ERR_KPC
    for view, dr, ttl in (("A", (5.25, 5.95), "A: distance over the design range 5.25-5.95 kpc (with distance and PM priors)"),
                          ("B", (OCEN_DISTANCE_KPC-sig, OCEN_DISTANCE_KPC+sig),
                           f"B: distance at the measurement, {OCEN_DISTANCE_KPC-sig:.2f}-{OCEN_DISTANCE_KPC+sig:.2f} kpc")):
        U = candidates(g, 17, dr, seed=11 if view == "A" else 12)
        mu, sd = em.predict(U); P = ER.physical(U, smax); lp = ln_prior(P)
        val = mu+lp; best = int(np.argmax(val)); ref = val[best]
        prof = corner(P, val-ref, sd, ttl, ROOT/f"plots/emulator_corner_{view}.png")
        res["views"][view] = dict(dist_range=list(dr), best=dict(params={k: float(P[k][best]) for k in ED.NAMES}, mu=float(mu[best]),
                                  ln_prior=float(lp[best]), sd=float(sd[best])), profiles=prof)
        print(view, "best mu %.0f (+ ln prior %.1f), sd %.0f at" % (mu[best], lp[best], sd[best]),
              {k: round(float(P[k][best]), 2) for k in ED.NAMES})
    dA, dB = res["views"]["A"]["best"], res["views"]["B"]["best"]
    print("cost of the measured distance (best B - best A, mu + prior): %.0f" % ((dB["mu"]+dB["ln_prior"])-(dA["mu"]+dA["ln_prior"])))
    (ROOT/"results/plot_data/emulator_surface_20261018.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
