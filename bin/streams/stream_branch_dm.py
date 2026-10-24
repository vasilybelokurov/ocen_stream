#!/usr/bin/env python3
"""Distance offset of the upper sky branch relative to the main stream, in the main stream's distance frame (2026-10-24).

All stars (no CMD mask) with G0 -> G0 - dm_main(phi1) (current quadratic relation, gaia_cmd_mask.json), so the distance gradient
along the stream and the different phi1 weighting of the two windows drop out (user: the phi1 weighting is important).
Windows (y = dphi2 - sky centre): main |y - m| < HW, branch |y - m2| < HW (dphi2_sky_final tracks), phi1 12-22; PM selection for
BOTH = the main stream's box |pm - frozen track| < (1.5, 1.0) mas/yr. Hess diagram and per-pixel background (hess_window: quadratic
in y from 1-deg bands at 6 <= |y| < 14), summed over 1-deg phi1 slices. One offset per window from the cluster-template
Poisson profile likelihood (stream_dm_fit.profile, domain(0)).
Error calibration: template stars (cluster annulus, shifted by the fitted offset, as many as the fitted amplitude) injected into
null windows (|y - (m - 2.5)| and |y - (m - 3.5)| < HW, Poisson-resampled), refitted; error scale = sd of (fit - truth)/(half
Delta lnL = 0.5 interval).
Usage: PYTHONPATH=src python bin/streams/stream_branch_dm.py -> results/plot_data/stream_branch_dm.json, plots/stream_branch_dm.png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import stream_branch_compare as SB, stream_dm_fit as DMF
from gaia_cmd_mask import dm_of

SB.HW = HW = 1.0; LO, HI = 12., 22.; NREP = 60


def main():
    R, sc, m, m2, pm = SB.tracks(); P = json.loads((ROOT/"results/plot_data/gaia_cmd_mask.json").read_text())
    S, clus = DMF.load(); u = S["phi1"]; y = S["dphi2"]-sc(u); S["g0"] = S["g0"]-np.array(dm_of(P, u))
    sel = (np.abs(S["pmra"]-pm["pmra"](u)) < 1.5) & (np.abs(S["pmdec"]-pm["pmdec"](u)) < 1.0)
    wins = dict(main=m, branch=m2, null1=lambda x: m(x)-2.5, null2=lambda x: m(x)-3.5); H = {}
    for a0 in np.arange(LO, HI, 1.):
        for w, fn in wins.items():
            H[w] = H.get(w, 0)+np.stack(SB.hess_window(S, sel, y, fn(u), a0, a0+1))
    dom = DMF.domain(0.)["all"]; fit = {w: DMF.profile(H[w][0], H[w][1], clus, 0., dom) for w in wins}
    for w, r in fit.items():
        print(f"{w:6s}: N-B {H[w][0].sum()-H[w][1].sum():+6.0f} | dm offset {r['dm']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] A {r['A']:.0f} sig {r['sig']:.1f}{' EDGE' if r['edge'] else ''}", flush=True)
    # injection calibration
    rng = np.random.default_rng(7); cal = {}
    for w in ("main", "branch"):
        z = []
        for i in range(NREP):
            nw = ("null1", "null2")[i % 2]; Hn = rng.poisson(H[nw][0]).astype(float); dm_t = fit[w]["dm"]
            idx = rng.choice(len(clus["C"][0]), int(round(fit[w]["A"])), replace=False)
            Hn += np.histogram2d(clus["C"][0][idx], clus["C"][1][idx]+dm_t, (DMF.EC, DMF.EG))[0]
            r = DMF.profile(Hn, H[nw][1], clus, 0., dom); z.append((r["dm"]-dm_t)/max(0.5*(r["hi"]-r["lo"]), DMF.DGRID))
        z = np.array(z); cal[w] = dict(bias=float(np.mean(z)), scale=float(np.std(z))); print(f"injection {w}: z mean {z.mean():+.2f} sd {z.std():.2f} ({NREP} reps)", flush=True)
    e = {w: 0.5*(fit[w]["hi"]-fit[w]["lo"])*max(cal[w]["scale"], 1.) for w in ("main", "branch")}
    d = fit["branch"]["dm"]-fit["main"]["dm"]; ed = float(np.hypot(e["main"], e["branch"]))
    print(f"branch - main: {d:+.2f} +- {ed:.2f} mag (calibrated) = {d/ed:+.1f} sigma; distance ratio {10**(d/5):.3f}", flush=True)
    json.dump(dict(fit={w: {k: v for k, v in r.items() if k not in ("dms", "L")} for w, r in fit.items()}, calibration=cal, d_dm=d, d_dm_err=ed, hw=HW, phi1=(LO, HI)),
              open(ROOT/"results/plot_data/stream_branch_dm.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True); ext = (DMF.EC[0], DMF.EC[-1], DMF.EG[0], DMF.EG[-1])
    cc, cg = 0.5*(DMF.EC[1:]+DMF.EC[:-1]), 0.5*(DMF.EG[1:]+DMF.EG[:-1])
    for a, w in zip(ax, ("main", "branch", "null1")):
        a.imshow(gaussian_filter(H[w][0]-H[w][1], 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8, extent=ext)
        T = DMF.template(clus, fit[w]["dm"]); a.contour(cc, cg, T.T, levels=[0.25*T.max()], colors="k", linewidths=0.8); a.invert_yaxis()
        a.set(xlabel=r"$(BP-RP)_0$", ylabel=r"$G_0$ - dm$_{main}(\phi_1)$", title=f"{w}: N-B {H[w][0].sum()-H[w][1].sum():.0f}; offset {fit[w]['dm']:+.2f} (template contour)")
    for w, col in (("main", "k"), ("branch", "C3"), ("null1", "0.6")):
        ax[3].plot(fit[w]["dms"], fit[w]["L"], color=col, label=f"{w}: {fit[w]['dm']:+.2f} ({fit[w]['sig']:.1f}$\\sigma$)")
    ax[3].set(ylim=(-10, 0.5), xlabel="dm offset from the main-stream relation [mag]", ylabel=r"$\Delta\ln L$", title=f"branch - main {d:+.2f} $\\pm$ {ed:.2f} (calibrated)"); ax[3].legend()
    fig.suptitle(f"Upper branch vs main stream in the main stream's distance frame, phi1 {LO:g}-{HI:g}, windows +-{HW:g} deg, main-stream PM box")
    fig.savefig(ROOT/"plots/stream_branch_dm.png", dpi=90); print("plots/stream_branch_dm.png")


if __name__ == "__main__":
    main()
