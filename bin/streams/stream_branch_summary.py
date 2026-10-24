#!/usr/bin/env python3
"""Summary figure: main stream vs upper sky branch -- sky, PMs, distance (2026-10-24).

Inputs (no refits): stream_spline_fit.json (dphi2_sky_final tracks), stream_branch_compare.json (per-bin 2-D PM fits),
stream_branch_pm_null.json (PM error calibration: errors x max(pull sd, 1)), stream_branch_dm.json (joint distance offset).
Recomputed for display only: the sky excess map (CMD pass + k = 2 PM masks, per-column quadratic background from
4 < |dphi2 - centre| < 12, as stream_spline_plot) and the main-frame Hess diagrams (as stream_branch_dm).
Panels: (a) sky excess with both tracks and the +-1 deg windows; (b, c) mu_ra*, mu_de offsets from the frozen main PM track;
(d) along-PM amplitude difference branch - main (calibrated); (e) dm offset likelihoods (main frame); (f, g) N - B CMDs in the main
stream's distance frame with the template contour at each fitted offset.
Usage: PYTHONPATH=src python bin/streams/stream_branch_summary.py -> plots/stream_branch_summary.png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import stream_branch_compare as SB, stream_dm_fit as DMF
from gaia_cmd_mask import dm_of

J = lambda n: json.loads((ROOT/f"results/plot_data/{n}.json").read_text())


def main():
    R, sc, m, m2, pm = SB.tracks(); G = np.array(R["grid"]); PC, NL, DM = J("stream_branch_compare"), J("stream_branch_pm_null"), J("stream_branch_dm")
    fig = plt.figure(figsize=(22, 10), constrained_layout=True); gs = fig.add_gridspec(2, 4)
    # (a) sky excess
    from stream_spline_plot import catalogue
    S = catalogue(); u, x = S["phi1"], S["dphi2"]; ev = lambda p, k="c": np.interp(u, G, R["tracks"][p][k])
    sel = np.ones(u.size, bool)
    for p in ("pmra", "pmdec"):
        sel &= np.abs(S[p]-ev(p)) < 2*np.sqrt(ev(p, "s")**2+S[p+"_error"]**2)
    E1, Ey = np.arange(0, 25.01, 1.), np.arange(-8, 8.01, 0.25); c1, cy = 0.5*(E1[1:]+E1[:-1]), 0.5*(Ey[1:]+Ey[:-1])
    y = x-sc(u); H = np.histogram2d(u[sel], y[sel], (E1, Ey))[0]; Bk = np.zeros_like(H)
    for j in range(len(c1)):
        off = (np.abs(cy) > 4) & (np.abs(cy) < 12); Bk[j] = np.polyval(np.polyfit(cy[off], H[j, off], 2), cy)
    X = gaussian_filter(H-Bk, 1.); X /= np.clip(X, 0, None).sum(1, keepdims=True)
    a = fig.add_subplot(gs[0, :2]); a.imshow(X.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.08, vmax=0.08, extent=(E1[0], E1[-1], Ey[0], Ey[-1]))
    g = np.linspace(0, 25, 251); g2 = np.linspace(10, 23, 131)
    a.plot(g, m(g), "k-", lw=1, label="main track"); a.plot(g2, m2(g2), "C3-", lw=1, label="branch track")
    for f_, col, rng in ((m, "k", (12, 22)), (m2, "C3", (12, 22))):
        gg = np.linspace(*rng, 101); a.plot(gg, f_(gg)+1, col+":", lw=0.8); a.plot(gg, f_(gg)-1, col+":", lw=0.8)
    a.set(xlabel=r"$\phi_1$ [deg]", ylabel=r"$\Delta\phi_2$ - sky centre [deg]", title="(a) sky excess (CMD + PM masks), column-normalised; dotted: +-1 deg windows used"); a.legend(loc="lower left")
    # (b, c) PM offsets per bin (calibrated errors)
    bins = PC["bins"][:2]; xc = np.array([0.5*sum(b["phi1"]) for b in bins]); sc_ = {q: [max(n[f"pull_{q}_sd"], 1.) for n in NL] for q in ("ra", "de")}
    for k, (q, lab) in enumerate((("ra", r"$\mu_{\alpha*}$"), ("de", r"$\mu_\delta$"))):
        a = fig.add_subplot(gs[0, 2+k])
        for j, (w, col) in enumerate((("main", "k"), ("branch", "C3"))):
            e = [b[w][f"mu_{q}_err"]*(sc_[q][i] if w == "branch" else 1.) for i, b in enumerate(bins)]
            a.errorbar(xc+0.25*j, [b[w][f"mu_{q}"] for b in bins], e, fmt="o", color=col, capsize=3, label=w)
        a.axhline(0, color="k", lw=0.5, ls=":"); a.set(xlabel=r"$\phi_1$ bin centre [deg]", ylabel=lab+" - frozen main track [mas/yr]",
                                                      title=f"({'bc'[k]}) {lab} offsets (branch errors calibrated)", xlim=(12, 20)); a.legend()
    # (d) along-PM amplitude difference
    a = fig.add_subplot(gs[1, 0]); D, E = [], []
    for i, b in enumerate(bins):
        c = 0.5*sum(b["phi1"]); A_, Dd = pm["pmra"](np.array([c]))[0]+b["main"]["mu_ra"], pm["pmdec"](np.array([c]))[0]+b["main"]["mu_de"]; ux, uy = np.array([A_, Dd])/np.hypot(A_, Dd)
        D.append((b["branch"]["mu_ra"]-b["main"]["mu_ra"])*ux+(b["branch"]["mu_de"]-b["main"]["mu_de"])*uy)
        E.append(np.hypot(ux*np.hypot(b["branch"]["mu_ra_err"]*sc_["ra"][i], b["main"]["mu_ra_err"]), uy*np.hypot(b["branch"]["mu_de_err"]*sc_["de"][i], b["main"]["mu_de_err"])))
    D, E = np.array(D), np.array(E); w_ = 1/E**2; Dm, Em = np.sum(w_*D)/w_.sum(), 1/np.sqrt(w_.sum())
    a.errorbar(xc, D, E, fmt="o", color="C3", capsize=3); a.axhline(0, color="k", lw=0.5, ls=":"); a.axhspan(Dm-Em, Dm+Em, color="C3", alpha=0.15)
    a.set(xlabel=r"$\phi_1$ bin centre [deg]", ylabel=r"$\Delta|\mu|$ branch - main [mas/yr]", title=f"(d) PM amplitude: combined {Dm:+.2f} $\\pm$ {Em:.2f} ({Dm/Em:+.1f}$\\sigma$)", xlim=(12, 20))
    # (e)-(g) distance in the main stream's frame
    S2, clus = DMF.load(); uS = S2["phi1"]; yS = S2["dphi2"]-sc(uS); S2["g0"] = S2["g0"]-np.array(dm_of(J("gaia_cmd_mask"), uS))
    selS = (np.abs(S2["pmra"]-pm["pmra"](uS)) < 1.5) & (np.abs(S2["pmdec"]-pm["pmdec"](uS)) < 1.0); SB.HW = DM["hw"]; Hh = {}
    for a0 in np.arange(*DM["phi1"], 1.):
        for w, fn in (("main", m), ("branch", m2)):
            Hh[w] = Hh.get(w, 0)+np.stack(SB.hess_window(S2, selS, yS, fn(uS), a0, a0+1))
    dom = DMF.domain(0.)["all"]; a = fig.add_subplot(gs[1, 1]); ext = (DMF.EC[0], DMF.EC[-1], DMF.EG[0], DMF.EG[-1]); cc, cg = 0.5*(DMF.EC[1:]+DMF.EC[:-1]), 0.5*(DMF.EG[1:]+DMF.EG[:-1])
    for w, col in (("main", "k"), ("branch", "C3")):
        r = DMF.profile(Hh[w][0], Hh[w][1], clus, 0., dom); a.plot(r["dms"], r["L"], color=col, label=f"{w}: {r['dm']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
    a.set(ylim=(-8, 0.5), xlabel=r"dm offset from the main-stream relation [mag]", ylabel=r"$\Delta\ln L$",
          title=f"(e) distance: branch - main {DM['d_dm']:+.2f} $\\pm$ {DM['d_dm_err']:.2f} mag (injection-calibrated)"); a.legend()
    for k, w in enumerate(("main", "branch")):
        a = fig.add_subplot(gs[1, 2+k]); a.imshow(gaussian_filter(Hh[w][0]-Hh[w][1], 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8, extent=ext)
        T = DMF.template(clus, DM["fit"][w]["dm"]); a.contour(cc, cg, T.T, levels=[0.25*T.max()], colors="k", linewidths=0.8); a.invert_yaxis()
        a.set(xlabel=r"$(BP-RP)_0$", ylabel=r"$G_0$ - dm$_{main}(\phi_1)$", title=f"({'fg'[k]}) {w} CMD, N - B, $\\phi_1$ 12-22 (main-stream distance frame)")
    fig.suptitle("Main stream vs upper sky branch (phi1 12-22): sky position, proper motions, distance")
    fig.savefig(ROOT/"plots/stream_branch_summary.png", dpi=85); print("plots/stream_branch_summary.png")


if __name__ == "__main__":
    main()
