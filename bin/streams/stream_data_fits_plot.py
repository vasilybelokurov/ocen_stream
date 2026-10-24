#!/usr/bin/env python3
"""The adopted data extraction in one 3 x 3 figure: data - background | fitted model | residual, rows sky, mu_ra*, mu_de (2026-10-24).

Sky row: single-component local fits (stream_sky_local.py --strip 0, 3-deg windows, phi1 0-24), shown over +-10 deg in v
(fit window +-6 dotted). Data: CMD pass + both frozen PM masks; background B = the fit's fixed band background (quadratic per
0.5-deg row, from 4-12 deg) x observed-area map; model = r N(v | c + g (phi1 - phi_k), s) x area, analytic per pixel.
PM rows: the accepted spline fits (keys pmra, pmdec), rendered by stream_spline_plot.py --keys pmra,pmdec in a subprocess
(its default argv set-up: mixture likelihood, not the sky's --extended) and stacked under the sky row.
Usage: PYTHONPATH=src python bin/streams/stream_data_fits_plot.py -> plots/stream_data_fits.png
"""
import json, subprocess, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]


def pm_rows():
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
    subprocess.run([sys.executable, "-W", "ignore", str(ROOT/"bin/streams/stream_spline_plot.py"), "--keys", "pmra,pmdec"], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    return ROOT/"plots/stream_spline_pmra_pmdec.png"


def sky_row(out):
    sys.argv = [sys.argv[0], "--strip", "0"]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from scipy.special import ndtr
    import stream_sky_local as SL
    from stream_spline_plot import catalogue
    L = json.loads((ROOT/"results/plot_data/stream_sky_local_nostrip.json").read_text()); d = SL.SF.load("dphi2")
    R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); G = np.array(R["grid"]); S = catalogue(); u = S["phi1"]
    ev = lambda p, k="c": np.interp(u, G, R["tracks"][p][k]); v = S["dphi2"]-np.interp(u, G, R["sky_window"]["centre"]); ok = np.ones(u.size, bool)
    for p in ("pmra", "pmdec"):
        ok &= np.abs(S[p]-ev(p)) < 2*np.sqrt(ev(p, "s")**2+S[p+"_error"]**2)
    E1, Ey = np.arange(0, 24.01, 0.5), np.arange(-10, 10.01, 0.25); c1, cy = 0.5*(E1[1:]+E1[:-1]), 0.5*(Ey[1:]+Ey[:-1])
    H = np.histogram2d(u[ok], v[ok], (E1, Ey))[0]; ar = d["area"]
    jr = np.clip(np.digitize(c1, ar["Eu"])-1, 0, len(ar["Eu"])-2); jy = np.clip(np.digitize(cy, ar["Ey"])-1, 0, len(ar["Ey"])-2)
    A = ar["A"][jr][:, jy]; cell = (ar["Eu"][1]-ar["Eu"][0])*(ar["Ey"][1]-ar["Ey"][0]); B = np.clip(d["bg_coef"][jr]@np.vander(cy, 3).T, 0, None)/cell*A*0.5*0.25
    M = np.zeros_like(H); fu = np.linspace(0, 0.5, 6)[:-1]+0.05                        # 0.1-deg sub-steps in phi1 per pixel
    for w in L["windows"]:
        for j in np.nonzero((c1 >= w["phi1"][0]) & (c1 < w["phi1"][1]))[0]:
            mu = w["c"]+w["g"]*(E1[j]+fu-0.5*sum(w["phi1"])); a_, b_ = np.clip(Ey[:-1], -6, 6), np.clip(Ey[1:], -6, 6)
            M[j] = w["r"]*0.1*np.sum(ndtr((b_[None, :]-mu[:, None])/w["s"])-ndtr((a_[None, :]-mu[:, None])/w["s"]), 0)*A[j]
    X = H-B; sb = np.sqrt(np.maximum(B, 1)); win = np.abs(cy) < 6
    chi = ((X-M)/np.sqrt(np.maximum(B+M, 1)))[:, win]
    fig, ax = plt.subplots(1, 3, figsize=(6.7*3, 4.6), constrained_layout=True)
    for a, im, t in zip(ax, (X/sb, M/sb, (X-M)/sb), ("data - background", "model (local fits, one component)", "residual (data - background - model)")):
        a.imshow(gaussian_filter(im, 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=(0, 24, -10, 10))
        for w in L["windows"]:
            x = np.array(w["phi1"]); a.plot(x, w["c"]+w["g"]*(x-0.5*sum(x)), "k-", lw=0.8)
        a.axhline(6, color="k", ls=":", lw=0.6); a.axhline(-6, color="k", ls=":", lw=0.6)
        a.set(xlim=(-8, 32), ylim=(-10, 10), xlabel=r"$\phi_1$ [deg]", ylabel=r"$\Delta\phi_2$ - sky centre [deg]", title=f"sky: {t}")
    fig.suptitle(f"Sky: single-component local fits (3-deg windows, no branch cut); solid: fitted track; dotted: +-6 fit window; residual sd inside "
                 f"the window {chi.std():.2f} (in units of sqrt(B + S)); data excess {X[:, win].sum():.0f}, model {M[:, win].sum():.0f}")
    fig.savefig(out, dpi=95); print(f"sky: residual sd {chi.std():.2f}, mean {chi.mean():+.2f}; excess {X[:, win].sum():.0f}, model {M[:, win].sum():.0f}")


def main():
    from PIL import Image
    tmp = ROOT/"plots/stream_data_fits_sky.png"; sky_row(tmp); pm = pm_rows()
    a, b = Image.open(tmp), Image.open(pm); w = max(a.width, b.width)
    a = a.resize((w, int(a.height*w/a.width))); b = b.resize((w, int(b.height*w/b.width)))
    c = Image.new("RGB", (w, a.height+b.height), "white"); c.paste(a, (0, 0)); c.paste(b, (0, a.height))
    c.save(ROOT/"plots/stream_data_fits.png"); tmp.unlink(); print("plots/stream_data_fits.png")


if __name__ == "__main__":
    main()
