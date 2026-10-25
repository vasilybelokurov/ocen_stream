#!/usr/bin/env python3
"""Explore alternative bar models by their non-axisymmetric force profile (2026-10-21; no sprays, no fits).

Target (bin/streams/diagnostics/bar_diag2.py): the step-2 best fits use the Hunter+2024 bar with amplitude A ~1.4 and size
S ~1.25-1.33, i.e. max_phi |F_m>0| / |F_axi| (z = 0) of ~0.30-0.40 at R 1.5-3, 0.34-0.36 at 4, 0.25-0.27 at 5, 0.19-0.21 at 6,
0.135-0.16 at 7 kpc (published bar: 0.23-0.29, 0.19, 0.13, 0.08, 0.04).
Candidates (all as non-axisymmetric parts added to the same axisymmetric host, so v_c is unchanged):
  hunter A,S        : the current construction, A x (baryon_full(x/S) - baryon_axi(x/S)) (AGAMA scale [A S, S]);
  sormani LB x f    : Sormani+2022 bar (oCen_bar example_mw_bar_potential.py: X-bar + two long-bar components): X-bar
                      unchanged, long bar x f (f = 2, 3; only its m > 0 force matters here);
  sormani LB len s  : long-bar components stretched in x, y by s (1.15, 1.3; mass kept), non-axi part only;
  ferrers           : AGAMA Ferrers bar (n = 2) of semi-axes a, 0.35a, 0.25a and mass M, non-axi part only (minus its axisymmetrised
                      version), added to the published Hunter bar.
m > 0 force measured directly on rings (ring mean of F_R, F_z removed): no axisymmetrised potentials are built (a
CylSpline/Multipole with symmetry 'a' from a non-axisymmetric density gave spurious differences, 2026-10-21).
Reported: the force-ratio profile at R = 1.5, 2, 3, 4, 5, 6, 7, 8 kpc for z = 0 and 1.5 kpc, and the extra non-axisymmetric
mass where defined.
Usage: PYTHONPATH=src python bin/streams/diagnostics/bar_options.py   (results/plot_data/bar_options.json, plots/bar_options.png)
"""
import json, os, sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(ROOT/"src"), str(ROOT/"bin/streams")]
RS = (1.5, 2., 3., 4., 5., 6., 7., 8.)
PHI = np.linspace(0, np.pi, 37)[:-1]


RING = np.linspace(0, 2*np.pi, 73)[:-1]


def ratio(force, axi, z):
    """max over a ring of |F - <F>_phi| (m > 0 part: the ring mean of F_R and F_z subtracted in cylindrical components; no
    axisymmetrised potential needed) over the mean |F_axi| on the ring. `force` may include any axisymmetric part."""
    out = []
    for R in RS:
        P = np.column_stack((R*np.cos(RING), R*np.sin(RING), np.full_like(RING, z))); F = force(P)
        c, s_ = np.cos(RING), np.sin(RING); FR = F[:, 0]*c+F[:, 1]*s_; Fp = -F[:, 0]*s_+F[:, 1]*c; Fz = F[:, 2]
        d = np.column_stack((FR-FR.mean(), Fp-Fp.mean(), Fz-Fz.mean()))
        out.append(float(np.max(np.linalg.norm(d, axis=1))/np.mean(np.linalg.norm(axi.force(P), axis=1))))
    return out


def main():
    from ocen_stream.restricted import agama_kpc, HUNTER24_DIR
    agama = agama_kpc()
    axi = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_axi.ini"))
    full = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_full.ini"))
    bax = agama.Potential(file=os.path.join(HUNTER24_DIR, "MWPotentialHunter24_baryon_axi.ini"))
    res = {}

    def hunter(A, S):
        f1 = agama.Potential(potential=full, scale=[A*S, S]); f2 = agama.Potential(potential=bax, scale=[A*S, S])
        return lambda P: f1.force(P)-f2.force(P)
    for lab, A, S in (("hunter published (A=S=1)", 1., 1.), ("hunter fitted 25 deg (A 1.38 S 1.24)", 1.38, 1.24),
                      ("hunter fitted 12.5 deg (A 1.37 S 1.33)", 1.37, 1.33)):
        h = hunter(A, S); res[lab] = dict(z0=ratio(h, axi, 0.), z1p5=ratio(h, axi, 1.5))
    # Sormani+2022 components
    spec = importlib.util.spec_from_file_location("mwbar", os.path.join(HUNTER24_DIR, "example_mw_bar_potential.py"))
    src = Path(os.path.join(HUNTER24_DIR, "example_mw_bar_potential.py")).read_text()
    src = src.split("def makePotentialModel")[0]                       # definitions only (the script plots at import)
    ns = {}; exec(compile(src, "example_mw_bar_potential_defs", "exec"), ns)
    body = src.split("def makeBarDensity")[1]
    import re
    block = body.split("numpy.array(")[1].split("\n    )")[0]
    pars = np.array([float(v) for v in re.findall(r"[-+]?\d+\.\d+e[-+]\d+", block)])
    assert len(pars) == 33, len(pars)
    xb = ns["makeXBar"](**dict(zip(["densityNorm", "x0", "y0", "z0", "cpar", "cperp", "m", "outerCutoffRadius", "alpha", "c", "n", "xc", "yc"], pars[0:13])))
    keys_lb = ["densityNorm", "x0", "y0", "scaleHeight", "cperp", "cpar", "outerCutoffRadius", "innerCutoffRadius", "outerCutoffStrength", "innerCutoffStrength"]
    lb1p, lb2p = dict(zip(keys_lb, pars[13:23])), dict(zip(keys_lb, pars[23:33]))

    def cyl(dens, sym="t"):
        return agama.Potential(type="CylSpline", density=agama.Density(dens, symmetry=sym), mmax=8, gridsizeR=25, gridsizez=25,
                               Rmin=0.1, Rmax=40, zmin=0.05, zmax=20, symmetry=sym)

    def nonaxi_of(dens):
        p = cyl(dens)
        return p, None, p.force                                        # ratio() removes the m = 0 part itself
    print("building Sormani components (CylSpline, mmax 8)...", flush=True)
    _, _, nx_x = nonaxi_of(xb)
    lb = lambda Q: ns["makeLongBar"](**lb1p)(Q)+ns["makeLongBar"](**lb2p)(Q)
    _, _, nx_lb = nonaxi_of(lb)
    res["sormani published (X + LB)"] = dict(z0=ratio(lambda Q: nx_x(Q)+nx_lb(Q), axi, 0.), z1p5=ratio(lambda Q: nx_x(Q)+nx_lb(Q), axi, 1.5))
    for f in (2., 3.):
        g = (lambda f: (lambda Q: nx_x(Q)+f*nx_lb(Q)))(f)
        res[f"sormani long bar non-axi x{f:g}"] = dict(z0=ratio(g, axi, 0.), z1p5=ratio(g, axi, 1.5))
    for s in (1.15, 1.3):
        def lbs(Q, s=s):
            Q2 = Q.copy(); Q2[:, 0] /= s; Q2[:, 1] /= s
            return lb(Q2)/s**2                                         # mass kept (stretched in x and y)
        _, _, nx_s = nonaxi_of(lbs)
        g = (lambda nx_s: (lambda Q: nx_x(Q)+nx_s(Q)))(nx_s)
        res[f"sormani long bar length x{s:g}"] = dict(z0=ratio(g, axi, 0.), z1p5=ratio(g, axi, 1.5))
        g2 = (lambda nx_s: (lambda Q: nx_x(Q)+2*nx_s(Q)))(nx_s)
        res[f"sormani long bar length x{s:g}, non-axi x2"] = dict(z0=ratio(g2, axi, 0.), z1p5=ratio(g2, axi, 1.5))
    h0 = hunter(1., 1.)
    for a, M in ((5., 5e9), (5., 1e10), (6., 5e9)):
        fer = agama.Potential(type="Ferrers", mass=M, scaleRadius=a, axisRatioY=0.35, axisRatioZ=0.25)
        g = (lambda fer: (lambda Q: h0(Q)+fer.force(Q)))(fer)
        res[f"hunter published + Ferrers a={a:g} M={M:.0e}"] = dict(z0=ratio(g, axi, 0.), z1p5=ratio(g, axi, 1.5))
    (ROOT/"results/plot_data/bar_options.json").write_text(json.dumps(dict(R=RS, profiles=res), indent=1))
    print(f"{'model':48s} z=0: R = " + " ".join(f"{r:5g}" for r in RS))
    for k, v in res.items():
        print(f"{k:48s} z0  " + " ".join(f"{x:5.3f}" for x in v["z0"]) + "   | z1.5 " + " ".join(f"{x:5.3f}" for x in v["z1p5"]))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for k, v in res.items():
        st = "k-" if "published" in k and "hunter" in k else ("r--" if "fitted" in k else "-")
        for j, z in enumerate(("z0", "z1p5")):
            ax[j].plot(RS, v[z], st if st != "-" else "o-", ms=3, lw=2 if st != "-" else 1, label=k)
    for j, t in enumerate(("z = 0", "z = 1.5 kpc")):
        ax[j].axvspan(6.4, 7.2, color="0.9", zorder=0); ax[j].axvspan(1.5, 2.2, color="0.95", zorder=0)
        ax[j].set(xlabel="R [kpc]", ylabel="max |F_nonaxi| / |F_axi|", title=f"{t}; shaded: omega Cen peri / apo ranges")
    ax[0].legend(fontsize=6.5)
    fig.savefig(ROOT/"plots/bar_options.png", dpi=110); print(ROOT/"plots/bar_options.png")


if __name__ == "__main__":
    main()
