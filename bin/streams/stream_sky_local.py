#!/usr/bin/env python3
"""Single-component sky track by local fits at fixed knots, main band only (2026-10-24; Fable/Codex review sky1comp).

Data: the extended sky set-up of stream_spline_fit.py (--extended --W 6 --hi 25 --nohole: CMD pass, both frozen PM masks,
v = dphi2 - sky-window centre, |v| < 6; background shape AND level fixed from area-corrected 4-12 deg bands; observed-area map).
Branch strip: stars with |v - m2(phi1)| < HW_STRIP at phi1 13-22 removed (m2 = upper-branch track of dphi2_sky_final); the same
strip is removed from the stream integral (and must be removed from model particles).
Per non-overlapping 3-deg phi1 window k (centre phi_k): extended Poisson likelihood
    lnL = sum_i ln[lam_b(u_i, v_i) + r N(v_i | c + g (u_i - phi_k), s)] - r int int N A du dv,
parameters c (track at phi_k), s (intrinsic width, bounded 0.1-4 deg), r (stream stars per deg); slope g FIXED to the gradient of
the two-component main track at phi_k (a free slope is unconstrained in the weak windows: c ran to the window edge). Errors: inverse
finite-difference Hessian. Poisson mocks drawn from the fitted model (+ fixed background) give the error calibration.
Usage: PYTHONPATH=src python bin/streams/stream_sky_local.py [--strip 1.2 | --strip 0 (no branch cut, suffix _nostrip)] [--nmock 50]
       -> results/plot_data/stream_sky_local.json, plots/stream_sky_local.png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
HW_STRIP = float(sys.argv[sys.argv.index("--strip")+1]) if "--strip" in sys.argv else 1.2
SUF = "" if HW_STRIP > 0 else "_nostrip"                                        # --strip 0: no branch cut
NMOCK = int(sys.argv[sys.argv.index("--nmock")+1]) if "--nmock" in sys.argv else 50
sys.argv += ["--extended", "--W", "6", "--hi", "25", "--nohole"]          # stream_spline_fit reads its set-up from argv at import
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
import stream_spline_fit as SF

EDGES = np.arange(0., 24.01, 3.); STRIP = (13., 22.); SMIN, SMAX = 0.1, 4.0


def strip_mask(u, v, m2):
    """True inside the excluded branch strip."""
    u = np.asarray(u, float); mm = m2(u.ravel()).reshape(u.shape)
    return (u >= STRIP[0]) & (u < STRIP[1]) & (np.abs(v-mm) < HW_STRIP)


def unpack(p):
    c, xs, lr = p; return c, SMIN+(SMAX-SMIN)/(1+np.exp(-xs)), np.exp(lr)


def nll(p, w):
    (c, s, r), g = unpack(p), w["g"]; mu = c+g*(w["u"]-w["pk"]); mg = c+g*(w["ug"]-w["pk"])
    lam = w["lam_b"]+r*np.exp(-0.5*((w["v"]-mu)/s)**2)/(s*np.sqrt(2*np.pi))
    G = np.exp(-0.5*((w["vg"][None, :]-mg[:, None])/s)**2)/(s*np.sqrt(2*np.pi))
    return -(np.sum(np.log(lam))-r*np.sum(np.trapezoid(G*w["Ag"], w["vg"], axis=1))*w["du"])


def hessian(f, p, h=1e-3):
    n = len(p); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i]*h, np.eye(n)[j]*h
            H[i, j] = H[j, i] = (f(p+e_i+e_j)-f(p+e_i-e_j)-f(p-e_i+e_j)+f(p-e_i-e_j))/(4*h*h)
    return H


def fit(w, p0):
    best = None
    for xs0 in (-1.5, 0., 1.5):                                       # starts in width: 0.7, 2.0, 3.3 deg
        q0 = np.array(p0, float); q0[1] = xs0
        r = minimize(nll, q0, args=(w,), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=8000))
        r = minimize(nll, r.x, args=(w,), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    p = best.x; C = np.linalg.pinv(hessian(lambda q: nll(q, w), p))
    c, s, rr = unpack(p); J = np.diag([1., (s-SMIN)*(1-(s-SMIN)/(SMAX-SMIN)), rr])   # d(c, s, r)/d(p)
    Cn = J@C@J.T; e = np.sqrt(np.clip(np.diag(Cn), 0, None))
    return dict(p=p.tolist(), c=c, g=w["g"], s=s, r=rr, c_err=e[0], s_err=e[1], r_err=e[2], at_bound=bool(s < SMIN+0.05 or s > SMAX-0.1), nll=float(best.fun))


def window(d, a0, a1, m2, m1):
    k = (d["u"] >= a0) & (d["u"] < a1); kg = (d["ug"] >= a0) & (d["ug"] < a1)
    keep = k & ~strip_mask(d["u"], d["v"], m2)
    Ag = d["Ag"][kg]*~strip_mask(d["ug"][kg][:, None], d["vg"][None, :], m2)
    pk = 0.5*(a0+a1); g = float(np.diff(m1(np.array([pk-0.05, pk+0.05])))[0]/0.1)
    return dict(u=d["u"][keep], v=d["v"][keep], lam_b=d["lam_b"][keep], ug=d["ug"][kg], vg=d["vg"], Ag=Ag, du=0.1, pk=pk, g=g,
                cf=d["bg_coef"], Eu=d["area"]["Eu"], cell=(d["area"]["Eu"][1]-d["area"]["Eu"][0])*(d["area"]["Ey"][1]-d["area"]["Ey"][0]))


def mock(w, res, rng):
    """Poisson realisation of fixed background + fitted stream on the window's 0.1-deg x vg grid (strip and area applied)."""
    ug, vg, dv = w["ug"], w["vg"], w["vg"][1]-w["vg"][0]; jg = np.clip(np.digitize(ug, w["Eu"])-1, 0, len(w["Eu"])-2)
    lb = np.clip(w["cf"][jg]@np.vander(vg, 3).T, 0, None)/w["cell"]
    mg = res["c"]+res["g"]*(ug-w["pk"]); lam = (lb+res["r"]*np.exp(-0.5*((vg[None, :]-mg[:, None])/res["s"])**2)/(res["s"]*np.sqrt(2*np.pi)))*w["Ag"]
    n = rng.poisson(lam*w["du"]*dv); iu, iv = np.nonzero(n); rep = n[iu, iv]
    u = np.repeat(ug[iu], rep)+rng.uniform(-0.05, 0.05, rep.sum()); v = np.clip(np.repeat(vg[iv], rep)+rng.uniform(-0.5, 0.5, rep.sum())*dv, vg[0], vg[-1])
    ju = np.clip(np.digitize(u, w["Eu"])-1, 0, len(w["Eu"])-2); bq = np.clip(np.sum(w["cf"][ju]*np.vander(v, 3), axis=1), 1e-3, None)
    wm = dict(w); wm.update(u=u, v=v, lam_b=bq/w["cell"]); return wm


def main():
    d = SF.load("dphi2"); m1, m2 = SF.sky_tracks(); rng = np.random.default_rng(5); out = []
    for a0, a1 in zip(EDGES[:-1], EDGES[1:]):
        w = window(d, a0, a1, m2, m1); p0 = [m1(np.array([w["pk"]]))[0], 0., np.log(30.)]
        res = fit(w, p0); res.update(phi1=(a0, a1), n_stars=int(len(w["u"])))
        Z = []
        for _ in range(NMOCK):
            rm = fit(mock(w, res, rng), res["p"]); Z.append([(rm[q]-res[q])/rm[q+"_err"] for q in ("c", "s", "r")]+[rm[q] for q in ("c", "s", "r")])
        Z = np.array(Z); res.update(mock=dict(n=NMOCK, z_mean=Z[:, :3].mean(0).tolist(), z_sd=Z[:, :3].std(0).tolist(), sd=Z[:, 3:].std(0).tolist(),
                                              mean=Z[:, 3:].mean(0).tolist(), at_bound_frac=float(np.mean((Z[:, 4] < SMIN+0.05) | (Z[:, 4] > SMAX-0.1)))))
        print(f"phi1 {a0:4.1f}-{a1:4.1f}: N {res['n_stars']:4d}; c {res['c']:+.2f} +- {res['c_err']:.2f} (mock sd {res['mock']['sd'][0]:.2f}); "
              f"s {res['s']:.2f} +- {res['s_err']:.2f} (mock sd {res['mock']['sd'][1]:.2f}); r {res['r']:.1f} +- {res['r_err']:.1f}/deg; "
              f"z sd c/s/r {np.round(res['mock']['z_sd'], 2).tolist()}, z mean {np.round(res['mock']['z_mean'], 2).tolist()}; bound {res['at_bound']}", flush=True)
        out.append(res)
    json.dump(dict(strip_hw=HW_STRIP, strip_phi1=STRIP, edges=EDGES.tolist(), smin=SMIN, smax=SMAX, windows=out,
                   note="v = dphi2 - sky-window centre (stream_masks.json); main band, branch strip removed; extended likelihood, fixed band background"),
              open(ROOT/f"results/plot_data/stream_sky_local{SUF}.json", "w"), indent=1)
    plot(d, out, m2)


def plot(d, out, m2):
    F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text()); m1, _ = SF.sky_tracks()
    fig = plt.figure(figsize=(20, 9.5), constrained_layout=True); gs = fig.add_gridspec(3, 8)
    a = fig.add_subplot(gs[0, :5]); E1, Ey = np.arange(0, 25.01, 0.5), np.arange(-6, 6.01, 0.25)
    H = np.histogram2d(d["u"], d["v"], (E1, Ey))[0]; jr = np.clip(np.digitize(0.5*(E1[1:]+E1[:-1]), d["area"]["Eu"])-1, 0, len(d["area"]["Eu"])-2)
    cy = 0.5*(Ey[1:]+Ey[:-1]); cell = (d["area"]["Eu"][1]-d["area"]["Eu"][0])*(d["area"]["Ey"][1]-d["area"]["Ey"][0])
    jy = np.clip(np.digitize(cy, d["area"]["Ey"])-1, 0, len(d["area"]["Ey"])-2)
    B = np.clip(d["bg_coef"][jr]@np.vander(cy, 3).T, 0, None)/cell*d["area"]["A"][jr][:, jy]*0.5*0.25      # the fit's own background
    from scipy.ndimage import gaussian_filter
    a.imshow(gaussian_filter((H-B)/np.sqrt(np.maximum(B, 1)), 1.).T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=(0, 25, -6, 6))
    g = np.linspace(0, 25, 251); g2 = np.linspace(*STRIP, 91)
    a.plot(g, m1(g), "k--", lw=0.8, label="two-component main track"); a.plot(g2, m2(g2)+HW_STRIP, "y-", lw=0.8); a.plot(g2, m2(g2)-HW_STRIP, "y-", lw=0.8, label="branch strip (excluded)")
    W1 = F["dphi2_W6ext_nohole"]; a.plot(W1["grid"], W1["m"], "C2:", lw=1, label="global 1-comp spline (W6ext_nohole)")
    pk = np.array([0.5*sum(r["phi1"]) for r in out]); c = np.array([r["c"] for r in out]); ce = np.array([r["c_err"] for r in out]); s = np.array([r["s"] for r in out])
    for r in out:
        x = np.array(r["phi1"]); a.plot(x, r["c"]+r["g"]*(x-0.5*sum(x)), "k-", lw=2)
    a.errorbar(pk, c, ce, fmt="ko", ms=4, capsize=3, label="local fits (track +- err)"); a.errorbar(pk, c, s, fmt="none", ecolor="k", alpha=0.3, lw=6)
    a.set(xlim=(0, 25), ylim=(-6, 6), xlabel=r"$\phi_1$ [deg]", ylabel="v = $\\Delta\\phi_2$ - centre [deg]", title="(a) data - fixed background, (N-B)/sqrt(B); local fits (grey: +-width)"); a.legend(fontsize=7, loc="lower right")
    a = fig.add_subplot(gs[0, 5:]); se = np.array([r["s_err"] for r in out])
    a.errorbar(pk, s, se, fmt="ko", capsize=3, label="local fits"); a.plot(W1["grid"], W1["s"], "C2:", label="global 1-comp spline")
    T = F["dphi2_sky_final"]; a.plot(T["grid"], T["s"], "k--", lw=0.8, label="two-component main"); a.axhline(SMAX, color="r", lw=0.5, ls=":")
    a.set(xlim=(0, 25), ylim=(0, 5), xlabel=r"$\phi_1$ [deg]", ylabel="intrinsic width [deg]", title="(b) width"); a.legend(fontsize=7)
    for k, (r, (a0, a1)) in enumerate(zip(out, zip(EDGES[:-1], EDGES[1:]))):
        a = fig.add_subplot(gs[1+k//4, 2*(k % 4):2*(k % 4)+2]); w = window(d, a0, a1, m2, m1); ev = np.arange(-6, 6.01, 0.3); cv = 0.5*(ev[1:]+ev[:-1])
        n = np.histogram(w["v"], ev)[0]; jg = np.clip(np.digitize(w["ug"], w["Eu"])-1, 0, len(w["Eu"])-2)
        lb = np.clip(w["cf"][jg]@np.vander(w["vg"], 3).T, 0, None)/w["cell"]*w["Ag"]; mg = r["c"]+r["g"]*(w["ug"]-w["pk"])
        ls = r["r"]*np.exp(-0.5*((w["vg"][None, :]-mg[:, None])/r["s"])**2)/(r["s"]*np.sqrt(2*np.pi))*w["Ag"]
        bb = np.interp(cv, w["vg"], lb.sum(0)*0.1)*0.3; ss = np.interp(cv, w["vg"], ls.sum(0)*0.1)*0.3
        a.bar(cv, n-bb, 0.3, color="0.7", yerr=np.sqrt(np.maximum(n, 1)), error_kw=dict(lw=0.6, ecolor="0.4")); a.plot(cv, ss, "C3-", label="local fit")
        a.axvline(m1(np.array([0.5*(a0+a1)]))[0], color="k", ls="--", lw=0.8)
        if a0 >= STRIP[0]-3:
            mm = m2(np.array([0.5*(a0+a1)]))[0]; a.axvspan(mm-HW_STRIP, mm+HW_STRIP, color="y", alpha=0.2)
        a.axhline(0, color="k", lw=0.4); a.set(title=f"phi1 {a0:.0f}-{a1:.0f}: c {r['c']:+.2f}, s {r['s']:.2f}, N {r['n_stars']}", xlabel="v [deg]")
    fig.savefig(ROOT/f"plots/stream_sky_local{SUF}.png", dpi=85); print(f"plots/stream_sky_local{SUF}.png")


if __name__ == "__main__":
    main()
