#!/usr/bin/env python3
"""Spline mixture model of the omega Cen stream along phi1 (Erkal, Koposov & Belokurov 2017, MNRAS 470, 60, Sec. 3.3:
https://ui.adsabs.harvard.edu/abs/2017MNRAS.470...60E -- knots placed greedily by AIC; Koposov et al. 2023, MNRAS 521, 4936:
https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.4936K -- measurements at spline knots for the dynamical fit). 2026-10-24.

Per observable q in (dphi2, pmra, pmdec), unbinned, all stars at once:
  data   : stars passing the CMD mask and the OTHER two frozen masks (results/plot_data/stream_masks.json, k = 2; PM fits: sky
           window |dphi2 - centre| < half-width; sky fit: both PM masks); v = q - frozen track(phi1); window |v| < W.
  model  : p(v | phi1) = f(phi1) N_D(v; m(phi1), s(phi1)^2 + e_i^2) + (1 - f(phi1)) b(v | phi1), both normalised over the
           star's domain D (window, minus the upper-branch hole for the sky fit); m, log s, logit f = natural cubic splines.
  b      : PMs -- empirical: off-sky stars (5 < |dphi2 - centre| < 14, same other mask), 2-D histogram in (phi1, v) smoothed,
           normalised per phi1; sky -- exp(a1(phi1) v + a2(phi1) v^2) normalised numerically (a1, a2 splines, fixed knots).
  knots  : start 4 per spline (ends and thirds); candidates at 25/50/75% of every interval of every spline, refitted (warm start);
           the best is kept if AIC = 2k - 2 lnL decreases; stop otherwise (minimum spacing 1 deg).
  errors : covariance = inverse Hessian of -lnL at the maximum (torch autograd).
Outputs: results/plot_data/stream_spline_fit.json (knots, values, covariance, AIC history), plots/stream_spline_fit.png
(data | model stream | residual, the cross-selection panels).
Usage: PYTHONPATH=src python bin/streams/stream_spline_fit.py [--q pmdec] [--test] [--null]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
torch.set_default_dtype(torch.float64)
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

W_SKY = float(sys.argv[sys.argv.index("--W")+1]) if "--W" in sys.argv else 7.0          # sky window half-width [deg]
SMAX = float(sys.argv[sys.argv.index("--smax")+1]) if "--smax" in sys.argv else None     # optional cap on the intrinsic width
EXTENDED = "--extended" in sys.argv                                                     # sky: extended likelihood, background level fixed
BGFIXED = "--bgfixed" in sys.argv or EXTENDED                                                       # sky: fixed empirical background
TWO = "--two" in sys.argv                                  # sky: second stream component (upper branch), phi1 TWO_LO..TWO_HI
TWO_LO, TWO_HI = map(float, sys.argv[sys.argv.index("--two-range")+1].split(",")) if "--two-range" in sys.argv else (10., 23.)
SFIX = int(sys.argv[sys.argv.index("--sfix")+1]) if "--sfix" in sys.argv else 0   # main-stream width: N fixed equally spaced knots, not searched
TAG = sys.argv[sys.argv.index("--tag")+1] if "--tag" in sys.argv else ""                  # result key suffix (variants)
CFG = dict(pmdec=dict(W=3.0, lo=-2., hi=28.), pmra=dict(W=4.0, lo=-2., hi=28.), dphi2=dict(W=W_SKY, lo=0., hi=float(sys.argv[sys.argv.index("--hi")+1]) if "--hi" in sys.argv else 28.))  # sky window (default +-7 deg; 6 and 10 tried)
NG = 281                                                     # sky: integration grid in v over [-W, W]
SQ2 = np.sqrt(2.)


def basis(knots, x):
    """natural cubic spline basis (len(x), len(knots)); x clipped to the knot range (constant extrapolation)."""
    knots = np.asarray(knots, float); xc = np.clip(x, knots[0], knots[-1]); I = np.eye(len(knots))
    if len(knots) == 2:
        t = (xc-knots[0])/(knots[1]-knots[0]); return np.column_stack((1-t, t))
    return np.column_stack([CubicSpline(knots, I[j], bc_type="natural")(xc) for j in range(len(knots))])


def Phi(z):
    return 0.5*(1+torch.erf(z/SQ2))


# ------------------------------------------------------------------ data
SKYWIN = sys.argv[sys.argv.index("--skywin")+1] if "--skywin" in sys.argv else None   # PMs: +-SKYWIN_HW sky window on the main or branch track


def sky_tracks():
    """main and upper-branch tracks (y = dphi2 - sky centre) of the two-component sky fit dphi2_sky_final."""
    F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())["dphi2_sky_final"]["fit"]
    return (lambda u: basis(F["m"]["knots"], u)@np.array(F["m"]["values"])), (lambda u: basis(F["m2"]["knots"], u)@np.array(F["m2"]["values"]))


SKYWIN_HW, BRANCH_RANGE = 1.0, (10., 23.)


def load(q, null_shift=0., skywin=SKYWIN):
    from gaia_cmd_mask import cmd_pass
    from gaia_footprint_plots import gc_masked
    import gaia_xsel_marked as GX
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float)
    keep = cmd_pass(2.0) & ~gc_masked(lg, raw["b"].astype(float)); S = {k: v[keep].astype(float) for k, v in raw.items() if v.dtype.kind == "f"}
    R = json.loads((ROOT/"results/plot_data/stream_masks.json").read_text()); G = np.array(R["grid"])
    u, x = S["phi1"], S["dphi2"]; ev = lambda p, key="c": np.interp(u, G, R["tracks"][p][key])
    skc, skh = np.interp(u, G, R["sky_window"]["centre"]), np.interp(u, G, R["sky_window"]["half_width"])
    pmm = {p: np.abs(S[p]-ev(p)) < 2*np.sqrt(ev(p, "s")**2+S[p+"_error"]**2) for p in ("pmra", "pmdec")}
    c = CFG[q]; rng = (u >= c["lo"]) & (u < c["hi"])
    if q == "dphi2":
        sel = rng & pmm["pmra"] & pmm["pmdec"]; v = x-skc+null_shift; e = np.zeros(u.size)
        up = np.array(GX.MARK["upper"]); vu = np.interp(u, up[:, 0], up[:, 1], left=np.nan, right=np.nan)-skc
        hlo, hhi = np.where(np.isfinite(vu), vu-1.25, 0.), np.where(np.isfinite(vu), vu+1.25, 0.)
        if "--nohole" in sys.argv:                                   # no upper-branch cut (user: see the model without cuts)
            hlo, hhi = np.zeros(u.size), np.zeros(u.size)
        on = sel & (np.abs(v) < c["W"]) & ~((v > hlo) & (v < hhi))
        # observed-area map A(phi1, y), y = dphi2 - centre: all footprint stars (GC masks applied), 0.5 deg x 0.25 deg cells,
        # fraction = counts / robust quadratic (in y) fit to log counts per phi1 row, clipped to [0, 1]; footprint edges
        # (|dphi2| < 15, b > 8) and GC-mask holes -> A < 1 (the background and stream normalisations integrate over A)
        ua, xa = raw["phi1"].astype(float), raw["dphi2"].astype(float); ka = ~gc_masked(lg, raw["b"].astype(float)) & (ua >= c["lo"]) & (ua < c["hi"])
        ya = xa[ka]-np.interp(ua[ka], G, R["sky_window"]["centre"]); Eu = np.arange(c["lo"], c["hi"]+1e-9, 0.5); Ey = np.arange(-13., 13.+1e-9, 0.25)   # area map to +-13 (background bands)
        Na = np.histogram2d(ua[ka], ya, (Eu, Ey))[0]; cy = 0.5*(Ey[1:]+Ey[:-1]); A = np.zeros_like(Na)
        for j in range(Na.shape[0]):
            ok = Na[j] > 0.5*np.median(Na[j][Na[j] > 0])
            for _ in range(3):
                cf = np.polyfit(cy[ok], np.log(Na[j][ok]), 2); r_ = np.log(np.maximum(Na[j], 1e-3))-np.polyval(cf, cy)
                ok = (Na[j] > 0) & (r_ > -0.3)
            A[j] = np.clip(Na[j]/np.exp(np.polyval(cf, cy)), 0, 1)
        vg = np.linspace(-c["W"], c["W"], NG); ju = np.clip(np.digitize(u[on], Eu)-1, 0, len(Eu)-2); jy = np.clip(np.digitize(vg, Ey)-1, 0, len(Ey)-2)
        Aw = A[ju][:, jy]*~((vg[None, :] > hlo[on][:, None]) & (vg[None, :] < hhi[on][:, None]))
        out = {}
        if BGFIXED:
            # FIXED empirical background (as for the PMs): stars of the same selection (both PM masks) at 4 <= |y| <= 12 deg,
            # per 0.5-deg phi1 row and 0.25-deg y cell, density = counts / A (cells with A > 0.5); quadratic in y per row,
            # coefficients smoothed over 3 rows; b(v | phi1) = quadratic x A (observed density), normalised over the star's domain
            ys = (x-skc)[sel]; Nb = np.histogram2d(u[sel], ys, (Eu, Ey))[0]; band = (np.abs(cy) >= 4) & (np.abs(cy) <= 12); cf = np.zeros((Nb.shape[0], 3))
            for j in range(Nb.shape[0]):
                ok = band & (A[j] > 0.5); cf[j] = np.polyfit(cy[ok], Nb[j][ok]/A[j][ok], 2) if ok.sum() >= 8 else np.nan
            cf = np.array([np.nanmean(cf[max(j-1, 0):j+2], 0) for j in range(len(cf))])
            bq = np.clip(np.sum(cf[ju]*np.vander(v[on], 3), axis=1), 1e-3, None)
            bqg = np.clip(cf[ju]@np.vander(vg, 3).T, 1e-3, None)*Aw; Zb = np.trapezoid(bqg, vg, axis=1)
            out = dict(lb=np.log(bq)-np.log(Zb), bqg=bqg, bg_coef=cf)
            if EXTENDED:
                # extended likelihood: background LEVEL fixed too. lambda_b = area-corrected band density per deg^2 at the star;
                # stream rate integrated over a 0.1-deg phi1 grid with the observed-area map
                cell = (Eu[1]-Eu[0])*(Ey[1]-Ey[0]); ug = np.arange(c["lo"]+0.05, c["hi"], 0.1); jg = np.clip(np.digitize(ug, Eu)-1, 0, len(Eu)-2)
                # stream integrals over the observed area minus the upper-branch hole (same exclusion as the stars)
                vug = np.interp(ug, up[:, 0], up[:, 1], left=np.nan, right=np.nan)-np.interp(ug, G, R["sky_window"]["centre"])
                hg = np.isfinite(vug)[:, None] & (np.abs(vg[None, :]-np.nan_to_num(vug)[:, None]) < 1.25) & ("--nohole" not in sys.argv)
                out.update(lam_b=bq/cell, ug=ug, Ag=A[jg][:, jy]*~hg, n_bg_expected=float(np.sum(np.clip(cf[jg]@np.vander(vg, 3).T, 0, None)*A[jg][:, jy])*(vg[1]-vg[0])*0.1/cell))
        k2 = np.linspace(TWO_LO, TWO_HI, 4); fitu = np.polyfit(up[:, 0], up[:, 1], 1)                         # marked upper branch, linear
        out["m2_init"] = (np.polyval(fitu, k2)-np.interp(k2, G, R["sky_window"]["centre"])).tolist()
        return dict(u=u[on], v=v[on], e=e[on], hlo=hlo[on], hhi=hhi[on], q=q, W=c["W"], lo=c["lo"], hi=c["hi"], track=None, Aw=Aw, vg=vg, **out,
                    area=dict(A=A, Eu=Eu, Ey=Ey), all=dict(u=u[sel], x=x[sel]))
    other = "pmdec" if q == "pmra" else "pmra"
    v = S[q]-ev(q)+null_shift; win = np.abs(x-skc) < skh
    if skywin is not None:                                            # main / branch: +-1 deg around the two-component sky tracks
        tm, tb = sky_tracks(); win = np.abs(x-skc-(tm(u) if skywin == "main" else tb(u))) < SKYWIN_HW
        if skywin == "branch":
            c = dict(c, lo=BRANCH_RANGE[0], hi=BRANCH_RANGE[1]); rng = (u >= c["lo"]) & (u < c["hi"])
    offb = (np.abs(x-skc) > 5) & (np.abs(x-skc) < 14)
    on = rng & pmm[other] & win & (np.abs(v) < c["W"]); off = rng & pmm[other] & offb & (np.abs(v) < c["W"])
    # empirical background b(v | phi1): 1-deg x 0.1 histogram of the off-sky stars, smoothed, normalised per phi1 row
    Eu = np.arange(c["lo"], c["hi"]+1e-9, 1.); Ev = np.linspace(-c["W"], c["W"], int(20*c["W"])+1)
    H = gaussian_filter(np.histogram2d(u[off], v[off], (Eu, Ev))[0]+0.5, (1., 2.)); H /= (H.sum(1, keepdims=True)*(Ev[1]-Ev[0]))
    cu, cv = 0.5*(Eu[1:]+Eu[:-1]), 0.5*(Ev[1:]+Ev[:-1])
    from scipy.interpolate import RegularGridInterpolator
    bfun = RegularGridInterpolator((cu, cv), H, bounds_error=False, fill_value=None)
    out = {}
    if EXTENDED:
        # background LEVEL fixed: per 1-deg phi1 row and 0.1 mas/yr v cell, counts in 1-deg dphi2 bands at 5 <= |y| < 14 (same
        # other-PM mask, smoothed by 2 cells in v), quadratic in y (constant if one-sided), integrated over the row's sky window
        # (median half-width) -> expected background counts in the on-window; lambda_b per (deg phi1 x mas/yr)
        yv = x-skc; Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb2 = np.abs(cb) >= 5; sel2 = rng & pmm[other] & (np.abs(v) < c["W"])
        H3 = np.histogramdd(np.column_stack((u[sel2], yv[sel2], v[sel2])), (Eu, Eb, Ev))[0]; Bt = np.zeros((len(Eu)-1, len(Ev)-1))
        for j in range(len(Eu)-1):
            Hj = gaussian_filter(H3[j], (0, 2.)); ob = offb2 & (H3[j].sum(1) > 0)
            dg = 3 if ((ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3) else 1
            coef = np.linalg.pinv(np.vander(cb[ob], dg))@Hj[ob]; kj = on & (u >= Eu[j]) & (u < Eu[j+1]); h = float(np.median(skh[kj])) if kj.any() else 3.
            yy = np.linspace(-h, h, 41); Bt[j] = np.clip(np.vander(yy, dg)@coef, 0, None).mean(0)*(2*h)
        lamg = Bt/((Eu[1]-Eu[0])*(Ev[1]-Ev[0]))
        lfun = RegularGridInterpolator((cu, cv), lamg, bounds_error=False, fill_value=None)
        ug = np.arange(c["lo"]+0.05, c["hi"], 0.1); jg = np.clip(np.digitize(ug, Eu)-1, 0, len(Eu)-2)
        eg = np.array([np.median(S[q+"_error"][on & (u >= Eu[j]) & (u < Eu[j+1])]) if np.any(on & (u >= Eu[j]) & (u < Eu[j+1])) else 0.3 for j in jg])
        out = dict(lam_b=np.clip(lfun(np.column_stack((u[on], v[on]))), 1e-3, None), ug=ug, eg=eg, n_bg_expected=float(Bt.sum()))
    return dict(u=u[on], v=v[on], e=S[q+"_error"][on], b=bfun(np.column_stack((u[on], v[on]))), bfun=bfun, q=q, W=c["W"], lo=c["lo"], hi=c["hi"],
                n_off=int(off.sum()), **out)


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, d, knots):
        self.d, self.knots = d, {k: np.array(sorted(v), float) for k, v in knots.items()}
        self.B = {k: torch.tensor(basis(kn, d["u"])) for k, kn in self.knots.items()}
        self.v, self.e, self.W = torch.tensor(d["v"]), torch.tensor(d["e"]), d["W"]; self.smin = 0.1 if d["q"] == "dphi2" else 0.05
        self.smax = SMAX if d["q"] == "dphi2" else None
        if d["q"] == "dphi2":
            self.vg = torch.tensor(d["vg"]); self.mask = torch.tensor(d["Aw"])          # observed-area weight x upper-branch hole
            if "lb" in d:
                self.lb = torch.tensor(d["lb"])
            self.ext = "lam_b" in d
            if self.ext:
                self.lam_b = torch.tensor(d["lam_b"]); self.ug = torch.tensor(d["ug"]); self.Ag = torch.tensor(d["Ag"])
                self.Bg = {k: torch.tensor(basis(kn, d["ug"])) for k, kn in self.knots.items()}
                tp = lambda x_: 1/(1+np.exp(-(x_-TWO_LO)/0.3))/(1+np.exp((x_-TWO_HI)/0.3))
                self.tap, self.tapg = torch.tensor(tp(d["u"])), torch.tensor(tp(d["ug"]))
        else:
            self.lb = torch.log(torch.tensor(np.maximum(d["b"], 1e-12)))
            self.ext = "lam_b" in d
            if self.ext:                                              # PM extended: per-grid-point median errors for the integral
                self.lam_b = torch.tensor(d["lam_b"]); self.ug = torch.tensor(d["ug"]); self.eg = torch.tensor(d["eg"])
                self.Bg = {k: torch.tensor(basis(kn, d["ug"])) for k, kn in self.knots.items()}

    def sizes(self):
        return {k: len(v) for k, v in self.knots.items()}

    def unpack(self, th):
        out, i = {}, 0
        for k, n in self.sizes().items():
            out[k] = th[i:i+n]; i += n
        return out

    def lnl(self, th, per_star=False):
        P = self.unpack(th); m = self.B["m"]@P["m"]; fl = self.B["f"]@P["f"]   # logit f
        s = self.width(self.B["s"]@P["s"])
        sig = torch.sqrt(s**2+self.e**2); W = self.W
        lg = -0.5*((self.v-m)/sig)**2-torch.log(sig)-0.5*np.log(2*np.pi)
        Z = Phi((W-m)/sig)-Phi((-W-m)/sig)
        if self.d["q"] == "dphi2":                                   # normalisations over the observed area (numerical)
            gz = torch.exp(-0.5*((self.vg[None, :]-m[:, None])/sig[:, None])**2)/(sig[:, None]*np.sqrt(2*np.pi))
            Z = torch.trapezoid(gz*self.mask, self.vg, dim=1)
            if "a1" in self.knots:
                a1, a2 = self.B["a1"]@P["a1"], self.B["a2"]@P["a2"]
                lbu = a1*self.v+a2*self.v**2
                gg = torch.exp(a1[:, None]*self.vg[None, :]+a2[:, None]*self.vg[None, :]**2)*self.mask
                lb = lbu-torch.log(torch.trapezoid(gg, self.vg, dim=1))
            else:
                lb = self.lb
        else:
            lb = self.lb
        if getattr(self, "ext", False):                              # extended: sum log(lambda_b + rate N) - integral of the stream rate
            lam = self.lam_b+torch.exp(fl+lg); nint2 = 0.
            if "m2" in self.knots:                                    # upper branch: rate x logistic taper to TWO_LO..TWO_HI
                m2, s2, f2 = self.B["m2"]@P["m2"], self.width(self.B["s2"]@P["s2"]), self.B["f2"]@P["f2"]
                lam = lam+torch.exp(f2-0.5*((self.v-m2)/s2)**2-torch.log(s2)-0.5*np.log(2*np.pi))*self.tap
                m2g, s2g, f2g = self.Bg["m2"]@P["m2"], self.width(self.Bg["s2"]@P["s2"]), self.Bg["f2"]@P["f2"]
                g2 = torch.exp(-0.5*((self.vg[None, :]-m2g[:, None])/s2g[:, None])**2)/(s2g[:, None]*np.sqrt(2*np.pi))
                nint2 = torch.sum(torch.exp(f2g)*self.tapg*torch.trapezoid(g2*self.Ag, self.vg, dim=1))*0.1
            L = torch.log(lam)
            mg, fg = self.Bg["m"]@P["m"], self.Bg["f"]@P["f"]; sg = self.width(self.Bg["s"]@P["s"])
            if self.d["q"] == "dphi2":
                gzg = torch.exp(-0.5*((self.vg[None, :]-mg[:, None])/sg[:, None])**2)/(sg[:, None]*np.sqrt(2*np.pi))
                nint = torch.sum(torch.exp(fg)*torch.trapezoid(gzg*self.Ag, self.vg, dim=1))*0.1
            else:
                sgt = torch.sqrt(sg**2+self.eg**2); nint = torch.sum(torch.exp(fg)*(Phi((W-mg)/sgt)-Phi((-W-mg)/sgt)))*0.1
            return L if per_star else L.sum()-nint-nint2
        L = torch.logaddexp(torch.nn.functional.logsigmoid(fl)+lg-torch.log(torch.clamp(Z, 1e-12)), torch.nn.functional.logsigmoid(-fl)+lb)   # stable at f -> 0, 1
        return L if per_star else L.sum()

    def fit(self, th0, maxiter=300):
        th = torch.tensor(th0, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1, max_iter=maxiter, tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); L = -self.lnl(th); L.backward(); return L
        opt.step(closure)
        return th.detach().numpy().copy(), float(self.lnl(th.detach()))

    def grad(self, th):
        t = torch.tensor(th, requires_grad=True); L = -self.lnl(t); return torch.autograd.grad(L, t)[0].numpy()

    def cov(self, th, h=1e-4):
        """inverse Hessian of -lnL by central differences of the autograd gradient (autograd's Hessian returned NaN)."""
        n = len(th); H = np.zeros((n, n))
        for i in range(n):
            a, b = th.copy(), th.copy(); a[i] += h; b[i] -= h; H[i] = (self.grad(a)-self.grad(b))/(2*h)
        H = 0.5*(H+H.T)
        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(H)

    def width(self, x):
        """intrinsic width from its spline value x: sqrt(e^2x + smin^2) (floor); with a cap (--smax): smin + (smax - smin) sigmoid(x)."""
        lib = torch if torch.is_tensor(x) else np
        if self.smax is None:
            return lib.sqrt(lib.exp(2*x)+self.smin**2)
        return self.smin+(self.smax-self.smin)/(1+lib.exp(-x))

    def curves(self, th, x):
        P = self.unpack(torch.tensor(th)); out = {}
        for k in ("m", "s", "f"):
            b = basis(self.knots[k], x)@P[k].numpy(); out[k] = b if k == "m" else (self.width(b) if k == "s" else (np.exp(b) if getattr(self, "ext", False) else 1/(1+np.exp(-b))))
        return out


def start_knots(lo, hi, n=4):
    return list(np.linspace(lo, hi, n))


def init_theta(M, prev=None):
    """initial values: from a previous model's curves at the new knots (warm start), else m = 0, s = 0.1 W, f = 0.05."""
    th = []
    for k, kn in M.knots.items():
        if prev is not None and k in prev[0].knots:
            pm, pth = prev; P = pm.unpack(torch.tensor(pth)); th.append(basis(pm.knots[k], kn)@P[k].numpy())
        else:
            th.append({"m": np.zeros(len(kn)), "s": np.full(len(kn), np.log(0.1*M.W)), "f": np.full(len(kn), np.log(20.) if getattr(M, "ext", False) else -3.),
                       "a1": np.zeros(len(kn)), "a2": np.zeros(len(kn)), "m2": np.array(M.d.get("m2_init", np.zeros(len(kn)))),
                       "s2": np.full(len(kn), np.log(0.6)), "f2": np.full(len(kn), np.log(8.))}[k])
    return np.concatenate(th)


MINSEP = float(sys.argv[sys.argv.index("--minsep")+1]) if "--minsep" in sys.argv else 1.0   # minimum knot spacing [deg]


def aic_search(d, verbose=True, min_sep=None, max_steps=25):
    """Greedy AIC knot insertion (Erkal+2017 Sec. 3.3) for the splines m, s, f (background splines fixed)."""
    min_sep = MINSEP if min_sep is None else min_sep
    knots = {k: start_knots(d["lo"], d["hi"]) for k in ("m", "s", "f")}
    if SFIX:
        knots["s"] = start_knots(d["lo"], d["hi"], SFIX)
    if TWO and d["q"] == "dphi2":                                    # upper branch: 4 fixed knots over TWO_LO..TWO_HI (not searched)
        k2 = list(np.linspace(TWO_LO, TWO_HI, 4)); knots |= dict(m2=k2, s2=k2, f2=k2)
    if d["q"] == "dphi2" and "lb" not in d:
        nb = int(sys.argv[sys.argv.index("--bgknots")+1]) if "--bgknots" in sys.argv else 3   # background spline knots (fixed, equally spaced)
        knots |= dict(a1=start_knots(d["lo"], d["hi"], nb), a2=start_knots(d["lo"], d["hi"], nb))
    M = Model(d, knots); th, L = M.fit(init_theta(M)); k = len(th); aic = 2*k-2*L; hist = [dict(knots={a: list(b) for a, b in M.knots.items()}, lnL=L, aic=aic)]
    if verbose:
        print(f"  {d['q']}: start lnL {L:.1f} AIC {aic:.1f} ({k} params, {len(d['u'])} stars)", flush=True)
    for step in range(max_steps):
        best = None
        for comp in (("m", "f") if SFIX else ("m", "s", "f")):
            kn = M.knots[comp]
            for a, b in zip(kn[:-1], kn[1:]):
                for t in (0.25, 0.5, 0.75):
                    x_ = a+t*(b-a)
                    if min(x_-a, b-x_) < min_sep:
                        continue
                    nk = {c: list(v) for c, v in M.knots.items()}; nk[comp] = sorted(nk[comp]+[x_]); M2 = Model(d, nk)
                    th2, L2 = M2.fit(init_theta(M2, (M, th)), maxiter=100)
                    if best is None or L2 > best[2]:
                        best = (M2, th2, L2, comp, x_)
        if best is None:
            break
        aic2 = 2*len(best[1])-2*best[2]
        if verbose:
            print(f"  step {step}: best new knot {best[3]} at {best[4]:.2f}: lnL {best[2]:.1f} AIC {aic2:.1f} (was {aic:.1f})", flush=True)
        if aic2 >= aic:
            break
        M, th = best[0], best[0].fit(best[1])[0]; L = float(M.lnl(torch.tensor(th))); aic = 2*len(th)-2*L
        hist.append(dict(knots={a: list(b) for a, b in M.knots.items()}, lnL=L, aic=aic))
    return M, th, hist


# ------------------------------------------------------------------ tests
def simulate(d, M, th, seed=1):
    """synthetic data set with the same phi1, errors and background as d, drawn from the fitted model (M, th)."""
    rng = np.random.default_rng(seed); u, e, W = d["u"], d["e"], d["W"]; c = M.curves(th, u); n = len(u)
    st = rng.random(n) < c["f"]; v = np.empty(n)
    sig = np.sqrt(c["s"]**2+e**2); vs = c["m"]+sig*rng.standard_normal(n)
    while np.any(st & (np.abs(vs) >= W)):
        bad = st & (np.abs(vs) >= W); vs[bad] = c["m"][bad]+sig[bad]*rng.standard_normal(bad.sum())
    vg = np.linspace(-W, W, 801)
    if d["q"] == "dphi2":                                         # exp-polynomial background with the upper-branch hole
        vg = d["vg"]
        if "bqg" in d:
            pb = d["bqg"]
        else:
            P = M.unpack(torch.tensor(th)); a1 = (M.B["a1"]@P["a1"]).numpy(); a2 = (M.B["a2"]@P["a2"]).numpy()
            pb = np.exp(a1[:, None]*vg[None, :]+a2[:, None]*vg[None, :]**2)*d["Aw"]
        acc = lambda vv: np.array([np.interp(a_, vg, w_) for a_, w_ in zip(vv, d["Aw"])])        # area acceptance of a stream draw
        rej = st & ((np.abs(vs) >= W) | (rng.random(n) > acc(vs)))
        while np.any(rej):
            vs[rej] = c["m"][rej]+sig[rej]*rng.standard_normal(rej.sum()); rej = rej & ((np.abs(vs) >= W) | (rng.random(n) > acc(vs)))
    else:
        pb = d["bfun"](np.column_stack((np.repeat(u, 1)[:, None].repeat(len(vg), 1).ravel(), np.tile(vg, n)))).reshape(n, len(vg))
    cdf = np.cumsum(pb, 1); cdf /= cdf[:, -1:]; r = rng.random(n); vb = vg[np.argmax(cdf >= r[:, None], 1)]
    vb = np.clip(vb+rng.uniform(-0.5, 0.5, n)*(vg[1]-vg[0]), -W+1e-6, W-1e-6)                 # de-grid, strictly inside the window
    v[:] = np.where(st, vs, vb); d2 = dict(d); d2["v"] = v
    if d["q"] != "dphi2":
        d2["b"] = d["bfun"](np.column_stack((u, v)))
    elif "bqg" in d:                                              # fixed background: re-evaluate at the mock values
        Zb = np.trapezoid(d["bqg"], d["vg"], axis=1); bv = np.array([np.interp(a_, d["vg"], w_) for a_, w_ in zip(v, d["bqg"])])
        d2["lb"] = np.log(np.maximum(bv, 1e-12))-np.log(Zb)
    return d2


def report(M, th, C, name):
    P = M.unpack(torch.tensor(th)); i0 = 0; out = {}
    for k, n in M.sizes().items():
        vals, err = P[k].numpy(), np.sqrt(np.diag(C)[i0:i0+n]); out[k] = dict(knots=M.knots[k].tolist(), values=vals.tolist(), err=err.tolist()); i0 += n
    print(f"  {name}: " + "; ".join(f"{k} knots {np.round(o['knots'], 1).tolist()}" for k, o in out.items() if k in ("m", "s", "f")), flush=True)
    return out


def main():
    qs = [sys.argv[sys.argv.index("--q")+1]] if "--q" in sys.argv else ["pmdec", "pmra", "dphi2"]
    res = {}
    for q in qs:
        t0 = time.time()
        if "--null" in sys.argv:                                         # stream-free: PM window shifted by 6 mas/yr / sky by 8 deg
            d = load(q, null_shift=(6. if q != "dphi2" else 8.)); M, th, hist = aic_search(d)
            c = M.curves(th, np.linspace(d["lo"], d["hi"], 61))
            print(f"NULL {q}: knots m {len(M.knots['m'])} s {len(M.knots['s'])} f {len(M.knots['f'])}; max f {c['f'].max():.4f}; "
                  f"lnL gain vs no stream: {hist[-1]['lnL']-float(M.lnl(torch.tensor(th)) if False else hist[0]['lnL']):.1f}", flush=True)
            continue
        d = load(q); M, th, hist = aic_search(d); C = M.cov(th); out = report(M, th, C, q)
        if "--test" in sys.argv and EXTENDED:                            # simulate() draws mixture mocks (f = probability); extended f is a rate
            sys.exit("--test does not support --extended: use stream_spline_sky_test.py (Poisson mocks)")
        if "--test" in sys.argv:                                         # recovery: refit synthetic data drawn from the fitted model
            zs = []
            for seed in (1, 2, 3):
                ds = simulate(d, M, th, seed); Ms = Model(ds, {k: list(v) for k, v in M.knots.items()}); ths, _ = Ms.fit(th.copy()); Cs = Ms.cov(ths)
                n = len(M.knots["m"]); zs.append((ths[:n]-th[:n])/np.sqrt(np.diag(Cs)[:n]))
                Ma, tha, ha = aic_search(ds, verbose=False)
                print(f"  TEST seed {seed}: track z at knots {np.round(zs[-1], 2).tolist()}; AIC search on the mock -> knots "
                      f"m {len(Ma.knots['m'])} s {len(Ma.knots['s'])} f {len(Ma.knots['f'])} (data: {len(M.knots['m'])}/{len(M.knots['s'])}/{len(M.knots['f'])})", flush=True)
            z = np.concatenate(zs); print(f"  TEST {q}: track z mean {z.mean():+.2f}, sd {z.std():.2f} (expect 0, 1; n = {z.size})", flush=True)
        G = np.linspace(d["lo"], d["hi"], 121); c = M.curves(th, G)
        res[q+TAG] = dict(smax=SMAX, fit=out, cov=C.tolist(), lnL=hist[-1]["lnL"], aic_history=hist, n_stars=len(d["u"]), window=d["W"], phi1_range=[d["lo"], d["hi"]],
                      grid=G.tolist(), m=c["m"].tolist(), s=c["s"].tolist(), f=c["f"].tolist(),
                      note="v = q - frozen track (stream_masks.json); track = frozen track + m")
        print(f"  {q} done in {time.time()-t0:.0f} s", flush=True)
    if res:
        f = ROOT/"results/plot_data/stream_spline_fit.json"; old = json.loads(f.read_text()) if f.exists() else {}
        old.update(res); f.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
