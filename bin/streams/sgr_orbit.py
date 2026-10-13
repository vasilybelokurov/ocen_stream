#!/usr/bin/env python3
"""Sgr (M54) and omega Cen orbits over the last --tback Myr: closest approaches, with Monte Carlo over the catalogue errors and
Chandrasekhar dynamical friction on Sgr.

Present-day phase space: $OCEN_CATALOGUES/gc_catalog_full.fits (NGC 6715 = M54, NGC 5139 = omega Cen; RA, DEC, DIST, PMRA,
PMDEC, RV and their *_ERR columns; Baumgardt & Vasiliev 2021 compilation). Frame: frames.py 'baumgardt'.
Dynamical friction (Chandrasekhar 1943; as used for Sgr e.g. by Vasiliev, Belokurov & Erkal 2021):
  a_df = -4 pi G^2 M rho(r) ln(Lambda) / v^3 [erf(X) - 2X/sqrt(pi) exp(-X^2)] v,  X = v / (sqrt(2) sigma(r)),
  rho(r) and sigma(r) from the host: spherically averaged density and the isotropic Jeans equation
  sigma^2(r) = (1/rho) int_r^inf rho dPhi/dr' dr'; ln(Lambda) = ln(r / (1.6 a_Sgr)) floored at 1 (Hashimoto+2003 form), a_Sgr = Plummer
  scale radius. Sgr mass fixed per run (--msgr); omega Cen feels no friction (as in our stream runs).
Integration: scipy DOP853 (rtol 1e-10) backwards from today, host forces from AGAMA (kpc, km/s, Msun; G = 4.30092e-6).
Usage: python bin/streams/sgr_orbit.py --mw McMillan17 --msgr 4e8 --asgr 1.0 --nmc 200
"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from astropy.table import Table
from scipy.integrate import solve_ivp
from scipy.special import erf
from ocen_dm.streams.restricted import AGAMA_T_MYR, agama_kpc, host_potential
from ocen_dm.streams.frames import to_model

G = 4.300917e-6


def catalogue(name):
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/gc_catalog_full.fits"))
    r = t[[str(n).strip() == name for n in t["NAME"]]][0]
    return {k: float(r[k]) for k in ("RA", "DEC", "DIST", "PMRA", "PMDEC", "RV", "DIST_ERR", "PMRA_ERR", "PMDEC_ERR", "RV_ERR")}


def jeans_sigma(P):
    """Isotropic Jeans sigma(r) and rho(r) of the spherically averaged host, as interpolators."""
    r = np.logspace(-2, 2.7, 600)
    pts = lambda rr: np.column_stack((rr, np.zeros_like(rr), np.zeros_like(rr)))
    # spherical average over directions (Gauss-Legendre in cos theta, uniform phi)
    mu, wmu = np.polynomial.legendre.leggauss(16); ph = np.linspace(0, 2*np.pi, 16, endpoint=False)
    rho = np.zeros_like(r); dphi = np.zeros_like(r)
    for m, wm in zip(mu, wmu):
        st = np.sqrt(1-m*m)
        for p in ph:
            n = np.array([st*np.cos(p), st*np.sin(p), m])
            x = r[:, None]*n[None, :]
            rho += wm/2/len(ph)*P.density(x)
            dphi += wm/2/len(ph)*(-(P.force(x)*n[None, :]).sum(1))
    integrand = rho*dphi
    cum = np.concatenate((np.cumsum((0.5*(integrand[1:]+integrand[:-1])*np.diff(r))[::-1])[::-1], [0.]))
    sig = np.sqrt(np.maximum(cum/np.maximum(rho, 1e-30), 1e-6))
    return lambda rr: np.interp(np.log(rr), np.log(r), rho), lambda rr: np.interp(np.log(rr), np.log(r), sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mw", default="McMillan17"); ap.add_argument("--msgr", type=float, default=4e8)
    ap.add_argument("--asgr", type=float, default=1.0, help="Sgr Plummer scale radius [kpc] (for ln Lambda)")
    ap.add_argument("--tback", type=float, default=1000.); ap.add_argument("--nmc", type=int, default=200); ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    agama_kpc(); P = host_potential(a.mw); rhof, sigf = jeans_sigma(P)
    cs, co = catalogue("NGC 6715"), catalogue("NGC 5139")
    rng = np.random.default_rng(a.seed)

    def draw(c, n):
        return to_model(np.full(n, c["RA"]), np.full(n, c["DEC"]), c["DIST"]+c["DIST_ERR"]*rng.standard_normal(n),
                        c["PMRA"]+c["PMRA_ERR"]*rng.standard_normal(n), c["PMDEC"]+c["PMDEC_ERR"]*rng.standard_normal(n),
                        c["RV"]+c["RV_ERR"]*rng.standard_normal(n))
    n = a.nmc+1
    S = draw(cs, n); O = draw(co, n)
    S[0] = to_model(cs["RA"], cs["DEC"], cs["DIST"], cs["PMRA"], cs["PMDEC"], cs["RV"])[0]
    O[0] = to_model(co["RA"], co["DEC"], co["DIST"], co["PMRA"], co["PMDEC"], co["RV"])[0]
    T = a.tback/AGAMA_T_MYR; tt = np.linspace(0, -T, int(a.tback*2)+1)      # 0.5 Myr output

    def rhs(t, y, M):
        x, v = y[:3], y[3:]
        acc = P.force(x[None, :])[0]
        if M > 0:
            r = np.linalg.norm(x); vv = np.linalg.norm(v); X = vv/(np.sqrt(2)*sigf(r))
            lnL = max(np.log(r/(1.6*a.asgr)), 1.)
            acc = acc - 4*np.pi*G**2*M*rhof(r)*lnL/vv**3*(erf(X)-2*X/np.sqrt(np.pi)*np.exp(-X*X))*v
        return np.concatenate((v, acc))
    res = []
    for i in range(n):
        so = solve_ivp(rhs, (0, -T), S[i], t_eval=tt, method="DOP853", rtol=1e-10, atol=1e-12, args=(a.msgr,)).y.T
        oo = solve_ivp(rhs, (0, -T), O[i], t_eval=tt, method="DOP853", rtol=1e-10, atol=1e-12, args=(0.,)).y.T
        d = np.linalg.norm(so[:, :3]-oo[:, :3], axis=1); k = d.argmin()
        res.append((d[k], -tt[k]*AGAMA_T_MYR, np.linalg.norm(so[k, 3:]-oo[k, 3:]), np.linalg.norm(so[:, :3], axis=1).min()))
        if i == 0:
            nominal = dict(sgr=so[::20].tolist(), ocen=oo[::20].tolist(), t_myr=(-tt[::20]*AGAMA_T_MYR).tolist())
    res = np.array(res)
    q = lambda c: np.percentile(res[:, c], [2.5, 50, 97.5])
    print(f"{a.mw} M_Sgr {a.msgr:.1e} (a {a.asgr} kpc), last {a.tback:.0f} Myr, {n} orbits (nominal + {a.nmc} MC):")
    print(f"  nominal: closest approach {res[0,0]:.2f} kpc at {res[0,1]:.0f} Myr ago, dv {res[0,2]:.0f} km/s; Sgr min r_gal {res[0,3]:.2f} kpc")
    print(f"  MC 2.5/50/97.5%: closest {np.round(q(0),2)} kpc; time {np.round(q(1))} Myr ago; dv {np.round(q(2))} km/s; Sgr min r_gal {np.round(q(3),2)} kpc")
    out = ROOT/f"results/plot_data/sgr_orbit_{Path(a.mw).stem}_{a.msgr:.0e}.json"
    out.write_text(json.dumps(dict(args=vars(a), sgr_obs=cs, ocen_obs=co, nominal=nominal, mc=res.tolist()), indent=1))


if __name__ == "__main__":
    main()
