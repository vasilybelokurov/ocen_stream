#!/usr/bin/env python3
"""Recovery test of the two-component extended sky model (stream_spline_fit.py --extended --two): Poisson mocks drawn from the
fitted model (fixed band background x observed-area map + main stream + upper branch) on the 0.1-deg phi1 x v grid, refitted
with the same knots; z-scores of the track knots (main and branch), and the AIC knot search on one mock (2026-10-24).
Usage: PYTHONPATH=src python bin/streams/stream_spline_sky_test.py --W 6 --hi 25 --extended --nohole --two --smax 1.5 --minsep 2 --key KEY
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
from stream_spline_fit import load, Model, aic_search

KEY = sys.argv[sys.argv.index("--key")+1]


def mock(d, M, th, rng):
    ug, vg, Ag = d["ug"], d["vg"], d["Ag"]; Eu = d["area"]["Eu"]; jg = np.clip(np.digitize(ug, Eu)-1, 0, len(Eu)-2)
    cell = (Eu[1]-Eu[0])*(d["area"]["Ey"][1]-d["area"]["Ey"][0]); dv = vg[1]-vg[0]
    lam = np.clip(d["bg_coef"][jg]@np.vander(vg, 3).T, 0, None)/cell*Ag                     # background per deg^2 (observed)
    P = M.unpack(torch.tensor(th)); g = lambda m_, s_: np.exp(-0.5*((vg[None, :]-m_[:, None])/s_[:, None])**2)/(s_[:, None]*np.sqrt(2*np.pi))
    m1, s1, r1 = (M.Bg["m"]@P["m"]).numpy(), M.width((M.Bg["s"]@P["s"]).numpy()), np.exp((M.Bg["f"]@P["f"]).numpy())
    lam = lam+r1[:, None]*g(m1, s1)*Ag
    if "m2" in M.knots:
        m2, s2, r2 = (M.Bg["m2"]@P["m2"]).numpy(), M.width((M.Bg["s2"]@P["s2"]).numpy()), np.exp((M.Bg["f2"]@P["f2"]).numpy())*M.tapg.numpy()
        lam = lam+r2[:, None]*g(m2, s2)*Ag
    n = rng.poisson(lam*0.1*dv); iu, iv = np.nonzero(n); rep = n[iu, iv]
    u = np.repeat(ug[iu], rep)+rng.uniform(-0.05, 0.05, rep.sum()); v = np.repeat(vg[iv], rep)+rng.uniform(-0.5, 0.5, rep.sum())*dv
    ok = (np.abs(v) < d["W"]) & (u >= d["lo"]) & (u < d["hi"]); u, v = u[ok], v[ok]
    ju = np.clip(np.digitize(u, Eu)-1, 0, len(Eu)-2); bq = np.clip(np.sum(d["bg_coef"][ju]*np.vander(v, 3), axis=1), 1e-3, None)
    dm = dict(d); dm.update(u=u, v=v, e=np.zeros(u.size), hlo=np.zeros(u.size), hhi=np.zeros(u.size), lam_b=bq/cell,
                            Aw=np.ones((u.size, len(vg))), lb=np.zeros(u.size))
    return dm


def main():
    F = json.loads((ROOT/"results/plot_data/stream_spline_fit.json").read_text())[KEY]; d = load("dphi2"); fit = F["fit"]
    kn = {k: v["knots"] for k, v in fit.items()}; M = Model(d, kn); th = np.concatenate([np.array(fit[k]["values"]) for k in M.sizes()])
    rng = np.random.default_rng(7); z1, z2 = [], []; sz = M.sizes(); off = np.cumsum([0]+list(sz.values()))[:-1]; idx = dict(zip(sz, off))
    for seed in range(3):
        dm = mock(d, M, th, rng); Ms = Model(dm, kn); ths, _ = Ms.fit(th.copy()); C = Ms.cov(ths); e = np.sqrt(np.abs(np.diag(C)))
        a, n = idx["m"], sz["m"]; z1.append((ths[a:a+n]-th[a:a+n])/e[a:a+n])
        if "m2" in sz:
            a, n = idx["m2"], sz["m2"]; z2.append((ths[a:a+n]-th[a:a+n])/e[a:a+n])
        print(f"mock {seed}: {len(dm['u'])} stars (data {len(d['u'])}); main track z {np.round(z1[-1], 2).tolist()}"
              + (f"; branch track z {np.round(z2[-1], 2).tolist()}" if z2 else ""), flush=True)
        if seed == 0:
            Ma, tha, ha = aic_search(dm, verbose=False)
            print(f"   AIC search on the mock: knots m {len(Ma.knots['m'])} s {len(Ma.knots['s'])} f {len(Ma.knots['f'])} "
                  f"(data: {sz['m']}/{sz['s']}/{sz['f']})", flush=True)
    z1 = np.concatenate(z1); print(f"main track z: mean {z1.mean():+.2f}, sd {z1.std():.2f} (n {z1.size})")
    if z2:
        z2 = np.concatenate(z2); print(f"branch track z: mean {z2.mean():+.2f}, sd {z2.std():.2f} (n {z2.size})")


if __name__ == "__main__":
    main()
