"""Cross-machine diagnostic (2026-10-29; docs/reviews/cross_machine_fable_codex_20261029.md): where do two machines first differ?

For one unit point u the chi2 pipeline is run stage by stage, exactly as pilots.Evaluator runs it, and every intermediate is saved:
  A host tables (jaxspray.host_for, JAX)       B today's centre state (jaxmodel.today_xv, JAX)
  C centre orbit back in time (C orbit_back)   D Hermite resampling (numpy) -> orb, orbit_r
  E release ICs (release_finish, JAX on GPU)   F particle integration (CUDA kernel: y, chi, attempts)
  G observables (JAX)                          H summaries m, e and chi2 (jaxscore)
The final chi2 is checked against pilots.Evaluator (must be bit-identical). With --feed OTHER.npz (a dump from another machine) each
stage is also re-run here on the OTHER machine's saved inputs and compared with its saved outputs: a stage that reproduces the other
machine's bits from its inputs is not a source; the first stage that does not is.
Usage: python -m ocen_stream.xdump --inputs DIR --u U.json --out DUMP.npz [--feed OTHER.npz] [--bank 7]
       (U.json: {"u": [9 numbers]} or the colab_ref.json format, first point)
"""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from . import jaxspray as J, jaxmodel as JM, jaxscore as JS, cspray, cuspray, pilots as PL


def _save_tb(tb, pre):
    return {pre+k: np.asarray(v) for k, v in tb.items()}


def _load_tb(z, pre):
    return {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}


def _cmp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    eq = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a == b)
    if eq.all():
        return "bit-identical"
    d = np.abs(a.astype(float)-b.astype(float)); rel = d/np.maximum(np.abs(b.astype(float)), 1e-300)
    return f"DIFFER: {int((~eq).sum())} of {a.size} elements, max |diff| {np.nanmax(d):.3e}, max rel {np.nanmax(rel):.3e}"


class Pipe:
    def __init__(s, inputs, bank_seed):
        I = Path(inputs); s.I = I; s.ht, s.inp, s.si = J.load_host(I/"host_tables.npz"), JM.load_inputs(I/"spray_inputs.npz"), JS.load(I/"score_inputs.npz")
        pts = json.loads((I/"points.json").read_text()); s.trel, s.keep = JM.schedule(s.inp, pts["_epoch_offset"])
        s.bank = {k: jnp.asarray(v) for k, v in JS.make_bank(len(s.keep), 8, bank_seed).items()}; s.seed = bank_seed
        s.obs = jax.jit(lambda y: JM.observables(y, s.inp)); s.c2 = jax.jit(lambda m, e: JS.chi2(m, e, s.si))       # as the Evaluator

    def host(s, p):                                                                      # A
        return J.host_for(s.ht, p["angle"], p["omega"], p["amp"], p["size"], p["halo_q"], p["vc"], s.inp["T"])

    def xv0(s, p):                                                                       # B
        return np.asarray(JM.today_xv(p["dist"], p["pmra"], p["pmdec"], s.inp))

    def resample(s, tk, yk, ak):                                                         # D (as jaxmodel.release_prepare)
        tcu = np.linspace(0., float(s.inp["T"]), 39113); H = lambda t: np.concatenate((cspray.hermite(t, tk, yk[:, :3], yk[:, 3:]), cspray.hermite(t, tk, yk[:, 3:], ak)), 1)
        return tcu, H(tcu), H(np.asarray(s.trel))

    def full_tb(s, tbh, tcu, orbu):
        i = s.inp; tcu = jnp.asarray(tcu)
        return dict(tbh, sat=i["sat"], lr0=i["lr0"], dlr=i["dlr"], nr=i["nr"], mtot=i["mtot"], G=i["G"], tc0=tcu[0], dtc=tcu[1]-tcu[0], orb=jnp.asarray(orbu))

    def finish(s, tb, orbit_r):                                                          # E
        tb2, ic, t0, _ = JM.release_finish(tb, jnp.asarray(orbit_r), s.trel, s.inp, s.keep); return tb2, np.asarray(ic), np.asarray(t0)

    def kernel(s, ic, t0, tb):                                                           # F
        (y, chi, natt), = cuspray.integrate([(ic, t0, tb)], JM.SPRAY_ACC); return np.asarray(y), np.asarray(chi), np.asarray(natt)

    def score(s, y, chi, t0):                                                            # G, H (as Evaluator._models)
        o = s.obs(jnp.asarray(y)); age = jnp.asarray((float(s.inp["T"])-t0)*JM.T_MYR)
        m, e = JS.measure_capped(o, jnp.asarray(chi*JM.T_MYR), age, s.si, s.bank); return {k: np.asarray(v) for k, v in o.items()}, np.asarray(m), np.asarray(e), float(s.c2(m, e))


def run(P, u):
    p = PL.to_params(u, P.ht); tbh = P.host(p); xv0 = P.xv0(p); tk, yk, ak = cspray.orbit_back(xv0, tbh, JM.ORBIT_ACC)
    tcu, orbu, orbit_r = P.resample(tk, yk, ak); tb = P.full_tb(tbh, tcu, orbu); tb2, ic, t0 = P.finish(tb, orbit_r)
    y, chi, natt = P.kernel(ic, t0, tb2); o, m, e, c2 = P.score(y, chi, t0)
    D = dict(u=u, params=np.array([p[k] for k in PL.NAMES]), xv0=xv0, tk=tk, yk=yk, ak=ak, tcu=tcu, orbu=orbu, orbit_r=orbit_r, ic=ic, t0=t0,
             y=y, chi=chi, natt=natt, m=m, e=e, chi2=np.float64(c2), **_save_tb(tbh, "tbh__"), **_save_tb(tb2, "tb__"),
             **{f"obs__{k}": v for k, v in o.items()})
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--u", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--feed"); ap.add_argument("--bank", type=int, default=7); a = ap.parse_args()
    U = json.loads(Path(a.u).read_text()); u = np.array(U["u"][0] if isinstance(U["u"][0], list) else U["u"], np.float64)
    P = Pipe(a.inputs, a.bank); env = PL.environment(); print("environment", json.dumps(env), flush=True)
    D = run(P, u); D["environment"] = np.array(json.dumps(env))
    ev = PL.Evaluator(a.inputs, tempfile.mkdtemp()); c_ev = float(ev(u[None], "xdump", seed=a.bank)[0])
    print(f"chi2 {D['chi2']!r}; Evaluator {c_ev!r}; dump reproduces the production path: {D['chi2'] == c_ev}", flush=True)
    np.savez_compressed(a.out, **D); print("saved", a.out, flush=True)
    if a.feed:
        B = np.load(a.feed); print("\nother machine:", str(B["environment"]), "\nother chi2", float(B["chi2"]), "| here", float(D["chi2"]))
        print("\n-- own outputs vs the other machine's (same u):")
        for k in ("params", "xv0", "tk", "yk", "orbu", "orbit_r", "ic", "y", "chi", "natt", "m", "e", "chi2"):
            print(f"  {k:8s} {_cmp(D[k], B[k])}")
        tbB, tbhB = _load_tb(B, "tb__"), _load_tb(B, "tbh__")
        print("  host tables:", {k: _cmp(np.asarray(D['tbh__'+k]), v) for k, v in tbhB.items() if np.asarray(v).size > 1})
        print("\n-- each stage re-run here on the OTHER machine's inputs vs its outputs (bit-identical = not a source):")
        tk, yk, ak = cspray.orbit_back(B["xv0"], tbhB, JM.ORBIT_ACC); print(f"  C orbit_back(xv0, host tables)  tk {_cmp(tk, B['tk'])}; yk {_cmp(yk, B['yk'])}")
        _, orbu, orbit_r = P.resample(B["tk"], B["yk"], B["ak"]); print(f"  D hermite(tk, yk, ak)          orbu {_cmp(orbu, B['orbu'])}; orbit_r {_cmp(orbit_r, B['orbit_r'])}")
        _, ic, _ = P.finish(P.full_tb(tbhB, B["tcu"], B["orbu"]), B["orbit_r"]); print(f"  E release_finish(tb, orbit_r)  ic {_cmp(ic, B['ic'])}")
        y, chi, natt = P.kernel(B["ic"], B["t0"], tbB); print(f"  F kernel(ic, t0, tb)           y {_cmp(y, B['y'])}; chi {_cmp(chi, B['chi'])}; attempts {_cmp(natt, B['natt'])}")
        o, m, e, c2 = P.score(B["y"], B["chi"], B["t0"]); print(f"  G observables(y)               l {_cmp(o['l'], B['obs__l'])}; pmdec {_cmp(o['pmdec'], B['obs__pmdec'])}")
        print(f"  H score(y, chi)                m {_cmp(m, B['m'])}; e {_cmp(e, B['e'])}; chi2 {c2!r} vs {float(B['chi2'])!r}")
        p = PL.to_params(B["u"], P.ht); print(f"  A host_for(params)             ", {k: _cmp(np.asarray(v), tbhB[k]) for k, v in P.host(p).items() if np.asarray(v).size > 1})
        print(f"  B today_xv(params)             {_cmp(P.xv0(p), B['xv0'])}")


if __name__ == "__main__":
    main()
