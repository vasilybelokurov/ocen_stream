#!/usr/bin/env python3
"""C spray (cspray) at the 5 test points: chi2 vs the AGAMA sprays and the JAX 1e-6 sprays (same release bank), and CPU timing (2026-10-26).

Release (JAX + the C centre orbit) and the whole integration in C (cspray.integrate, all CPU cores, tolerance 1e-6), observables, then
the knot score with fixed per-slot banks 7/8/9, each spray its own chi selection.
Usage: PYTHONPATH=src python bin/streams/cspray_check.py -> results/gpu/cspray_check/*.npz, results/plot_data/cspray_check.json
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
G = ROOT/"results/gpu"; OUT = G/"cspray_check"; SC = G/"spray_check"; TS = G/"tolscan"


def sprays():
    import jax; jax.config.update("jax_enable_x64", True)
    from ocen_stream import jaxspray as J, jaxmodel as JM, cspray as CS
    OUT.mkdir(parents=True, exist_ok=True); pts = json.loads((G/"points.json").read_text()); off = pts.pop("_epoch_offset")
    ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz"); trel, keep = JM.schedule(inp, off); CS.lib()
    for name, p in pts.items():
        t1 = time.time(); tb, ic, t0, _ = JM.release(p, ht, inp, trel, keep); ic.block_until_ready(); t_rel = time.time()-t1
        t1 = time.time(); y, chi, natt = CS.integrate(np.asarray(ic), np.asarray(t0), tb, JM.SPRAY_ACC); t_int = time.time()-t1
        o = JM.observables(jax.numpy.asarray(y), inp)
        np.savez(OUT/f"{name}_c.npz", age_cut_myr=700., chi=chi*JM.T_MYR, age=np.asarray((inp["T"]-t0)*JM.T_MYR), d=np.asarray(o["dist"]), natt=natt,
                 **{k: np.asarray(o[k]) for k in ("l", "b", "pmra", "pmdec", "vlos")})
        print(f"{name}: release {t_rel:.2f} s, C integration {t_int:.1f} s ({natt.sum()/1e6:.1f} M attempts, slowest {natt.max()}, failed {np.sum(natt < 0)})", flush=True)


def main():
    if "--spray" in sys.argv:
        return sprays()
    subprocess.run([sys.executable, __file__, "--spray"], check=True)
    sys.argv = [sys.argv[0]]
    import stream_model_compare as MC
    from ocen_stream.jaxscore import make_bank
    D = MC.data_vector(); O = MC.Obs(); res = {}
    for f in sorted(OUT.glob("*_c.npz")):
        name = f.stem[:-2]; r = res[name] = {}
        for lab, ref in (("agama", SC/f"{name}_agama.npz"), ("jax1e-6", TS/f"{name}_acc1e-06.npz")):
            d = []
            for seed in (7, 8, 9):
                b = make_bank(179200, MC.NBOOT, seed); d.append(MC.chi2(D, MC.measure(O, D, str(f), None, bank=b))["total"]-MC.chi2(D, MC.measure(O, D, str(ref), None, bank=b))["total"])
            r[lab] = d
        print(f"{name}: dchi2 C - AGAMA {' / '.join(f'{x:+.2f}' for x in r['agama'])}; C - JAX 1e-6 {' / '.join(f'{x:+.2f}' for x in r['jax1e-6'])}", flush=True)
    (ROOT/"results/plot_data/cspray_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
