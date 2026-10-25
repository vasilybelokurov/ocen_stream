#!/usr/bin/env python3
"""How loose can the DOP853 tolerance be before the fit notices? (2026-10-25)

At the 5 test points: JAX DOP853 sprays at ACC = 1e-6, 1e-5, 1e-4 (each particle's attempt count recorded), compared with the
1e-8 JAX sprays (results/gpu/spray_check/*_jax_dop853.npz) and the AGAMA sprays (same bank), scored with fixed per-slot banks 7/8/9
(each spray its own chi selection). Reported: total and slowest attempts (work), dchi2 vs 1e-8 and vs AGAMA, max |summary shift|
in sigma_data vs 1e-8.
Usage: PYTHONPATH=src python bin/streams/gpu_tolerance_scan.py -> results/gpu/tolscan/*.npz, results/plot_data/gpu_tolerance_scan.json
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
G = ROOT/"results/gpu"; OUT = G/"tolscan"; SC = G/"spray_check"; ACCS = (1e-6, 1e-5, 1e-4)


def sprays():
    import jax; jax.config.update("jax_enable_x64", True)
    from ocen_stream import jaxspray as J, jaxmodel as JM
    OUT.mkdir(parents=True, exist_ok=True); pts = json.loads((G/"points.json").read_text()); off = pts.pop("_epoch_offset")
    ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz"); trel, keep = JM.schedule(inp, off)
    for name, p in pts.items():
        tb, ic, t0, _ = JM.release(p, ht, inp, trel, keep)
        for acc in ACCS:
            f = OUT/f"{name}_acc{acc:.0e}.npz"
            if f.exists():
                continue
            t1 = time.time(); fin = np.zeros(len(ic), int); y, chi, natt = JM.dop853_chi(ic, t0, tb, acc, 32, finish=fin); o = JM.observables(jax.numpy.asarray(y), inp)
            np.savez(f, age_cut_myr=700., chi=chi, age=np.asarray((inp["T"]-t0)*JM.T_MYR), d=np.asarray(o["dist"]), finish=fin,
                     **{k: np.asarray(o[k]) for k in ("l", "b", "pmra", "pmdec", "vlos")})
            print(f"{name} acc {acc:.0e}: {time.time()-t1:.0f} s CPU; attempts total {fin.sum()/1e6:.1f} M, median {np.median(fin):.0f}, slowest {natt}", flush=True)


def main():
    if "--spray" in sys.argv:
        return sprays()
    subprocess.run([sys.executable, __file__, "--spray"], check=True)
    sys.argv = [sys.argv[0]]
    import stream_model_compare as MC
    from ocen_stream.jaxscore import make_bank
    D = MC.data_vector(); O = MC.Obs(); sd = np.r_[D["sky_ce"], D["sky_se"], np.sqrt(np.diag(D["pmra_C"])), np.sqrt(np.diag(D["pmdec_C"]))]
    vec = lambda M: np.r_[M["sky_c"], M["sky_s"], M["pmra"], M["pmdec"]]; res = {}
    pts = [k for k in json.loads((G/"points.json").read_text()) if not k.startswith("_")]
    for name in pts:
        ref8, ag = SC/f"{name}_jax_dop853.npz", SC/f"{name}_agama.npz"; r = res[name] = {}
        banks = [make_bank(len(np.load(ref8)["chi"]), MC.NBOOT, s) for s in (7, 8, 9)]
        sc = lambda f, b: MC.measure(O, D, str(f), None, bank=b)
        M8 = [sc(ref8, b) for b in banks]; MA = [sc(ag, b) for b in banks]
        c8 = [MC.chi2(D, M)["total"] for M in M8]; ca = [MC.chi2(D, M)["total"] for M in MA]; r["chi2_1e-8"] = c8; r["chi2_agama"] = ca
        for acc in ACCS:
            f = OUT/f"{name}_acc{acc:.0e}.npz"; z = np.load(f); Mx = [sc(f, b) for b in banks]; cx = [MC.chi2(D, M)["total"] for M in Mx]
            sh = max(float(np.nanmax(np.abs((vec(a)-vec(b))/sd))) for a, b in zip(Mx, M8))
            r[f"{acc:.0e}"] = dict(attempts_M=float(z["finish"].sum()/1e6), slowest=int(z["finish"].max()), dchi2_vs_1e8=[x-y for x, y in zip(cx, c8)],
                                   dchi2_vs_agama=[x-y for x, y in zip(cx, ca)], max_shift_sigma=sh)
            q = r[f"{acc:.0e}"]; print(f"{name} acc {acc:.0e}: attempts {q['attempts_M']:.1f} M (slowest {q['slowest']}); dchi2 vs 1e-8 "
                                     f"{' / '.join(f'{d:+.2f}' for d in q['dchi2_vs_1e8'])}; vs AGAMA {' / '.join(f'{d:+.2f}' for d in q['dchi2_vs_agama'])}; "
                                     f"max |shift| {sh:.3f} sigma", flush=True)
        print(f"{name}: 1e-8 vs AGAMA {' / '.join(f'{a-b:+.2f}' for a, b in zip(c8, ca))}", flush=True)
    (ROOT/"results/plot_data/gpu_tolerance_scan.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
