#!/usr/bin/env python3
"""Stragglers of the JAX DOP853 spray: what happens to chi2 if particles still unfinished after CAP attempts are dropped (2026-10-25)?

On a GPU the last ~0.1% of particles (trapped near the cluster centre, up to ~13-17k attempts) take ~5 of the ~9.5 s of integration
(gpu_dop853_trace.py). Particles are independent, so one full DOP853 spray per test point with each particle's finishing attempt
recorded (dop853_chi(finish=...)) gives every cap at once: dropped = finish > CAP (their chi set to -1, i.e. outside the chi > 0
selection). Scored with fixed per-slot banks 7/8/9 (as the GPU score) against the uncapped spray. Also reports, for the dropped
particles: number, age, distance from the cluster centre today, chi sign and whether they would pass the masks' phi1 range.
Usage: PYTHONPATH=src python bin/streams/gpu_straggler_test.py -> results/gpu/straggler/*.npz, results/plot_data/gpu_straggler_test.json
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
G = ROOT/"results/gpu"; OUT = G/"straggler"; CAPS = (800, 1600, 3200)


def sprays():
    import jax; jax.config.update("jax_enable_x64", True)
    from ocen_stream import jaxspray as J, jaxmodel as JM
    OUT.mkdir(parents=True, exist_ok=True); pts = json.loads((G/"points.json").read_text()); off = pts.pop("_epoch_offset")
    ht, inp = J.load_host(G/"host_tables.npz"), JM.load_inputs(G/"spray_inputs.npz"); trel, keep = JM.schedule(inp, off)
    for name, p in pts.items():
        if (OUT/f"{name}.npz").exists():
            continue
        t1 = time.time(); tb, ic, t0, rj = JM.release(p, ht, inp, trel, keep); fin = np.zeros(len(ic), int)
        y, chi, natt = JM.dop853_chi(ic, t0, tb, 1e-8, 32, finish=fin); o = JM.observables(np.asarray(y), inp)
        dc = np.linalg.norm(y[:, :3]-np.asarray(tb["orb"])[np.argmax(np.asarray(tb["tcs"]))][:3], axis=1)   # distance from the centre today
        np.savez(OUT/f"{name}.npz", age_cut_myr=700., chi=chi, age=np.asarray((inp["T"]-t0)*JM.T_MYR), d=np.asarray(o["dist"]), finish=fin,
                 dcen=dc, rj=np.asarray(rj), **{k: np.asarray(o[k]) for k in ("l", "b", "pmra", "pmdec", "vlos")})
        print(f"{name}: {time.time()-t1:.0f} s; slowest {natt} attempts; unfinished at caps {CAPS}: {[int((fin > c).sum()) for c in CAPS]}", flush=True)


def main():
    if "--spray" in sys.argv:
        return sprays()
    subprocess.run([sys.executable, __file__, "--spray"], check=True)
    sys.argv = [sys.argv[0]]
    import stream_model_compare as MC
    from ocen_stream.jaxscore import make_bank
    from ocen_stream.path import project_gc
    D = MC.data_vector(); O = MC.Obs(); res = {}
    for f in sorted(OUT.glob("*.npz")):
        if "_cap" in f.stem:
            continue
        z = dict(np.load(f)); name = f.stem; r = res[name] = {}; base = []
        for seed in (7, 8, 9):
            base.append(MC.chi2(D, MC.measure(O, D, str(f), None, bank=make_bank(len(z["chi"]), MC.NBOOT, seed)))["total"])
        r["chi2"] = base
        for cap in CAPS:
            drop = z["finish"] > cap; zc = dict(z, chi=np.where(drop, -1., z["chi"])); fc = OUT/f"{name}_cap{cap}.npz"; np.savez(fc, **zc)
            dc = [MC.chi2(D, MC.measure(O, D, str(fc), None, bank=make_bank(len(z["chi"]), MC.NBOOT, s)))["total"]-b for s, b in zip((7, 8, 9), base)]
            u = project_gc(z["l"][drop], z["b"][drop], O.GC)[0] if drop.any() else np.array([])
            r[f"cap{cap}"] = dict(n_drop=int(drop.sum()), dchi2=dc, n_drop_selected=int(np.sum(drop & (z["chi"] > 0) & (z["age"] < 700))),
                                  age_med=float(np.median(z["age"][drop])) if drop.any() else None,
                                  dcen_over_rj_med=float(np.median(z["dcen"][drop]/z["rj"][drop])) if drop.any() else None,
                                  phi1_range=[float(u.min()), float(u.max())] if u.size else None)
            q = r[f"cap{cap}"]; print(f"{name} cap {cap}: dropped {q['n_drop']} ({q['n_drop_selected']} in the chi>0 selection), median age "
                                      f"{q['age_med']} Myr, median distance from centre {q['dcen_over_rj_med']} r_J, phi1 {q['phi1_range']}; "
                                      f"dchi2 {' / '.join(f'{d:+.2f}' for d in dc)} (chi2 {' / '.join(f'{b:.1f}' for b in base)})", flush=True)
    (ROOT/"results/plot_data/gpu_straggler_test.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
