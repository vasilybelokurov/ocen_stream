#!/usr/bin/env python3
"""Fixed-step integrators vs AGAMA's adaptive integrator for spray particles (2026-10-25; adapted from Fable's design-review check).

Set-up: best free-fit host (angle 14, Omega 33, A 1.27, S 1.44, q 0.8, v_c 222; d 5.67, PM (-3.254, -6.725)) + the moving progenitor
(frozen A_nodm mass profile), NP trailing particles released uniformly over the last 700 Myr (release model of spray.py), all
integrated to today. Forces from AGAMA for every scheme, so only the integration (and float32 rounding) error is measured.
Reference: AGAMA accuracy 1e-10. Schemes: AGAMA 1e-6 / 1e-8; kick-drift-kick leapfrog (2nd order) and Yoshida (1990) 4th-order
symplectic composition, fixed step per particle (its age split into equal steps), time passed to every force call; float32 by
rounding the state after each sub-step. Also records each particle's closest approach to the progenitor centre (in r_J units).
Usage: PYTHONPATH=src python bin/streams/integrator_test.py [NP] -> results/plot_data/integrator_test.npz, plots/integrator_test.png
"""
import sys, time, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR, FrozenCore, satellite_potential
from ocen_stream.frames import frame_r0, to_model, OCEN_OBS
from ocen_stream.spray import rj_vj_R, release_ic

agama = agama_kpc(); T = 1955.58/AGAMA_T_MYR; NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
host = host_potential("x", bar_omega=33., bar_angle_deg=14., t_today=T, bar_amp=1.27, bar_size=1.44,
                      axi_variant=dict(halo_q=0.8, vc_target=222., r0_kpc=frame_r0("gravity21")))
today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], 5.67, -3.254, -6.725, OCEN_OBS["vlos"], "gravity21")[0]
prof = json.loads((ROOT/"results/nbody/A_nodm/model_profiles.json").read_text())
r_kpc, M_enc = np.array(prof["r_pc"])/1e3, np.array(prof["enclosed"]["total"])
_, back = agama.orbit(potential=host, ic=today, timestart=T, time=-T, trajsize=2, accuracy=1e-12)
tc, orb = agama.orbit(potential=host, ic=back[-1], timestart=0., time=T, trajsize=int(np.ceil(T*AGAMA_T_MYR/0.05))+1, accuracy=1e-12)
rng = np.random.default_rng(7); tr = np.sort(rng.uniform(T-700./AGAMA_T_MYR, T, NP))
orbit_r = np.array([np.interp(tr, tc, orb[:, i]) for i in range(6)]).T
rj, vj, R = rj_vj_R(host, orbit_r, tr, lambda rr: np.interp(np.log(rr), np.log(r_kpc), M_enc))
ic, arm = release_ic(orbit_r, rj, vj, R, rng); keep = arm == 1; ic, t0, rj0 = ic[keep], np.repeat(tr, 2)[keep], np.repeat(rj, 2)[keep]
sat = satellite_potential(np.zeros((0, 3)), np.zeros(0), core=FrozenCore(r_kpc, M_enc))
total = agama.Potential(host, agama.Potential(potential=sat, center=np.column_stack((tc, orb))))
prog = lambda t: np.array([np.interp(t, tc, orb[:, i]) for i in range(3)]).T
print(f"{len(ic)} trailing particles; r_J median {np.median(rj0)*1e3:.0f} pc", flush=True)


def agama_run(acc):
    t1 = time.time(); xv = np.vstack(agama.orbit(potential=total, ic=ic, timestart=t0, time=T-t0, trajsize=1, accuracy=acc, verbose=False)[:, 1])
    return xv, time.time()-t1


W1 = 1/(2-2**(1/3)); W0 = -2**(1/3)/(2-2**(1/3))
SCHEMES = dict(leapfrog=((0.5, 0.5), (1.,)), yoshida4=((W1/2, (W0+W1)/2, (W0+W1)/2, W1/2), (W1, W0, W1)))   # drift / kick coefficients


def fixed_step(scheme, dt_myr, f32=False):
    """drift-kick-...-drift composition (DKD leapfrog / Yoshida-4), per-particle step = age / round(age / dt); returns final xv,
    force evaluations per particle (max), wall time and the closest approach to the progenitor [r_J]."""
    cd, ck = SCHEMES[scheme]; n = np.maximum(np.rint((T-t0)/(dt_myr/AGAMA_T_MYR)).astype(int), 1); dti = (T-t0)/n
    cast = (lambda a: a.astype(np.float32).astype(np.float64)) if f32 else (lambda a: a)
    x, v = cast(ic[:, :3].copy()), cast(ic[:, 3:].copy()); dmin = np.full(len(x), np.inf); t1 = time.time()
    for k in range(n.max()):
        h = np.where(k < n, dti, 0.)[:, None]; frac = 0.
        for i in range(len(cd)):
            x = cast(x+cd[i]*h*v); frac += cd[i]
            if i < len(ck):
                tt = t0+(k+frac)*dti; v = cast(v+ck[i]*h*total.force(x, t=tt))
                if i == 0:
                    dmin = np.minimum(dmin, np.where(k < n, np.linalg.norm(x-prog(tt), axis=1), np.inf))
    return np.hstack((x, v)), len(ck)*n.max(), time.time()-t1, dmin/rj0


def main():
    ref, tref = agama_run(1e-10); dkpc = np.linalg.norm(ref[:, :3]-np.array([-8.2, 0., 0.02]), axis=1)     # rough heliocentric distance
    to_mu = lambda xv: np.linalg.norm(xv[:, 3:]-ref[:, 3:], axis=1)/(4.74047*dkpc)                              # |dv| as mas/yr at the star
    res = {}
    for acc in (1e-6, 1e-8):
        xv, el = agama_run(acc); res[f"agama {acc:.0e}"] = dict(mu=to_mu(xv), nf=np.nan, sec=el)
    for scheme, dts in (("leapfrog", (0.2, 0.1, 0.05, 0.02, 0.01)), ("yoshida4", (0.8, 0.4, 0.2, 0.1, 0.05))):
        for dt in dts:
            xv, nf, el, dmin = fixed_step(scheme, dt); res[f"{scheme} {dt}"] = dict(mu=to_mu(xv), nf=nf, sec=el, dt=dt)
    xv, nf, el, _ = fixed_step("yoshida4", 0.2, f32=True); res["yoshida4 0.2 f32"] = dict(mu=to_mu(xv), nf=nf, sec=el, dt=0.2)
    P = (50, 90, 99, 99.9)
    for k, r in res.items():
        print(f"{k:22s} force evals {r['nf']:>7} ({r['sec']:5.0f} s): |dmu| mas/yr pct {P}: " + " ".join(f"{np.percentile(r['mu'], p):.1e}" for p in P), flush=True)
    np.savez(ROOT/"results/plot_data/integrator_test.npz", dmin_rj=dmin, **{k.replace(" ", "_"): r["mu"] for k, r in res.items()})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for k, r in res.items():
        if k.startswith("agama") or k in ("leapfrog 0.02", "yoshida4 0.2", "yoshida4 0.2 f32", "leapfrog 0.1", "yoshida4 0.8"):
            s = np.sort(r["mu"]); ax[0].plot(s, 1-np.arange(s.size)/s.size, label=k, ls="--" if k.startswith("agama") else "-")
    ax[0].set(xscale="log", yscale="log", xlabel="|dv| vs AGAMA 1e-10, as mas/yr", ylabel="fraction of particles with larger error",
              title="error distribution (survival function)", ylim=(1/len(ic), 1.05)); ax[0].legend(fontsize=8)
    for p, ls in zip((50, 90, 99), (":", "-", "--")):
        for scheme, col in (("leapfrog", "C0"), ("yoshida4", "C3")):
            ks = [k for k in res if k.startswith(scheme+" ") and "f32" not in k]
            ax[1].plot([res[k]["nf"] for k in ks], [np.percentile(res[k]["mu"], p) for k in ks], "o"+ls, color=col, label=f"{scheme} {p}%")
        ax[1].axhline(np.percentile(res["agama 1e-08"]["mu"], p), color="k", ls=ls, lw=0.8, label=f"AGAMA 1e-8 {p}%")
    ax[1].set(xscale="log", yscale="log", xlabel="force evaluations per particle (700 Myr)", ylabel="|dv| percentile [mas/yr]",
              title="accuracy vs cost"); ax[1].legend(fontsize=7, ncol=2)
    ax[2].scatter(dmin, res["yoshida4 0.2"]["mu"], s=5, label="yoshida4 dt 0.2"); ax[2].scatter(dmin, res["leapfrog 0.02"]["mu"], s=5, label="leapfrog dt 0.02")
    ax[2].scatter(dmin, res["agama 1e-08"]["mu"], s=5, label="AGAMA 1e-8", c="k")
    ax[2].set(xscale="log", yscale="log", xlabel="closest approach to the progenitor centre [r_J at release]", ylabel="|dv| [mas/yr]",
              title="where the errors come from"); ax[2].legend(fontsize=8)
    fig.suptitle(f"Integrator test: {len(ic)} trailing particles (< 700 Myr) in the best-fit barred host + moving progenitor; forces from AGAMA")
    fig.savefig(ROOT/"plots/integrator_test.png", dpi=85); print("plots/integrator_test.png")


if __name__ == "__main__":
    main()
