"""Whole spray model in JAX at any parameter point (GPU pipeline stage 2, 2026-10-25).

Port of grid4_common.make_spray / spray.spray_unwrapped (Fardal release, trailing arm, max age, chi ordering) on the host of
jaxspray.host_for. Inputs (results/gpu/spray_inputs.npz, bin/streams/gpu_spray_export.py): frame constants (ICRS <-> model
frame is affine: x_model = F (A x_icrs + b), v_model = F (A v_icrs + v_sun), F = diag(-1, 1, 1); ICRS -> Galactic rotation), omega
Cen's position / v_los, the progenitor profile (Phi_sat(ln r) B-spline and M(<r) table) and a frozen bank of release offsets
(the normal draws of spray.release_ic, in its order).
Steps: present-day point from (distance, PMs); centre orbit integrated backwards from T to 0 with Yoshida-4 (host only, step
DT_ORBIT) and sampled at every step (the integrator is time-reversible, so no forward pass is needed); release epochs
t_k = T - window + (k + offset) window / n_rel; r_J from the host Hessian along r (jvp) and M(<r_J) by fixed-point iteration;
Fardal offsets from the bank; trailing particles younger than max_age; Yoshida-4 to T with chi = int (|r| - |r_prog|) dt.
"""
import numpy as np
import jax
import jax.numpy as jnp
from . import jaxspray as J

T_MYR = 977.79222168; KMS_MASYR = 4.740470463533348


def load_inputs(path, dtype=jnp.float64):
    z = np.load(path); f = lambda a: jnp.asarray(a, dtype)
    return dict(A=f(z["A"]), b=f(z["b"]), vsun=f(z["vsun"]), Agal=f(z["Agal"]), ra=f(z["ra"]), dec=f(z["dec"]), vlos=f(z["vlos"]),
                sat=f(np.pad(z["sat_coef"], (2, 2), mode="edge")), lr0=f(z["lr0"]), dlr=f(z["dlr"]), nr=z["sat_coef"].size,
                mtot=f(z["sat_mtot"]), G=f(z["G"]), lrm=f(np.log(z["r_kpc"])), mr=f(z["M_enc"]), bank=f(z["bank"]), T=f(z["T"]),
                window=f(z["window_myr"]/T_MYR), nrel=int(z["nrel"]), max_age=f(z["max_age_myr"]/T_MYR))


def today_xv(d, pmra, pmdec, inp):
    """model-frame phase-space point of omega Cen for distance [kpc] and PMs [mas/yr] (ra, dec, v_los fixed)."""
    a, de = jnp.deg2rad(inp["ra"]), jnp.deg2rad(inp["dec"])
    n = jnp.array([jnp.cos(de)*jnp.cos(a), jnp.cos(de)*jnp.sin(a), jnp.sin(de)])
    ea = jnp.array([-jnp.sin(a), jnp.cos(a), 0.]); ed = jnp.array([-jnp.sin(de)*jnp.cos(a), -jnp.sin(de)*jnp.sin(a), jnp.cos(de)])
    v = inp["vlos"]*n+KMS_MASYR*d*(pmra*ea+pmdec*ed); F = jnp.array([-1., 1., 1.])
    return jnp.concatenate((F*(inp["A"]@(d*n)+inp["b"]), F*(inp["A"]@v+inp["vsun"])))


def observables(xv, inp):
    """model frame (N, 6) -> l, b [deg], pmra*, pmdec [mas/yr], vlos [km/s], dist [kpc] (inverse of the affine frame map)."""
    F = jnp.array([-1., 1., 1.]); Ai = jnp.linalg.inv(inp["A"])
    p = (F*xv[:, :3]-inp["b"])@Ai.T; v = (F*xv[:, 3:]-inp["vsun"])@Ai.T
    d = jnp.linalg.norm(p, axis=1); n = p/d[:, None]; a = jnp.arctan2(n[:, 1], n[:, 0]); de = jnp.arcsin(n[:, 2])
    ea = jnp.stack((-jnp.sin(a), jnp.cos(a), 0*a), 1); ed = jnp.stack((-jnp.sin(de)*jnp.cos(a), -jnp.sin(de)*jnp.sin(a), jnp.cos(de)), 1)
    g = n@inp["Agal"].T
    return dict(l=jnp.rad2deg(jnp.arctan2(g[:, 1], g[:, 0])) % 360., b=jnp.rad2deg(jnp.arcsin(g[:, 2])), dist=d,
                pmra=jnp.sum(v*ea, 1)/(KMS_MASYR*d), pmdec=jnp.sum(v*ed, 1)/(KMS_MASYR*d), vlos=jnp.sum(v*n, 1))


def centre_orbit(xv0, tb, T, n_steps):
    """host-only orbit from T back to 0 by Yoshida-4 with n_steps equal steps; returns (tc ascending, orbit (n_steps+1, 6))."""
    cd, ck = J.YOSHIDA; h = -T/n_steps

    def step(s, k):
        x, v = s; t = T+k*h; frac = 0.
        for i in range(4):
            x = x+cd[i]*h*v; frac += cd[i]
            if i < 3:
                v = v+ck[i]*h*(-jax.grad(lambda y: jnp.sum(J.phi_host(y, jnp.full(1, t+frac*h), tb)))(x[None])[0])
        return (x, v), jnp.concatenate((x, v))
    _, traj = jax.lax.scan(step, (xv0[:3], xv0[3:]), jnp.arange(n_steps))
    orb = jnp.concatenate((xv0[None], traj))[::-1]
    return jnp.linspace(0., T, n_steps+1), orb


def rj_vj_R(orbit, t, tb, inp, n_iter=30):
    """Jacobi radius, v_J and host->satellite rotation matrices along the orbit (as spray.rj_vj_R; d2Phi/dr2 from the Hessian)."""
    x = orbit[:, :3]; v = orbit[:, 3:]; r = jnp.linalg.norm(x, axis=1); Lv = jnp.cross(x, v); L = jnp.linalg.norm(Lv, axis=1)
    e0 = x/r[:, None]; e2 = Lv/L[:, None]; Rm = jnp.stack((e0, jnp.cross(e2, e0), e2), 1)
    g = lambda y: jax.grad(lambda z: jnp.sum(J.phi_host(z, t, tb)))(y)
    d2 = jnp.sum(e0*jax.jvp(g, (x,), (e0,))[1], 1)
    Om = L/r**2; den = jnp.maximum(Om**2-d2, 1e-6)
    m_of_r = lambda rr: jnp.interp(jnp.log(rr), inp["lrm"], inp["mr"])
    rj = jax.lax.fori_loop(0, n_iter, lambda _, rj: (inp["G"]*m_of_r(rj)/den)**(1/3), jnp.full_like(r, 0.1))
    return rj, Om*rj, Rm


def release_ic(orbit, rj, vj, Rm, bank):
    """Fardal (gala-modified) offsets, interleaved (trailing, leading) per release point, from bank = (4, 2N) normal draws."""
    s = jnp.tile(jnp.array([1., -1.]), rj.size); rj2, vj2, R2 = jnp.repeat(rj, 2)*s, jnp.repeat(vj, 2)*s, jnp.repeat(Rm, 2, axis=0)
    rx = bank[0]*0.5+2.0; rz = bank[1]*0.5*rj2; rvy = (bank[2]*0.5+0.3)*vj2*rx; rvz = bank[3]*0.5*vj2; rx = rx*rj2; z0 = 0*rx
    ic = jnp.repeat(orbit, 2, axis=0)
    dx = jnp.einsum("ni,nij->nj", jnp.stack((rx, z0, rz), 1), R2); dv = jnp.einsum("ni,nij->nj", jnp.stack((z0, rvy, rvz), 1), R2)
    return jnp.concatenate((ic[:, :3]+dx, ic[:, 3:]+dv), 1), s


def yoshida4_chi(ic, t0, tb, dt):
    """Yoshida-4 (as jaxspray.yoshida4) accumulating chi = int (|r| - |r_prog|) dt [kpc Myr] (trapezoid over the steps)."""
    cd, ck = J.YOSHIDA; dur = tb["T"]-t0; n = jnp.maximum(jnp.rint(dur/dt), 1); h0 = dur/n
    rp = lambda x, t: jnp.linalg.norm(x, axis=1)-jnp.linalg.norm(J.centre(t, tb), axis=1)

    def step(k, s):
        x, v, chi, dmin = s; on = k < n; h = jnp.where(on, h0, 0.)[:, None]; frac = 0.; f0 = rp(x, t0+k*h0)
        for i in range(4):
            x = x+cd[i]*h*v; frac += cd[i]
            if i < 3:
                tk = t0+(k+frac)*h0; v = v+ck[i]*h*J.accel(x, tk, tb)
                dmin = jnp.where(on, jnp.minimum(dmin, jnp.linalg.norm(x-J.centre(tk, tb), axis=1)), dmin)
        chi = chi+jnp.where(on, 0.5*(f0+rp(x, t0+(k+1)*h0))*h0, 0.)
        return x, v, chi, dmin
    x, v, chi, dmin = jax.lax.fori_loop(0, jnp.max(n).astype(jnp.int32), step, (ic[:, :3], ic[:, 3:], jnp.zeros(len(ic)), jnp.full(len(ic), jnp.inf)))
    return jnp.concatenate((x, v), 1), chi*T_MYR, dmin


def dopri8_chi(ic, t0, tb, rtol=1e-6, atol=1e-6, max_steps=200000):
    """adaptive Dopri8 per particle (diffrax, vmapped) for (x, v, chi): the re-integration of particles that pass close to the
    progenitor centre, where the fixed step is inaccurate. Returns (final (N, 6), chi [kpc Myr])."""
    import diffrax as dfx

    def rhs(t, y, a):
        tt = t[None]; x = y[None, :3]
        dchi = jnp.linalg.norm(y[:3])-jnp.linalg.norm(J.centre(tt, tb)[0])
        return jnp.concatenate((y[3:6], J.accel(x, tt, tb)[0], dchi[None]))

    def one(y0, ts):
        s = dfx.diffeqsolve(dfx.ODETerm(rhs), dfx.Dopri8(), t0=ts, t1=tb["T"], dt0=None, y0=jnp.concatenate((y0, jnp.zeros(1))),
                            saveat=dfx.SaveAt(t1=True), stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol), max_steps=max_steps)
        return s.ys[-1]
    y = jax.vmap(one)(ic, t0)
    return y[:, :6], y[:, 6]*T_MYR


def schedule(inp, offset):
    """release epochs t_k (N,) and the indices of the kept particles (trailing, younger than max_age) among the 2N interleaved
    ones: fixed by the bank configuration, independent of the parameters."""
    T, W, n = float(inp["T"]), float(inp["window"]), inp["nrel"]
    trel = T-W+(np.arange(n)+offset)*W/n; t0 = np.repeat(trel, 2); arm = np.tile([1, -1], n)
    return trel, np.nonzero((arm == 1) & ((T-t0) < float(inp["max_age"])))[0]


def release(params, ht, inp, trel, keep, n_orbit=39112):
    """centre orbit and the kept initial conditions at params = dict(angle, omega, amp, size, halo_q, vc, dist, pmra, pmdec);
    returns tb (host + progenitor tables for jaxspray), ic, t0. The orbit at the release epochs is linearly interpolated from the
    0.05-Myr samples, as spray.spray_unwrapped."""
    T = inp["T"]; tb = J.host_for(ht, params["angle"], params["omega"], params["amp"], params["size"], params["halo_q"], params["vc"], T)
    tc, orb = centre_orbit(today_xv(params["dist"], params["pmra"], params["pmdec"], inp), tb, T, n_orbit)
    tb = dict(tb, sat=inp["sat"], lr0=inp["lr0"], dlr=inp["dlr"], nr=inp["nr"], mtot=inp["mtot"], G=inp["G"], tc0=tc[0], dtc=tc[1]-tc[0], orb=orb)
    trel = jnp.asarray(trel); orbit_r = jax.vmap(lambda c: jnp.interp(trel, tc, c), 1, 1)(orb)
    rj, vj, Rm = rj_vj_R(orbit_r, trel, tb, inp)
    ic, _ = release_ic(orbit_r, rj, vj, Rm, inp["bank"])
    return tb, ic[keep], jnp.repeat(trel, 2)[keep], jnp.repeat(rj, 2)[keep]


def spray(params, ht, inp, trel, keep, dt=0.2/T_MYR, flag_rj=0.5, rtol=1e-6, pad=512):
    """the model stream today: final phase space (kept particles), chi [kpc Myr], age [Myr] and the number of re-integrated
    particles. Yoshida-4 for all; particles that came within flag_rj tidal radii (at release) of the progenitor centre are
    re-integrated with adaptive Dopri8 (rtol = atol); flag_rj = 0 disables. The flagged set is padded to a multiple of pad
    (fewer recompilations)."""
    tb, ic, t0, rj = release(params, ht, inp, trel, keep)
    xv, chi, dmin = yoshida4_chi(ic, t0, tb, dt)
    f = np.nonzero(np.asarray(dmin < flag_rj*rj))[0]
    if f.size:
        fp = np.resize(f, -(-f.size//pad)*pad); y, c = dopri8_chi(ic[fp], t0[fp], tb, rtol, rtol)
        xv, chi = xv.at[f].set(y[:f.size]), chi.at[f].set(c[:f.size])
    return xv, chi, (inp["T"]-t0)*T_MYR, f.size
