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


def release(params, ht, inp, trel, keep):
    """centre orbit and the kept initial conditions at params = dict(angle, omega, amp, size, halo_q, vc, dist, pmra, pmdec);
    returns tb (host + progenitor tables for jaxspray), ic, t0, r_J. The centre orbit is AGAMA's DOP853 at ORBIT_ACC (adaptive steps,
    centre_orbit_dop853, run on the CPU device: a single orbit is latency-bound on a GPU; was 39112 fixed Yoshida steps: 7 s on an
    A100); the state at the release epochs is cubic Hermite in its steps (spray.spray_unwrapped interpolates linearly between
    0.05-Myr samples)."""
    T = inp["T"]; tb = J.host_for(ht, params["angle"], params["omega"], params["amp"], params["size"], params["halo_q"], params["vc"], T)
    cpu = jax.devices("cpu")[0]; dev = jax.devices()[0]                  # one orbit = ~6k sequential steps: latency-bound on a GPU
    a = jax.device_put((today_xv(params["dist"], params["pmra"], params["pmdec"], inp), {k: v for k, v in tb.items()}, T), cpu)
    with jax.default_device(cpu):
        tcu, orbu, orbit_r = jax.device_put(_orbit_resample_jit(*a, jnp.asarray(trel)), dev)
    # the particles see the centre on a uniform 0.05-Myr grid (O(1) lookup in every force call; a search over the adaptive steps made
    # the GPU spray 3x slower)
    tb = dict(tb, sat=inp["sat"], lr0=inp["lr0"], dlr=inp["dlr"], nr=inp["nr"], mtot=inp["mtot"], G=inp["G"], tc0=tcu[0], dtc=tcu[1]-tcu[0], orb=orbu)
    ic, rj = _release_ic_jit(orbit_r, jnp.asarray(trel), tb, inp)
    return tb, ic[keep], jnp.repeat(jnp.asarray(trel), 2)[keep], jnp.repeat(rj, 2)[keep]


def _orbit_resample(xv0, tb, T, trel, n_grid=39112):
    """adaptive centre orbit, resampled on a uniform grid (Hermite in the DOP853 steps), and its state at the release epochs."""
    tcs, orb, acc_c, n = centre_orbit_dop853(xv0, tb, T, acc=ORBIT_ACC)
    orb = jnp.where(n < 32767, orb, jnp.nan)                         # buffer exhausted before t = 0: poison the orbit, never silent
    tcu = jnp.linspace(0., T, n_grid+1); H = lambda t: jnp.concatenate((J._hermite(t, tcs, orb[:, :3], orb[:, 3:]), J._hermite(t, tcs, orb[:, 3:], acc_c)), 1)
    return tcu, H(tcu), H(trel)


def _release_ic(orbit_r, trel, tb, inp):
    rj, vj, Rm = rj_vj_R(orbit_r, trel, tb, inp)
    return release_ic(orbit_r, rj, vj, Rm, inp["bank"])[0], rj


def spray(params, ht, inp, trel, keep, dt=0.2/T_MYR, flag_rj=0.5, rtol=1e-6, pad=512, method="yoshida", acc=1e-8, n_att=32, handoff=None):
    """the model stream today: final phase space (kept particles), chi [kpc Myr], age [Myr] and the number of re-integrated
    particles (method 'yoshida') or the attempts of the slowest particle (method 'dop853': AGAMA's integrator at accuracy acc for
    every particle). Yoshida-4: particles that came within flag_rj tidal radii (at release) of the progenitor centre are
    re-integrated with adaptive Dopri8 (rtol = atol); flag_rj = 0 disables. The flagged set is padded to a multiple of pad
    (fewer recompilations)."""
    tb, ic, t0, rj = release(params, ht, inp, trel, keep)
    if method == "dop853":
        y, chi, natt = dop853_chi(ic, t0, tb, acc, n_att, handoff=handoff)
        return jnp.asarray(y), jnp.asarray(chi), (inp["T"]-t0)*T_MYR, natt
    xv, chi, dmin = yoshida4_chi(ic, t0, tb, dt)
    f = np.nonzero(np.asarray(dmin < flag_rj*rj))[0]
    if f.size:
        fp = np.resize(f, -(-f.size//pad)*pad); y, c = dopri8_chi(ic[fp], t0[fp], tb, rtol, rtol)
        xv, chi = xv.at[f].set(y[:f.size]), chi.at[f].set(c[:f.size])
    return xv, chi, (inp["T"]-t0)*T_MYR, f.size


# ---- AGAMA's DOP853 (math_ode.cpp, OdeStepperDOP853 + initTimeStep + orbit.cpp accuracy factor), work-efficient on a GPU ----
_C = (0.05260015195876773187856, 0.07890022793815159781784, 0.11835034190722739672676, 0.28164965809277260327324,
      0.33333333333333333333333, 0.25, 0.30769230769230769230769, 0.65128205128205128205128, 0.6, 0.85714285714285714285714, 1.0)
_A = ((0.05260015195876773187856,),
      (0.01972505698453789945446, 0.05917517095361369836338),
      (0.02958758547680684918169, 0., 0.08876275643042054754507),
      (0.24136513415926668550237, 0., -0.88454947932828608534486, 0.92483400326179200311574),
      (0.03703703703703703703704, 0., 0., 0.17082860872947387127960, 0.12546768756682242501669),
      (0.037109375, 0., 0., 0.17025221101954403931498, 0.06021653898045596068502, -0.017578125),
      (0.03709200011850479271088, 0., 0., 0.17038392571223999381021, 0.10726203044637328465181, -0.01531943774862440175279,
       0.00827378916381402288758),
      (0.62411095871607571711443, 0., 0., -3.36089262944694129406857, -0.86821934684172600681819, 27.5920996994467083049416,
       20.1540675504778934086187, -43.4898841810699588477366),
      (0.47766253643826436589043, 0., 0., -2.48811461997166764192642, -0.59029082683684299637145, 21.2300514481811942347289,
       15.2792336328824235832597, -33.2882109689848629194453, -0.02033120170850862613582),
      (-0.93714243008598732571704, 0., 0., 5.18637242884406370830024, 1.09143734899672957818500, -8.14978701074692612513997,
       -18.5200656599969598641566, 22.7394870993505042818970, 2.49360555267965238987089, -3.04676447189821950038237),
      (2.27331014751653820792360, 0., 0., -10.5344954667372501984067, -2.00087205822486249909676, -17.9589318631187989172766,
       27.9488845294199600508500, -2.85899827713502369474066, -8.87285693353062954433549, 12.3605671757943030647266,
       0.64339274601576353035597))
_B = {0: 0.05429373411656876223805, 5: 4.45031289275240888144114, 6: 1.89151789931450038304282, 7: -5.80120396001058478146721,
      8: 0.31116436695781989440892, 9: -0.15216094966251607855618, 10: 0.20136540080403034837478, 11: 0.04471061572777259051769}
_BHH = {0: 0.24409448818897637795276, 8: 0.73384668828161185734136, 11: 0.02205882352941176470588}
_ER = {0: 0.01312004499419488073250, 5: -1.22515644637620444072057, 6: -0.49575894965725019152141, 7: 1.66437718245498653696153,
       8: -0.35032884874997368168865, 9: 0.33417911871301747902973, 10: 0.08192320648511571246571, 11: -0.02235530786388629525884}


def _deriv(y, t, tb, with_acc=False):
    """dy/dt for (x, v) (N, 6), the chi integrand |r| - |r_prog| and, optionally, AGAMA's accuracy factor
    min(1, |E| / max(|Epot|, Ekin)) (orbit.cpp OrbitIntegrator::eval)."""
    x, v = y[:, :3], y[:, 3:]
    if with_acc:
        pot, vjp = jax.vjp(lambda z: J.phi(z, t, tb), x); a = -vjp(jnp.ones_like(pot))[0]
        ek = 0.5*jnp.sum(v*v, 1); fac = jnp.minimum(1., jnp.abs(pot+ek)/jnp.maximum(jnp.abs(pot), ek))
    else:
        a, fac = J.accel(x, t, tb), None
    dchi = jnp.linalg.norm(x, axis=1)-jnp.linalg.norm(J.centre(t, tb), axis=1)
    return jnp.concatenate((v, a), 1), dchi, fac


def _init_step(y, t, tb, acc):
    """AGAMA initTimeStep (accAbs = 0) times the accuracy factor; returns (h, k1)."""
    k1, _, fac = _deriv(y, t, tb, True)
    h1 = jnp.clip(jnp.sqrt(jnp.sum(y*y, 1)/jnp.sum(k1*k1, 1))*0.01, 1e-12, 1e12)
    k2 = _deriv(y+h1[:, None]*k1, t+h1, tb)[0]; xt = y+0.5*h1[:, None]*(k1+k2); k3 = _deriv(xt, t+h1, tb)[0]
    d0 = jnp.maximum(jnp.abs(y), jnp.abs(xt)); d1 = jnp.maximum(jnp.abs(k1), jnp.abs(k3))
    d2 = jnp.abs((3*k2-k1-2*k3)/h1[:, None]); d3 = jnp.abs(6*(k3-k2)/(h1**2)[:, None])
    h2 = jnp.min((d0*d2+d1*d1)/(d1*d3+d2*d2), 1)
    return jnp.clip(acc**(1/8)*jnp.sqrt(h2), 1e-12, 1e12)*fac, k1


def _attempts(s, T, tb, acc, n_att):
    """n_att DOP853 step attempts for every particle (accept / reject / finish as AGAMA's doStep + OrbitIntegrator::run)."""
    def one(_, s):
        t, y, k1, h, chi, nbad, prev, done = s
        rem = T-t; dt = jnp.minimum(h, rem); K = [k1]; D = [_deriv(y, t, tb)[1]]
        for i, (c, a) in enumerate(zip(_C, _A)):
            yi = y+dt[:, None]*sum(aj*K[j] for j, aj in enumerate(a) if aj)
            if i < 10:
                k, d, _ = _deriv(yi, t+c*dt, tb); K.append(k); D.append(d)
            else:
                k, d, _ = _deriv(yi, t+dt, tb); K.append(k); D.append(d)
        kb = sum(b*K[j] for j, b in _B.items()); ynew = y+dt[:, None]*kb
        k13, d13, fac = _deriv(ynew, t+dt, tb, True)
        sk = acc*jnp.maximum(jnp.abs(y), jnp.abs(ynew))*fac[:, None]
        e3 = jnp.sum(((kb-sum(b*K[j] for j, b in _BHH.items()))/sk)**2, 1); e5 = jnp.sum((sum(b*K[j] for j, b in _ER.items())/sk)**2, 1)
        den = jnp.sqrt(6*(e5+0.01*e3)); err = jnp.where(den == 0, 0., e5*dt/den); f8 = err**0.125
        ok = (err <= 1) | (nbad >= 3) & (err > 0.5*prev)                                # AGAMA: accept, or give up after 4 bad tries
        worse = err > 0.5*prev; nb = jnp.where(worse, nbad+1, 0)
        hrej = dt*jnp.maximum(0.333, 0.9/jnp.where(nb >= 2, 1/0.333, f8))
        hacc = jnp.where(err <= 1, dt*jnp.minimum(6., 0.9/f8), dt)
        go = ~done; acc_ = go & ok
        chi = jnp.where(acc_, chi+dt*sum(b*D[j] for j, b in _B.items()), chi)
        t = jnp.where(acc_, jnp.where(dt == rem, T, t+dt), t); y = jnp.where(acc_[:, None], ynew, y); k1 = jnp.where(acc_[:, None], k13, k1)
        h = jnp.where(go, jnp.where(ok, hacc, hrej), h); nbad = jnp.where(go, jnp.where(ok, 0, nb), nbad)
        prev = jnp.where(go, jnp.where(ok, jnp.inf, err), prev); done = done | (t >= T)
        return t, y, k1, h, chi, nbad, prev, done
    return jax.lax.fori_loop(0, n_att, one, s)


_attempts_jit = jax.jit(_attempts, static_argnums=(4,))
_init_jit = jax.jit(_init_step)


def dop853_chi(ic, t0, tb, acc=1e-8, n_att=32, min_size=256, trace=None, finish=None, handoff=None, max_att=200000):
    """AGAMA's DOP853 for every particle from t0 to T, carrying chi [kpc Myr]. Work-efficient on a GPU: n_att attempts per call,
    then the unfinished particles are compacted into a power-of-two array (finished ones stop costing anything). Returns
    (final (N, 6), chi, number of accepted+rejected attempts of the slowest particle). trace (optional list): appended with
    (attempts so far, active particles, array size, seconds) per call, to see where the time goes. finish (optional (N,) array):
    filled with the attempt count (to n_att resolution) at which each particle reached T. handoff (optional, attempts): once
    reached, the unfinished particles and the tables move to the CPU device and the same loop continues there (no change to the
    result: on a GPU the last ~0.1% of particles, which pass through the cluster core, otherwise take ~half the time, latency-
    bound; docs/reviews/straggler_fable_20261025.md). A no-op when the default device is already the CPU."""
    T = tb["T"]; n = len(ic); t = jnp.asarray(t0); y = jnp.asarray(ic); h, k1 = _init_jit(y, t, tb, acc)
    out_y, out_chi = np.zeros((n, 6)), np.zeros(n); idx = np.arange(n); calls = 0
    s = (t, y, k1, h, jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.full(n, jnp.inf), jnp.zeros(n, bool))
    while True:
        t_c = __import__("time").perf_counter(); n_live = int(np.sum(idx >= 0)) if trace is not None else 0
        s = _attempts_jit(s, T, tb, acc, n_att); calls += 1
        if trace is not None:
            s[7].block_until_ready(); trace.append((calls*n_att, n_live, len(idx), __import__("time").perf_counter()-t_c))
        done = np.asarray(s[7]); live = idx >= 0
        fin = done & live; out_y[idx[fin]] = np.asarray(s[1])[fin]; out_chi[idx[fin]] = np.asarray(s[4])[fin]
        if finish is not None:
            finish[idx[fin]] = calls*n_att
        act = np.nonzero(~done & live)[0]
        if act.size == 0:
            return out_y, out_chi*T_MYR, calls*n_att
        if calls*n_att > max_att:
            raise RuntimeError(f"dop853_chi: {act.size} particles unfinished after {max_att} attempts (non-finite state or stalled step?)")
        if handoff is not None and calls*n_att >= handoff and s[1].devices() != {cpu := jax.devices("cpu")[0]}:
            s, tb, T = jax.device_put((s, tb, T), cpu); min_size = 64                      # small arrays are cheap on a CPU
        m = max(min_size, 1 << (2*int(np.ceil(np.log2(act.size)/2))))                     # powers of 4: fewer compiled shapes
        if m < len(idx):                                                                  # compact: keep the unfinished ones
            sel = np.concatenate((act, np.full(m-act.size, act[0])))
            s = tuple(a[sel] for a in s); s = s[:7]+(s[7].at[act.size:].set(True),)
            idx = np.concatenate((idx[act], np.full(m-act.size, -1)))
        else:
            idx = np.where(done, -1, idx)


def centre_orbit_dop853(xv0, tb, T, acc=1e-12, nmax=32768):
    """host-only orbit from T back to 0 with AGAMA's DOP853 (accuracy acc, as spray.spray_unwrapped's 1e-12), integrated in
    tau = T - t (dx/dtau = -v, dv/dtau = -a(x, T - tau)); every accepted step is stored. Returns (knots t ascending (nmax,),
    states (nmax, 6), accelerations (nmax, 3), number of steps); unused buffer entries repeat the t = 0 state at t < 0."""
    def f(tau, y, with_acc=False):
        t = T-tau; x = y[None, :3]
        if with_acc:
            pot, vjp = jax.vjp(lambda z: J.phi_host(z, jnp.full(1, t), tb), x); a = -vjp(jnp.ones(1))[0][0]
            ek = 0.5*jnp.sum(y[3:]**2); fac = jnp.minimum(1., jnp.abs(pot[0]+ek)/jnp.maximum(jnp.abs(pot[0]), ek))
            return jnp.concatenate((-y[3:], -a)), fac
        a = -jax.grad(lambda z: jnp.sum(J.phi_host(z, jnp.full(1, t), tb)))(x)[0]                  # host only
        return jnp.concatenate((-y[3:], -a)), None
    k1, fac = f(0., xv0, True)
    h1 = jnp.clip(jnp.sqrt(jnp.sum(xv0**2)/jnp.sum(k1**2))*0.01, 1e-12, 1e12); k2 = f(h1, xv0+h1*k1)[0]
    xt = xv0+0.5*h1*(k1+k2); k3 = f(h1, xt)[0]
    d0 = jnp.maximum(jnp.abs(xv0), jnp.abs(xt)); d1 = jnp.maximum(jnp.abs(k1), jnp.abs(k3))
    d2 = jnp.abs((3*k2-k1-2*k3)/h1); d3 = jnp.abs(6*(k3-k2)/h1**2); h0 = jnp.clip(acc**(1/8)*jnp.sqrt(jnp.min((d0*d2+d1*d1)/(d1*d3+d2*d2))), 1e-12, 1e12)*fac
    buf_t = jnp.full(nmax, -1.); buf_y = jnp.zeros((nmax, 6)); buf_a = jnp.zeros((nmax, 3))
    buf_t = buf_t.at[0].set(0.); buf_y = buf_y.at[0].set(xv0); buf_a = buf_a.at[0].set(-k1[3:])

    def body(s):
        tau, y, k1, h, nbad, prev, n, bt, by, ba = s
        rem = T-tau; dt = jnp.minimum(h, rem); K = [k1]
        for c, a in zip(_C, _A):
            K.append(f(tau+c*dt, y+dt*sum(aj*K[j] for j, aj in enumerate(a) if aj))[0])
        kb = sum(b*K[j] for j, b in _B.items()); ynew = y+dt*kb; k13, fac = f(tau+dt, ynew, True)
        sk = acc*jnp.maximum(jnp.abs(y), jnp.abs(ynew))*fac
        e3 = jnp.sum(((kb-sum(b*K[j] for j, b in _BHH.items()))/sk)**2); e5 = jnp.sum((sum(b*K[j] for j, b in _ER.items())/sk)**2)
        den = jnp.sqrt(6*(e5+0.01*e3)); err = jnp.where(den == 0, 0., e5*dt/den); f8 = err**0.125
        worse = err > 0.5*prev; nb = jnp.where(worse, nbad+1, 0); ok = (err <= 1) | ((nbad >= 3) & worse)
        hn = jnp.where(ok, jnp.where(err <= 1, dt*jnp.minimum(6., 0.9/f8), dt), dt*jnp.maximum(0.333, 0.9/jnp.where(nb >= 2, 1/0.333, f8)))
        tau2 = jnp.where(ok, jnp.where(dt == rem, T, tau+dt), tau); n2 = jnp.where(ok, n+1, n)
        bt = jnp.where(ok, bt.at[n2].set(tau2), bt); by = jnp.where(ok, by.at[n2].set(ynew), by); ba = jnp.where(ok, ba.at[n2].set(-k13[3:]), ba)
        return (tau2, jnp.where(ok, ynew, y), jnp.where(ok, k13, k1), hn, jnp.where(ok, 0, nb), jnp.where(ok, jnp.inf, err), n2, bt, by, ba)
    s = jax.lax.while_loop(lambda s: (s[0] < T) & (s[6] < nmax-1), body, (0., xv0, k1, h0, 0, jnp.inf, 0, buf_t, buf_y, buf_a))
    n, bt, by, ba = s[6], s[7], s[8], s[9]
    t = jnp.where(jnp.arange(nmax) <= n, T-bt, -1.-jnp.arange(nmax)*1e-3)                 # unused entries: t < 0, strictly ordered
    by = jnp.where((jnp.arange(nmax) <= n)[:, None], by, by[n]); ba = jnp.where((jnp.arange(nmax) <= n)[:, None], ba, ba[n])
    o = jnp.argsort(t)
    return t[o], by[o], ba[o], n


_orbit_jit = jax.jit(centre_orbit_dop853, static_argnames=("acc", "nmax"))
_orbit_resample_jit = jax.jit(_orbit_resample)
_release_ic_jit = jax.jit(_release_ic)
ORBIT_ACC = 1e-10            # same accuracy vs AGAMA as 1e-12 (0.06 pc over the last 700 Myr, = AGAMA's own closure); 5.7k steps
