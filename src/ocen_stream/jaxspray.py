"""JAX particle integration for the spray in the rotating-bar host (GPU benchmark, 2026-10-25).

Inputs come from bin/streams/gpu_bench_export.py (results/gpu/bench_inputs.npz): the host as cos(m phi') Fourier modes on a grid
uniform in (u, w) = (asinh(R / rs), asinh(z / zs)) with cubic B-spline coefficients (mirror boundary at R = 0 and z = 0, where the
potential is even), the bar angle theta(t) = theta_T + Omega_b (t - T), the progenitor potential Phi_sat(ln r) (B-spline), and its
centre orbit sampled uniformly in time (cubic Hermite in time). Forces are -grad Phi by autodiff (conservative, consistent with the
interpolated potential). Units as AGAMA here: kpc, km/s, time unit kpc/(km/s) = 977.79 Myr; Omega_b in km/s/kpc = rad per unit.
Integrators (particles independent; each from its release time t0 to T):
  yoshida4(...): Yoshida (1990) 4th-order symplectic drift-kick composition, fixed step per particle (age / round(age / dt));
  dopri8(...):   diffrax Dopri8 (Dormand-Prince 8(5,3)) with a PID step-size controller, vmapped over particles.
Precision follows jax_enable_x64 and the dtype passed to load(); float64 is the default for validation.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

W1 = 1/(2-2**(1/3)); W0 = -2**(1/3)/(2-2**(1/3))
YOSHIDA = ((W1/2, (W0+W1)/2, (W0+W1)/2, W1/2), (W1, W0, W1))           # drift, kick coefficients


def load(path, dtype=jnp.float64):
    """Benchmark tables as a dict of device arrays; B-spline coefficients padded by 2 cells (mirror / edge) for the 4x4 stencil."""
    z = np.load(path)
    f = lambda a: jnp.asarray(a, dtype)
    host = np.pad(z["host_coef"], ((0, 0), (2, 2), (2, 2)), mode="reflect")                 # mirror about u = 0 and w = 0
    sat = np.pad(z["sat_coef"], (2, 2), mode="edge")
    tb = dict(host=f(host), m=f(z["host_m"]), du=f(z["du"]), dw=f(z["dw"]), rs=f(z["rs"]), zs=f(z["zs"]), nu=z["host_coef"].shape[1],
              nw=z["host_coef"].shape[2], theta_T=f(z["theta_T"]), omega=f(z["omega_b"]), T=f(z["T"]), sat=f(sat), lr0=f(z["lr0"]),
              dlr=f(z["dlr"]), nr=z["sat_coef"].size, mtot=f(z["sat_mtot"]), G=f(z["G"]), tc0=f(z["tc"][0]),
              dtc=f(z["tc"][1]-z["tc"][0]), orb=f(z["orb"]))
    return tb


def _w(t):
    """cubic B-spline weights of nodes i-1, i, i+1, i+2 at fractional position t in [0, 1)."""
    t2, t3 = t*t, t*t*t
    return jnp.stack(((1-t)**3/6, (3*t3-6*t2+4)/6, (-3*t3+3*t2+3*t+1)/6, t3/6), -1)


def _spline2(c, x, y, nx, ny):
    """tensor cubic B-spline at grid coordinates (x, y) (in node units) of padded coefficients c (..., nx+4, ny+4); points beyond
    the grid are clamped to its last cell."""
    i = jnp.clip(jnp.floor(x), 0, nx-2); j = jnp.clip(jnp.floor(y), 0, ny-2)
    wx, wy = _w(x-i), _w(y-j); k = jnp.arange(4)
    I = (i.astype(jnp.int32)+1)[..., None]+k; J = (j.astype(jnp.int32)+1)[..., None]+k
    return jnp.einsum("...nkl,nk,nl->...n", c[..., I[:, :, None], J[:, None, :]], wx, wy)


def _spline1(c, x, n):
    i = jnp.clip(jnp.floor(x), 0, n-2); k = jnp.arange(4)
    return jnp.sum(c[(i.astype(jnp.int32)+1)[..., None]+k]*_w(x-i), -1)


def centre(t, tb):
    """progenitor centre position at times t (N,) by cubic Hermite interpolation of the sampled orbit."""
    s = (t-tb["tc0"])/tb["dtc"]; k = jnp.clip(jnp.floor(s), 0, tb["orb"].shape[0]-2); h = (s-k)[:, None]; k = k.astype(jnp.int32)
    p0, p1, v0, v1 = tb["orb"][k, :3], tb["orb"][k+1, :3], tb["orb"][k, 3:]*tb["dtc"], tb["orb"][k+1, 3:]*tb["dtc"]
    return (2*h**3-3*h**2+1)*p0+(h**3-2*h**2+h)*v0+(-2*h**3+3*h**2)*p1+(h**3-h**2)*v1


def phi_host(x, t, tb):
    """host potential at positions x (N, 3) and times t (N,)."""
    R = jnp.hypot(x[:, 0], x[:, 1]); u = jnp.arcsinh(R/tb["rs"])/tb["du"]; w = jnp.arcsinh(jnp.abs(x[:, 2])/tb["zs"])/tb["dw"]
    modes = _spline2(tb["host"], u, w, tb["nu"], tb["nw"])                                    # (n_modes, N)
    ang = jnp.arctan2(x[:, 1], x[:, 0])-(tb["theta_T"]+tb["omega"]*(t-tb["T"]))
    return jnp.sum(modes*jnp.cos(tb["m"][:, None]*ang[None, :]), 0)


def phi_sat(x, t, tb):
    """progenitor potential (frozen profile carried along its orbit); -G M / r beyond the table."""
    r = jnp.linalg.norm(x-centre(t, tb), axis=1); s = (jnp.log(r)-tb["lr0"])/tb["dlr"]
    return jnp.where(s < tb["nr"]-1, _spline1(tb["sat"], jnp.maximum(s, 0.), tb["nr"]), -tb["G"]*tb["mtot"]/r)


def phi(x, t, tb):
    return phi_host(x, t, tb)+phi_sat(x, t, tb)


def accel(x, t, tb):
    """-grad Phi at (N, 3) positions; particles are independent, so the gradient of the sum is per particle."""
    return -jax.grad(lambda y: jnp.sum(phi(y, t, tb)))(x)


@jax.jit
def yoshida4(ic, t0, tb, dt):
    """fixed-step Yoshida-4 from t0 (N,) to T; per-particle step age / round(age / dt). Returns (final (N, 6), force evals)."""
    cd, ck = YOSHIDA; dur = tb["T"]-t0; n = jnp.maximum(jnp.rint(dur/dt), 1); h0 = dur/n

    def step(k, s):
        x, v = s; h = jnp.where(k < n, h0, 0.)[:, None]; frac = 0.
        for i in range(4):
            x = x+cd[i]*h*v; frac += cd[i]
            if i < 3:
                v = v+ck[i]*h*accel(x, t0+(k+frac)*h0, tb)
        return x, v
    x, v = jax.lax.fori_loop(0, jnp.max(n).astype(jnp.int32), step, (ic[:, :3], ic[:, 3:]))
    return jnp.concatenate((x, v), 1), 3*jnp.max(n)


@partial(jax.jit, static_argnames=("max_steps",))
def dopri8(ic, t0, tb, rtol=1e-8, atol=1e-8, max_steps=100000):
    """adaptive Dopri8 per particle (vmapped). Returns (final (N, 6), steps per particle)."""
    import diffrax as dfx
    term = dfx.ODETerm(lambda t, y, a: jnp.concatenate((y[3:], accel(y[None, :3], t[None], tb)[0])))

    def one(y0, ts):
        s = dfx.diffeqsolve(term, dfx.Dopri8(), t0=ts, t1=tb["T"], dt0=None, y0=y0, saveat=dfx.SaveAt(t1=True),
                            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol), max_steps=max_steps)
        return s.ys[-1], s.stats["num_steps"]
    return jax.vmap(one)(ic, t0)


def run(method, ic, t0, tb, batch=None, **kw):
    """integrate in particle batches (bounds device memory); method 'yoshida4' (kw dt) or 'dopri8' (kw rtol, atol)."""
    f = dict(yoshida4=lambda a, b: yoshida4(a, b, tb, kw["dt"]), dopri8=lambda a, b: dopri8(a, b, tb, **kw))[method]
    batch = batch or len(ic); out, aux = [], []
    for s in range(0, len(ic), batch):
        r = f(jnp.asarray(ic[s:s+batch]), jnp.asarray(t0[s:s+batch])); out.append(r[0]); aux.append(jnp.atleast_1d(r[1]))
    return jnp.concatenate(out), jnp.concatenate(aux)
