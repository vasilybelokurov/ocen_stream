"""Smooth stream path for the stream-54 northern arm and along/across-stream coordinates (s, x).

Construction (build_path): ridge points in a gnomonic (tangent-plane) projection centred on (l0, b0) = (-48, 28) deg:
(1) the Stage-0 ridge (mixture centres of l in 2-deg b bins) up to b = 36; (2) around the top of the arc, the median b of members
in 2-deg l bins from l = -46 to -28 (members with b > 30). A parametric smoothing spline (scipy splprep, cubic) is fitted to the
ordered points, with smoothing s chosen so the curve stays within max_dev deg of every ridge point; the curve is extended by
`extend` deg along the end tangents and sampled densely. project(): s = arc length [deg] from the start (near the cluster) to
the nearest curve point; x = signed perpendicular distance [deg] (positive to the left of the direction of increasing s).
The same fixed path is used for data and all models.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

L0, B0 = -48.0, 28.0


def gnomonic(l, b, l0=L0, b0=B0):
    """Tangent-plane coordinates (x east-ish in l, y in b) [deg] about (l0, b0)."""
    l, b, l0r, b0r = np.radians(l), np.radians(b), np.radians(l0), np.radians(b0)
    c = np.sin(b0r)*np.sin(b)+np.cos(b0r)*np.cos(b)*np.cos(l-l0r)
    x = np.cos(b)*np.sin(l-l0r)/c
    y = (np.cos(b0r)*np.sin(b)-np.sin(b0r)*np.cos(b)*np.cos(l-l0r))/c
    return np.degrees(x), np.degrees(y)


def ridge_points(l, b, track):
    """Ordered ridge points (l, b): Stage-0 track up to b = 36, then median b in l bins (-46..-28) for members with b > 30."""
    tb, tl = np.array(track["b"]), np.array(track["l"]["mu"])
    k = np.isfinite(tl) & (tb <= 36)
    pts = list(zip(tl[k], tb[k]))
    for lo in np.arange(-46., -28., 2.):
        m = (l >= lo) & (l < lo+2) & (b > 30)
        if m.sum() >= 10:
            pts.append((lo+1., float(np.median(b[m]))))
    return np.array(pts)


def build_path(l, b, track, max_dev=1.0, extend=3.0, nsamp=4000):
    pts = ridge_points(l, b, track)
    X, Y = gnomonic(pts[:, 0], pts[:, 1])
    best = None
    for s in np.logspace(-1, 2.5, 40)[::-1]:          # largest smoothing that keeps every point within max_dev
        tck, u = splprep([X, Y], s=s, k=3)
        cx, cy = splev(np.linspace(0, 1, 2000), tck)
        dev = cKDTree(np.column_stack((cx, cy))).query(np.column_stack((X, Y)))[0].max()
        if dev <= max_dev:
            best = (s, tck, dev); break
    s_used, tck, dev = best
    t = np.linspace(0, 1, nsamp); cx, cy = splev(t, tck)
    # extend along end tangents
    def ext(i0, i1, n=200):
        dx, dy = cx[i1]-cx[i0], cy[i1]-cy[i0]; nrm = np.hypot(dx, dy)
        f = np.linspace(0, extend, n)[1:]
        return cx[i1]+f*dx/nrm, cy[i1]+f*dy/nrm
    sx, sy = ext(1, 0); ex, ey = ext(-2, -1)
    cx = np.concatenate((sx[::-1], cx, ex)); cy = np.concatenate((sy[::-1], cy, ey))
    arc = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))))
    return dict(cx=cx.tolist(), cy=cy.tolist(), arc=arc.tolist(), ridge=pts.tolist(), smoothing=float(s_used), max_dev=float(dev),
                centre=[L0, B0], extend=extend)


def project(l, b, path):
    """Along-stream s [deg] and signed across-stream x [deg] for points (l, b) on a path from build_path()."""
    cx, cy, arc = np.asarray(path["cx"]), np.asarray(path["cy"]), np.asarray(path["arc"])
    X, Y = gnomonic(np.asarray(l), np.asarray(b), *path["centre"])
    _, i = cKDTree(np.column_stack((cx, cy))).query(np.column_stack((X, Y)))
    j = np.clip(i, 1, len(cx)-2)
    tx, ty = cx[j+1]-cx[j-1], cy[j+1]-cy[j-1]; tn = np.hypot(tx, ty); tx, ty = tx/tn, ty/tn
    dx, dy = X-cx[i], Y-cy[i]
    s = arc[i]+dx*tx+dy*ty
    x = tx*dy-ty*dx
    return s, x


def save(path, fname):
    with open(fname, "w") as f:
        json.dump(path, f)


def load(fname):
    with open(fname) as f:
        return json.load(f)


# ---------------- great-circle frame + low-order polynomial track (publishable form) ----------------
def _unit(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.stack((np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)), axis=-1)


def gc_rotation(pole_lb, origin_lb):
    """Rotation matrix from Galactic unit vectors to the stream frame: z = pole, x = origin projected onto the equator."""
    z = _unit(*pole_lb); o = _unit(*origin_lb)
    x = o-np.dot(o, z)*z; x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.vstack((x, y, z))


def to_stream(l, b, Rm):
    v = _unit(np.asarray(l), np.asarray(b)) @ Rm.T
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])), np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


def fit_gc_frame(ridge_lb, origin_lb, max_deg=4, max_dev=1.0):
    """Pole minimising sum phi2^2 of the ridge points (origin fixed at origin_lb), then the lowest-degree polynomial
    phi2(phi1) with all ridge points within max_dev deg. Returns dict(pole, origin, coeffs (highest power first), degree, dev)."""
    from scipy.optimize import minimize
    rl, rb = ridge_lb[:, 0], ridge_lb[:, 1]
    def cost(p):
        Rm = gc_rotation(p, origin_lb); _, f2 = to_stream(rl, rb, Rm); return np.sum(f2**2)
    best = min((minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
                for x0 in ([-40., -40.], [140., 40.], [0., -50.], [100., 0.], [-120., 30.])), key=lambda r: r.fun)
    pole = [float(((best.x[0]+180) % 360)-180), float(best.x[1])]
    Rm = gc_rotation(pole, origin_lb); f1, f2 = to_stream(rl, rb, Rm)
    for deg in range(1, max_deg+1):
        c = np.polyfit(f1, f2, deg); dev = np.abs(np.polyval(c, f1)-f2).max()
        if dev <= max_dev:
            break
    return dict(pole=pole, origin=list(origin_lb), coeffs=c.tolist(), degree=int(deg), max_dev=float(dev),
                phi1_range=[float(f1.min()), float(f1.max())])


def project_gc(l, b, frame):
    """Along-stream phi1 and across-stream dphi2 = phi2 - poly(phi1) [deg]."""
    Rm = gc_rotation(frame["pole"], frame["origin"]); f1, f2 = to_stream(l, b, Rm)
    return f1, f2-np.polyval(frame["coeffs"], f1)


def chord_gc_frame(ridge_lb, origin_lb, end_lb, max_deg=4, max_dev=1.0):
    """Great circle through origin_lb and end_lb (chord frame: bisects the arc's turn), pole = origin x end; then the
    lowest-degree polynomial phi2(phi1) with all ridge points within max_dev deg."""
    z = np.cross(_unit(*origin_lb), _unit(*end_lb)); z /= np.linalg.norm(z)
    pole = [float(np.degrees(np.arctan2(z[1], z[0]))), float(np.degrees(np.arcsin(z[2])))]
    Rm = gc_rotation(pole, origin_lb); f1, f2 = to_stream(ridge_lb[:, 0], ridge_lb[:, 1], Rm)
    for deg in range(1, max_deg+1):
        c = np.polyfit(f1, f2, deg); dev = np.abs(np.polyval(c, f1)-f2).max()
        if dev <= max_dev:
            break
    return dict(pole=pole, origin=list(origin_lb), end=list(end_lb), coeffs=c.tolist(), degree=int(deg), max_dev=float(dev),
                phi1_range=[float(f1.min()), float(f1.max())])
