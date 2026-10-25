#!/usr/bin/env python3
"""Geometry of the bar, the Sun and omega Cen's orbit (2026-10-23).

Host and orbit: the best step-2 strong-bar fit (12.5 deg; results/streams/profile_step2_20261020/runs/final_a12.5_3.json); orbit
integrated back 700 Myr (the age of the oldest debris in the score). Model frame: Galactic Centre at the origin, Sun on the +x
axis, Galactic rotation anticlockwise (L_z > 0); omega Cen's orbit is retrograde (L_z < 0).
(a) Face-on view today: midplane density of the bar term (full minus axisymmetric host: the bar's non-axisymmetric
    overdensity) of the strong bar (filled log contours, red outline) and of the published bar (black dashed, same level), omega Cen's orbit coloured by time ago, its position today, the Sun, corotation radius of the bar, pericentre and
    apocentre circles; arrows for the sense of Galactic and bar rotation and of omega Cen's motion.
(b) The same orbit in the frame rotating with the bar (bar major axis fixed at its present angle).
(c) Meridional view: |z| against R.
(d) Torque of the bar term on omega Cen, tau_z / (R |F_axi|), along the orbit for the strong and the published bar
    (bin/streams/diagnostics/orbit_torque.py), with R(t) on a second axis.
Output: plots/bar_geometry.{png,pdf}
Usage: PYTHONPATH=src python bin/streams/geometry_figure.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def coloured_line(ax, x, y, c, cmap="viridis", lw=1.0):
    seg = np.stack([np.column_stack((x[:-1], y[:-1])), np.column_stack((x[1:], y[1:]))], axis=1)
    lc = LineCollection(seg, cmap=cmap, array=0.5*(c[1:]+c[:-1]), linewidth=lw); ax.add_collection(lc); return lc


def main():
    from ocen_stream.restricted import host_potential, agama_kpc, AGAMA_T_MYR
    from ocen_stream.frames import to_model, OCEN_OBS, frame_r0, FRAMES
    agama = agama_kpc(); T = 1955.58/AGAMA_T_MYR; r0 = frame_r0("gravity21")
    p = json.loads((ROOT/"results/streams/profile_step2_20261020/runs/final_a12.5_3.json").read_text())["params"]
    host = lambda A, S: host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=A, bar_size=S,
                                       axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=r0))
    H, H0, Ha = host(p["amp"], p["size"]), host(1., 1.), host(0., 1.)
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    t, o = agama.orbit(potential=H, ic=today, timestart=T, time=-700./AGAMA_T_MYR, trajsize=7001, accuracy=1e-10)
    ago = (T-t)*AGAMA_T_MYR; x, y, z = o[:, 0], o[:, 1], o[:, 2]; R = np.hypot(x, y)
    ang = np.deg2rad(p["angle"])+p["omega"]*(t-T)                     # bar major axis angle at time t (AGAMA units: rad)
    xb = x*np.cos(ang-np.deg2rad(p["angle"]))+y*np.sin(ang-np.deg2rad(p["angle"]))
    yb = -x*np.sin(ang-np.deg2rad(p["angle"]))+y*np.cos(ang-np.deg2rad(p["angle"]))   # rotated so the bar sits at its present angle
    # corotation: Omega_c(R) = v_c / R = Omega_b
    Rg = np.linspace(2, 12, 501); vc = np.sqrt(-Rg*Ha.force(np.column_stack((Rg, 0*Rg, 0*Rg)))[:, 0]); Rcr = np.interp(0, (vc/Rg-p["omega"])[::-1], Rg[::-1])
    L = 9.; g = np.linspace(-L, L, 241); X, Y = np.meshgrid(g, g); P = np.column_stack((X.ravel(), Y.ravel(), 0*X.ravel()))
    ra = Ha.density(P).reshape(X.shape)
    rho = H.density(P, t=T).reshape(X.shape)-ra; rho0 = H0.density(P, t=T).reshape(X.shape)-ra   # bar overdensity
    fig, ax = plt.subplots(2, 2, figsize=(10, 9.4), constrained_layout=True); a, b, c, d = ax.ravel()
    lev = np.arange(7.5, 10.6, 0.5); LV = 8.0
    for A_ in (a, b):
        A_.contourf(X, Y, np.log10(np.clip(rho, 1e5, None)), levels=lev, cmap="Greys", alpha=0.6)
        A_.contour(X, Y, np.log10(np.clip(rho0, 1e5, None)), levels=[LV], colors="k", linewidths=1.0, linestyles="--")
        A_.contour(X, Y, np.log10(np.clip(rho, 1e5, None)), levels=[LV], colors="C3", linewidths=1.2)
        for rr, st, lab in ((Rcr, "C3:", "corotation"), (R.min(), "0.4:", None), (R.max(), "0.4:", None)):
            A_.add_patch(plt.Circle((0, 0), rr, fill=False, ls=":", color=st[:-1], lw=1))
        A_.plot(r0, 0, "*", color="gold", mec="k", ms=13, zorder=5); A_.text(r0, -0.8, "Sun", ha="center", fontsize=8)
        A_.plot(0, 0, "k+", ms=8)
        A_.set(xlim=(-L, L), ylim=(-L, L), aspect="equal", xlabel="x [kpc]", ylabel="y [kpc]")
    lc = coloured_line(a, x, y, ago); coloured_line(b, xb, yb, ago)
    for A_ in (a, b):
        A_.plot(x[0], y[0], "o", color="C1", mec="k", ms=8, zorder=6); A_.text(x[0]+0.3, y[0]+0.3, r"$\omega$ Cen today", fontsize=8)
    cb = fig.colorbar(lc, ax=[a, b], location="top", shrink=0.6, pad=0.01); cb.set_label("time ago [Myr]")
    # sense of rotation: Galaxy/bar anticlockwise; omega Cen moves the other way
    a.annotate("", xy=(-6.2, 4.2), xytext=(-4.2, 6.2), arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
    a.text(-6.9, 6.3, f"Galaxy and bar, $\\Omega_b$ = {p['omega']:.1f}", color="C3", fontsize=8)
    v_ = today[3:5]/np.linalg.norm(today[3:5]); a.annotate("", xy=(x[0]+1.8*v_[0], y[0]+1.8*v_[1]), xytext=(x[0], y[0]),
                                                           arrowprops=dict(arrowstyle="->", color="C1", lw=1.5))
    a.text(0.02, 0.02, f"red line: strong bar (A {p['amp']:.2f}, S {p['size']:.2f}), angle {p['angle']:.1f}$^\\circ$\n"
           f"dashed: published bar; bar overdensity $10^{{8}}$ M$_\\odot$ kpc$^{{-3}}$\n"
           f"dotted circles: pericentre {R.min():.1f}, apocentre {R.max():.1f}, corotation {Rcr:.1f} kpc",
           transform=a.transAxes, fontsize=7, va="bottom", bbox=dict(fc="w", ec="none", alpha=0.8))
    a.set_title("(a) face-on, today (inertial frame)"); b.set_title("(b) the same orbit in the frame rotating with the bar")
    # (c) meridional
    coloured_line(c, R, np.abs(z), ago); c.set(xlim=(0, 8), ylim=(0, 3), xlabel="R [kpc]", ylabel="|z| [kpc]", title="(c) meridional view")
    c.axvspan(4, 8, color="0.93", zorder=0); c.text(6, 2.8, "R > 4 kpc: 66 per cent of the time", ha="center", fontsize=8)
    # (d) torque along the orbit
    Fa = np.linalg.norm(Ha.force(o[:, :3]), axis=1)
    tau = lambda h: (lambda F: x*F[:, 1]-y*F[:, 0])(np.array([h.force(xi, t=ti)-Ha.force(xi) for xi, ti in zip(o[:, :3], t)]))
    d.plot(ago, tau(H)/(R*Fa), color="C3", lw=0.8, label="strong bar")
    d.plot(ago, tau(H0)/(R*Fa), color="k", lw=0.8, label="published bar")
    d.axhline(0, color="0.5", lw=0.5); d.set(xlabel="time ago [Myr]", ylabel=r"bar torque $\tau_z$ / ($R\,|F_{\rm axi}|$)", xlim=(700, 0),
                                              title="(d) torque of the bar on $\\omega$ Cen along its orbit")
    d2 = d.twinx(); d2.plot(ago, R, color="C0", lw=0.8, alpha=0.6); d2.set_ylabel("R [kpc]", color="C0"); d2.set_ylim(0, 16)
    d.legend(fontsize=8, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT/f"plots/bar_geometry.{ext}", dpi=150)
    print(ROOT/"plots/bar_geometry.png", f"corotation {Rcr:.2f} kpc")


if __name__ == "__main__":
    main()
