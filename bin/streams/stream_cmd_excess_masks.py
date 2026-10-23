#!/usr/bin/env python3
"""CMD excess of the omega Cen stream selected by the data-driven PM + sky masks (2026-10-23).

Stars: footprint (G < 20, parallax cut, GC masks), dereddened (SFD x 0.86). Selection: both PM masks at k = 2
(|pm - c(phi1)| < 2 sqrt(s(phi1)^2 + e_i^2)) and the sky window |dphi2 - centre| < half-width from the mask file; NO CMD cut.
Background per CMD pixel: quadratic in y = dphi2 - centre fitted to 1-deg bands at 5 <= |y| < 14 (constant where one-sided),
integrated over the window (as the PM panels of stream_masks.py; no red-control normalisation).
Panels: 4-deg phi1 bins, excess Hess diagram with the CMD mask (k = 2, dm(phi1) relation of gaia_cmd_mask.json; dashed: the
template-frame G0 = 16.5 limit above which the CMD cut is not applied).
Usage: PYTHONPATH=src python bin/streams/stream_cmd_excess_masks.py [--masks orbitstart] -> plots/stream_cmd_excess_masks[_orbitstart].png
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

TAG = sys.argv[sys.argv.index("--masks")+1] if "--masks" in sys.argv else ""
BINS = [(a, a+4) for a in range(-2, 30, 4)]
EC, EG = np.arange(-0.2, 2.01, 0.04), np.arange(12, 21.51, 0.1)


def main():
    from gaia_footprint_plots import gc_masked
    from gaia_cmd_mask import CMDMask, G_BRIGHT
    raw = dict(np.load(ROOT/"results/streams/gaia_footprint_dr3.npz")); z0 = np.load(ROOT/"results/streams/gaia_footprint_dered.npz")
    assert np.array_equal(z0["source_id"], raw["source_id"])
    lg = np.where(raw["l"] > 180, raw["l"]-360, raw["l"]).astype(float); keep = ~gc_masked(lg, raw["b"].astype(float)) & np.isfinite(z0["c0"])
    u, x = raw["phi1"].astype(float)[keep], raw["dphi2"].astype(float)[keep]; c0, g0 = z0["c0"].astype(float)[keep], z0["g0"].astype(float)[keep]
    R = json.loads((ROOT/f"results/plot_data/stream_masks{'_' + TAG if TAG else ''}.json").read_text()); G = np.array(R["grid"])
    pm = np.ones(u.size, bool)
    for q in ("pmra", "pmdec"):
        c, s = np.interp(u, G, R["tracks"][q]["c"]), np.interp(u, G, R["tracks"][q]["s"]); e = raw[q+"_error"].astype(float)[keep]
        pm &= np.abs(raw[q].astype(float)[keep]-c) < 2*np.sqrt(s**2+e**2)
    yv = x-np.interp(u, G, R["sky_window"]["centre"]); hw = np.interp(u, G, R["sky_window"]["half_width"])
    P = json.loads((ROOT/"results/plot_data/gaia_cmd_mask.json").read_text()); M = CMDMask(P); Z = np.load(ROOT/"results/plot_data/gaia_cmd_mask_errors.npz")
    dmf = lambda p: np.minimum(np.polyval(P["dm_relation"]["coef"], p), 0.)
    cg_, gg_ = np.linspace(-0.2, 2.0, 221), np.linspace(12.0, 21.6, 241); CG, GG = np.meshgrid(cg_, gg_)
    Eb = np.arange(-14, 14.01, 1.); cb = 0.5*(Eb[1:]+Eb[:-1]); offb = np.abs(cb) >= 5
    fig, ax = plt.subplots(2, 4, figsize=(20, 11), constrained_layout=True, sharey=True)
    for a, (a0, a1) in zip(ax.ravel(), BINS):
        k = pm & (u >= a0) & (u < a1); h = float(np.median(hw[k & (np.abs(yv) < 3)])) if np.any(k & (np.abs(yv) < 3)) else 3.
        on = k & (np.abs(yv) < hw); Hn = np.histogram2d(c0[on], g0[on], (EC, EG))[0]
        ks = k & (np.abs(yv) < 14); H3 = np.histogramdd(np.column_stack((yv[ks], c0[ks], g0[ks])), (Eb, EC, EG))[0]
        ob = offb & (H3.sum((1, 2)) > 0); both = (ob & (cb < 0)).sum() >= 3 and (ob & (cb > 0)).sum() >= 3; dg = 3 if both else 1
        coef = np.tensordot(np.linalg.pinv(np.vander(cb[ob], dg)), H3[ob], 1); yy = np.linspace(-h, h, 41)
        B = np.clip(np.tensordot(np.vander(yy, dg), coef, 1), 0, None).mean(0)*(2*h)
        dm = float(dmf(0.5*(a0+a1)))
        e_c = np.interp(GG.ravel()+0.3, Z["gcen"], Z["med_sc"]); e_g = np.interp(GG.ravel()+0.3, Z["gcen"], Z["med_sg"])
        zg = M.z(CG.ravel(), GG.ravel(), e_c, e_g, dm).reshape(CG.shape); zg[~np.isfinite(zg)] = 99
        Cc, Gc = np.meshgrid(0.5*(EC[1:]+EC[:-1]), 0.5*(EG[1:]+EG[:-1]), indexing="ij")             # mask evaluated at pixel centres
        zp = M.z(Cc.ravel(), Gc.ravel(), np.interp(Gc.ravel()+0.3, Z["gcen"], Z["med_sc"]), np.interp(Gc.ravel()+0.3, Z["gcen"], Z["med_sg"]), dm).reshape(Cc.shape)
        inpix = (zp < 2) & (Gc-dm >= G_BRIGHT)
        Nin, Bin = Hn[inpix].sum(), B[inpix].sum(); N, Bt = Hn.sum(), B.sum()
        a.imshow(gaussian_filter(Hn-B, 1.).T, origin="lower", extent=(EC[0], EC[-1], EG[0], EG[-1]), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        a.contour(cg_, gg_, zg, levels=[2.], colors="k", linewidths=0.9); a.axhline(G_BRIGHT+dm, color="k", ls=":", lw=0.8)
        a.set(xlim=(-0.2, 2.0), ylim=(21.5, 12), xlabel=r"(BP-RP)$_0$",
              title=f"$\\phi_1$ {a0}-{a1}, dm {dm:+.2f}, sky $\\pm${h:.1f}$^\\circ$\nin CMD mask: {Nin-Bin:.0f} on B {Bin:.0f} (S/N {(Nin-Bin)/np.sqrt(max(Bin, 1)):.1f}); all: {N-Bt:.0f} on B {Bt:.0f}"); a.title.set_fontsize(10)
        print(f"phi1 {a0}-{a1}: in mask {Nin-Bin:.0f} (B {Bin:.0f}, S/N {(Nin-Bin)/np.sqrt(max(Bin, 1)):.1f}); all {N-Bt:.0f} (B {Bt:.0f})", flush=True)
    for a in ax[:, 0]:
        a.set_ylabel(r"G$_0$")
    fig.suptitle(f"CMD excess (on-sky minus per-pixel off-sky interpolation) of stars in the data-driven PM + sky masks "
                 f"({'orbit start' if TAG == 'orbitstart' else 'marked start'}); black: CMD mask k = 2 at dm($\\phi_1$); dotted: RGB limit (no CMD cut above)")
    out = ROOT/f"plots/stream_cmd_excess_masks{'_' + TAG if TAG else ''}.png"; fig.savefig(out, dpi=95); print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
