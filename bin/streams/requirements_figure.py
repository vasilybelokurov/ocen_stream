#!/usr/bin/env python3
"""Summary figure: what stream 54 requires to be fitted well (2026-10-22).

(a) Non-axisymmetric force the fit demands vs the published bar: max_phi |F_m>0| / |F_axi| against R at z = 0 and 1.5 kpc
    (results/plot_data/bar_options.json): fitted 12.5-deg bar (Hunter A 1.37, S 1.33), published Hunter/Sormani bar, and the
    Sormani long bar with 2x mass and 1.3x length; omega Cen's pericentre and apocentre ranges shaded.
(b) Fit quality of the model variants (256k ln L + prior relative to the best; results/streams/profile_step2_20261020/profile.json,
    results/streams/fixedbar_20261021/result.json; excess-pattern test 32k, relative to its own fitted reference).
(c) Residual tracks (data - model) along the stream in Delta phi2, mu_alpha*, mu_delta (2-deg phi1 bins; as plot_bestfit_real.py)
    for the best inflated-bar model (12.5 deg) and the best published-bar model (27 deg, Omega_b 44.8).
(d) omega Cen's L_z over the last Gyr in the fitted 12.5-deg host and in the same host with the published bar (A = S = 1).
Output: plots/stream_requirements.png
Usage: PYTHONPATH=src python bin/streams/requirements_figure.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def best_final(pattern):
    r = max((json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT/pattern))), key=lambda r: r["total_jk"])
    return Path(pattern).parent/f"{r['run_id']}.npz", r["params"]


def lz_history(p, amp, size):
    from ocen_dm.streams.restricted import host_potential, agama_kpc, AGAMA_T_MYR
    from ocen_dm.streams.frames import to_model, OCEN_OBS, frame_r0
    agama = agama_kpc(); T = 1955.58/AGAMA_T_MYR; r0 = frame_r0("gravity21")
    host = host_potential("x", bar_omega=p["omega"], bar_angle_deg=p["angle"], t_today=T, bar_amp=amp, bar_size=size,
                          axi_variant=dict(halo_q=p["halo_q"], vc_target=p["vc"], r0_kpc=r0))
    today = to_model(OCEN_OBS["ra"], OCEN_OBS["dec"], p["dist"], p["pmra"], p["pmdec"], OCEN_OBS["vlos"], "gravity21")[0]
    t, orb = agama.orbit(potential=host, ic=today, timestart=T, time=-1000./AGAMA_T_MYR, trajsize=2001, accuracy=1e-10)
    return (T-t)*AGAMA_T_MYR, orb[:, 0]*orb[:, 4]-orb[:, 1]*orb[:, 3]


def main():
    from plot_bestfit_real import load, resid
    fig = plt.figure(figsize=(17, 11), constrained_layout=True); gs = fig.add_gridspec(2, 3)
    # (a)
    B = json.loads((ROOT/"results/plot_data/bar_options.json").read_text()); R = B["R"]; pr = B["profiles"]
    ax = fig.add_subplot(gs[0, 0])
    for key, lab, st in (("hunter fitted 12.5 deg (A 1.37 S 1.33)", "fit's demand (12.5-deg best)", "C3-"),
                         ("hunter published (A=S=1)", "published bar (Hunter = Sormani+2022)", "k-"),
                         ("sormani long bar length x1.3, non-axi x2", "long bar 2x mass, 1.3x length", "C2--")):
        ax.plot(R, pr[key]["z0"], st, lw=2, label=f"{lab}, z = 0"); ax.plot(R, pr[key]["z1p5"], st, lw=1, alpha=0.6)
    ax.axvspan(1.5, 2.2, color="0.93"); ax.axvspan(6.4, 7.2, color="0.88")
    ax.text(1.85, 0.47, "peri", ha="center", fontsize=8); ax.text(6.8, 0.47, "apo", ha="center", fontsize=8)
    ax.set(xlabel="R [kpc]", ylabel=r"max$_\phi$ $|F_{m>0}|$ / $|F_{\rm axi}|$", ylim=(0, 0.5),
           title="(a) non-axisymmetric force: demanded vs published\n(thin: z = 1.5 kpc)"); ax.legend(fontsize=7.5)
    # (b)
    prof = json.loads((ROOT/"results/streams/profile_step2_20261020/profile.json").read_text())
    fb = json.loads((ROOT/"results/streams/fixedbar_20261021/result.json").read_text())
    best = max(d["mean"]+d.get("ln_prior", 0) for d in prof.values())
    rows = [(f"inflated bar, {a} deg", prof[a]["mean"]+prof[a]["ln_prior"]-best) for a in ("12.5", "17.5", "22.5", "25", "27.5", "32.5")]
    rows += [("published bar, 27 deg, Om 44.8", fb["from25_om42"]["mean"]-best), ("published bar, 27 deg, Om 33.4", fb["from25"]["mean"]-best)]
    ax = fig.add_subplot(gs[0, 1]); y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=["C3"]*6+["k", "0.5"]); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for yi, (_, v) in zip(y, rows):
        ax.text(v-40, yi, f"{v:.0f}", ha="right", va="center", fontsize=8, color="w" if v < -600 else "k")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlabel=r"$\Delta$ ln L (relative to the best; statistical sd ~30-80, T ~ 2)", title="(b) fit quality: inflated vs published bar")
    # (d)
    f12, p12 = best_final("results/streams/profile_step2_20261020/runs/final_a12.5_[0-9].json")
    ax = fig.add_subplot(gs[0, 2])
    for amp, size, lab, st in ((p12["amp"], p12["size"], "fitted bar (A 1.37, S 1.33)", "C3-"), (1.0, 1.0, "published bar (A = S = 1)", "k-")):
        age, lz = lz_history(p12, amp, size); ax.plot(age, lz, st, lw=1.2, label=lab)
    ax.axvspan(0, 700, color="0.95", zorder=0); ax.text(350, ax.get_ylim()[0] if False else -470, "debris in the score (< 700 Myr)", ha="center", fontsize=8)
    ax.set(xlabel="time ago [Myr]", ylabel=r"$L_z$ of omega Cen [kpc km/s]", title="(d) torque history: omega Cen's L_z\n(same 12.5-deg parameters)")
    ax.legend(fontsize=8); ax.invert_xaxis()
    # (c)
    ffb, pfb = best_final("results/streams/fixedbar_20261021/runs/final_from25_om42_[0-9].json")
    e = np.arange(-2., 28.1, 2.)
    r1, r2 = resid(load(ROOT/f12), e), resid(load(ROOT/ffb), e)
    for j, (q, yl) in enumerate((("dphi2", r"$\Delta\phi_2$ residual [deg]"), ("pmra", r"$\mu_{\alpha*}$ residual [mas/yr]"), ("pmdec", r"$\mu_\delta$ residual [mas/yr]"))):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(r1["c"], r1[q], "o-", color="C3", label="inflated bar, 12.5 deg (best)")
        ax.plot(r2["c"], r2[q], "s-", color="k", label="published bar, 27 deg, Omega_b 44.8")
        ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel=r"$\phi_1$ [deg]", ylabel=yl, ylim=(-3, 3.2))
        if j == 0:
            ax.set_title("(c) data - model along the stream (2-deg bins)"); ax.legend(fontsize=8)
    fig.suptitle("Stream 54 (omega Cen): what a good fit requires -- a bar-locked non-axisymmetric force at 4-7 kpc ~2.5-3.5x the published bar's",
                 fontsize=13)
    fig.savefig(ROOT/"plots/stream_requirements.png", dpi=110); print(ROOT/"plots/stream_requirements.png")


if __name__ == "__main__":
    main()
