#!/usr/bin/env python3
"""chi2 (current objective) vs the normalised Gaussian -2 lnL = chi2 + log|C_data + C_model| over the rescored models (2026-10-25).

Question (Fable vs Codex): does dropping log|C| matter? C_model (particle-bootstrap errors) varies between models, so plain chi2
rewards models with larger model errors (fewer particles in a window, broader debris). Uses the per-point model values and errors
saved by stream_model_rescore.py --tag vec; same covariance assembly as stream_model_compare.chi2 (sky diagonal, PM C_data + diag
C_model); log|C| relative to log|C_data| (a constant), missing windows skipped in log|C| (they carry the 1e3 penalty in chi2).
Usage: PYTHONPATH=src python bin/streams/likelihood_norm_compare.py -> plots/likelihood_norm_compare.png, results/plot_data/likelihood_norm_compare.json
"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT/"bin/streams"), str(ROOT/"src")]
sys.argv = [sys.argv[0]]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import stream_model_compare as MC

PAR = ("angle", "omega", "amp", "size", "halo_q", "vc", "dist", "pmra", "pmdec")


def logdet_excess(D, M):
    """log|C_data + C_model| - log|C_data| for the 25-point vector."""
    out = 0.
    for q, e in (("sky_c", "sky_ce"), ("sky_s", "sky_se")):
        ok = np.isfinite(M[q]); out += np.sum(np.log1p(M[e][ok]**2/D[e][ok]**2))
    for q in ("pmra", "pmdec"):
        ok = np.isfinite(M[q]); Cd = D[q+"_C"][np.ix_(ok, ok)]
        out += np.linalg.slogdet(Cd+np.diag(M[q+"_e"][ok]**2))[1]-np.linalg.slogdet(Cd)[1]
    return float(out)


def main():
    D = MC.data_vector(); rows = [r for r in csv.DictReader(open(ROOT/"results/plot_data/stream_model_rescore_vec.csv")) if r.get("chi2_total")]
    rec = []
    for r in rows:
        M = {k: np.array(json.loads(r["v_"+k]), float) for k in ("sky_c", "sky_s", "pmra", "pmdec", "sky_ce", "sky_se", "pmra_e", "pmdec_e")}
        c = MC.chi2(D, M); ld = logdet_excess(D, M)
        rec.append(dict(file=r["file"], set=r["set"], n=int(r["n_particles"]), chi2=c["total"], logdet=ld, norm=c["total"]+ld,
                        err_ratio=float(np.nanmedian(np.r_[M["sky_se"]/D["sky_se"], M["pmra_e"]/np.sqrt(np.diag(D["pmra_C"]))])),
                        **{k: float(r[k]) if r[k] not in ("", "nan") else np.nan for k in PAR}))
    C2, LD, NO, N = (np.array([x[k] for x in rec]) for k in ("chi2", "logdet", "norm", "n"))
    ib, jb = np.argmin(C2), np.argmin(NO); top = lambda a, k: set(np.argsort(a)[:k])
    summ = dict(n_models=len(rec), best_chi2=rec[ib], best_norm=rec[jb], top10_overlap=len(top(C2, 10) & top(NO, 10)),
                top50_overlap=len(top(C2, 50) & top(NO, 50)), logdet_top50=[float(np.percentile(LD[list(top(C2, 50))], p)) for p in (5, 50, 95)],
                spearman_top200=float(np.corrcoef(np.argsort(np.argsort(C2))[list(top(C2, 200))], np.argsort(np.argsort(NO))[list(top(C2, 200))])[0, 1]))
    (ROOT/"results/plot_data/likelihood_norm_compare.json").write_text(json.dumps(dict(summary=summ, models=rec), indent=1))
    print(json.dumps(summ, indent=1))
    good = C2 < 600; fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    a = ax[0, 0]; s = a.scatter(C2[good], NO[good], c=np.log10(N[good]), s=6, cmap="viridis"); fig.colorbar(s, ax=a, label="log10 particles")
    a.plot([150, 600], [150, 600], "k:"); a.set(xlabel="chi2 (current)", ylabel="chi2 + log|C| - log|C_data|", title="objectives (models with chi2 < 600)")
    a = ax[0, 1]; a.scatter(C2[good], LD[good], c=np.log10(N[good]), s=6, cmap="viridis")
    a.set(xlabel="chi2", ylabel="log|C| - log|C_data|", title="log-determinant term vs chi2")
    a = ax[0, 2]; a.scatter(N[good], LD[good], s=6); a.set(xscale="log", xlabel="particles kept", ylabel="log|C| - log|C_data|", title="log|C| vs particle number")
    for a, k in zip(ax[1], ("angle", "omega", "amp")):
        x = np.array([q[k] for q in rec])
        for y, lab, col in ((C2, "chi2", "k"), (NO, "chi2 + log|C|", "C3")):
            e = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), 14); j = np.digitize(x, e)
            env = np.array([(0.5*(e[i-1]+e[i]), y[(j == i) & good].min()) for i in range(1, len(e)) if ((j == i) & good).any()])
            a.plot(*env.T, "-o", color=col, ms=3, label=f"lower envelope: {lab}")
        a.axvline(x[ib], color="k", ls="--", lw=0.8); a.axvline(x[jb], color="C3", ls=":", lw=1.2); a.set(xlabel=k, ylabel="objective", ylim=(140, 450)); a.legend(fontsize=8)
    fig.suptitle(f"chi2 vs normalised Gaussian -2lnL over {len(rec)} rescored models; best by chi2: {Path(rec[ib]['file']).stem} ({C2[ib]:.0f}), "
                 f"by -2lnL: {Path(rec[jb]['file']).stem}; top-50 overlap {summ['top50_overlap']}/50")
    fig.savefig(ROOT/"plots/likelihood_norm_compare.png", dpi=80); print("plots/likelihood_norm_compare.png")


if __name__ == "__main__":
    main()
