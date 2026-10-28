#!/usr/bin/env python3
"""Draw the measured data vector and a model's stored measurements onto an existing stream_model_compare map figure (2026-10-28).

No recomputation: the figure (plots/stream_model_compare_<model>.png) is loaded as an image; the residual-column panel frames are
found from the image (dark spine pixels); data and model values are mapped into them with the panels' axis ranges
(sky: phi1 0-24, v -10..10 deg; pmra: -2..28, -28..4; pmdec: -2..28, -18..2 mas/yr; as in stream_model_compare.model_maps).
Data: sky local fits (results/plot_data/stream_sky_local_nostrip.json; window phi1 6-9 dropped, grey), PM spline knots
(stream_spline_fit.json) as frozen track (stream_masks.json) + fitted offset. Model: results/plot_data/stream_model_compare_<tag>.json.
Sky: track c + g (phi1 - centre) over each window (solid) and +- width (dashed); PMs: knot values +- 1 sigma.
Usage: python bin/streams/overlay_measurements.py --png plots/stream_model_compare_final_c0_2.png --json stream_model_compare_c0
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mi

ROOT = Path(__file__).resolve().parents[2]; J = lambda n: json.loads((ROOT/f"results/plot_data/{n}.json").read_text())
RANGES = (("sky", (0, 24), (-10, 10)), ("pmra", (-2, 28), (-28, 4)), ("pmdec", (-2, 28), (-18, 2)))
DROP = ((6., 9.),); MC = "#1baf7a"


def frames(im):
    """pixel boxes (x0, x1, y0, y1) of the three right-column panels from the dark spine pixels."""
    dark = im[..., :3].mean(2) < 0.35; H, W = dark.shape
    rows = np.where(dark[:, W//2:].sum(1) > 300)[0]; cols = np.where(dark[H//4:, :].sum(0) > 150)[0]
    group = lambda a: [g[0] for g in np.split(a, np.where(np.diff(a) > 3)[0]+1)]
    r = [x for x in group(rows) if 20 < x < H-20] + [rows[rows > H-60].min()] if False else group(rows)
    r = [x for x in r if x > 20]; c = [x for x in group(cols) if x > W//2]
    y = sorted(r)[:6]; x0, x1 = c[-2], c[-1]
    return [(x0, x1, y[2*i], y[2*i+1]) for i in range(3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--png", required=True); ap.add_argument("--json", required=True); a = ap.parse_args()
    im = mi.imread(ROOT/a.png); H, W = im.shape[:2]; B = frames(im)
    M = J(a.json)["models"][0]; L = J("stream_sky_local_nostrip"); F = J("stream_spline_fit"); R = J("stream_masks"); G = np.array(R["grid"])
    E = 190; fig = plt.figure(figsize=((W+E)/80, H/80), dpi=80); ax = fig.add_axes([0, 0, W/(W+E), 1]); ax.imshow(im); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    for (q, (xa, xb), (ya, yb)), (x0, x1, y0, y1) in zip(RANGES, B):
        px = lambda x: x0+(np.asarray(x)-xa)/(xb-xa)*(x1-x0); py = lambda y: y1-(np.asarray(y)-ya)/(yb-ya)*(y1-y0)
        if q == "sky":
            win = [w for w in L["windows"] if tuple(w["phi1"]) not in DROP]
            for w in L["windows"]:
                a0, a1 = w["phi1"]; x = np.array([a0, a1]); pk = 0.5*(a0+a1); i = win.index(w) if w in win else None
                vals = [(w["c"], w["s"], "0.5", "dropped window")] if i is None else [(w["c"], w["s"], "k", "data"), (M["sky_c"][i], M["sky_s"][i], MC, "model")]
                for c, s, col, lab in vals:
                    ax.plot(px(x), py(c+w["g"]*(x-pk)), color=col, lw=2.5, label=lab if a0 in (0., 6.) else None)
                    for sg in (-1, 1):
                        ax.plot(px(x), py(c+sg*s+w["g"]*(x-pk)), color=col, lw=1.3, ls="--")
                if i is not None:
                    ax.errorbar(px(pk), py(w["c"]), yerr=w["c_err"]*(y1-y0)/(yb-ya), color="k", capsize=2, lw=1.2)
        else:
            f = F[q]; n = len(f["fit"]["m"]["knots"]); k = np.array(f["fit"]["m"]["knots"]); t = np.interp(k, G, R["tracks"][q]["c"])
            d, de = np.array(f["fit"]["m"]["values"]), np.sqrt(np.diag(np.array(f["cov"])[:n, :n])); m, me = np.array(M[q], float), np.array(M[q+"_e"], float)
            s = (y1-y0)/(yb-ya)
            ax.errorbar(px(k), py(t+d), yerr=de*s, fmt="s", color="k", ms=6, capsize=3, lw=1.5, label="data knots" if q == "pmra" else None)
            ax.errorbar(px(k), py(t+m), yerr=me*s, fmt="o", color=MC, mec="k", ms=6, capsize=2, lw=1.5, label="model knots" if q == "pmra" else None)
    fig.legend(*ax.get_legend_handles_labels(), loc="center right", fontsize=9, frameon=False, title="right column:\nmeasurements", title_fontsize=9)
    out = ROOT/a.png.replace(".png", "_measured.png"); fig.savefig(out, dpi=80); print(out)


if __name__ == "__main__":
    main()
