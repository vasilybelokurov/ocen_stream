#!/usr/bin/env python3
"""Stage 0: observed track of Ibata+2024 stream 54 (northern arm) with src/ocen_dm/streams/track.py.
Selection: stream 54, b > 15, -75 < l < -20 (excludes the cluster region b < 15). Gaia DR3 PM errors from
results/plot_data/stream5455_gaia_errors.npz. Variants: all members; G < 19 (catalogue dereddened G). Distance-modulus track:
results/plot_data/stream54_cmd_distance.json (relative). v_los: per star (29).
Usage: python bin/streams/measure_track.py -> results/plot_data/stream54_track.json, plots/stream54_track.png
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/"src"))
import numpy as np, astropy.coordinates as coord, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from ocen_dm.streams.track import measure

w = lambda x: np.where(x > 180, x-360, x)


def main():
    t = Table.read(os.path.expandvars("$OCEN_CATALOGUES/streamfinder_ibata2024_dr3.fits")); st = np.asarray(t["Stream"])
    sel = np.isin(st, [54, 55]); t = t[sel]
    er = np.load(ROOT/"results/plot_data/stream5455_gaia_errors.npz")
    assert np.all(er["source_id"] == np.asarray(t["Gaia"], np.int64))
    g = coord.SkyCoord(np.asarray(t["RAdeg"], float), np.asarray(t["DEdeg"], float), unit="deg").galactic
    l, b = w(g.l.deg), g.b.deg
    arm = (np.asarray(t["Stream"]) == 54) & (b > 15) & (l < -20) & (l > -75)
    G = np.asarray(t["Gmag"], float)
    out = {}
    for name, k in (("all", arm), ("G<19", arm & (G < 19))):
        r = measure(l[k], b[k], np.asarray(t["pmRA"], float)[k], np.asarray(t["pmDE"], float)[k], er["pmra_error"][k], er["pmdec_error"][k])
        out[name] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else {a: c.tolist() for a, c in vv.items()}) for kk, vv in r.items()}
        print(f"\n{name}: N = {k.sum()}")
        print("  b    n    l (+-)           pmra (+-) [width]       pmdec (+-) [width]")
        for j, bc in enumerate(r["b"]):
            print(f"  {bc:4.0f} {r['n'][j]:4d}  {r['l']['mu'][j]:6.2f} {r['l']['sig_mu'][j]:4.2f}   {r['pmra']['mu'][j]:6.2f} {r['pmra']['sig_mu'][j]:4.2f} [{r['pmra']['width'][j]:4.2f}]"
                  f"   {r['pmdec']['mu'][j]:6.2f} {r['pmdec']['sig_mu'][j]:4.2f} [{r['pmdec']['width'][j]:4.2f}]")
    v = np.asarray(t["VHel"], float); ev = np.asarray(t["e_VHel"], float); hv = arm & np.isfinite(v) & (ev < 300)
    out["vlos_stars"] = dict(b=b[hv].tolist(), l=l[hv].tolist(), v=v[hv].tolist(), e=ev[hv].tolist())
    out["distance_modulus_track"] = json.loads((ROOT/"results/plot_data/stream54_cmd_distance.json").read_text())
    (ROOT/"results/plot_data/stream54_track.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for a, q in zip(ax, ("l", "pmra", "pmdec")):
        a.plot(b[arm], {"l": l, "pmra": np.asarray(t["pmRA"], float), "pmdec": np.asarray(t["pmDE"], float)}[q][arm], ".", ms=2, color="0.75")
        for name, c in (("all", "k"), ("G<19", "C3")):
            r = out[name]; a.errorbar(np.array(r["b"])+(0.3 if c == "C3" else 0), r[q]["mu"], yerr=r[q]["sig_mu"], fmt="o", color=c, ms=4, label=name)
        a.set_xlabel("b [deg]"); a.set_ylabel(q); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0].invert_yaxis(); ax[1].set_ylim(-20, 0); ax[2].set_ylim(-14, -3)
    fig.suptitle("Stream 54 northern-arm track (mixture fit per 2-deg b bin, block-bootstrap errors)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT/"plots/stream54_track.png", dpi=85)


if __name__ == "__main__":
    main()
