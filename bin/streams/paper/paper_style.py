"""Shared matplotlib style for the stream-54 paper figures (MNRAS: column 3.32 in, page 6.97 in, 8-9 pt text)."""
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "ocen-df-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN, PAGE = 3.32, 6.97
COLOURS = dict(data="0.15", null="#1f77b4", bar28="#d95f02", bar16="#7570b3", alt="#1b9e77", grey="0.6")
LABELS = dict(null="no bar (state refitted)", bar28="bar, 28 deg", bar16="bar, 16 deg")


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
    })
    return plt
