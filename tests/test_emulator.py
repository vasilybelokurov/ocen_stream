"""Stochastic-kriging emulator on a 9-D toy with known mean and heteroscedastic noise (layout as the real training set)."""
import numpy as np
from scipy.stats import qmc

from ocen_dm.streams.emulator import group_replicates, NoiseModel, MeanEmulator, kfold_validate


def _toy(seed=1):
    rng = np.random.default_rng(seed)
    U = qmc.Sobol(9, scramble=True, seed=seed).random_base2(8)                  # 256 points
    c = np.array([4000, 3000, 2500, 800, 1500, 300, 300, 600, 400.])
    m = lambda x: -22000-np.sum(c*(x-0.4)**2, axis=1)-1500*x[:, 0]*x[:, 2]      # smooth, anisotropic, with an interaction
    s2 = lambda x: (30+120*x[:, 2]*x[:, 3])**2                                    # noise grows with "amp x size"
    rows = []
    rep = set(range(0, 256, 4))                                                   # 64 points with 3 runs
    for i, x in enumerate(U):
        for j in range(3 if i in rep else 1):
            sd = np.sqrt(s2(x[None])[0])
            rows.append((i, m(x[None])[0]+rng.normal(0, sd), s2(x[None])[0]*np.exp(rng.normal(0, 0.35))))   # jk var: noisy proxy
    key = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); v = np.array([r[2] for r in rows])
    return U, m, s2, key, y, v


def test_noise_model_and_coverage():
    U, m, s2, key, y, v = _toy()
    g = group_replicates(U[key], y, v, key)
    nm = NoiseModel().fit(g["X"], g["S2"], g["r"], g["vjk_mean"])
    s2hat = nm.predict(g["X"], g["vjk_mean"])
    ratio = s2hat/s2(g["X"])
    assert 0.6 < np.median(ratio) < 1.6 and np.percentile(ratio, 90)/np.percentile(ratio, 10) < 4
    val = kfold_validate(g["X"], g["ybar"], s2hat/g["r"], K=5, n_restarts=2)
    assert 0.7 < val["z_sd"] < 1.4 and 0.88 < val["cover95"] <= 1.0
