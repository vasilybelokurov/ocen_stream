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


def test_noise_model_unbiased_over_seeds():
    """64 points x 3 runs pin s2 to ~ a factor 2 per point; no bias on average (mean ln ratio over 4 toys within 0.3)."""
    lr = []
    for s in (1, 2, 3, 4):
        U, m, s2, key, y, v = _toy(s)
        g = group_replicates(U[key], y, v, key)
        r = NoiseModel().fit(g["X"], g["S2"], g["r"], g["vjk_mean"]).predict(g["X"], g["vjk_mean"])/s2(g["X"])
        assert np.percentile(r, 90)/np.percentile(r, 10) < 6
        lr.append(np.mean(np.log(r)))
    assert abs(np.mean(lr)) < 0.3


def test_kfold_coverage():
    U, m, s2, key, y, v = _toy()
    g = group_replicates(U[key], y, v, key)
    s2hat = NoiseModel().fit(g["X"], g["S2"], g["r"], g["vjk_mean"]).predict(g["X"], g["vjk_mean"])
    val = kfold_validate(g["X"], g["ybar"], s2hat/g["r"], K=5, n_restarts=2)
    assert 0.7 < val["z_sd"] < 1.4 and 0.88 < val["cover95"] <= 1.0
