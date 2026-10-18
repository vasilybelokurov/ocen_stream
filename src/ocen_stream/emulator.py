"""Stochastic-kriging emulator of the stream log-likelihood (2026-10-18).

Model (Ankenman, Nelson & Staum 2010, Operations Research 58, 371, 2010OpRes..58..371A; heteroskedastic GP as in Binois,
Gramacy & Ludkovski, arXiv:1611.05902):
    y_ij = m(x_i) + e_ij,   e_ij ~ N(0, s2(x_i)),   j = 1..r_i replicates at design point x_i,
with m the infinite-particle ln L (here the jackknife-corrected total) and s2 the simulation-noise variance.
1. Noise surface. At points with r_i >= 2 replicates the sample variance S2_i satisfies, for Gaussian noise,
   E[ln S2_i] = ln s2(x_i) + psi(k/2) - ln(k/2), Var[ln S2_i] = psi1(k/2), k = r_i - 1 (psi = digamma, psi1 = trigamma).
   ln s2(x) = a + b ln v_jk(x) + g(x): v_jk is the run's own jackknife variance (available at every point, mean over the
   point's runs), g a GP fitted to the bias-adjusted ln S2_i with the known Var[ln S2_i] as per-point noise. Nothing is set
   by hand: a, b, the GP hyper-parameters and the noise level come from the replicates.
2. Mean. GP (constant mean + Matern-5/2 ARD kernel) on the replicate means ybar_i with per-point noise s2(x_i)/r_i fixed
   from step 1 (no extra free noise term).
Inputs are in the unit cube (the design's coordinates). Validation: K-fold over design points (all replicates of a point
leave together); z_i = (ybar_i - mu_i)/sqrt(var_mu_i + s2_i/r_i) should have sd ~ 1 and 95 % coverage ~ 0.95.
"""
import numpy as np
from scipy.special import digamma, polygamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

__all__ = ["group_replicates", "NoiseModel", "MeanEmulator", "kfold_validate"]


def group_replicates(X, y, vjk, key):
    """Collapse runs to design points. X (n, d) unit coords, y (n,) totals, vjk (n,) jackknife variances, key (n,) point ids.
    Returns dict(X (m, d), ybar, r, S2 (nan if r < 2), vjk_mean, keys)."""
    keys = list(dict.fromkeys(key)); key = np.asarray(key)
    Xg, yb, r, S2, vj = [], [], [], [], []
    for k in keys:
        s = key == k
        Xg.append(X[s][0]); yb.append(y[s].mean()); r.append(int(s.sum()))
        S2.append(y[s].var(ddof=1) if s.sum() >= 2 else np.nan); vj.append(vjk[s].mean())
    return dict(X=np.array(Xg), ybar=np.array(yb), r=np.array(r), S2=np.array(S2), vjk_mean=np.array(vj), keys=keys)


class NoiseModel:
    """ln s2(x) = a + b ln vjk(x) + GP(x), fitted to replicate sample variances (see module docstring)."""

    def fit(self, X, S2, r, vjk):
        ok = np.isfinite(S2) & (r >= 2) & (S2 > 0)
        k = (r[ok]-1).astype(float)
        z = np.log(S2[ok]) - (digamma(k/2)-np.log(k/2))          # unbiased for ln s2
        vz = polygamma(1, k/2)                                       # its sampling variance
        A = np.column_stack((np.ones(ok.sum()), np.log(vjk[ok])))
        W = 1/vz
        self.coef = np.linalg.solve(A.T @ (A*W[:, None]), A.T @ (W*z))   # weighted least squares
        res = z - A @ self.coef
        kern = ConstantKernel(0.1, (1e-4, 10.))*RBF(np.full(X.shape[1], 0.5), (0.05, 20.))
        self.gp = GaussianProcessRegressor(kern, alpha=vz, normalize_y=False, n_restarts_optimizer=3, random_state=0)
        self.gp.fit(X[ok], res)
        self.n_fit = int(ok.sum()); self.resid_sd = float(np.std(res)); self.mean_vz = float(np.mean(vz))
        return self

    def predict_log(self, X, vjk):
        return self.coef[0]+self.coef[1]*np.log(vjk)+self.gp.predict(X)

    def predict(self, X, vjk):
        return np.exp(self.predict_log(X, vjk))


def _features(X, trend):
    if trend == "linear":
        return X
    if trend == "quadratic":
        from sklearn.preprocessing import PolynomialFeatures
        return PolynomialFeatures(2, include_bias=False).fit_transform(X)
    raise ValueError(trend)


class MeanEmulator:
    """GP on replicate means with fixed per-point noise s2/r. trend: 'constant' (default), 'linear' or 'quadratic' --
    a ridge-regularised polynomial mean (alpha by generalised CV, weights 1/noise) with the GP on its residuals
    (2026-10-18; trend-coefficient uncertainty is not propagated)."""

    def __init__(self, n_restarts=5, nu=2.5, trend="constant"):
        self.n_restarts, self.nu, self.trend = n_restarts, nu, trend

    def _trend(self, X):
        return 0. if self.trend == "constant" else self.reg.predict(_features(X, self.trend))

    def fit(self, X, ybar, noise_var):
        d = X.shape[1]
        if self.trend != "constant":
            from sklearn.linear_model import RidgeCV
            self.reg = RidgeCV(alphas=np.logspace(-4, 3, 22)).fit(_features(X, self.trend), ybar, sample_weight=1/noise_var)
            ybar = ybar-self.reg.predict(_features(X, self.trend))
        self.ymu, self.ysd = float(ybar.mean()), float(ybar.std())
        kern = ConstantKernel(1.0, (1e-3, 1e6))*Matern(np.full(d, 0.5), (0.02, 50.), nu=self.nu)
        self.gp = GaussianProcessRegressor(kern, alpha=noise_var/self.ysd**2, normalize_y=False,
                                           n_restarts_optimizer=self.n_restarts, random_state=0)
        self.gp.fit(X, (ybar-self.ymu)/self.ysd)
        return self

    def predict(self, X, return_std=True):
        m, s = self.gp.predict(X, return_std=True); t = self._trend(X)
        return (t+self.ymu+self.ysd*m, self.ysd*s) if return_std else t+self.ymu+self.ysd*m


def kfold_validate(X, ybar, noise_var, K=5, seed=0, **kw):
    """K-fold over design points. Returns dict(z, mu, sd_pred, rmse, z_sd, cover95, cover68)."""
    n = len(ybar); idx = np.random.default_rng(seed).permutation(n); folds = np.array_split(idx, K)
    mu = np.zeros(n); sp = np.zeros(n)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        em = MeanEmulator(**kw).fit(X[tr], ybar[tr], noise_var[tr])
        mu[f], sp[f] = em.predict(X[f])
    tot = np.sqrt(sp**2+noise_var)
    z = (ybar-mu)/tot
    return dict(z=z, mu=mu, sd_pred=sp, rmse=float(np.sqrt(np.mean((ybar-mu)**2))), z_sd=float(np.std(z)),
                cover95=float(np.mean(np.abs(z) < 1.96)), cover68=float(np.mean(np.abs(z) < 1.0)))
