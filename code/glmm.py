"""鄉鎮隨機截距的二項模型（含 offset），邊際最大概似；回傳經驗貝氏（收縮）估計。
O_j ~ Binomial(n_j, expit(off_j + X_j·beta + u_j)),  u_j ~ N(0, tau^2)
off_j = logit(E_j / n_j)，E_j 為只看房屋組成時的期望繼承棟數。
積分：對 z = u/tau 在 [-7, 7] 取 4,001 點的細格點（梯形法）。大鄉鎮的後驗很窄（SD 約 0.02），
固定節點的 Gauss–Hermite 會讓後驗落在單一節點上，所以改用細格點。"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
_z = np.linspace(-7, 7, 4001); _lw = -0.5*_z**2 - 0.5*np.log(2*np.pi) + np.log(_z[1] - _z[0])
def _ll_mat(par, O, n, off, X):
    beta, tau = par[:-1], np.exp(par[-1])
    eta = (off + X @ beta)[:, None] + tau*_z[None, :]
    lp = -np.logaddexp(0, -eta); l1p = -np.logaddexp(0, eta)
    return (gammaln(n+1) - gammaln(O+1) - gammaln(n-O+1))[:, None] + O[:, None]*lp + (n-O)[:, None]*l1p + _lw[None, :]
def fit(O, n, off, X=None):
    O, n, off = map(lambda a: np.asarray(a, float), (O, n, off))
    X = np.ones((len(O), 1)) if X is None else np.asarray(X, float)
    f = lambda par: -logsumexp(_ll_mat(par, O, n, off, X), axis=1).sum()
    r = minimize(f, np.r_[np.zeros(X.shape[1]), np.log(0.3)], method="L-BFGS-B", bounds=[(None, None)]*X.shape[1] + [(np.log(1e-3), np.log(3))])
    M = _ll_mat(r.x, O, n, off, X); W = np.exp(M - logsumexp(M, axis=1, keepdims=True))
    tau = np.exp(r.x[-1]); u = tau*(W @ _z); v = np.maximum(tau**2*(W @ _z**2) - u**2, 0)
    return {"beta": r.x[:-1], "tau": tau, "u": u, "u_sd": np.sqrt(v), "ll": -r.fun, "fixed": X @ r.x[:-1]}
