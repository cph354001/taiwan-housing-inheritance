"""rev1014 核對 6：鄉鎮平均 offset 對逐棟 offset。
舊：O_j ~ Bin(n_j, expit(logit(E_j/n_j) + α + u_j))（12_composition_vs_place.py、glmm.py）
新：D_i ~ Bern(expit(logit(p_i) + α + u_j))，p_i 為 112 年逐棟組成模型預測（以 110–111 年訓練；data/bcd_with_pcomp.parquet，即舊版 E_j 的來源）。
為加速，同鄉鎮內依 p_i 分成 B 個分位箱；箱內：Σ D_i·logit(p_i) 精確保留（不含參數），log(1+e^η) 以箱內平均 logit 近似。B=20、50、100 比較收斂。
積分與 glmm.py 相同：z∈[-7,7] 4,001 點梯形法。輸出 out/rev1014_glmm_bldoffset.json、out/rev1014_u_compare_112.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit, logsumexp
from scipy.stats import spearmanr, pearsonr
from common import *
_z = np.linspace(-7, 7, 4001); _lw = -0.5*_z**2 - 0.5*np.log(2*np.pi) + np.log(_z[1] - _z[0])
t = pd.read_csv(OUT/"township_master_112.csv").merge(pd.read_csv(OUT/"lvr_township.csv")[["house_cd","turn_old","lv_stock_old"]], on="house_cd", how="left")
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet"); d = d[(d.year == 112) & d.house_cd.isin(t.house_cd)].copy()
assert len(d) == t.n.sum(), (len(d), t.n.sum())
d["lo"] = logit(np.clip(d.p_comp, 1e-6, 1-1e-6))
code = {h: i for i, h in enumerate(t.house_cd)}; d["g"] = d.house_cd.map(code)
def make_bins(B):
    d["r"] = d.groupby("g").lo.rank(method="first", pct=True); d["bin"] = np.minimum((d.r*B).astype(int), B-1)
    x = d.groupby(["g","bin"]).agg(n=("D","size"), O=("D","sum"), lo=("lo","mean")).reset_index().sort_values(["g","bin"])
    return x
def fit(x, n_town):
    g = x.g.values; starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    n, O, lo = x.n.values.astype(float), x.O.values.astype(float), x.lo.values
    def mat(par):
        a, tau = par[0], np.exp(par[1]); eta = (lo + a)[:, None] + tau*_z[None, :]
        ll = O[:, None]*eta - n[:, None]*np.logaddexp(0, eta)               # 常數項 Σ D_i·(logit p_i − 箱內平均) 與參數無關，略去
        return np.add.reduceat(ll, starts, axis=0) + _lw[None, :]
    f = lambda par: -logsumexp(mat(par), axis=1).sum()
    r = minimize(f, np.r_[0.0, np.log(0.3)], method="L-BFGS-B", bounds=[(None, None), (np.log(1e-3), np.log(3))])
    M = mat(r.x); W = np.exp(M - logsumexp(M, axis=1, keepdims=True)); tau = np.exp(r.x[1])
    u = tau*(W @ _z); v = np.maximum(tau**2*(W @ _z**2) - u**2, 0)
    return {"alpha": float(r.x[0]), "tau": float(tau), "u": u, "u_sd": np.sqrt(v), "converged": bool(r.success), "nll": float(r.fun)}
R = {"n_town": len(t), "n_bld": int(len(d))}
old = t[["house_cd","name","county","u","u_sd","turn_old","lv_stock_old","lp","alpha"]].copy()
R["old"] = {"alpha": float(t.alpha.iloc[0]), "tau": float(pd.read_csv(OUT/"composition_vs_place_summary.csv", index_col=0).loc[112, "tau_place"])}
sig = lambda u, s: np.where(u - 1.96*s > 0, 1, np.where(u + 1.96*s < 0, -1, 0))
big = old.lv_stock_old >= 1000
R["old"]["spearman_u_turn_old_244"] = float(spearmanr(old.u[big], old.turn_old[big])[0])
# 檢查：箱數 = 1 時應退化成舊模型的 offset 嗎？不會（舊 offset 是 logit(平均 p)，箱=1 是平均 logit p），所以另跑「舊 offset 寫成同一函式」做實作驗證
xo = pd.DataFrame({"g": np.arange(len(t)), "n": t.n.values, "O": t.O.values, "lo": t.off.values})
fo = fit(xo, len(t)); R["impl_check_old_offset_refit"] = {"alpha": fo["alpha"], "tau": fo["tau"], "corr_u_with_stored": float(np.corrcoef(fo["u"], t.u)[0,1]), "max_abs_diff_u": float(np.abs(fo["u"] - t.u).max())}
s_old = sig(old.u.values, old.u_sd.values)
for B in [20, 50, 100]:
    f = fit(make_bins(B), len(t)); u, s = f["u"], f["u_sd"]; s_new = sig(u, s)
    A_old, A_new = set(np.flatnonzero(s_old != 0)), set(np.flatnonzero(s_new != 0))
    R[f"B{B}"] = {"alpha": f["alpha"], "tau": f["tau"], "converged": f["converged"],
        "pearson_old_new": float(pearsonr(old.u, u)[0]), "spearman_old_new": float(spearmanr(old.u, u)[0]),
        "mean_abs_diff": float(np.abs(old.u - u).mean()), "max_abs_diff": float(np.abs(old.u - u).max()),
        "sd_u_old": float(old.u.std()), "sd_u_new": float(np.std(u, ddof=1)),
        "slope_new_on_old": float(np.polyfit(old.u, u, 1)[0]),
        "n_sig_old": len(A_old), "n_sig_new": len(A_new), "n_sig_both": len(A_old & A_new), "jaccard_sig": len(A_old & A_new)/len(A_old | A_new),
        "sign_class_agreement_3way": float((s_old == s_new).mean()), "sig_pos_old_new": [int((s_old == 1).sum()), int((s_new == 1).sum())], "sig_neg_old_new": [int((s_old == -1).sum()), int((s_new == -1).sum())],
        "sig_opposite_sign": int(((s_old*s_new) == -1).sum()),
        "spearman_u_new_turn_old_244": float(spearmanr(u[big.values], old.turn_old[big])[0]),
        "ref25_P5_P95_new_with_alpha": [float(expit(logit(.25) + f["alpha"] + np.quantile(u, .05))), float(expit(logit(.25) + f["alpha"] + np.quantile(u, .95)))],
        "u_P5_P95_new": [float(np.quantile(u, .05)), float(np.quantile(u, .95))]}
    x = old.lp - old.lp.mean(); R[f"B{B}"]["pop_slope_u_new"] = float((x*(u - u.mean())).sum()/(x*x).sum()); R[f"B{B}"]["pop_slope_u_old"] = float((x*(old.u - old.u.mean())).sum()/(x*x).sum())
    if B == 50: old["u_new"], old["u_sd_new"], old["sig_old"], old["sig_new"] = u, s, s_old, s_new
    print(B, json.dumps(R[f"B{B}"], ensure_ascii=False, default=float))
# 逐棟 offset 的組成項（鄉鎮的平均 logit p_i）與舊 offset（logit 平均 p_i）的差異
mlo = d.groupby("g").lo.mean().values; R["offset_diff"] = {"mean(logit(Ebar)-mean logit p)": float((t.off.values - mlo).mean()), "sd": float((t.off.values - mlo).std()), "corr_with_old_u": float(np.corrcoef(t.off.values - mlo, t.u)[0,1])}
old.to_csv(OUT/"rev1014_u_compare_112.csv", index=False, encoding="utf-8-sig")
json.dump(R, open(OUT/"rev1014_glmm_bldoffset.json","w"), ensure_ascii=False, indent=1, default=float); print(json.dumps({k: v for k, v in R.items() if not k.startswith("B")}, ensure_ascii=False, indent=1, default=float))
