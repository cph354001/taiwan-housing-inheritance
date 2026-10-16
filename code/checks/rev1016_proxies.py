"""rev1016 審稿建議 11 與 14。輸出 out/rev1016_proxies.json、out/rev1016_proxies_reg.csv、out/rev1016_denominator_sims.csv
 11：主要迴歸 u ~ log 買賣率 + log 死亡代理（272 鄉鎮，標準化，縣市群集 SE；同 21_township_supply_vs_sales.py 的 reg），
     死亡代理並列：(a) 主設定屋主死亡釋出率 m；(a') 只留屋主年齡結構 m_own_nat；(b) 不含模擬屋主年齡的：65+ 死亡率（死亡/65+人口）、
     粗死亡率、年齡標準化死亡比 SMR、報告用的 65+ 死亡人數÷模擬存量 m65；(c) 65+ 人口比例（戶政）。
 14：買賣率分母（模擬建物棟數×10）加入乘法誤差：
     隨機 log 常態 σ = 0.05/0.10/0.20/0.30（各 500 次），鄉鎮層級共同誤差（同一鄉鎮的既有建物存量、老屋存量、全部存量同乘）；
     另做老屋存量獨立誤差的版本；地區型態誤差：人口最少 20% 鄉鎮系統性高估／低估 10%、20%，以及沿 log 人口的連續梯度。
     重算主要迴歸係數與 u 對老屋買賣率的 Spearman（老屋存量≥1000，鄉鎮集合固定為原始樣本）。並以 SIMEX 外推示意既有誤差的方向。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from common import *
rng = np.random.default_rng(20261016)
S = pd.read_csv(OUT/"township_supply_sales.csv").set_index("house_cd")
M = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)].copy()
M["mort65"] = M.deaths65/M.pop65_ris*100; M["cdr"] = M.deaths/M.pop_ris*100; M["sh65"] = M.pop65_ris/M.pop_ris*100
OUTJ = {"n_town": int(len(M))}
def reg(df, yv, xb, xm, cluster=True):
    Xd = pd.DataFrame({"b": np.log(xb), "m": np.log(xm)}, index=df.index); Z = (Xd - Xd.mean())/Xd.std(); yy = (yv - yv.mean())/yv.std()
    f = sm.OLS(yy, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": df.county}) if cluster else sm.OLS(yy, sm.add_constant(Z)).fit()
    return f
# ---------- 11 ----------
PROX = {"(a) 屋主死亡釋出率（主設定）": "m", "(a') 只留屋主年齡結構（全國死亡率）": "m_own_nat", "(a'') 鄉鎮年齡別死亡率直接套用": "m_own_loc",
        "(b1) 65+ 死亡率（65+死亡÷65+人口，純戶政）": "mort65", "(b2) 粗死亡率（純戶政）": "cdr", "(b3) 年齡標準化死亡比 SMR（純戶政）": "smr_death",
        "(b4) 65+ 死亡人數÷模擬存量（報告 §3.4 的版本）": "m65", "(c) 65+ 人口比例（戶政）": "sh65", "(c') 屋主 65+ 比例（模擬屋主檔）": "owner_65"}
rows = {}
for yname, ycol in [("u", "u"), ("原始繼承占比logit", "logit_raw"), ("存量log(繼承÷買賣)", "odds_stock")]:
    for lab, c in PROX.items():
        z = M[M[c] > 0]; f = reg(z, z[ycol], z.bx, z[c])
        r1 = np.corrcoef(np.log(z[c]), z[ycol])[0,1]
        rows[(yname, lab)] = {"n": int(len(z)), "b_買賣率": f.params["b"], "se_買賣率": f.bse["b"], "b_死亡代理": f.params["m"], "se_死亡代理": f.bse["m"], "R2": f.rsquared,
                              "代理單獨R2": r1**2, "corr(log代理, log買賣率)": float(np.corrcoef(np.log(z[c]), np.log(z.bx))[0,1]), "corr(log代理, log主設定m)": float(np.corrcoef(np.log(z[c]), np.log(z.m))[0,1]),
                              "Spearman(代理, y)": spearmanr(z[c], z[ycol])[0]}
R11 = pd.DataFrame(rows).T; R11.index.names = ["y", "死亡代理"]; R11.to_csv(OUT/"rev1016_proxies_reg.csv", encoding="utf-8-sig"); print(R11.round(3).to_string())
OUTJ["11_主要迴歸_各死亡代理"] = {f"{a}｜{b}": {k: float(v) for k, v in r.items()} for (a, b), r in rows.items()}
# 三個變數同時放（買賣率＋主設定 m＋65+ 人口比例），看 m 是否只是在代理高齡人口
Xd = pd.DataFrame({"log買賣率": np.log(M.bx), "log釋出率": np.log(M.m), "log 65+人口比例": np.log(M.sh65), "log 65+死亡率": np.log(M.mort65)})
for cols in [["log買賣率","log釋出率","log 65+人口比例"], ["log買賣率","log 65+人口比例","log 65+死亡率"], ["log買賣率","log釋出率","log 65+人口比例","log 65+死亡率"]]:
    Z = (Xd[cols] - Xd[cols].mean())/Xd[cols].std(); f = sm.OLS((M.u - M.u.mean())/M.u.std(), sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": M.county})
    OUTJ.setdefault("11b_多代理同時", {})[" + ".join(cols)] = {**{c: [float(f.params[c]), float(f.bse[c])] for c in cols}, "R2": float(f.rsquared)}
print(json.dumps(OUTJ["11b_多代理同時"], ensure_ascii=False, indent=1))

# ---------- 14 ----------
LT = pd.read_csv(OUT/"lvr_township.csv").set_index("house_cd")
B = LT[LT.lv_stock_old >= 1000].dropna(subset=["u","turn_old"]).copy()           # 與 17_real_sales_lvr.py lvr_vs_sim 相同（244 個）
base = {"b_買賣率": float(reg(M, M.u, M.bx, M.m).params["b"]), "b_釋出率": float(reg(M, M.u, M.bx, M.m).params["m"]), "R2": float(reg(M, M.u, M.bx, M.m).rsquared),
        "m65版_b_買賣率": float(reg(M, M.u, M.bx, M.m65).params["b"]), "m65版_b_死亡": float(reg(M, M.u, M.bx, M.m65).params["m"]),
        "Spearman(u,老屋買賣率)": float(spearmanr(B.u, B.turn_old)[0]), "n_reg": int(len(M)), "n_spearman": int(len(B))}
OUTJ["14_基準"] = base; print(base)
lpM = np.log(M["pop"]); lpB = np.log(B["pop"])
def one(eM, eB, eB_old=None):
    """eM：主樣本鄉鎮的 log 分母誤差；eB：Spearman 樣本鄉鎮的 log 分母誤差（共同）；eB_old：老屋存量另加的獨立誤差。"""
    bx = M.bx*np.exp(-eM); m65 = M.m65*np.exp(-eM)                            # 分母乘 exp(e) → 率乘 exp(-e)
    f = reg(M, M.u, bx, M.m, cluster=False); f2 = reg(M, M.u, bx, m65, cluster=False)
    to = B.turn_old*np.exp(-eB - (0 if eB_old is None else eB_old))
    return [f.params["b"], f.params["m"], f.rsquared, f2.params["b"], f2.params["m"], spearmanr(B.u, to)[0]]
COLS = ["b_買賣率", "b_釋出率", "R2", "m65版_b_買賣率", "m65版_b_死亡", "Spearman(u,老屋買賣率)"]
def summ(arr, lab):
    a = np.asarray(arr); out = {}
    for i, c in enumerate(COLS):
        out[c] = {"median": float(np.median(a[:, i])), "p2.5": float(np.percentile(a[:, i], 2.5)), "p97.5": float(np.percentile(a[:, i], 97.5))}
    out["符號維持比例"] = {"b_買賣率<0": float((a[:, 0] < 0).mean()), "b_釋出率>0": float((a[:, 1] > 0).mean()), "Spearman<0": float((a[:, 5] < 0).mean()),
                     "b_買賣率 的絕對值 > 基準 的比例": float((np.abs(a[:, 0]) > abs(base["b_買賣率"])).mean()), "|Spearman| > 基準 的比例": float((np.abs(a[:, 5]) > abs(base["Spearman(u,老屋買賣率)"])).mean())}
    return out
SIM = {}; long = []
# 共同鄉鎮誤差：同一鄉鎮的誤差在兩個樣本一致
allcd = M.index.union(B.index)
for sg in [0.05, 0.10, 0.20, 0.30]:
    for mode in ["鄉鎮共同誤差", "老屋存量另加獨立誤差"]:
        res = []
        for r in range(500):
            e = pd.Series(rng.normal(0, sg, len(allcd)), index=allcd)
            eo = pd.Series(rng.normal(0, sg, len(B)), index=B.index) if mode != "鄉鎮共同誤差" else None
            res.append(one(e.reindex(M.index).values, e.reindex(B.index).values, None if eo is None else eo.values))
        SIM[f"σ={sg:.2f}｜{mode}"] = summ(res, ""); long += [{"設定": f"σ={sg:.2f}｜{mode}", **dict(zip(COLS, x))} for x in res]
        print(sg, mode, {k: round(v["median"], 3) for k, v in SIM[f"σ={sg:.2f}｜{mode}"].items() if isinstance(v, dict) and "median" in v}, SIM[f"σ={sg:.2f}｜{mode}"]["符號維持比例"])
# 地區型態：人口最少 20% 的鄉鎮（以 272 主樣本的人口五分位切點）分母系統性偏誤
cut = M["pop"].quantile(0.2)
for dlt in [0.10, -0.10, 0.20, -0.20]:
    k = f"人口最少20%鄉鎮分母{'高' if dlt > 0 else '低'}估{abs(dlt)*100:.0f}%"
    eM = np.where(M["pop"] <= cut, np.log(1 + dlt), 0.0); eB = np.where(B["pop"] <= cut, np.log(1 + dlt), 0.0)
    det = one(eM, eB); SIM[k + "（無隨機誤差）"] = dict(zip(COLS, map(float, det)))
    res = []
    for r in range(500):
        e = pd.Series(rng.normal(0, 0.10, len(allcd)), index=allcd)
        res.append(one(eM + e.reindex(M.index).values, eB + e.reindex(B.index).values))
    SIM[k + "＋隨機σ=0.10"] = summ(res, ""); long += [{"設定": k + "＋隨機σ=0.10", **dict(zip(COLS, x))} for x in res]
    print(k, {c: round(v, 3) for c, v in SIM[k + "（無隨機誤差）"].items()})
# 連續梯度：分母 log 誤差 = g × (log人口 − 中位數)/(第90 − 第10 百分位的 log 人口差)，使第 10 與第 90 百分位鄉鎮相差 g（10%、20%）
q10, q90, med = np.log(M["pop"]).quantile([.1, .9, .5])
for g in [0.10, -0.10, 0.20, -0.20]:
    k = f"人口梯度：小鄉鎮相對大鄉鎮分母{'高' if g > 0 else '低'}估{abs(g)*100:.0f}%（P10 對 P90）"
    eM = -np.log(1 + g)*(lpM - med)/(q90 - q10); eB = -np.log(1 + g)*(lpB - med)/(q90 - q10)
    SIM[k] = dict(zip(COLS, map(float, one(eM.values, eB.values)))); print(k, {c: round(v, 3) for c, v in SIM[k].items()})
# 若誤差與「繼承偏高」本身有關（最壞情況）：u 高的鄉鎮分母被低估 → 量到的買賣率偏高（會削弱負相關）或反之
for g in [0.10, -0.10, 0.20, -0.20]:
    k = f"誤差與u相關：u 每高 1 SD，分母{'高' if g > 0 else '低'}估{abs(g)*100:.0f}%"
    zM = (M.u - M.u.mean())/M.u.std(); zB = (B.u - M.u.mean())/M.u.std()
    SIM[k] = dict(zip(COLS, map(float, one(np.log(1 + g)*zM.values, np.log(1 + g)*zB.values)))); print(k, {c: round(v, 3) for c, v in SIM[k].items()})
OUTJ["14_分母誤差"] = SIM
pd.DataFrame(long).to_csv(OUT/"rev1016_denominator_sims.csv", index=False, encoding="utf-8-sig")
# SIMEX 示意：假設既有分母誤差 σ0，用 λ=(σ_add/σ0)² 對中位數做二次外推到 λ=−1
SX = {}
for s0 in [0.10, 0.20]:
    lam = np.r_[0, [(s/s0)**2 for s in [0.05, 0.10, 0.20, 0.30]]]
    for c in ["b_買賣率", "b_釋出率", "Spearman(u,老屋買賣率)"]:
        y = np.r_[base[c], [SIM[f"σ={s:.2f}｜鄉鎮共同誤差"][c]["median"] for s in [0.05, 0.10, 0.20, 0.30]]]
        sel = lam <= 9.01
        SX[f"σ0={s0:.2f}｜{c}"] = float(np.polyval(np.polyfit(lam[sel], y[sel], 2), -1))
OUTJ["14_SIMEX外推(示意)"] = SX; print(SX)
# 既有的抽樣誤差（模擬檔約為 1/10 抽樣）：S_ex/10、老屋存量/10 的 Poisson CV
OUTJ["14_既有抽樣誤差參考"] = {"主樣本 既有建物模擬棟數 中位數": float((M.S_ex/10).median()), "對應 Poisson CV 中位數": float(np.median(1/np.sqrt(M.S_ex/10))), "最大": float(np.max(1/np.sqrt(M.S_ex/10))),
                         "老屋樣本 模擬棟數 中位數": float((B.lv_stock_old/10).median()), "對應 Poisson CV 中位數(老屋)": float(np.median(1/np.sqrt(B.lv_stock_old/10))), "最大(老屋)": float(np.max(1/np.sqrt(B.lv_stock_old/10)))}
json.dump(OUTJ, open(OUT/"rev1016_proxies.json", "w"), ensure_ascii=False, indent=1, default=float)
print("done")
