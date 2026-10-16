"""問題二的模擬資料內部檢驗：贈與（生前的家內移轉）與買賣的地方效果。
對同一批建物（買賣／贈與／繼承、屋齡已知、110–112 年），用與 12_composition_vs_place.py 相同的流程，
把結果變數換成「是贈與」與「是買賣」，得到各鄉鎮的地方效果 u_C、u_B（扣除屋齡、建材、樓層、用途後）。
 - 贈與與屋主死亡無關。若 u_C 與 u_D（繼承）在鄉鎮之間高度相關，地方差異量到的是「房子在家族內傳、不進市場」的傾向，不是死亡。
 - u_B 可直接對照真實買賣率（實價登錄），是登記原因欄位的另一個外部驗證。"""
import pandas as pd, numpy as np, json, lightgbm as lgb
from scipy.special import logit
from scipy.stats import spearmanr
import statsmodels.api as sm
from common import *
import glmm
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet"); town = pd.read_csv(DATA/"town.csv")
for c in ["materials_group_cd","floor_group_cd","purpose_group_cd"]: d[c] = d[c].astype(str).astype("category")
FEATS = ["age","materials_group_cd","floor_group_cd","purpose_group_cd"]
def place_effect(d, target):
    d = d.copy(); d["Y"] = (d.reason == target).astype(int); d["p"] = np.nan
    for y in [110,111,112]:
        te = d.year == y
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, verbose=-1, random_state=1)
        m.fit(d.loc[~te, FEATS], d.loc[~te, "Y"]); d.loc[te, "p"] = m.predict_proba(d.loc[te, FEATS])[:, 1]
    out = {}
    for y in [110,111,112]:
        t = d[d.year == y].groupby("house_cd").agg(n=("Y","size"), O=("Y","sum"), E=("p","sum")).reset_index().merge(town, on="house_cd"); t = t[t.n >= 50].copy()
        t["E"] *= t.O.sum()/t.E.sum(); f = glmm.fit(t.O, t.n, logit(t.E/t.n)); t["u"], t["u_sd"], t["tau"] = f["u"], f["u_sd"], f["tau"]
        out[y] = t.set_index("house_cd")
    return out
UC, UB = place_effect(d, "C"), place_effect(d, "B")
UD = {y: pd.read_csv(OUT/f"township_effects_{y}.csv").set_index("house_cd") for y in [110,111,112]}
rows = []
for y in [110,111,112]:
    m = pd.DataFrame({"u_D": UD[y].u, "u_C": UC[y].u, "u_B": UB[y].u, "county": UD[y].county}).dropna()
    dm = lambda x: x - x.groupby(m.county).transform("mean")
    rows.append({"年度": y, "鄉鎮數": len(m), "tau_C": UC[y].tau.iloc[0], "tau_B": UB[y].tau.iloc[0], "tau_D": pd.read_csv(OUT/"composition_vs_place_summary.csv", index_col=0).loc[y, "tau_place"],
                 "corr(u_D,u_C)": m.u_D.corr(m.u_C), "Spearman(u_D,u_C)": spearmanr(m.u_D, m.u_C)[0], "縣市內corr(u_D,u_C)": dm(m.u_D).corr(dm(m.u_C)),
                 "corr(u_D,u_B)": m.u_D.corr(m.u_B), "corr(u_C,u_B)": m.u_C.corr(m.u_B)})
R = pd.DataFrame(rows); R.to_csv(OUT/"gift_vs_inherit_place_effect.csv", index=False, encoding="utf-8-sig"); print(R.round(3).to_string())
# 對照真實流量（112 年；272 個本島鄉鎮，與 §3.4 相同樣本）
S = pd.read_csv(OUT/"township_supply_sales.csv").set_index("house_cd"); S = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)]
M = S[["county","lp","bx","m","u","odds_stock"]].join(UC[112].u.rename("u_C")).join(UB[112].u.rename("u_B")).dropna()
M["lb"], M["lm"] = np.log(M.bx), np.log(M.m)
def reg(yv):
    X = pd.DataFrame({"log買賣率": M.lb, "log釋出率": M.lm}); Z = (X - X.mean())/X.std(); yy = (yv - yv.mean())/yv.std()
    f = sm.OLS(yy, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": M.county})
    return {"標準化係數_買賣率": f.params["log買賣率"], "se_買賣率": f.bse["log買賣率"], "標準化係數_釋出率": f.params["log釋出率"], "se_釋出率": f.bse["log釋出率"], "R2": f.rsquared,
            "Spearman_買賣率": spearmanr(M.lb, yv)[0], "Spearman_釋出率": spearmanr(M.lm, yv)[0]}
RR = pd.DataFrame({"繼承 u_D": reg(M.u), "贈與 u_C": reg(M.u_C), "買賣 u_B": reg(M.u_B)}).T
RR.to_csv(OUT/"gift_sales_inherit_vs_real_rates.csv", encoding="utf-8-sig"); print(RR.round(3).to_string())
# 繼承的地方效果裡，與贈與共同的部分 vs 獨有的部分，各自跟死亡率、買賣率的關係
f_cd = sm.OLS(M.u, sm.add_constant(M.u_C)).fit(); M["u_D_common"], M["u_D_resid"] = f_cd.fittedvalues, f_cd.resid
extra = {"鄉鎮數": int(len(M)), "corr(u_D,u_C)_272": float(M.u.corr(M.u_C)), "u_D 對 u_C 的 R2": float(f_cd.rsquared),
         "u_D 中與贈與共同的部分：Spearman(對買賣率, 對釋出率)": [float(spearmanr(M.u_D_common, M.lb)[0]), float(spearmanr(M.u_D_common, M.lm)[0])],
         "u_D 中贈與解釋不了的部分：Spearman(對買賣率, 對釋出率)": [float(spearmanr(M.u_D_resid, M.lb)[0]), float(spearmanr(M.u_D_resid, M.lm)[0])],
         "u_B 對真實買賣率 Spearman": float(spearmanr(M.u_B, M.bx)[0]), "u_B 對真實買賣率 控制log人口後 Pearson": float(np.corrcoef(sm.OLS(M.u_B, sm.add_constant(M.lp)).fit().resid, sm.OLS(M.lb, sm.add_constant(M.lp)).fit().resid)[0,1]),
         "u_C 對真實買賣率 Spearman": float(spearmanr(M.u_C, M.bx)[0]), "u_C 對釋出率 Spearman": float(spearmanr(M.u_C, M.m)[0])}
json.dump(extra, open(OUT/"gift_vs_inherit_extra.json","w"), ensure_ascii=False, indent=1); print(json.dumps(extra, ensure_ascii=False, indent=1))
pd.DataFrame({"u_D": UD[112].u, "u_C": UC[112].u, "u_B": UB[112].u, "name": UD[112].name}).to_csv(OUT/"place_effects_BCD_112.csv", encoding="utf-8-sig")
