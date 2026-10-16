"""主題一：鄉鎮的繼承占比，多少是房屋組成、多少是地方本身？
步驟
 1. 組成模型：只用建物條件（精確屋齡、建材、總樓層組、用途）預測每棟「繼承」的機率。以另外兩個年度訓練、預測本年度（樣本外）。
 2. 間接標準化：E_j = 該鄉鎮各棟預測機率之和；O_j = 實際繼承棟數。
 3. 地方效果：鄉鎮隨機截距二項模型（offset = logit(E_j/n_j)），取收縮後的 u_j。
 4. 變異分解、跨年度信度、樣本外預測力。
分母：登記原因為買賣／贈與／繼承、屋齡已知的建物，每棟一筆。只用 110–112 年（屋齡到每一歲才有資訊，見 11_generator_forensics）。"""
import pandas as pd, numpy as np, json, lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from common import *
import glmm
b = pd.read_parquet(DATA/"bld.parquet"); town = pd.read_csv(DATA/"town.csv")
d = b[b.year.isin([110,111,112]) & b.reason.isin(list("BCD")) & b.age.notna()].merge(town, left_on="countyntownship_cd", right_on="house_cd")
d = d[d["pop"].notna()].copy(); d["D"] = (d.reason == "D").astype(int)
for c in ["materials_group_cd","floor_group_cd","purpose_group_cd"]: d[c] = d[c].fillna("NA").astype("category")
FEATS = ["age","materials_group_cd","floor_group_cd","purpose_group_cd"]
audit = {"n_by_year": d.groupby("year").size().to_dict(), "townships": int(d.house_cd.nunique())}
def gbm(train, test, feats=FEATS):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, random_state=1)
    m.fit(train[feats], train.D); return m.predict_proba(test[feats])[:, 1], m
d["p_comp"] = np.nan
for y in [110,111,112]:
    te = d.year == y; d.loc[te, "p_comp"], _ = gbm(d[~te], d[te])
# 對照：直接用同年度全國「建材×樓層×屋齡(歲)×用途」格子平均（先前研究 §4.8 的寫法，但屋齡改為每一歲）
d["p_cell"] = d.groupby(["year"]+FEATS, observed=True).D.transform("mean")
def township_table(g):
    t = g.groupby("house_cd").agg(n=("D","size"), O=("D","sum"), E=("p_comp","sum"), E_cell=("p_cell","sum"), age_mean=("age","mean")).reset_index().merge(town, on="house_cd")
    t = t[t.n >= 50].copy()
    t["E"] *= t.O.sum()/t.E.sum()                      # 校準到本年度全國水準（把年度間的整體位移放進常數）
    t["raw"] = t.O/t.n; t["exp"] = t.E/t.n; t["off"] = logit(t.exp)
    return t
T = {y: township_table(d[d.year == y]) for y in [110,111,112]}
res = {}; DEC = {}
for y, t in T.items():
    f0 = glmm.fit(t.O, t.n, np.zeros(len(t)))                                   # 沒有任何調整
    f1 = glmm.fit(t.O, t.n, t.off)                                              # 扣除房屋組成
    Xc = pd.get_dummies(t.county, drop_first=False).astype(float).values
    f2 = glmm.fit(t.O, t.n, t.off, Xc)                                          # 再扣除縣市
    t["u0"], t["u"], t["u_sd"], t["u_within"], t["county_fx"] = f0["u"], f1["u"], f1["u_sd"], f2["u"], f2["fixed"]
    t["alpha"] = f1["beta"][0]
    res[y] = {"tau_raw": f0["tau"], "tau_place": f1["tau"], "tau_place_within_county": f2["tau"], "alpha": f1["beta"][0], "n_town": len(t), "n_bld": int(t.n.sum())}
    # 變異分解（logit 尺度，鄉鎮等權）：η_j = off_j（房屋組成）+ u_j（地方，收縮估計）
    def decomp(x, w=None):
        w = np.ones(len(x)) if w is None else np.asarray(w, float); cv = np.cov(np.c_[x.off, x.u].T, aweights=w, bias=True)
        tt_ = cv[0,0] + cv[1,1] + 2*cv[0,1]; return {"只來自房屋組成": cv[0,0]/tt_, "只來自地方": cv[1,1]/tt_, "兩者重疊": 2*cv[0,1]/tt_, "組成與地方的相關": cv[0,1]/np.sqrt(cv[0,0]*cv[1,1]), "模型總變異": tt_, "觀察到的logit變異(含抽樣雜訊)": np.var(logit((x.O + .5)/(x.n + 1)))}
    DEC[(y, "鄉鎮等權")] = decomp(t); DEC[(y, "建物數加權")] = decomp(t, t.n); DEC[(y, "鄉鎮等權、不含離島")] = decomp(t[~t.county.isin(ISLANDS)])
    v_comp, v_place, cov = np.var(t.off), np.var(t.u), np.cov(t.off, t.u, bias=True)[0,1]; tot = v_comp + v_place + 2*cov
    res[y].update({"var_total_logit": tot, "var_comp": v_comp, "var_place": v_place, "cov_comp_place": cov,
                   "share_comp": v_comp/tot, "share_place": v_place/tot, "share_cov": 2*cov/tot, "corr_comp_place": cov/np.sqrt(v_comp*v_place)})
    # 百分點尺度（對照先前研究的 62%）：鄉鎮實際占比被期望占比解釋的比例
    w = t.n; r2 = lambda yv, x, w=None: np.corrcoef(yv, x)[0,1]**2
    res[y].update({"R2_raw_on_exp_equalweight": r2(t.raw, t.exp), "R2_raw_on_cellexp": r2(t.raw, t.E_cell/t.n),
                   "sd_raw_pp": t.raw.std()*100, "sd_exp_pp": t.exp.std()*100})
    t.to_csv(OUT/f"township_effects_{y}.csv", index=False, encoding="utf-8-sig")
R = pd.DataFrame(res).T; R.to_csv(OUT/"composition_vs_place_summary.csv", encoding="utf-8-sig"); print(R.T.round(4).to_string())
DD = pd.DataFrame(DEC).T; DD.index.names = ["year","權重"]; DD.to_csv(OUT/"variance_decomposition.csv", encoding="utf-8-sig"); print(DD.round(3).to_string())
# 跨年度信度：三個年度是各自獨立產製的模擬檔
m = T[110][["house_cd","n","O","E"]].merge(T[111][["house_cd","n","O","E"]], on="house_cd", suffixes=("_110","_111")).merge(T[112][["house_cd","n","O","E","u"]].rename(columns={"n":"n_112","O":"O_112","E":"E_112","u":"u_112"}), on="house_cd")
for y in [110,111,112]: m[f"lr_{y}"] = np.log((m[f"O_{y}"]+0.5)/(m[f"E_{y}"]+0.5))
big = m[(m[["n_110","n_111","n_112"]].min(axis=1) >= 300)]
rel = big[["lr_110","lr_111","lr_112"]].corr(); print("log(O/E) 跨年相關（≥300棟的鄉鎮，n=%d）"%len(big)); print(rel.round(3))
rel.to_csv(OUT/"place_effect_cross_year_corr.csv")
audit["cross_year_corr_logOE_min300"] = rel.values[np.triu_indices(3,1)].round(3).tolist(); audit["n_town_min300"] = len(big)
# 樣本外預測：用 110+111 年估地方效果，預測 112 年每一棟
tr, te = d[d.year < 112].copy(), d[d.year == 112].copy()
_, m_tr = gbm(tr, tr); tr["p_tr"] = m_tr.predict_proba(tr[FEATS])[:, 1]; te["p_comp"] = m_tr.predict_proba(te[FEATS])[:, 1]   # 這一段的組成模型只用 110–111 年
tt = tr.groupby("house_cd").agg(n=("D","size"), O=("D","sum"), E=("p_tr","sum")).reset_index().merge(town, on="house_cd"); tt["E"] *= tt.O.sum()/tt.E.sum()
ft = glmm.fit(tt.O, tt.n, logit(tt.E/tt.n)); tt["u_tr"] = ft["u"]
ct = tt.groupby("county").agg(O=("O","sum"), E=("E","sum")); ct["c_tr"] = np.log(ct.O/ct.E)
te = te.merge(tt[["house_cd","u_tr"]], on="house_cd", how="left").merge(ct[["c_tr"]], left_on="county", right_index=True, how="left").fillna({"u_tr":0,"c_tr":0})
base = te.D.mean(); cal = lambda p: expit(logit(np.clip(p,1e-5,1-1e-5)) + (logit(base) - logit(np.clip(p,1e-5,1-1e-5)).mean())*0)   # 不另做校準
preds = {"只用全國平均": np.full(len(te), tr.D.mean()), "只知道鄉鎮（不看房子）": te.house_cd.map(tr.groupby("house_cd").D.mean()).fillna(tr.D.mean()).values,
         "房屋條件": te.p_comp.values, "房屋條件＋縣市": expit(logit(te.p_comp) + te.c_tr).values, "房屋條件＋鄉鎮": expit(logit(te.p_comp) + ft["beta"][0] + te.u_tr).values}
pm = []
for k, p in preds.items():
    p = np.clip(p, 1e-5, 1-1e-5); g = te.assign(p=p).groupby("house_cd").agg(o=("D","mean"), p=("p","mean"), n=("D","size")); g = g[g.n >= 300]
    pm.append({"模型": k, "logloss": log_loss(te.D, p), "AUC": roc_auc_score(te.D, p) if np.std(p) > 0 else 0.5, "Brier": brier_score_loss(te.D, p),
               "鄉鎮層級R2": 1 - ((g.o-g.p)**2).sum()/((g.o-g.o.mean())**2).sum(), "鄉鎮層級RMSE_pp": np.sqrt(((g.o-g.p)**2).mean())*100, "評估鄉鎮數(≥300棟)": len(g)})
PM = pd.DataFrame(pm); PM["logloss相對改善%"] = (1 - PM.logloss/PM.logloss.iloc[0])*100
PM.to_csv(OUT/"oos_prediction_112.csv", index=False, encoding="utf-8-sig"); print(PM.round(4).to_string())
audit["tau_train_110_111"] = float(ft["tau"])
d[["year","building_sn","house_cd","age","materials_group_cd","floor_group_cd","purpose_group_cd","reason","D","p_comp"]].to_parquet(DATA/"bcd_with_pcomp.parquet", index=False)
(OUT/"audit_composition.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=float))
