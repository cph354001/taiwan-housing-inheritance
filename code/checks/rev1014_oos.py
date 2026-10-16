"""rev1014 核對 3：重跑 12_composition_vs_place.py 第 65–81 行（樣本外預測表），逐項確認有沒有用到 112 年結果。輸出 out/rev1014_oos.json
與原程式唯一的差異：另加一個「刻意用 112 年總數校準」的對照列，量化若有洩漏會差多少。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss
from common import *
import glmm
b = pd.read_parquet(DATA/"bld.parquet"); town = pd.read_csv(DATA/"town.csv")
d = b[b.year.isin([110,111,112]) & b.reason.isin(list("BCD")) & b.age.notna()].merge(town, left_on="countyntownship_cd", right_on="house_cd")
d = d[d["pop"].notna()].copy(); d["D"] = (d.reason == "D").astype(int)
for c in ["materials_group_cd","floor_group_cd","purpose_group_cd"]: d[c] = d[c].fillna("NA").astype("category")
FEATS = ["age","materials_group_cd","floor_group_cd","purpose_group_cd"]
def gbm(train):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, random_state=1)
    return m.fit(train[FEATS], train.D)
tr, te = d[d.year < 112].copy(), d[d.year == 112].copy()
m_tr = gbm(tr); tr["p_tr"] = m_tr.predict_proba(tr[FEATS])[:, 1]; te["p_comp"] = m_tr.predict_proba(te[FEATS])[:, 1]
tt = tr.groupby("house_cd").agg(n=("D","size"), O=("D","sum"), E=("p_tr","sum")).reset_index().merge(town, on="house_cd"); k_tr = tt.O.sum()/tt.E.sum(); tt["E"] *= k_tr
ft = glmm.fit(tt.O, tt.n, logit(tt.E/tt.n)); tt["u_tr"] = ft["u"]
ct = tt.groupby("county").agg(O=("O","sum"), E=("E","sum")); ct["c_tr"] = np.log(ct.O/ct.E)
te = te.merge(tt[["house_cd","u_tr"]], on="house_cd", how="left").merge(ct[["c_tr"]], left_on="county", right_index=True, how="left").fillna({"u_tr":0,"c_tr":0})
preds = {"只用全國平均": np.full(len(te), tr.D.mean()), "只知道鄉鎮": te.house_cd.map(tr.groupby("house_cd").D.mean()).fillna(tr.D.mean()).values,
         "房屋條件": te.p_comp.values, "房屋條件＋縣市": expit(logit(te.p_comp) + te.c_tr).values, "房屋條件＋鄉鎮": expit(logit(te.p_comp) + ft["beta"][0] + te.u_tr).values}
# 對照（刻意洩漏）：把「房屋條件」各列在 logit 尺度平移，使 112 年預測總數等於 112 年實際總數
from scipy.optimize import brentq
for k in ["房屋條件","房屋條件＋縣市","房屋條件＋鄉鎮"]:
    p0 = np.clip(preds[k], 1e-5, 1-1e-5); s = brentq(lambda c: expit(logit(p0)+c).sum() - te.D.sum(), -3, 3); preds[k+"（對照：以112年總數校準）"] = expit(logit(p0)+s)
pm = []
for k, p in preds.items():
    p = np.clip(p, 1e-5, 1-1e-5); g = te.assign(p=p).groupby("house_cd").agg(o=("D","mean"), p=("p","mean"), n=("D","size")); g = g[g.n >= 300]
    pm.append({"模型": k, "logloss": log_loss(te.D, p), "AUC": roc_auc_score(te.D, p) if np.std(p) > 0 else 0.5, "RMSE_pp": np.sqrt(((g.o-g.p)**2).mean())*100, "評估鄉鎮數": len(g), "預測總數÷實際總數": float(p.sum()/te.D.sum())})
PM = pd.DataFrame(pm); PM["logloss相對改善%"] = (1 - PM.logloss/PM.logloss.iloc[0])*100
orig = pd.read_csv(OUT/"oos_prediction_112.csv")
R = {"table": PM.to_dict("records"), "orig_RMSE": orig.set_index("模型")["鄉鎮層級RMSE_pp"].to_dict(), "alpha_tr": float(ft["beta"][0]), "tau_tr": float(ft["tau"]), "k_tr(訓練年校準常數)": float(k_tr),
     "train_D_mean": float(tr.D.mean()), "test_D_mean": float(te.D.mean()), "n_train": len(tr), "n_test": len(te)}
json.dump(R, open(OUT/"rev1014_oos.json","w"), ensure_ascii=False, indent=1, default=float); print(PM.round(4).to_string()); print({k: v for k, v in R.items() if k != "table"})
