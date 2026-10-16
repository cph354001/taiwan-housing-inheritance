"""rev1016 核對 08：組成模型（LightGBM，留一年訓練）逐棟機率的校準，以及分組偏差會不會變成「地方效果」。
資料：data/bcd_with_pcomp.parquet（12_composition_vs_place.py 輸出；p_comp 為以另外兩年訓練的樣本外預測）。
樣本：與 GLMM 相同，限該年度 township_effects_{y}.csv 中的鄉鎮（≥50 棟）。
 (a) 校準後 = p_comp × k_y，k_y = ΣO/ΣE（12 的第 81 行，乘法縮放）；(b) 原始 = p_comp。
 1. calibration-in-the-large、logistic recalibration logit(y) ~ a + b·logit(p)、十分位校準表。
 2. 依屋齡帶、建材、總樓層組、用途、屋齡帶×樓層的 O/E（Σ實際/Σ預測；同時列 Σ預測/Σ實際）。
 3. 分組再校準（年內 O/E 因子）後重估 E_j、u，與原 u 比較；112 年再跑 21 的主要迴歸（272 個本島鄉鎮、標準化、縣市群集 SE）。
輸出 out/rev1016_calib.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.special import logit, expit
from scipy.stats import spearmanr, kendalltau
from common import *
import glmm
np.random.seed(20261016)
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet")
for c in ["materials_group_cd","floor_group_cd","purpose_group_cd"]: d[c] = d[c].astype(str)
AB = [-1, 9, 19, 29, 39, 49, 59, 200]; ABL = ["0-9","10-19","20-29","30-39","40-49","50-59","60+"]
d["ab"] = pd.cut(d.age, AB, labels=ABL).astype(str)
d["ab_fl"] = d.ab + "|" + d.floor_group_cd
d["cell4"] = d.ab + "|" + d.floor_group_cd + "|" + d.materials_group_cd + "|" + d.purpose_group_cd
S = pd.read_csv(OUT/"township_supply_sales.csv").set_index("house_cd"); S = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)]
def reg(u):                                  # 與 21_township_supply_vs_sales.py 的 reg(main, "u") 相同
    M = S[["county","m","bx"]].join(u.rename("u")).dropna()
    X = pd.DataFrame({"log釋出率": np.log(M.m), "log買賣率": np.log(M.bx)}); Z = (X - X.mean())/X.std(); yy = (M.u - M.u.mean())/M.u.std()
    f = sm.OLS(yy, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": M.county})
    return {"n": len(M), "買賣率": round(f.params["log買賣率"], 4), "se_買賣率": round(f.bse["log買賣率"], 4), "釋出率": round(f.params["log釋出率"], 4), "se_釋出率": round(f.bse["log釋出率"], 4), "R2": round(f.rsquared, 4)}
def calib(y, p):
    lp = logit(np.clip(p, 1e-6, 1-1e-6)); f = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    q = pd.qcut(p, 10, labels=False, duplicates="drop"); dec = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(n=("y","size"), 預測均值=("p","mean"), 實際比例=("y","mean"))
    dec["實際/預測"] = dec["實際比例"]/dec["預測均值"]
    return {"平均預測": float(p.mean()), "實際比例": float(y.mean()), "Σ預測/Σ實際": float(p.sum()/y.sum()),
            "recal_截距a": float(f.params[0]), "recal_斜率b": float(f.params[1]), "se_a": float(f.bse[0]), "se_b": float(f.bse[1]),
            "十分位": dec.round(4).reset_index().to_dict("records"), "十分位最大|實際-預測|_pp": float((dec["實際比例"] - dec["預測均值"]).abs().max()*100)}
R = {"說明": "O/E=Σ實際/Σ預測（>1 表示預測偏低）；E/O=Σ預測/Σ實際（<1 表示被低估）。校準後 p = p_comp × k_y（k_y 由 ≥50 棟鄉鎮的 ΣO/ΣE 而來）。",
     "年度": {}}
U = {}
for y in [110, 111, 112]:
    T = pd.read_csv(OUT/f"township_effects_{y}.csv"); dy = d[(d.year == y) & d.house_cd.isin(T.house_cd)].copy()
    assert len(dy) == T.n.sum()
    k = dy.D.sum()/dy.p_comp.sum(); dy["p_cal"] = np.clip(dy.p_comp*k, 0, 1 - 1e-6)
    r = {"n_bld": len(dy), "n_town": len(T), "k_y(校準常數)": float(k), "原始": calib(dy.D.values, dy.p_comp.values), "校準後": calib(dy.D.values, dy.p_cal.values)}
    # 2. 分組 O/E（以校準後 p；原始只差常數 k）
    grp = {}
    for g, col in [("屋齡帶", "ab"), ("建材", "materials_group_cd"), ("總樓層組", "floor_group_cd"), ("用途", "purpose_group_cd"), ("屋齡帶×樓層", "ab_fl")]:
        x = dy.groupby(col).agg(n=("D","size"), O=("D","sum"), E=("p_cal","sum"), E_raw=("p_comp","sum"))
        x["O/E(校準後)"] = x.O/x.E; x["E/O(校準後)"] = x.E/x.O; x["O/E(原始)"] = x.O/x.E_raw
        x["z"] = (x.O - x.E)/np.sqrt((dy.p_cal*(1 - dy.p_cal)).groupby(dy[col]).sum())
        if col == "ab_fl": x = x[x.n >= 2000]
        grp[g] = x.round(4).reset_index().to_dict("records")
    r["分組"] = grp
    # 3. 分組再校準：E'_j = Σ p_i × (O_g/E_g)，g = 屋齡帶×樓層；更細版 g = 屋齡帶×樓層×建材×用途（<500 棟的格退回屋齡帶×樓層）
    f_abfl = dy.groupby("ab_fl").D.transform("sum")/dy.groupby("ab_fl").p_comp.transform("sum")
    c4n = dy.groupby("cell4").D.transform("size"); f_c4 = dy.groupby("cell4").D.transform("sum")/dy.groupby("cell4").p_comp.transform("sum")
    f_c4 = np.where(c4n >= 500, f_c4, f_abfl)
    f_ab = dy.groupby("ab").D.transform("sum")/dy.groupby("ab").p_comp.transform("sum")
    dy["p_ab"], dy["p_abfl"], dy["p_c4"] = dy.p_comp*f_ab, dy.p_comp*f_abfl, dy.p_comp*f_c4
    tw = dy.groupby("house_cd").agg(n=("D","size"), O=("D","sum"), E=("p_cal","sum"), E_ab=("p_ab","sum"), E_abfl=("p_abfl","sum"), E_c4=("p_c4","sum"),
                                     old40=("age", lambda a: (a >= 40).mean()), old50=("age", lambda a: (a >= 50).mean()), age_mean=("age","mean"),
                                     sh_matA=("materials_group_cd", lambda a: (a == "A").mean()), sh_fl1=("floor_group_cd", lambda a: (a == "1").mean())).reindex(T.house_cd)
    assert np.allclose(tw.E.values*(T.O.sum()/tw.E.sum()), T.E.values, rtol=1e-6)
    tw["u"] = T.set_index("house_cd").u
    f0 = glmm.fit(tw.O, tw.n, logit(T.E.values/T.n.values)); r["實作檢查_重估原u_最大差"] = float(np.abs(f0["u"] - tw.u.values).max())
    comp = {}
    for c in ["old40","old50","age_mean","sh_matA","sh_fl1"]: comp[c] = float(spearmanr(tw[c], tw.u)[0])
    comp["說明"] = "鄉鎮組成與原 u 的 Spearman"
    rows = {}
    for lab, col in [("屋齡帶", "E_ab"), ("屋齡帶×樓層", "E_abfl"), ("屋齡帶×樓層×建材×用途", "E_c4")]:
        E2 = tw[col]*(tw.O.sum()/tw[col].sum())
        bias = np.log(E2/(tw.E*tw.O.sum()/tw.E.sum()))   # 鄉鎮的期望值被分組偏差改變多少（log）
        f = glmm.fit(tw.O, tw.n, logit(E2/tw.n)); un = pd.Series(f["u"], index=tw.index)
        rows[lab] = {"tau_新": float(f["tau"]), "tau_原": float(f0["tau"]), "alpha_新": float(f["beta"][0]), "Pearson(u原,u新)": float(np.corrcoef(tw.u, un)[0,1]),
                     "Spearman(u原,u新)": float(spearmanr(tw.u, un)[0]), "Kendall(u原,u新)": float(kendalltau(tw.u, un)[0]),
                     "mean|Δu|": float((un - tw.u).abs().mean()), "max|Δu|": float((un - tw.u).abs().max()), "sd(log E'/E)": float(bias.std()),
                     "Spearman(log E'/E, u原)": float(spearmanr(bias, tw.u)[0]), "Spearman(old50, u新)": float(spearmanr(tw.old50, un)[0]), "Spearman(old40, u新)": float(spearmanr(tw.old40, un)[0])}
        U[(y, lab)] = un
        rows[lab]["主要迴歸(新u)"] = reg(un)
    r["鄉鎮組成與u"] = comp; r["分組再校準"] = rows; r["主要迴歸(原u)"] = reg(tw.u)
    R["年度"][y] = r
    print(y, "k", round(k, 4), "recal raw", round(r["原始"]["recal_截距a"], 3), round(r["原始"]["recal_斜率b"], 3), "cal", round(r["校準後"]["recal_截距a"], 3), round(r["校準後"]["recal_斜率b"], 3))
    print(pd.DataFrame(grp["屋齡帶"])[["ab","n","O/E(校準後)","z"]].to_string())
    print(json.dumps(comp, ensure_ascii=False)); print(json.dumps(rows, ensure_ascii=False, default=float, indent=0)); print("orig reg", r["主要迴歸(原u)"])
R["說明_主要迴歸"] = "u 對 log 買賣率、log 屋主死亡釋出率（皆 112 年資料）；272 個本島鄉鎮；標準化係數，縣市群集 SE；與 21_township_supply_vs_sales.py 相同。報告值為 112 年原 u：買賣率 −0.48、釋出率 +0.38。110、111 年列出僅供參考（迴歸變數仍是 112 年的發生率）。"
# 4. 樣本外預測表（12_composition_vs_place.py 第 117–125 行）是否用到 112 年總數：引用 rev1014_oos.json
o = json.load(open(OUT/"rev1014_oos.json"))
R["樣本外表_再確認"] = {"說明": "12 第 119 行的校準常數只用訓練年（110–111）ΣO/ΣE；第 123 行的 cal() 未被呼叫；預測總數/實際總數 < 1 證明未對齊 112 年總數",
                      "k_tr": o["k_tr(訓練年校準常數)"], "各列預測總數÷實際總數": {x["模型"]: round(x["預測總數÷實際總數"], 4) for x in o["table"]},
                      "RMSE_pp": {x["模型"]: round(x["RMSE_pp"], 3) for x in o["table"]}}
json.dump(R, open(OUT/"rev1016_calib.json", "w"), ensure_ascii=False, indent=1, default=float)
print("done")
