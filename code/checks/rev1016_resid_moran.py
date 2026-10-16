"""rev1016 核對 10：主要迴歸殘差的空間自相關。
主要迴歸（21_township_supply_vs_sales.py 的 reg(main, "u")；報告 §3.4）：272 個本島鄉鎮，
  z(u) ~ z(log 買賣率) + z(log 屋主死亡釋出率)，縣市群集 SE。另一條加入 u_C（贈與地方效果，25_gift_place_effect.py）。
空間權重：與 13b_place_correlates.py 相同（twtown2010.json 依縣市＋鄉鎮名併合、Queen 相鄰、列標準化），限 272 個鄉鎮重建。
  無鄰居的鄉鎮（孤島）：主設定照 13b 不處理（esda 對孤島的空間落後值為 0）；敏感度另把孤島連到最近的鄉鎮（對稱）。
 1. 殘差 Moran's I（999 次置換）。
 2. spreg ML_Error / ML_Lag；以縣市（19 群）與 k-means（25 區塊，TWD97 座標）做區塊重抽 1000 次。
 3. u 用 110、111、112 三個年度各自的估計（發生率皆為 112 年資料）；u_C 依 25 的流程重估三個年度（112 年與存檔比對）。
輸出 out/rev1016_resid_moran.json"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, geopandas as gpd, lightgbm as lgb
import statsmodels.api as sm
from scipy.special import logit
from shapely.geometry import Polygon, MultiPolygon
from libpysal.weights import Queen, W as Wobj
from esda.moran import Moran
from sklearn.cluster import KMeans
import spreg
from common import *
import glmm
SEED = 20261016; rng = np.random.default_rng(SEED)
# ---- 樣本（與 21 相同）
S = pd.read_csv(OUT/"township_supply_sales.csv"); S = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)].copy()
S["lb"], S["lm"] = np.log(S.bx), np.log(S.m)
for y in [110, 111]: S = S.merge(pd.read_csv(OUT/f"township_effects_{y}.csv")[["house_cd","u"]].rename(columns={"u": f"u_{y}"}), on="house_cd", how="left")
S["u_112"] = S.u
# ---- u_C（25_gift_place_effect.py 的 place_effect，原樣）
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
        t["E"] *= t.O.sum()/t.E.sum(); f = glmm.fit(t.O, t.n, logit(t.E/t.n)); t["u"] = f["u"]
        out[y] = t.set_index("house_cd").u
    return out
UC = place_effect(d, "C")
for y in [110,111,112]: S[f"uC_{y}"] = S.house_cd.map(UC[y])
uc_saved = pd.read_csv(OUT/"place_effects_BCD_112.csv", index_col=0).u_C
chk_uC = float(np.abs(S.uC_112 - S.house_cd.map(uc_saved)).max())
# ---- 幾何（13b 第 46–60 行）
def _geom(gj):
    polys = [gj["coordinates"]] if gj["type"] == "Polygon" else gj["coordinates"]; out = []
    for rings in polys:
        rings = [r for r in rings if len(r) >= 4]
        if rings: out.append(Polygon(rings[0], rings[1:]).buffer(0))
    return MultiPolygon([q for p in out for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]) if not q.is_empty])
_j = json.load(open(ROOT/"external/shp/twtown2010.json"))
g = gpd.GeoDataFrame([f["properties"] for f in _j["features"]], geometry=[_geom(f["geometry"]) for f in _j["features"]], crs=4326)
fix = {"臺北縣":"新北市","台北縣":"新北市","桃園縣":"桃園市","臺中縣":"臺中市","台中縣":"臺中市","臺南縣":"臺南市","台南縣":"臺南市","高雄縣":"高雄市"}
g["county"] = g.county.str.replace("台","臺").replace(fix); g["stem"] = g.town.str.replace("台","臺").str[:-1]
g.loc[g.county.isin(["新竹市","嘉義市"]), "stem"] = "（全市"
S["stem"] = S.apply(lambda r: r["name"].replace(r["county"], "", 1)[:-1], axis=1)
g = g.dissolve(by=["county","stem"], as_index=False).merge(S, on=["county","stem"], how="inner").reset_index(drop=True)
assert len(g) == len(S), (len(g), len(S))
Wq = Queen.from_dataframe(g, use_index=False, silence_warnings=True); islands = list(Wq.islands)
cent = np.c_[g.to_crs(3826).geometry.centroid.x, g.to_crs(3826).geometry.centroid.y]
nb = {i: list(Wq.neighbors[i]) for i in range(len(g))}
for i in islands:
    dd = np.hypot(*(cent - cent[i]).T); dd[i] = np.inf; j = int(np.argmin(dd)); nb[i] = nb[i] + [j]; nb[j] = nb[j] + [i]
Wf = Wobj(nb, silence_warnings=True)
Wq.transform = "r"; Wf.transform = "r"
R = {"n": len(g), "孤島(Queen 無鄰居)": [g.loc[i, "name"] for i in islands], "u_C 112 與存檔最大差": chk_uC, "seed": SEED}
COUNTY = g.county.values
KM = KMeans(n_clusters=25, n_init=20, random_state=SEED).fit_predict(cent)
LAB = {"lb": "log買賣率", "lm": "log釋出率"}
def zs(a): a = np.asarray(a, float); return (a - a.mean())/a.std(ddof=1)
def ols(df_idx, ycol, xcols):
    y = zs(g.loc[df_idx, ycol]); X = np.c_[np.ones(len(df_idx)), *[zs(g.loc[df_idx, c]) for c in xcols]]
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]
def block_boot(ycol, xcols, groups, B=1000):
    ug = np.unique(groups); idx = {k: np.flatnonzero(groups == k) for k in ug}; out = []
    for _ in range(B):
        pick = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]); out.append(ols(pick, ycol, xcols))
    out = np.array(out); xcols = [LAB.get(c, c) for c in xcols]; return {c: [round(float(np.percentile(out[:, i], 2.5)), 3), round(float(np.percentile(out[:, i], 97.5)), 3)] for i, c in enumerate(xcols)} | {"符號維持比例": {c: float((np.sign(out[:, i]) == np.sign(np.median(out[:, i]))).mean()) for i, c in enumerate(xcols)}}
def analyse(ycol, xcols):
    lab = lambda c: LAB.get(c, c)
    y = zs(g[ycol]); X = np.c_[[zs(g[c]) for c in xcols]].T
    f = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": COUNTY}); res = y - sm.add_constant(X) @ f.params
    np.random.seed(SEED); mq = Moran(res, Wq, permutations=999); np.random.seed(SEED); mf = Moran(res, Wf, permutations=999)
    np.random.seed(SEED); my = Moran(y, Wq, permutations=999)
    r = {"OLS": {lab(c): [round(float(f.params[i+1]), 3), round(float(f.bse[i+1]), 3)] for i, c in enumerate(xcols)} | {"R2": round(float(f.rsquared), 3)},
         "Moran_y(Queen)": {"I": round(my.I, 3), "p": float(my.p_sim)},
         "Moran_殘差(Queen, 同13b)": {"I": round(mq.I, 3), "p_置換": float(mq.p_sim), "z": round(mq.z_sim, 2)},
         "Moran_殘差(Queen+孤島連最近)": {"I": round(mf.I, 3), "p_置換": float(mf.p_sim), "z": round(mf.z_sim, 2)}}
    names = [lab(c) for c in xcols]
    sem = spreg.ML_Error(y[:, None], X, w=Wf, name_x=names, name_y=ycol); lag = spreg.ML_Lag(y[:, None], X, w=Wf, name_x=names, name_y=ycol)
    r["SEM(ML, W=Queen+孤島連最近)"] = {nm: [round(float(sem.betas[i+1, 0]), 3), round(float(sem.std_err[i+1]), 3), [round(float(sem.betas[i+1, 0] - 1.96*sem.std_err[i+1]), 3), round(float(sem.betas[i+1, 0] + 1.96*sem.std_err[i+1]), 3)]] for i, nm in enumerate(names)} | {"lambda": [round(float(sem.betas[-1, 0]), 3), round(float(sem.std_err[-1]), 3)], "AIC": round(float(sem.aic), 1)}
    r["SLM(ML lag)"] = {nm: [round(float(lag.betas[i+1, 0]), 3), round(float(lag.std_err[i+1]), 3)] for i, nm in enumerate(names)} | {"rho": [round(float(lag.betas[-1, 0]), 3), round(float(lag.std_err[-1]), 3)], "AIC": round(float(lag.aic), 1)}
    rho = float(lag.betas[-1, 0]); r["SLM(ML lag)"]["總效果(β/(1-ρ))"] = {nm: round(float(lag.betas[i+1, 0])/(1 - rho), 3) for i, nm in enumerate(names)}
    ols_aic = sm.OLS(y, sm.add_constant(X)).fit().aic; r["OLS_AIC"] = round(float(ols_aic), 1)
    np.random.seed(SEED); r["SEM殘差Moran(Queen+孤島)"] = round(Moran(sem.e_filtered[:, 0], Wf, permutations=999).I, 3)
    r["區塊重抽_縣市19群_95%"] = block_boot(ycol, xcols, COUNTY)
    r["區塊重抽_kmeans25_95%"] = block_boot(ycol, xcols, KM)
    return r
for y in [110, 111, 112]:
    R[f"u_{y} ~ 買賣率 + 釋出率"] = analyse(f"u_{y}", ["lb", "lm"])
    R[f"u_{y} ~ 買賣率 + 釋出率 + u_C"] = analyse(f"u_{y}", ["lb", "lm", f"uC_{y}"])
    print(y, json.dumps(R[f"u_{y} ~ 買賣率 + 釋出率"], ensure_ascii=False)); print(y, json.dumps(R[f"u_{y} ~ 買賣率 + 釋出率 + u_C"], ensure_ascii=False))
R["說明"] = "係數皆為標準化（y、X 各自 z 分數）；OLS 為 [係數, 縣市群集SE]；SEM/SLM 為 [係數, ML SE]；區塊重抽為百分位 95% 區間（每次重抽重新標準化）。發生率（買賣率、釋出率）三個年度都用同一份 112 年資料。"
json.dump(R, open(OUT/"rev1016_resid_moran.json", "w"), ensure_ascii=False, indent=1, default=float)
print(R["孤島(Queen 無鄰居)"], chk_uC)
