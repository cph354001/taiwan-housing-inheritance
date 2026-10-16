"""地方效果 u_j（扣除房屋組成後的鄉鎮繼承傾向，logit 尺度）與鄉鎮特徵的關係；空間自相關與地圖。
全部是鄉鎮層級的生態關聯。"""
import pandas as pd, numpy as np, json, geopandas as gpd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from common import *
t = pd.read_csv(OUT/"township_effects_112.csv").merge(pd.read_csv(DATA/"town_cov.csv").drop(columns=["name","county","region","pop"]), on="house_cd")
t["lp"] = np.log2(t["pop"]); t["logit_raw"] = logit((t.O + 0.5)/(t.n + 1))
dm = lambda s: s - s.groupby(t.county).transform("mean")
# 1) 人口每加倍：原始、組成、地方 三條斜率（logit 尺度；原始 ≈ 組成 + 地方）
rows = []
for lab, w in [("鄉鎮等權", np.ones(len(t))), ("建物數加權", t.n.values)]:
    for scope, f in [("全國", lambda s: s - np.average(s, weights=w)), ("縣市內", lambda s: s - s.groupby(t.county).transform(lambda v: np.average(v, weights=w[v.index])))]:
        x = f(t.lp); sl = lambda yv: float((w*x*f(yv)).sum()/(w*x*x).sum())
        rows.append({"權重": lab, "範圍": scope, "原始": sl(t.logit_raw), "房屋組成": sl(t.off), "地方效果": sl(t.u), "原始勝算倍數": np.exp(sl(t.logit_raw)), "地方勝算倍數": np.exp(sl(t.u))})
S = pd.DataFrame(rows); S.to_csv(OUT/"population_slopes.csv", index=False, encoding="utf-8-sig"); print(S.round(4).to_string())
# 2) 相關矩陣（縣市內）
vars_ = {"lp":"戶籍人口(log2)","owner_65":"屋主65歲以上比例","owner_univ":"屋主大學以上比例","res_65":"居民65歲以上比例","res_local":"居民在地出生比例","res_move":"居民平均遷徙次數","res_stay":"遷入現址年數中位數","off":"房屋組成(期望值logit)"}
cor = pd.DataFrame({"全國": {v: spearmanr(t[k], t.u, nan_policy="omit")[0] for k, v in vars_.items()}, "縣市內": {v: spearmanr(dm(t[k]), dm(t.u), nan_policy="omit")[0] for k, v in vars_.items()},
                    "縣市內_不含離島": {v: spearmanr(dm(t[k])[~t.county.isin(ISLANDS)], dm(t.u)[~t.county.isin(ISLANDS)], nan_policy="omit")[0] for k, v in vars_.items()}})
cor.to_csv(OUT/"place_effect_correlates.csv", encoding="utf-8-sig"); print(cor.round(3).to_string())
# 3) 多變數（縣市固定效果、標準化係數、鄉鎮重抽 bootstrap 區間）
X = ["lp","owner_65","res_65","res_local","res_move"]
z = t.dropna(subset=X + ["u"]).copy()
def std_coef(df, cols):
    D = pd.get_dummies(df.county).astype(float).values
    A = np.c_[D, ((df[cols] - df[cols].mean())/df[cols].std()).values]; yv = df.u.values
    bta = np.linalg.lstsq(A, yv, rcond=None)[0]; r = yv - A @ bta
    r0 = yv - D @ np.linalg.lstsq(D, yv, rcond=None)[0]
    return bta[-len(cols):], 1 - (r**2).sum()/(r0**2).sum()
rng = np.random.default_rng(1); out = {}
for name, cols in [("只放人口", ["lp"]), ("人口＋屋主高齡", ["lp","owner_65"]), ("全部", X)]:
    bta, r2 = std_coef(z, cols); bs = np.array([std_coef(z.iloc[rng.integers(0, len(z), len(z))], cols)[0] for _ in range(500)])
    out[name] = {**{vars_[c]: "%.3f [%.3f, %.3f]"%(bta[i], *np.percentile(bs[:, i], [2.5, 97.5])) for i, c in enumerate(cols)}, "縣市內R2": round(r2, 3), "n": len(z)}
M = pd.DataFrame(out); M.to_csv(OUT/"place_effect_models.csv", encoding="utf-8-sig"); print(M.to_string())
# 4) 以「同一棟參考老屋」表示地方效果：全國 40–49 年、1–5 層建物的平均繼承占比為基準
AL = t.alpha.iloc[0]                       # 截距：u=0 是一般鄉鎮，不是全國建物平均，換算時要加回
t["p_ref"] = expit(logit(0.25) + AL + t.u)
q = t.u.quantile([.05,.25,.5,.75,.95]); print("u 分位數", q.round(3).to_dict(), "→ 參考老屋(基準25%)", expit(logit(0.25)+AL+q).round(3).to_dict())
t["pop_q"] = pd.qcut(t["pop"], 5, labels=["最少20%","次少","中間","次多","最多20%"])
Q = t.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"鄉鎮數": len(g), "人口中位數": g["pop"].median(), "原始繼承占比": g.O.sum()/g.n.sum(), "只看房屋的期望": g.E.sum()/g.n.sum(),
     "地方效果u平均": g.u.mean(), "參考老屋繼承占比": expit(logit(0.25) + AL + g.u.mean()), "實際÷期望": g.O.sum()/g.E.sum(), "屋主65+": np.average(g.owner_65, weights=g.owner_n)}), include_groups=False)
Q.to_csv(OUT/"population_quintiles.csv", encoding="utf-8-sig"); print(Q.round(3).to_string())
t.to_csv(OUT/"township_master_112.csv", index=False, encoding="utf-8-sig")
# 5) 地圖與 Moran's I
from shapely.geometry import Polygon, MultiPolygon
def _geom(gj):                                  # 來源檔有少數退化的環（<4 點），略過並修復
    polys = [gj["coordinates"]] if gj["type"] == "Polygon" else gj["coordinates"]
    out = []
    for rings in polys:
        rings = [r for r in rings if len(r) >= 4]
        if rings: out.append(Polygon(rings[0], rings[1:]).buffer(0))
    return MultiPolygon([q for p in out for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]) if not q.is_empty])
_j = json.load(open(ROOT/"external/shp/twtown2010.json"))
g = gpd.GeoDataFrame([f["properties"] for f in _j["features"]], geometry=[_geom(f["geometry"]) for f in _j["features"]], crs=4326)
fix = {"臺北縣":"新北市","台北縣":"新北市","桃園縣":"桃園市","臺中縣":"臺中市","台中縣":"臺中市","臺南縣":"臺南市","台南縣":"臺南市","高雄縣":"高雄市"}
g["county"] = g.county.str.replace("台","臺").replace(fix); g["stem"] = g.town.str.replace("台","臺").str[:-1]
g.loc[g.county.isin(["新竹市","嘉義市"]), "stem"] = "（全市"
tt = t.copy(); tt["stem"] = tt.apply(lambda r: r["name"].replace(r["county"], "", 1)[:-1], axis=1)
g = g.dissolve(by=["county","stem"], as_index=False).merge(tt, on=["county","stem"], how="left")
print("地圖鄉鎮 %d，對上地方效果 %d；未對上的分析鄉鎮：%s"%(len(g), g.u.notna().sum(), sorted(set(tt.name) - set(g.name.dropna()))))
from libpysal.weights import Queen
from esda.moran import Moran
gm = g[g.u.notna()].reset_index(drop=True); W = Queen.from_dataframe(gm, use_index=False, silence_warnings=True); W.transform = "r"
np.random.seed(0); mi = {k: Moran(gm[k].values, W, permutations=999) for k in ["u","u_within","off"]}
mor = {k: {"I": round(v.I, 3), "置換p": float(v.p_sim), "z": round(v.z_sim, 1)} for k, v in mi.items()}; mor["孤島數"] = len(W.islands); print(mor)
json.dump(mor, open(OUT/"moran.json","w"), ensure_ascii=False, indent=1)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 7.2))
main = g.cx[119.9:122.1, 21.8:25.4]
for ax, col, title, cmap, vmin, vmax in [(axes[0], "exp", "只看房屋組成時的期望繼承占比", "Oranges", 0.03, 0.35), (axes[1], "u", "調整房屋組成後的鄉鎮偏離（繼承勝算相對一般鄉鎮的倍數）", "RdBu_r", -0.6, 0.6)]:
    main.plot(ax=ax, color="#eceef1", edgecolor="white", linewidth=0.15)
    main[main[col].notna()].plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.15, legend=True, legend_kwds={"shrink": 0.45, "pad": 0.01})
    ax.set_axis_off(); ax.set_title(title, loc="left", fontsize=12)
cb = fig.axes[-1]; tk = [-0.5, -0.25, 0, 0.25, 0.5]; cb.set_yticks(tk); cb.set_yticklabels([f"×{np.exp(v):.2f}" for v in tk])
fig.suptitle("房屋組成預期繼承占比高的地方，與調整後偏離高的地方，並不完全重疊（112 年，本島；淺灰＝未納入）", x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.savefig(FIG/"fig03_map.png")
