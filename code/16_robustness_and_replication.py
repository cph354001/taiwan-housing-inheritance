"""(1) 重現先前研究的頭條數字；(2) 地方效果在不同設定下的穩健性；(3) 模擬檢核：收縮模型能否還原已知的 τ。"""
import pandas as pd, numpy as np, json
from scipy.special import expit, logit
from scipy.stats import spearmanr
from common import *
import glmm
own = pd.read_parquet(DATA/"own.parquet", columns=["year","building_sn","register_reason_group_cd","b_age","floor_group_cd","county_cd"])
r = own[own.year == 112]; rep = {}
for k, lab in [("B","買賣"),("C","贈與"),("D","繼承")]:
    g = r[r.register_reason_group_cd == k]; a = g.b_age[g.b_age >= 0]; fl = g.floor_group_cd[g.floor_group_cd.isin(list("012345"))]
    rep[lab] = {"紀錄數": len(g), "屋齡中位數": a.median(), "屋齡Q1": a.quantile(.25), "屋齡Q3": a.quantile(.75), "1–5層占比%": (fl == "1").mean()*100}
tot = r.groupby("county_cd").register_reason_group_cd.apply(lambda s: (s == "D").mean()*100)
rep["臺北原始繼承占比%(全部紀錄)"] = tot["A"]; rep["桃園原始繼承占比%(全部紀錄)"] = tot["H"]
print(pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).round(1))
# 縣市層級：原始、期望、O/E（分母 B+C+D、每棟一筆、屋齡已知，112 年）
t = pd.read_csv(OUT/"township_effects_112.csv")
cty = t.groupby("county").agg(n=("n","sum"), O=("O","sum"), E=("E","sum")); cty["原始%"] = cty.O/cty.n*100; cty["只看房屋的期望%"] = cty.E/cty.n*100; cty["O/E"] = cty.O/cty.E
cty = cty.sort_values("原始%", ascending=False); cty.to_csv(OUT/"county_raw_vs_expected.csv", encoding="utf-8-sig"); print(cty.round(2).to_string())
rep["臺北_桃園"] = {"原始差pp": cty.loc["臺北市","原始%"] - cty.loc["桃園市","原始%"], "期望差pp": cty.loc["臺北市","只看房屋的期望%"] - cty.loc["桃園市","只看房屋的期望%"], "臺北O/E": cty.loc["臺北市","O/E"], "桃園O/E": cty.loc["桃園市","O/E"]}
# 老屋（≥40 年、1–5 層）原始鄉鎮繼承占比（先前研究第三層的寫法）
b = pd.read_parquet(DATA/"bld.parquet"); town = pd.read_csv(DATA/"town.csv")
o = b[(b.year == 112) & b.reason.isin(list("BCD")) & (b.age >= 40) & (b.floor_group_cd == "1")]
g = o.groupby("countyntownship_cd").reason.agg(n="size", share=lambda s: (s == "D").mean()); g = g[g.n >= 100]
rep["老屋鄉鎮原始占比"] = {"鄉鎮數": len(g), "最小%": g.share.min()*100, "最大%": g.share.max()*100, "P5%": g.share.quantile(.05)*100, "P95%": g.share.quantile(.95)*100, "SD_pp": g.share.std()*100, "平均屋齡的鄉鎮SD(歲)": o.groupby("countyntownship_cd").age.mean().reindex(g.index).std()}
json.dump(rep, open(OUT/"replication_handoff.json","w"), ensure_ascii=False, indent=1, default=float); print(json.dumps(rep, ensure_ascii=False, indent=1, default=float)[:1500])
# ---------- 穩健性 ----------
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet").merge(town, on="house_cd")
d["p_cell"] = d.groupby(["year","age","materials_group_cd","floor_group_cd","purpose_group_cd"], observed=True).D.transform("mean")
d["lp"] = np.log2(d["pop"])
def run(g, pcol="p_comp", ycol="D", nmin=50):
    x = g.groupby("house_cd").agg(n=(ycol,"size"), O=(ycol,"sum"), E=(pcol,"sum"), lp=("lp","first"), county=("county","first")).reset_index(); x = x[x.n >= nmin]; x["E"] *= x.O.sum()/x.E.sum()
    f = glmm.fit(x.O, x.n, logit(x.E/x.n)); x["u"] = f["u"]
    dmx = x.lp - x.groupby("county").lp.transform("mean"); dmu = x.u - x.groupby("county").u.transform("mean")
    return x, {"鄉鎮數": len(x), "τ": f["tau"], "u~人口Spearman": spearmanr(x.lp, x.u)[0], "縣市內Spearman": spearmanr(dmx, dmu)[0], "人口加倍的勝算倍數": float(np.exp(np.polyfit(x.lp, x.u, 1)[0]))}
base, _ = run(d[d.year == 112]); rows = []
def add(lab, g, **kw):
    x, s = run(g, **kw); m = x.merge(base[["house_cd","u"]], on="house_cd", suffixes=("","_base")); s["與主設定的相關"] = np.corrcoef(m.u, m.u_base)[0,1]; rows.append({"設定": lab, **s})
y = d[d.year == 112]
add("主設定：112 年，分母 買賣＋贈與＋繼承，GBM 組成模型", y)
add("組成模型改用格子平均（建材×樓層×每歲屋齡×用途）", y, pcol="p_cell")
bd = y[y.reason.isin(["B","D"])].copy(); add("分母只留 買賣＋繼承", bd)
fam = y.copy(); fam["F"] = fam.reason.isin(["C","D"]).astype(int); fam["pF"] = fam.groupby(["age","materials_group_cd","floor_group_cd","purpose_group_cd"], observed=True).F.transform("mean"); add("結果改為「家內移轉」＝贈與＋繼承", fam, pcol="pF", ycol="F")
add("排除澎湖、金門、連江", y[~y.county.isin(ISLANDS)])
add("只留 ≥300 棟的鄉鎮", y, nmin=300)
add("只看 40 年以上、1–5 層", y[(y.age >= 40) & (y.floor_group_cd == "1")])
add("只看未滿 40 年", y[y.age < 40])
add("110 年", d[d.year == 110]); add("111 年", d[d.year == 111])
RB = pd.DataFrame(rows); RB.to_csv(OUT/"robustness_place_effect.csv", index=False, encoding="utf-8-sig"); print(RB.round(3).to_string())
# ---------- 模擬檢核：已知 τ，重新生成、重新估計 ----------
x = base; rng = np.random.default_rng(3); sim = []
for tau_true in [0.0, 0.15, 0.28]:
    est = []
    for _ in range(60):
        u = rng.normal(0, tau_true, len(x)); O = rng.binomial(x.n.values, expit(logit(x.E/x.n).values + u))
        f = glmm.fit(O, x.n.values, logit(x.E/x.n).values); est.append((f["tau"], np.corrcoef(f["u"], u)[0,1] if tau_true > 0 else np.nan, np.mean(np.abs(f["u"] - u) <= 1.96*f["u_sd"])))
    e = np.array(est); sim.append({"真實τ": tau_true, "估計τ平均": e[:,0].mean(), "估計τ的SD": e[:,0].std(), "u與真值相關": np.nanmean(e[:,1]), "95%區間涵蓋率": e[:,2].mean()})
SC = pd.DataFrame(sim); SC.to_csv(OUT/"simulation_check_glmm.csv", index=False, encoding="utf-8-sig"); print(SC.round(3).to_string())
# ---------- 補充：報告中其他數字的出處 ----------
extra = {}
# (a) 屋齡不明的建物在各人口五等分的比例，以及把它們算進來的原始繼承占比
tm = pd.read_csv(OUT/"township_master_112.csv")[["house_cd","pop"]]; tm["pop_q"] = pd.qcut(tm["pop"], 5, labels=["最少20%","次少","中間","次多","最多20%"])
a = b[(b.year == 112) & b.reason.isin(list("BCD"))].merge(tm, left_on="countyntownship_cd", right_on="house_cd")
UQ = a.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"繼承類屋齡不明%": g.age[g.reason == "D"].isna().mean()*100, "買賣贈與類屋齡不明%": g.age[g.reason != "D"].isna().mean()*100,
      "繼承占比%（屋齡已知）": (g.reason[g.age.notna()] == "D").mean()*100, "繼承占比%（含屋齡不明）": (g.reason == "D").mean()*100}), include_groups=False)
UQ.to_csv(OUT/"unknown_age_by_population.csv", encoding="utf-8-sig"); print(UQ.round(1).to_string())
# (b) 屋主與建物在哪一層對接（各年度）
ow = pd.read_parquet(DATA/"own.parquet", columns=["year","gender_cd","age","education_cd","county_cd","countyntownship_cd","register_reason_group_cd","b_age"])
ow = ow[(ow.gender_cd != 3) & ow.age.notna()].copy(); ow["s65"] = (ow.age >= 65).astype(float); ow["univ"] = (ow.education_cd >= 4).astype(float); ow["Dm"] = (ow.register_reason_group_cd == "D").astype(float); ow["ba"] = ow.b_age.where(ow.b_age >= 0)
rows = []
for yv, g in ow.groupby("year"):
    tt = g.groupby("countyntownship_cd").agg(n=("age","size"), s65=("s65","mean"), univ=("univ","mean")); tt = tt[tt.n >= 500]; tt["cty"] = tt.index.str[0]
    r65 = (tt.s65 - tt.groupby("cty").s65.transform("mean")).std()/np.sqrt((tt.s65*(1 - tt.s65)/tt.n).mean()); run_ = (tt.univ - tt.groupby("cty").univ.transform("mean")).std()/np.sqrt((tt.univ*(1 - tt.univ)/tt.n).mean())
    gg = g.dropna(subset=["ba"]); w = gg[["age","ba","Dm"]] - gg.groupby("countyntownship_cd")[["age","ba","Dm"]].transform("mean")
    rows.append({"year": yv, "鄉鎮間65歲以上比例SD÷二項SD": r65, "鄉鎮間大學以上比例SD÷二項SD": run_, "鄉鎮內相關_屋主年齡與屋齡": w.age.corr(w.ba), "鄉鎮內相關_屋主年齡與是否繼承": w.age.corr(w.Dm)})
FL = pd.DataFrame(rows); FL.to_csv(OUT/"forensic_owner_link.csv", index=False, encoding="utf-8-sig"); print(FL.round(3).to_string())
# (c) 地方效果在不同類型房屋之間的一致性
segs = {"未滿30年": y[y.age < 30], "30–39年": y[(y.age >= 30) & (y.age < 40)], "40–49年": y[(y.age >= 40) & (y.age < 50)], "50年以上": y[y.age >= 50], "1–5層": y[y.floor_group_cd == "1"], "6層以上": y[y.floor_group_cd.isin(["2","3","4","5"])]}
SE = {}
for k, g in segs.items():
    x = g.groupby("house_cd").agg(n=("D","size"), O=("D","sum"), E=("p_comp","sum")).reset_index(); x = x[x.n >= 300]; x["lr"] = np.log((x.O + .5)/(x.E*x.O.sum()/x.E.sum() + .5)); SE[k] = x
sc = []
for p, q in [("未滿30年","40–49年"), ("30–39年","40–49年"), ("40–49年","50年以上"), ("1–5層","6層以上")]:
    m = SE[p].merge(SE[q], on="house_cd"); sc.append({"比較": f"{p} 對 {q}", "鄉鎮數(兩類皆≥300棟)": len(m), "log(實際÷期望)的相關": np.corrcoef(m.lr_x, m.lr_y)[0,1]})
pd.DataFrame(sc).to_csv(OUT/"place_effect_by_segment.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(sc).round(2).to_string())
# (d) 臺北－桃園：每歲屋齡直接標準化（兩市合併權重、共同屋齡範圍）
x = y[y.house_cd.str[0].isin(["A","H"])].copy(); x["cty"] = x.house_cd.str[0]
g2 = x.groupby(["age","cty"]).agg(n=("D","size"), D=("D","mean")).unstack().dropna(); w = g2["n"].sum(1)
extra["臺北桃園_每歲屋齡直接標準化%"] = {"臺北": float((g2["D"]["A"]*w).sum()/w.sum()*100), "桃園": float((g2["D"]["H"]*w).sum()/w.sum()*100), "臺北原始": float(x[x.cty == "A"].D.mean()*100), "桃園原始": float(x[x.cty == "H"].D.mean()*100)}
x["ab"] = np.select([x.age < 35, x.age < 45], ["未滿35年", "35–44年"], "45年以上"); extra["臺北桃園_分屋齡帶%"] = (x.groupby(["ab","cty"]).D.mean()*100).round(1).unstack().to_dict()
# (e) 人建地檔鄉鎮居民數 × 10 對真實戶籍人口；112 年人建地檔筆數
tc = pd.read_csv(DATA/"town_cov.csv").dropna(subset=["pop","res_n"])
extra["居民數x10對真實人口"] = {"log相關": float(np.corrcoef(np.log(tc.res_n), np.log(tc["pop"]))[0,1]), "比值中位數": float((tc.res_n*10/tc["pop"]).median()), "112年人建地檔筆數(已對應鄉鎮)": int(tc.res_n.sum())}
extra["屋主檔規模"] = {"手冊111年母體筆數": 10766461, "111年模擬筆數": int((own.year == 111).sum()), "比值": float((own.year == 111).sum()/10766461)}
json.dump(extra, open(OUT/"report_extra_numbers.json","w"), ensure_ascii=False, indent=1); print(json.dumps(extra, ensure_ascii=False, indent=1))
# ---------- 把屋齡不明的建物納入（LightGBM 可直接處理缺值；屋齡不明成為一種房屋條件） ----------
import lightgbm as lgb
al = b[b.year.isin([110,111,112]) & b.reason.isin(list("BCD"))].merge(town, left_on="countyntownship_cd", right_on="house_cd"); al = al[al["pop"].notna()].copy()
al["D"] = (al.reason == "D").astype(int); al["lp"] = np.log2(al["pop"])
for c in ["materials_group_cd","floor_group_cd","purpose_group_cd"]: al[c] = al[c].fillna("NA").astype("category")
F = ["age","materials_group_cd","floor_group_cd","purpose_group_cd"]
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, verbose=-1, random_state=1).fit(al.loc[al.year < 112, F], al.loc[al.year < 112, "D"])
a112 = al[al.year == 112].copy(); a112["p_all"] = m.predict_proba(a112[F])[:, 1]
xa, sa = run(a112, pcol="p_all"); mm = xa.merge(base[["house_cd","u"]], on="house_cd", suffixes=("","_base")); sa["與主設定的相關"] = np.corrcoef(mm.u, mm.u_base)[0,1]
tq = xa.merge(tm[["house_cd","pop_q"]], on="house_cd").groupby("pop_q", observed=True).u.mean(); bq = base.merge(tm[["house_cd","pop_q"]], on="house_cd").groupby("pop_q", observed=True).u.mean()
sa.update({"設定": "納入屋齡不明的建物（屋齡缺值作為一種條件）", "棟數": int(xa.n.sum()), "最少20%鄉鎮u平均": float(tq.iloc[0]), "最多20%鄉鎮u平均": float(tq.iloc[-1]), "主設定_最少20%": float(bq.iloc[0]), "主設定_最多20%": float(bq.iloc[-1])})
pd.DataFrame([sa]).to_csv(OUT/"robustness_include_unknown_age.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame([sa]).round(3).T.to_string())
