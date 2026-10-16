"""主題二：繼承占比高，是「繼承多」還是「買賣少」？
存量占比只能識別發生率的比值；要分開兩者需要流量。這裡把真實登記流量（分子）除以模擬資料的建物存量（分母，模擬規模約為母體的 1/10），
得到每 100 棟建物每年的買賣、贈與、繼承登記數（以下稱周轉率／發生率）。
 (a) 22 縣市 × 108–112 年：內政部統計處「建物所有權登記—按區域別登記項目別分」（棟）
 (b) 臺中、臺南、新北、彰化 112 年各區（先前研究已解析；臺中／臺南／彰化為住宅筆數、新北為建物棟數）
"""
import pandas as pd, numpy as np, re, json
from scipy.stats import spearmanr
from common import *
b = pd.read_parquet(DATA/"bld.parquet"); town = pd.read_csv(DATA/"town.csv")
SCALE = 10.0   # 手冊：模擬筆數約為母體的 10%；以表 32–34 與本檔筆數核對，見報告
# ---------- (a) 縣市 × 年 ----------
rows = []
for y in range(108, 113):
    for line in open(REAL/f"raw/moi{y}.csv", encoding="utf-8-sig"):
        m = re.match(r'"(\d+)年/ (\S+)/ (.+?)",([\d\-]+),', line)
        if m: rows.append((int(m[1]), m[2], m[3], 0 if m[4] == "-" else int(m[4])))
r = pd.DataFrame(rows, columns=["year","county","item","n"]).pivot_table(index=["year","county"], columns="item", values="n", aggfunc="sum").reset_index()
cc = pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str)
st = b.groupby(["year","county_cd"]).size().rename("sim_n").reset_index().merge(cc[["county_cd","county_name"]], on="county_cd")
st["stock"] = st.sim_n*SCALE
# 存量中的占比（各年、全部建物、分母 B+C+D）與老屋（≥40 年、1–5 層）占比
x = b[b.reason.isin(list("BCD"))]
sh = x.groupby(["year","county_cd"]).reason.value_counts(normalize=True).unstack().add_prefix("stock_share_").reset_index()
C = r.merge(st, left_on=["year","county"], right_on=["year","county_name"]).merge(sh, on=["year","county_cd"])
for k, col in [("B","買賣"),("D","繼承"),("A","所有權第一次登記"),("T","移轉登記")]: C["r_"+k] = C[col]/C.stock*100
C["r_C"] = (C["贈與"] + C["夫妻贈與"])/C.stock*100
C["flow_share_D"] = C["繼承"]/(C["買賣"]+C["贈與"]+C["繼承"])
C = C[C.county != "連江縣"]
C.to_csv(OUT/"real_rates_county_year.csv", index=False, encoding="utf-8-sig")
nat = C.groupby("year")[["買賣","繼承","贈與","夫妻贈與","所有權第一次登記","移轉登記","stock"]].sum(); 
for k, col in [("買賣率","買賣"),("繼承率","繼承"),("第一次登記率","所有權第一次登記"),("移轉率","移轉登記")]: nat[k] = nat[col]/nat.stock*100
print(nat.round(3).to_string())
P = C.groupby("county").agg(r_B=("r_B","mean"), r_D=("r_D","mean"), r_C=("r_C","mean"), r_A=("r_A","mean"), stock=("stock","mean"), sD=("stock_share_D","mean"), fD=("flow_share_D","mean")).sort_values("r_B")
print(P.round(3).to_string())
def vdec(df, a="r_D", bb="r_B", grp=None):
    la, lb = np.log(df[a]), np.log(df[bb])
    if grp is not None: la, lb = la - la.groupby(df[grp]).transform("mean"), lb - lb.groupby(df[grp]).transform("mean")
    lo = la - lb; v = np.var(lo)
    return {"n": len(df), "sd_log_r_D": la.std(), "sd_log_r_B": lb.std(), "corr": np.corrcoef(la, lb)[0,1], "var_log_odds": v,
            "share_from_D": np.cov(la, lo, bias=True)[0,1]/v, "share_from_B": np.cov(-lb, lo, bias=True)[0,1]/v,
            "p90/p10_r_D": np.exp(np.quantile(la,.9)-np.quantile(la,.1)), "p90/p10_r_B": np.exp(np.quantile(lb,.9)-np.quantile(lb,.1))}
out = {"縣市（五年平均，21 縣市）": vdec(P), "縣市（排除澎湖金門）": vdec(P.drop(["澎湖縣","金門縣"]))}
# ---------- (b) 四縣市分區 ----------
X = pd.read_csv(REAL/"districts_real_112.csv")
s112 = b[b.year == 112]
S = s112.groupby("countyntownship_cd").agg(sim_n=("building_sn","size")).reset_index()
S["sim_res"] = S.countyntownship_cd.map(s112[s112.purpose_group_cd.isin(["A","F","G","L"])].groupby("countyntownship_cd").size())
old = s112[s112.reason.isin(list("BCD"))]
S["stock_share_D"] = S.countyntownship_cd.map(old.groupby("countyntownship_cd").reason.apply(lambda s: (s == "D").mean()))
S["old_share"] = S.countyntownship_cd.map(s112[s112.age.notna()].groupby("countyntownship_cd").age.apply(lambda s: (s >= 40).mean()))
X = X.merge(S, left_on="house_cd", right_on="countyntownship_cd", how="left")
eff = pd.read_csv(OUT/"township_effects_112.csv")[["house_cd","u","exp","raw"]]
X = X.merge(eff, on="house_cd", how="left")
X["stock"] = np.where(X.unit.str.contains("住宅"), X.sim_res, X.sim_n)*SCALE
for k in "BCD": X["r_"+k] = X[k]/X.stock*100
X = X[(X.B + X.C + X.D >= 30) & (X.sim_n >= 100) & (X.D > 0) & (X.B > 0)].copy(); X["lp"] = np.log2(X.pop_real)
X.to_csv(OUT/"real_rates_district_112.csv", index=False, encoding="utf-8-sig")
print(X.groupby("city").agg(n=("district","size"), r_B=("r_B","median"), r_D=("r_D","median"), r_C=("r_C","median")).round(3))
out["四縣市分區（扣除縣市平均，%d 區）"%len(X)] = vdec(X, grp="city")
for c, g in X.groupby("city"): out[c] = vdec(g)
V = pd.DataFrame(out).T; V.to_csv(OUT/"real_rate_variance_decomp.csv", encoding="utf-8-sig"); print(V.round(3).to_string())
# 與人口的關係（城市內）：人口每加倍，買賣率、繼承率各差多少倍
rows = []
dm = lambda s, g: s - s.groupby(g).transform("mean")
for k in ["r_B","r_D","r_C"]:
    yv = dm(np.log(X[k].clip(lower=1e-3)), X.city); xv = dm(X.lp, X.city); sl = (xv*yv).sum()/(xv*xv).sum()
    rows.append({"率": k, "人口加倍的倍數": np.exp(sl), "城市內相關": np.corrcoef(xv, yv)[0,1], "Spearman": spearmanr(xv, yv)[0]})
    for c, g in X.groupby("city"): rows[-1][c] = spearmanr(g.lp, g[k])[0]
PR = pd.DataFrame(rows); PR.to_csv(OUT/"real_rates_vs_population.csv", index=False, encoding="utf-8-sig"); print(PR.round(3).to_string())
# 模擬資料的存量占比、地方效果，與真實的買賣率、繼承率（城市內 Spearman）
Xu = X.dropna(subset=["u"]); dmu = lambda s: s - s.groupby(Xu.city).transform("mean")
lk = []
for a, lab in [("stock_share_D","模擬：存量繼承占比"), ("exp","模擬：只看房屋的期望占比"), ("u","模擬：地方效果 u")]:
    lk.append({"模擬指標": lab, "真實流量繼承占比": spearmanr(dmu(Xu[a]), dmu(Xu.real))[0], "真實買賣率": spearmanr(dmu(Xu[a]), dmu(np.log(Xu.r_B)))[0], "真實繼承率": spearmanr(dmu(Xu[a]), dmu(np.log(Xu.r_D)))[0], "n": len(Xu)})
LK = pd.DataFrame(lk); LK.to_csv(OUT/"sim_vs_real_district.csv", index=False, encoding="utf-8-sig"); print(LK.round(3).to_string())
# 縣市層級：屋主年齡結構（屋主檔、自然人，五年平均）與真實發生率
own = pd.read_parquet(DATA/"own.parquet", columns=["year","gender_cd","age","county_cd"]); o = own[(own.gender_cd != 3) & own.age.notna()]
oa = o.groupby("county_cd").age.agg(owner_65=lambda s: (s >= 65).mean(), owner_age="mean").reset_index().merge(cc[["county_cd","county_name"]], on="county_cd")
P2 = P.merge(oa, left_index=True, right_on="county_name").set_index("county_name")
P2.to_csv(OUT/"real_rates_county_mean.csv", encoding="utf-8-sig")
oc = {k: {"屋主65歲以上比例": spearmanr(P2.owner_65, P2[k])[0], "存量繼承占比(模擬)": spearmanr(P2.sD, P2[k])[0]} for k in ["r_B","r_D","fD"]}
print(pd.DataFrame(oc).round(3)); pd.DataFrame(oc).to_csv(OUT/"county_owner_age_vs_real.csv", encoding="utf-8-sig")
print("縣市：存量占比(模擬) vs 流量占比(真實) Spearman %.3f Pearson %.3f"%(spearmanr(P2.sD, P2.fD)[0], np.corrcoef(P2.sD, P2.fD)[0,1]))
