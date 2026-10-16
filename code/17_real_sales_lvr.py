"""外部佐證（全國鄉鎮、含屋齡）：內政部不動產交易實價查詢服務網「不動產買賣」開放資料，111S3–112S4 六季。
真實買賣的建物有鄉鎮、建築完成年月、總樓層數、主要建材，可以算「同類房子在各鄉鎮每年被買賣的比例」：
   買賣周轉率 = 真實買賣棟數（年化） ÷ 模擬存量 × 10
分母用模擬資料的建物存量（鄉鎮 × 屋齡組 × 樓層組），分子用真實交易。"""
import pandas as pd, numpy as np, glob, re, json
from scipy.stats import spearmanr
from scipy.special import logit
from common import *
import glmm
LET = {"a":"臺北市","b":"臺中市","c":"基隆市","d":"臺南市","e":"高雄市","f":"新北市","g":"宜蘭縣","h":"桃園市","i":"嘉義市","j":"新竹縣","k":"苗栗縣","m":"南投縣","n":"彰化縣","o":"新竹市","p":"雲林縣","q":"嘉義縣","t":"屏東縣","u":"花蓮縣","v":"臺東縣","w":"金門縣","x":"澎湖縣","z":"連江縣"}
SEASONS = ["111S3","111S4","112S1","112S2","112S3","112S4"]
CN = {c: i for i, c in enumerate("零一二三四五六七八九")}
def cn2int(s):
    if not isinstance(s, str) or not s.endswith("層"): return np.nan
    s = s[:-1]
    if s.isdigit(): return int(s)
    try:
        if "十" in s:
            a, _, c = s.partition("十"); return (CN[a] if a else 1)*10 + (CN[c] if c else 0)
        return CN[s]
    except KeyError: return np.nan
parts = []
for s in SEASONS:
    for f in glob.glob(str(ROOT/f"external/lvr/{s}/*_lvr_land_a.csv")):
        x = pd.read_csv(f, dtype=str, skiprows=[1], usecols=["鄉鎮市區","交易標的","交易年月日","交易筆棟數","總樓層數","建物型態","主要用途","主要建材","建築完成年月","備註"], on_bad_lines="skip", encoding="utf-8-sig")
        x["county"] = LET[f.split("/")[-1][0]]; x["season"] = s; parts.append(x)
L = pd.concat(parts, ignore_index=True)
L["n_bld"] = pd.to_numeric(L["交易筆棟數"].str.extract(r"建物(\d+)")[0], errors="coerce").fillna(0)
audit = {"rows": len(L), "rows_with_building": int((L.n_bld > 0).sum()), "buildings_total": int(L.n_bld.sum()), "by_season_buildings": L.groupby("season").n_bld.sum().astype(int).to_dict()}
L = L[(L.n_bld > 0) & L["交易標的"].fillna("").str.contains("房地|建物")].copy()      # 排除只有車位或土地的案件
audit["rows_used"] = len(L); audit["buildings_used"] = int(L.n_bld.sum())
L["special"] = L["備註"].fillna("").str.contains("親友|特殊關係")
yr = pd.to_numeric(L["建築完成年月"].str.strip().str[:-4], errors="coerce")       # 民國年
L["age"] = np.where(yr.between(1, 113), 112 - yr, np.nan)
L["floors"] = L["總樓層數"].map(cn2int)
L["fl"] = np.select([L.floors.between(1,5), L.floors.between(6,10), L.floors >= 11], ["1","2","3+"], default="NA")
L["ab"] = pd.cut(L.age, [-1,9,19,29,39,49,200], labels=["0-9","10-19","20-29","30-39","40-49","50+"]).astype(str).replace("nan","NA")
town = pd.read_csv(DATA/"town.csv"); nm = dict(zip(town.name, town.house_cd))
L["name"] = L.county + L["鄉鎮市區"].fillna("").str.replace("台","臺")
L["house_cd"] = L.name.map(nm); L.loc[L.county == "新竹市", "house_cd"] = "O01"; L.loc[L.county == "嘉義市", "house_cd"] = "I01"
audit["unmatched_names"] = L[L.house_cd.isna()].name.value_counts().head(10).to_dict(); audit["age_known_share"] = float((L.ab != "NA").mean()); audit["special_relation_share"] = float(np.average(L.special, weights=L.n_bld))
YEARS = len(SEASONS)/4
# 模擬存量（112 年，全部登記原因）
b = pd.read_parquet(DATA/"bld.parquet"); s = b[b.year == 112].copy()
s["ab"] = pd.cut(s.age, [-1,9,19,29,39,49,200], labels=["0-9","10-19","20-29","30-39","40-49","50+"]).astype(str).replace("nan","NA")
s["fl"] = s.floor_group_cd.map({"1":"1","2":"2","3":"3+","4":"3+","5":"3+"}).fillna("NA")
stock = s.groupby(["countyntownship_cd","ab","fl"]).size().rename("stock").reset_index().rename(columns={"countyntownship_cd":"house_cd"}); stock["stock"] *= 10
sales = L.dropna(subset=["house_cd"]).groupby(["house_cd","ab","fl"]).n_bld.sum().div(YEARS).rename("sales").reset_index()
cell = stock.merge(sales, on=["house_cd","ab","fl"], how="outer").fillna({"stock":0,"sales":0})
cell.to_csv(OUT/"lvr_cells.csv", index=False)
nat = cell.groupby(["ab","fl"])[["stock","sales"]].sum(); nat["rate"] = nat.sales/nat.stock*100
audit["national_rate_all"] = float(cell.sales.sum()/cell.stock.sum()*100); audit["annual_sales_buildings"] = float(cell.sales.sum())
print(nat.round(2).to_string())
by_age = cell[cell.ab != "NA"].groupby("ab")[["stock","sales"]].sum(); by_age["rate"] = by_age.sales/by_age.stock*100; print(by_age.round(2)); by_age.to_csv(OUT/"lvr_rate_by_age.csv")
# 鄉鎮：整體周轉率、老屋（≥40 年且 1–5 層）周轉率、以及扣除房屋組成後的周轉地方效果
ok = cell[(cell.ab != "NA") & (cell.fl != "NA") & (cell.stock > 0)].copy()
ok["rate_nat"] = ok.groupby(["ab","fl"]).sales.transform("sum")/ok.groupby(["ab","fl"]).stock.transform("sum")
ok["E"] = ok.stock*ok.rate_nat
T = ok.groupby("house_cd").agg(lv_stock=("stock","sum"), lv_sales=("sales","sum"), lv_E=("E","sum")).reset_index()
o = ok[ok.ab.isin(["40-49","50+"]) & (ok.fl == "1")].groupby("house_cd").agg(lv_stock_old=("stock","sum"), lv_sales_old=("sales","sum"), lv_E_old=("E","sum")).reset_index()
T = T.merge(o, on="house_cd", how="left").merge(pd.read_csv(OUT/"township_master_112.csv"), on="house_cd")
T["turn"] = T.lv_sales/T.lv_stock*100; T["turn_old"] = T.lv_sales_old/T.lv_stock_old*100; T["smr"] = T.lv_sales/T.lv_E; T["smr_old"] = T.lv_sales_old/T.lv_E_old
T["v"] = np.log((T.lv_sales + 0.5)/(T.lv_E + 0.5))            # 真實買賣周轉的地方效果（log 比值）
T.to_csv(OUT/"lvr_township.csv", index=False, encoding="utf-8-sig")
dm = lambda x: x - x.groupby(T.county).transform("mean")
big = T.lv_stock_old >= 1000
rows = []
for a, la in [("u","模擬：繼承的地方效果 u"), ("raw","模擬：原始繼承占比"), ("lp","戶籍人口(log2)"), ("res_65","居民65歲以上比例")]:
    for k, lk in [("turn","整體買賣周轉率"), ("v","同類房屋買賣周轉（扣除屋齡×樓層）"), ("turn_old","老屋買賣周轉率")]:
        m = T[[a,k,"county"]].replace([np.inf,-np.inf], np.nan).dropna() if k != "turn_old" else T.loc[big, [a,k,"county"]].dropna()
        dmm = lambda x: x - x.groupby(m.county).transform("mean")
        rows.append({"變數": la, "真實指標": lk, "全國Spearman_不含金門": spearmanr(m[m.county != "金門縣"][a], m[m.county != "金門縣"][k])[0], "全國Spearman": spearmanr(m[a], m[k])[0], "縣市內Spearman": spearmanr(dmm(m[a]), dmm(m[k]))[0], "鄉鎮數": len(m)})
R = pd.DataFrame(rows); R.to_csv(OUT/"lvr_vs_sim.csv", index=False, encoding="utf-8-sig"); print(R.round(3).to_string())
T["pop_q"] = pd.qcut(T["pop"], 5, labels=["最少20%","次少","中間","次多","最多20%"])
Q = T.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"鄉鎮數": len(g), "整體周轉率%": g.lv_sales.sum()/g.lv_stock.sum()*100, "老屋周轉率%": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100, "同類房屋周轉比(O/E)": g.lv_sales.sum()/g.lv_E.sum(), "老屋O/E": g.lv_sales_old.sum()/g.lv_E_old.sum()}), include_groups=False)
Q.to_csv(OUT/"lvr_population_quintiles.csv", encoding="utf-8-sig"); print(Q.round(3).to_string())
T["u_q"] = pd.qcut(T.u, 5, labels=["u最低20%","次低","中間","次高","u最高20%"])
U = T.groupby("u_q", observed=True).apply(lambda g: pd.Series({"鄉鎮數": len(g), "u平均": g.u.mean(), "整體周轉率%": g.lv_sales.sum()/g.lv_stock.sum()*100, "老屋周轉率%": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100, "同類房屋周轉比(O/E)": g.lv_sales.sum()/g.lv_E.sum()}), include_groups=False)
U.to_csv(OUT/"lvr_by_place_effect.csv", encoding="utf-8-sig"); print(U.round(3).to_string())


# ---------- 敏感度與限制的量化 ----------
# (1) 屋齡不明的低樓層建物若全部視為老屋（分子、分母都加入）
okn = cell[(cell.fl == "1") & cell.ab.isin(["40-49","50+","NA"])].groupby("house_cd")[["stock","sales"]].sum().rename(columns={"stock":"s_alt","sales":"x_alt"})
unk = cell[(cell.fl == "1") & (cell.ab == "NA")].groupby("house_cd").stock.sum().rename("s_unk")
allc = cell.groupby("house_cd")[["stock","sales"]].sum().rename(columns={"stock":"s_all","sales":"x_all"})
T2 = T.merge(okn, on="house_cd", how="left").merge(unk, on="house_cd", how="left").merge(allc, on="house_cd", how="left").fillna({"s_unk": 0})
# (2) 老屋買賣的建物型態與特殊關係交易
Lo = L[(L.fl == "1") & L.ab.isin(["40-49","50+"])].dropna(subset=["house_cd"]).copy()
Lo["透天"] = Lo["建物型態"].fillna("").str.contains("透天"); Lo["公寓"] = Lo["建物型態"].fillna("").str.contains("公寓")
ty = Lo.groupby("house_cd").apply(lambda g: pd.Series({"x_town": g.n_bld[g["透天"]].sum(), "x_apt": g.n_bld[g["公寓"]].sum(), "x_special": g.n_bld[g.special].sum(), "x_oldall": g.n_bld.sum()}), include_groups=False)
T2 = T2.merge(ty, on="house_cd", how="left")
S = T2.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"老屋周轉率%（主設定）": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100, "老屋周轉率%（屋齡不明視為老屋）": g.x_alt.sum()/g.s_alt.sum()*100,
    "屋齡不明低樓層÷已知老屋存量%": g.s_unk.sum()/g.lv_stock_old.sum()*100, "全部建物周轉率%（含屋齡樓層不明）": g.x_all.sum()/g.s_all.sum()*100,
    "老屋買賣中透天厝%": g.x_town.sum()/g.x_oldall.sum()*100, "老屋買賣中公寓%": g.x_apt.sum()/g.x_oldall.sum()*100, "老屋買賣中特殊關係交易%": g.x_special.sum()/g.x_oldall.sum()*100,
    "老屋周轉率%（排除特殊關係交易）": (g.x_oldall.sum() - g.x_special.sum())/YEARS/g.lv_stock_old.sum()*100}), include_groups=False)
S.to_csv(OUT/"lvr_sensitivity_quintiles.csv", encoding="utf-8-sig"); print(S.round(2).T.to_string())
# (3) 各縣市涵蓋率，與「既有建物（屋齡 ≥10 年）」的買賣率
import re
off = {}
for y in [111, 112]:
    for line in open(REAL/f"raw/moi{y}.csv", encoding="utf-8-sig"):
        m = re.match(r'"(\d+)年/ (\S+)/ 買賣",([\d\-]+),', line)
        if m and m[3] != "-": off[(y, m[2])] = int(m[3])
Lc = L.groupby("county").n_bld.sum()/YEARS
cov = pd.DataFrame({"lvr": Lc}); cov["official"] = [ (0.5*off.get((111, c), np.nan) + off.get((112, c), np.nan))/1.5 for c in cov.index ]; cov["coverage"] = cov.lvr/cov.official
ex = cell[~cell.ab.isin(["0-9","NA"])].copy(); ex["cty"] = ex.house_cd.str[0]
cc = pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str); ex = ex.merge(cc[["county_cd","county_name"]], left_on="cty", right_on="county_cd")
E2 = ex.groupby("county_name")[["stock","sales"]].sum(); E2["r_B_exist"] = E2.sales/E2.stock*100
CM = pd.read_csv(OUT/"real_rates_county_mean.csv").merge(E2[["r_B_exist"]], left_on="county_name", right_index=True).merge(cov[["coverage"]], left_on="county_name", right_index=True)
CM["r_B_exist_adj"] = CM.r_B_exist/CM.coverage
def vd(df, bcol):
    la, lb = np.log(df.r_D), np.log(df[bcol]); lo = la - lb; v = np.var(lo)
    return {"n": len(df), "corr(log買賣率,log繼承率)": np.corrcoef(la, lb)[0,1], "來自買賣率%": np.cov(-lb, lo, bias=True)[0,1]/v*100, "來自繼承率%": np.cov(la, lo, bias=True)[0,1]/v*100, "買賣率最高÷最低": df[bcol].max()/df[bcol].min()}
noP = CM[~CM.county_name.isin(["澎湖縣","金門縣"])]
VD = pd.DataFrame({"全部買賣登記（官方，含新成屋第一次出售）": vd(CM, "r_B"), "既有建物買賣（實價登錄，屋齡≥10年）": vd(CM, "r_B_exist"), "同上，以各縣市涵蓋率校正": vd(CM, "r_B_exist_adj"),
                   "全部買賣登記，不含澎湖金門": vd(noP, "r_B"), "既有建物買賣，不含澎湖金門": vd(noP, "r_B_exist")}).T
# 21 個點的穩定性：逐一拿掉一個縣市、以及縣市重抽
def _c(df, bcol="r_B_exist"):
    la, lb = np.log(df.r_D), np.log(df[bcol]); lo = la - lb; return np.corrcoef(la, lb)[0,1], np.cov(-lb, lo, bias=True)[0,1]/np.var(lo)*100
loo = pd.DataFrame([{"拿掉": c, **dict(zip(["相關","來自買賣率%"], _c(CM[CM.county_name != c])))} for c in CM.county_name]); rngb = np.random.default_rng(5)
bs = np.array([_c(CM.iloc[rngb.integers(0, len(CM), len(CM))]) for _ in range(2000)]); bs0 = np.array([_c(CM.iloc[rngb.integers(0, len(CM), len(CM))], "r_B") for _ in range(2000)])
STAB = {"既有建物_相關_逐一拿掉範圍": [float(loo["相關"].min()), float(loo["相關"].max())], "影響最大的縣市": loo.sort_values("相關").iloc[[0,-1]].to_dict("records"),
        "既有建物_相關_bootstrap95": np.percentile(bs[:,0], [2.5,97.5]).tolist(), "既有建物_買賣率占比_bootstrap95": np.percentile(bs[:,1], [2.5,97.5]).tolist(),
        "全部登記_相關_bootstrap95": np.percentile(bs0[:,0], [2.5,97.5]).tolist(), "全部登記_買賣率占比_bootstrap95": np.percentile(bs0[:,1], [2.5,97.5]).tolist()}
json.dump(STAB, open(OUT/"county_existing_sales_stability.json","w"), ensure_ascii=False, indent=1); print(STAB)
VD.to_csv(OUT/"county_existing_sales_decomp.csv", encoding="utf-8-sig"); CM.to_csv(OUT/"lvr_county.csv", index=False, encoding="utf-8-sig"); print(VD.round(3).to_string())
audit["county_coverage_min_max"] = [float(cov.coverage.min()), float(cov.coverage.max())]; audit["county_coverage"] = cov.coverage.round(3).to_dict(); audit["coverage_national"] = float(cov.lvr.sum()/cov.official.sum())
json.dump(audit, open(OUT/"audit_lvr.json","w"), ensure_ascii=False, indent=1, default=float); print(audit)
# (4) 只看「老屋買賣幾乎都是透天厝」的鄉鎮（產品相同），人口梯度是否還在
H = T2[(T2.x_town/T2.x_oldall >= 0.85) & (T2.lv_stock_old >= 1000)].copy(); H["g"] = pd.qcut(H["pop"], 3, labels=["人口少1/3","中","人口多1/3"])
HT = H.groupby("g", observed=True).apply(lambda g: pd.Series({"鄉鎮數": len(g), "人口中位數": g["pop"].median(), "老屋周轉率%": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100, "排除特殊關係%": (g.x_oldall.sum() - g.x_special.sum())/YEARS/g.lv_stock_old.sum()*100}), include_groups=False)
HT.loc["Spearman(人口,周轉率)"] = [len(H), np.nan, spearmanr(H.lp, H.turn_old)[0], np.nan]; HT.loc["Spearman(地方效果u,周轉率)"] = [len(H), np.nan, spearmanr(H.u, H.turn_old)[0], np.nan]
HT.to_csv(OUT/"lvr_townhouse_only.csv", encoding="utf-8-sig"); print(HT.round(3).to_string())
# (5) 以建材把產品再對齊一層：只看「磚造、1–5 層、屋齡 40 年以上」（兩邊資料都有建材；磚造低樓層幾乎都是透天厝）
Lb = L[(L.fl == "1") & L.ab.isin(["40-49","50+"]) & L["主要建材"].fillna("").str.contains("磚")].dropna(subset=["house_cd"])
xb = Lb.groupby("house_cd").n_bld.sum().div(YEARS).rename("x_brick"); xb_ns = Lb[~Lb.special].groupby("house_cd").n_bld.sum().div(YEARS).rename("x_brick_ns")
sb = s[(s.materials_group_cd == "B") & (s.floor_group_cd == "1") & (s.age >= 40)].groupby("countyntownship_cd").size().mul(10).rename("s_brick")
T3 = T2.merge(xb, on="house_cd", how="left").merge(xb_ns, on="house_cd", how="left").merge(sb, left_on="house_cd", right_index=True, how="left").fillna({"x_brick": 0, "x_brick_ns": 0})
T3["turn_brick"] = T3.x_brick/T3.s_brick*100
BR = T3.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"磚造老屋存量(棟)": g.s_brick.sum(), "年買賣棟數": g.x_brick.sum(), "磚造老屋周轉率%": g.x_brick.sum()/g.s_brick.sum()*100, "排除特殊關係%": g.x_brick_ns.sum()/g.s_brick.sum()*100,
     "磚造買賣中透天厝%": np.average(Lb[Lb.house_cd.isin(g.house_cd)]["建物型態"].fillna("").str.contains("透天"), weights=Lb[Lb.house_cd.isin(g.house_cd)].n_bld)*100}), include_groups=False)
B3 = T3[T3.s_brick >= 500]; BR.loc["Spearman(人口,周轉率)；鄉鎮數"] = [len(B3), np.nan, spearmanr(B3.lp, B3.turn_brick)[0], np.nan, np.nan]; BR.loc["Spearman(地方效果u,周轉率)"] = [len(B3), np.nan, spearmanr(B3.u, B3.turn_brick)[0], np.nan, np.nan]
BR.to_csv(OUT/"lvr_brick_only.csv", encoding="utf-8-sig"); print(BR.round(3).to_string()); print(L["主要建材"].value_counts().head(8))
# (5b) 磚造與非磚造在存量（模擬）與買賣（真實）中的比重是否一致，以及非磚造老屋的周轉率
T3["x_nb"] = T3.lv_sales_old - T3.x_brick; T3["s_nb"] = T3.lv_stock_old - T3.s_brick.fillna(0)
MB = T3.groupby("pop_q", observed=True).apply(lambda g: pd.Series({"老屋存量中磚造%(模擬)": g.s_brick.sum()/g.lv_stock_old.sum()*100, "老屋買賣中磚造%(真實)": g.x_brick.sum()/g.lv_sales_old.sum()*100,
     "非磚造老屋周轉率%": g.x_nb.sum()/g.s_nb.sum()*100, "磚造老屋周轉率%": g.x_brick.sum()/g.s_brick.sum()*100, "全部老屋周轉率%": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100}), include_groups=False)
MB.to_csv(OUT/"lvr_brick_vs_nonbrick.csv", encoding="utf-8-sig"); print(MB.round(2).to_string())
# (5c) 模擬存量的「其他」建材在小鄉鎮占老屋近三成，真實買賣中只占 4%；若那其實是登記方式不同的磚造屋，應把磚造與其他合併看
mat = L["主要建材"].fillna("")
L["mat4"] = np.select([mat.str.contains("磚"), mat.str.contains("鋼|Ｒ|RC|混凝土"), mat.str.contains("金屬|鐵皮|鐵造")], ["B","A","C"], default="D")
Lo2 = L[(L.fl == "1") & L.ab.isin(["40-49","50+"])].dropna(subset=["house_cd"])
xm = Lo2.groupby(["house_cd","mat4"]).n_bld.sum().div(YEARS).unstack(fill_value=0).reindex(columns=list("ABCD"), fill_value=0).add_prefix("x_m")
sm = s[(s.floor_group_cd == "1") & (s.age >= 40)].groupby(["countyntownship_cd","materials_group_cd"]).size().mul(10).unstack(fill_value=0).reindex(columns=list("ABCD"), fill_value=0).add_prefix("s_m").rename_axis("house_cd")
T4 = T2[["house_cd","pop_q","u","lp"]].merge(xm, on="house_cd", how="left").merge(sm, on="house_cd", how="left"); T4[[c for c in T4.columns if c[:3] in ("x_m","s_m")]] = T4[[c for c in T4.columns if c[:3] in ("x_m","s_m")]].fillna(0)
def _mt(g):
    r = {f"{k}_存量%": g[f"s_m{k}"].sum()/sum(g[f"s_m{j}"].sum() for j in "ABCD")*100 for k in "ABCD"}
    r.update({f"{k}_買賣%": g[f"x_m{k}"].sum()/sum(g[f"x_m{j}"].sum() for j in "ABCD")*100 for k in "ABCD"})
    r.update({f"{k}_周轉率%": g[f"x_m{k}"].sum()/max(g[f"s_m{k}"].sum(), 1)*100 for k in "ABCD"})
    r["磚造+其他_周轉率%"] = (g.x_mB.sum() + g.x_mD.sum())/(g.s_mB.sum() + g.s_mD.sum())*100; r["磚造_周轉率%"] = g.x_mB.sum()/g.s_mB.sum()*100
    return pd.Series(r)
MT = T4.groupby("pop_q", observed=True).apply(_mt, include_groups=False); MT.to_csv(OUT/"lvr_material_mix_quintiles.csv", encoding="utf-8-sig"); print(MT.round(2).T.to_string())
T4["turn_bd"] = (T4.x_mB + T4.x_mD)/(T4.s_mB + T4.s_mD)*100; B4 = T4[(T4.s_mB + T4.s_mD) >= 500]
json.dump({"磚造+其他_最少÷最多": float(MT["磚造+其他_周轉率%"].iloc[0]/MT["磚造+其他_周轉率%"].iloc[-1]), "磚造_最少÷最多": float(MT["磚造_周轉率%"].iloc[0]/MT["磚造_周轉率%"].iloc[-1]),
           "Spearman(u, 磚造+其他周轉率)": float(spearmanr(B4.u, B4.turn_bd)[0]), "鄉鎮數": int(len(B4))}, open(OUT/"lvr_brick_plus_other.json","w"), ensure_ascii=False, indent=1)
