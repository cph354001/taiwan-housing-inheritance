"""問題二的鄉鎮層級版本：把「繼承多」與「買賣少」在全國鄉鎮上用真實資料分開。
 繼承的來源：內政部戶政司各鄉鎮市區死亡人數（單一年齡，ODRP031，110–112 年平均），
            以「全國各年齡每人持有建物棟數」（模擬屋主檔 112 年 ÷ 112 年 6 月底單一年齡戶籍人口，ODRP014）加權，
            得到各鄉鎮「每年因屋主死亡而釋出的建物棟數」的期望值 ED（簡單版，假設各地每人持有棟數相同）。
            主設定改用各鄉鎮自己的屋主年齡分布：m = Σ_a（該鄉鎮建物由 a 歲組屋主持有的比例 × 該鄉鎮 a 歲組死亡率），
            因為鄉下每人登記建物棟數明顯較少（未辦保存登記的房屋多），簡單版會高估鄉下的釋出率。兩者都是代理指標，不是繼承登記。
 買賣：實價登錄屋齡 ≥10 年建物的年化買賣棟數（17_real_sales_lvr.py 的 lvr_cells.csv）。
 分母：模擬存量 ×10。
 檢核：在 21 縣市上，主設定的釋出率（與簡單版 ED÷存量）與真實繼承登記率（官方登記統計）比對。"""
import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
from common import *
RIS = ROOT/"external/ris"
AGES = [f"{a:03d}" for a in range(100)] + ["100up"]
town = pd.read_csv(DATA/"town.csv").dropna(subset=["name"]); nm = dict(zip(town.name, town.house_cd))
def to_cd(site):
    s = site.replace("台", "臺").replace("　", "").strip()
    s = {"高雄市三民一": "高雄市三民區", "高雄市三民二": "高雄市三民區", "高雄市鳳山一": "高雄市鳳山區", "高雄市鳳山二": "高雄市鳳山區"}.get(s, s)   # 108 年以戶政事務所分列
    if s.startswith("新竹市"): return "O01"
    if s.startswith("嘉義市"): return "I01"
    return nm.get(s)
# ---- 死亡人數（鄉鎮 × 單一年齡，兩性合計）
D = {}
for y in [108, 109, 110, 111, 112]:
    r = pd.DataFrame(json.load(open(RIS/f"death_{y}.json"))["responseData"])
    r = r[r.according == "按登記日期分"].copy(); r["house_cd"] = r.site_id.map(to_cd)
    miss = r[r.house_cd.isna()].site_id.tolist()
    if miss: print(y, "死亡資料對不到的鄉鎮：", miss)
    M = pd.DataFrame({a: r[f"death_age_{a}_m"].astype(int) + r[f"death_age_{a}_f"].astype(int) for a in AGES}); M["house_cd"] = r.house_cd.values
    D[y] = M.dropna(subset=["house_cd"]).groupby("house_cd").sum()
    print(y, "死亡人數合計", int(D[y].values.sum()))
# ---- 戶籍人口（112 年 6 月底，鄉鎮 × 單一年齡）
P = pd.concat([pd.DataFrame(json.load(open(RIS/f"pop_11206_p{p}.json"))["responseData"]) for p in range(1, 5)])
P["house_cd"] = P.site_id.map(to_cd); print("人口資料對不到的鄉鎮：", sorted(set(P[P.house_cd.isna()].site_id)))
PM = pd.DataFrame({a: P[f"people_age_{a}_m"].astype(int) + P[f"people_age_{a}_f"].astype(int) for a in AGES}); PM["house_cd"] = P.house_cd.values
PM = PM.dropna(subset=["house_cd"]).groupby("house_cd").sum(); print("人口合計", int(PM.values.sum()))
# ---- 全國各年齡每人持有建物棟數 h(a)（屋主檔 112 年自然人；每列是一筆持有）
own = pd.read_parquet(DATA/"own.parquet", columns=["year","gender_cd","age","countyntownship_cd"]); own = own[own.year == 112]; o = own[(own.gender_cd != 3) & own.age.notna()]
cnt = o.age.clip(upper=100).astype(int).value_counts().reindex(range(101), fill_value=0).values*10
h = pd.Series(cnt/PM.sum().values, index=AGES)
h5 = h.rolling(5, center=True, min_periods=1).mean()            # 稍微平滑
print("每人持有棟數：30歲 %.3f, 50歲 %.3f, 70歲 %.3f, 85歲 %.3f, 95歲 %.3f" % tuple(h5.iloc[[30,50,70,85,95]]))
pd.DataFrame({"age": AGES, "h": h.values, "h_smooth": h5.values, "pop": PM.sum().values, "deaths_112": D[112].sum().values}).to_csv(OUT/"ownership_by_age.csv", index=False)
old = AGES[65:]
def supply(years):
    Dm = sum(D[y] for y in years)/len(years)
    return pd.DataFrame({"deaths": Dm.sum(axis=1), "deaths65": Dm[old].sum(axis=1), "ED": (Dm*h5.values).sum(axis=1)})
S = supply([110, 111, 112]).join(supply([108, 109, 110, 111, 112]).add_suffix("_5y"))
S["pop_ris"] = PM.sum(axis=1); S["pop65_ris"] = PM[old].sum(axis=1)
# ---- 主設定：鄉鎮自己的屋主年齡分布 × 鄉鎮自己的年齡別死亡率
BANDS = [(0,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,100)]; bl = [f"{a}-{b}" for a, b in BANDS]
band = lambda M: pd.DataFrame({l: M[AGES[a:b+1]].sum(axis=1) for l, (a, b) in zip(bl, BANDS)})
D3 = band((D[110] + D[111] + D[112])/3); PB = band(PM); q_loc = (D3/PB).reindex(PB.index); q_nat = D3.sum()/PB.sum()
smr = D3.sum(axis=1)/(PB*q_nat).sum(axis=1)                       # 鄉鎮標準化死亡比（全年齡）
q_smr = pd.DataFrame(np.outer(smr, q_nat), index=smr.index, columns=bl)   # 較穩定的版本：全國年齡別死亡率 × 鄉鎮 SMR
own["band"] = pd.cut(own.age.clip(upper=100), [-1,39,49,59,69,79,89,100], labels=bl)
W = own[own.gender_cd != 3].groupby(["countyntownship_cd","band"], observed=False).size().unstack(fill_value=0).rename_axis("house_cd")
n_rec = own.groupby("countyntownship_cd").size().rename_axis("house_cd")             # 全部持有紀錄（含法人、年齡不明）
Wn = W.div(n_rec, axis=0)                                              # 各年齡組屋主持有的比例（法人與年齡不明不會死亡，計為 0）
S["m_own"] = (Wn*q_smr.reindex(Wn.index)).sum(axis=1)*100             # 主設定：每百筆持有每年因屋主死亡而釋出的筆數
S["m_own_loc"] = (Wn*q_loc.reindex(Wn.index)).sum(axis=1)*100         # 直接用鄉鎮年齡別死亡率（小鄉鎮較吵）
S["m_own_nat"] = (Wn*q_nat).sum(axis=1)*100                           # 只留屋主年齡結構的差異（死亡率用全國）
S["n_rec"] = n_rec*10; S["smr_death"] = smr; S["owner_80"] = Wn[["80-89","90-100"]].sum(axis=1)
# ---- 存量與買賣
cell = pd.read_csv(OUT/"lvr_cells.csv", dtype={"ab": str, "fl": str}, keep_default_na=False)
allc = cell.groupby("house_cd")[["stock","sales"]].sum().rename(columns={"stock":"S_all","sales":"X_all"})
ex = cell[~cell.ab.isin(["0-9","NA"])].groupby("house_cd")[["stock","sales"]].sum().rename(columns={"stock":"S_ex","sales":"X_ex"})
b = pd.read_parquet(DATA/"bld.parquet", columns=["year","countyntownship_cd","register_reason_group_cd"]); b = b[b.year == 112]
R = b.groupby(["countyntownship_cd","register_reason_group_cd"]).size().unstack(fill_value=0).rename_axis("house_cd")
T = pd.read_csv(OUT/"township_master_112.csv").set_index("house_cd").join(S).join(allc).join(ex).join(R[["B","C","D"]].add_prefix("n_"))
T = T.dropna(subset=["ED","S_all"]).copy(); print("鄉鎮數", len(T))
T["m_naive"] = T.ED/T.S_all*100; T["m"] = T.m_own; T["m65"] = T.deaths65/T.S_all*100           # 每百棟每年的「屋主死亡釋出」
T["bx"] = T.X_ex/T.S_ex*100; T["ba"] = T.X_all/T.S_all*100            # 買賣率：既有建物、全部
T["odds_stock"] = np.log((T.n_D + 0.5)/(T.n_B + 0.5))                   # 存量中 繼承÷買賣
T.reset_index().to_csv(OUT/"township_supply_sales.csv", index=False, encoding="utf-8-sig")
# ---- 檢核：縣市層級 ED÷存量 對 真實繼承登記率
CM = pd.read_csv(OUT/"real_rates_county_mean.csv"); T["rel"] = T.m_own*T.n_rec/100
cc = T.groupby("county").agg(ED=("ED","sum"), d65=("deaths65_5y","sum"), S=("S_all","sum"), rel=("rel","sum"), n_rec=("n_rec","sum"))
CM = CM.merge(cc, left_on="county_name", right_index=True); CM["m_naive"] = CM.ED/CM.S*100; CM["m"] = CM.rel/CM.n_rec*100; CM["m65"] = CM.d65/CM.S*100
chk = {"縣市數": len(CM), "Pearson(log 真實繼承登記率, log ED率)": float(np.corrcoef(np.log(CM.r_D), np.log(CM.m))[0,1]), "Spearman": float(spearmanr(CM.r_D, CM.m)[0]),
       "Pearson(log 真實繼承登記率, log 65歲以上死亡率)": float(np.corrcoef(np.log(CM.r_D), np.log(CM.m65))[0,1]),
       "不含澎湖金門 Pearson": float(np.corrcoef(np.log(CM[~CM.county_name.isin(["澎湖縣","金門縣"])].r_D), np.log(CM[~CM.county_name.isin(["澎湖縣","金門縣"])].m))[0,1]),
       "簡單版 Pearson(log 真實繼承登記率, log ED率)": float(np.corrcoef(np.log(CM.r_D), np.log(CM.m_naive))[0,1]), "簡單版斜率": float(np.polyfit(np.log(CM.m_naive), np.log(CM.r_D), 1)[0]),
       "全國 主設定釋出(筆/年)": float(CM.rel.sum()), "全國 ED(棟/年)": float(CM.ED.sum()), "全國真實繼承登記(棟/年)": float((CM.r_D*CM.stock/100).sum()), "真實÷ED 中位數": float((CM.r_D/CM.m).median()),
       "log(真實÷ED) 的縣市標準差": float(np.log(CM.r_D/CM.m).std()), "log 真實繼承率 對 log ED率 的斜率": float(np.polyfit(np.log(CM.m), np.log(CM.r_D), 1)[0])}
# 四縣市分區的真實繼承登記（單位不一，看縣市內排序）
dz = pd.read_csv(OUT/"real_rates_district_112.csv").merge(T[["m","m65","bx","m_naive"]], left_on="house_cd", right_index=True)
chk["四縣市分區：縣市內 Spearman(真實繼承登記率, ED率)"] = {c: float(spearmanr(g.r_D, g.m)[0]) for c, g in dz.groupby("city")}
chk["四縣市分區：縣市內 Spearman(真實買賣登記率, 實價既有建物買賣率)"] = {c: float(spearmanr(g.r_B, g.bx)[0]) for c, g in dz.groupby("city")}
json.dump(chk, open(OUT/"supply_proxy_check.json","w"), ensure_ascii=False, indent=1); print(json.dumps(chk, ensure_ascii=False, indent=1))
CM.to_csv(OUT/"county_supply_check.csv", index=False, encoding="utf-8-sig")
_dz, _CM = dz, CM
# ---- 鄉鎮層級的拆解：log(m/b) 的變異來自哪一邊
def decomp(df, mcol="m", bcol="bx", within=False):
    la, lb = np.log(df[mcol]), np.log(df[bcol])
    if within: la, lb = la - la.groupby(df.county).transform("mean"), lb - lb.groupby(df.county).transform("mean")
    lo = la - lb; v = np.var(lo)
    return {"鄉鎮數": len(df), "sd(log 釋出率)": la.std(), "sd(log 買賣率)": lb.std(), "相關": np.corrcoef(la, lb)[0,1], "來自買賣率%": np.cov(-lb, lo, bias=True)[0,1]/v*100, "來自釋出率%": np.cov(la, lo, bias=True)[0,1]/v*100}
ok = T[(T.S_ex >= 2000) & (T.X_ex > 0)].copy(); main = ok[~ok.county.isin(ISLANDS)]
rows = {"全部鄉鎮（既有建物存量≥2000棟）": decomp(ok), "不含離島三縣": decomp(main), "不含離島，扣除縣市平均": decomp(main, within=True),
        "不含離島，改用 65 歲以上死亡人數÷存量": decomp(main, "m65"), "不含離島，簡單版 ED÷存量": decomp(main, "m_naive"), "不含離島，鄉鎮年齡別死亡率": decomp(main[main.m_own_loc > 0], "m_own_loc"), "不含離島，只留屋主年齡結構": decomp(main, "m_own_nat"), "不含離島，買賣含新成屋": decomp(main, "m", "ba"), "不含離島，既有建物存量≥5000棟": decomp(main[main.S_ex >= 5000])}
rng = np.random.default_rng(7); bs = np.array([[decomp(main.iloc[rng.integers(0, len(main), len(main))])[k] for k in ["相關","來自買賣率%"]] for _ in range(2000)])
VD = pd.DataFrame(rows).T; VD.to_csv(OUT/"township_supply_sales_decomp.csv", encoding="utf-8-sig"); print(VD.round(3).to_string())
boot = {"相關_95%": np.percentile(bs[:,0], [2.5,97.5]).tolist(), "來自買賣率%_95%": np.percentile(bs[:,1], [2.5,97.5]).tolist()}
# 以縣市為單位重抽（鄉鎮在縣市內相關）
cts = main.county.unique(); grp = {c: main[main.county == c] for c in cts}
bc = np.array([[decomp(pd.concat([grp[c] for c in rng.choice(cts, len(cts))]))[k] for k in ["相關","來自買賣率%"]] for _ in range(1000)])
boot["縣市群集重抽_相關_95%"] = np.percentile(bc[:,0], [2.5,97.5]).tolist(); boot["縣市群集重抽_來自買賣率%_95%"] = np.percentile(bc[:,1], [2.5,97.5]).tolist()
# ---- 模擬存量的繼承占比，對得上哪一邊？
import statsmodels.api as sm
def reg(df, y, within=False, w=None):
    X = pd.DataFrame({"log釋出率": np.log(df.m), "log買賣率": np.log(df.bx)}); yy = df[y]
    if within: X = X - X.groupby(df.county).transform("mean"); yy = yy - yy.groupby(df.county).transform("mean")
    Z = (X - X.mean())/X.std(); f = sm.OLS((yy - yy.mean())/yy.std(), sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": df.county})
    f2 = sm.OLS(yy, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": df.county})
    return {"n": len(df), "R2": f.rsquared, "標準化係數_釋出率": f.params["log釋出率"], "標準化係數_買賣率": f.params["log買賣率"], "se_釋出率": f.bse["log釋出率"], "se_買賣率": f.bse["log買賣率"],
            "原始斜率_釋出率": f2.params["log釋出率"], "原始斜率_買賣率": f2.params["log買賣率"], "單獨R2_釋出率": np.corrcoef(X["log釋出率"], yy)[0,1]**2, "單獨R2_買賣率": np.corrcoef(X["log買賣率"], yy)[0,1]**2,
            "Spearman_釋出率": spearmanr(X["log釋出率"], yy)[0], "Spearman_買賣率": spearmanr(X["log買賣率"], yy)[0]}
RG = pd.DataFrame({"存量 log(繼承÷買賣)": reg(main, "odds_stock"), "存量 log(繼承÷買賣)，縣市內": reg(main, "odds_stock", True),
                   "原始繼承占比 logit": reg(main, "logit_raw"), "原始繼承占比 logit，縣市內": reg(main, "logit_raw", True),
                   "地方效果 u（扣除房屋組成）": reg(main, "u"), "地方效果 u，縣市內": reg(main, "u", True)}).T
RG.to_csv(OUT/"township_stock_vs_flows.csv", encoding="utf-8-sig"); print(RG.round(3).to_string())
# 穩態下 存量 log(繼承÷買賣) 應約等於 log(釋出率÷買賣率)＋常數
lo = np.log(main.m/main.bx); boot["存量log勝算 對 log(釋出÷買賣) 的相關"] = float(np.corrcoef(lo, main.odds_stock)[0,1]); boot["斜率"] = float(np.polyfit(lo, main.odds_stock, 1)[0]); boot["Spearman"] = float(spearmanr(lo, main.odds_stock)[0])
# 橫軸不含任何模擬資料的版本：65 歲以上死亡人數 ÷ 存量（純戶政資料）對存量 log 勝算
lo65 = np.log(main.m65/main.bx); boot["純真實版(65歲以上死亡率÷買賣率)_相關"] = float(np.corrcoef(lo65, main.odds_stock)[0,1]); boot["純真實版_Spearman"] = float(spearmanr(lo65, main.odds_stock)[0])
# 控制人口規模（log 戶籍人口）後的偏相關：存量 log 勝算 對 log(釋出÷買賣)；以及 u 對真實老屋買賣率（§3.5）
def _partial(x_, y_, z_):
    X_ = sm.add_constant(np.asarray(z_, float)); rx = np.asarray(x_, float) - X_ @ np.linalg.lstsq(X_, np.asarray(x_, float), rcond=None)[0]; ry = np.asarray(y_, float) - X_ @ np.linalg.lstsq(X_, np.asarray(y_, float), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0,1]), float(spearmanr(rx, ry)[0])
boot["控制log人口_存量log勝算對log(釋出÷買賣)_Pearson, Spearman"] = _partial(main.odds_stock, lo, main.lp)
boot["只用log人口預測存量log勝算_相關"] = float(abs(np.corrcoef(main.lp, main.odds_stock)[0,1]))
LT = pd.read_csv(OUT/"lvr_township.csv"); LT = LT[(LT.lv_stock_old >= 1000)].dropna(subset=["u","turn_old","lp"]); LT = LT[LT.turn_old > 0]
boot["控制log人口_u對log老屋買賣率_Pearson, Spearman（鄉鎮數）"] = [*_partial(LT.u, np.log(LT.turn_old), LT.lp), int(len(LT))]
boot["u對log老屋買賣率_未控制_Pearson, Spearman"] = [float(np.corrcoef(LT.u, np.log(LT.turn_old))[0,1]), float(spearmanr(LT.u, LT.turn_old)[0])]
boot["縣市內相關"] = float(np.corrcoef(lo - lo.groupby(main.county).transform("mean"), main.odds_stock - main.odds_stock.groupby(main.county).transform("mean"))[0,1])
# ---- 拆解比例對樣本、加權與代理指標的敏感度（稽核三）
def share_w(la, lb, w=None):
    w = np.ones(len(la)) if w is None else np.asarray(w, float); lo = la - lb; c = lambda a_, b_: np.average((a_ - np.average(a_, weights=w))*(b_ - np.average(b_, weights=w)), weights=w)
    return c(-lb, lo)/c(lo, lo)*100
lm, lbx = np.log(main.m), np.log(main.bx)
SEN = {f"既有建物存量≥{k:,}棟（{int((ok2.S_ex >= k).sum())} 個鄉鎮）": share_w(np.log(ok2[ok2.S_ex >= k].m), np.log(ok2[ok2.S_ex >= k].bx)) for ok2 in [T[(T.X_ex > 0) & ~T.county.isin(ISLANDS)]] for k in [500, 2000, 5000, 10000, 20000]}
SEN["以建物存量加權（272 個鄉鎮）"] = share_w(lm, lbx, main.S_all)
sl = lambda y_: np.polyfit(main.lp, y_, 1)[0]; SEN["只看隨人口規模變動的部分（對 log 人口的斜率）"] = -sl(lbx)/(sl(lm) - sl(lbx))*100
# 用真實繼承登記檢查：同一批地區，真實繼承率與代理指標各自得到的拆解
lc = pd.read_csv(OUT/"lvr_county.csv")[["county_name","r_B_exist"]]; C2 = _CM.merge(lc, on="county_name"); C19 = C2[~C2.county_name.isin(["澎湖縣","金門縣"])]
dmn = lambda x_, g_: np.log(x_) - np.log(x_).groupby(g_).transform("mean"); z = _dz[(_dz.r_D > 0) & (_dz.bx > 0)]
CHK = pd.DataFrame({"21 縣市": {"真實繼承登記": share_w(np.log(C2.r_D), np.log(C2.r_B_exist)), "代理指標（主設定）": share_w(np.log(C2.m), np.log(C2.r_B_exist)), "簡單版": share_w(np.log(C2.m_naive), np.log(C2.r_B_exist))},
     "19 縣市（不含澎湖金門）": {"真實繼承登記": share_w(np.log(C19.r_D), np.log(C19.r_B_exist)), "代理指標（主設定）": share_w(np.log(C19.m), np.log(C19.r_B_exist)), "簡單版": share_w(np.log(C19.m_naive), np.log(C19.r_B_exist))},
     f"四縣市 {len(z)} 區（扣除縣市平均）": {"真實繼承登記": share_w(dmn(z.r_D, z.city), dmn(z.bx, z.city)), "代理指標（主設定）": share_w(dmn(z.m, z.city), dmn(z.bx, z.city)), "簡單版": share_w(dmn(z.m_naive, z.city), dmn(z.bx, z.city))}}).T
CHK["相關(真實繼承率, 買賣率)"] = [np.corrcoef(np.log(C2.r_D), np.log(C2.r_B_exist))[0,1], np.corrcoef(np.log(C19.r_D), np.log(C19.r_B_exist))[0,1], np.corrcoef(dmn(z.r_D, z.city), dmn(z.bx, z.city))[0,1]]
CHK["相關(代理指標, 買賣率)"] = [np.corrcoef(np.log(C2.m), np.log(C2.r_B_exist))[0,1], np.corrcoef(np.log(C19.m), np.log(C19.r_B_exist))[0,1], np.corrcoef(dmn(z.m, z.city), dmn(z.bx, z.city))[0,1]]
CHK["代理sd÷真實sd"] = [np.log(C2.m).std()/np.log(C2.r_D).std(), np.log(C19.m).std()/np.log(C19.r_D).std(), dmn(z.m, z.city).std()/dmn(z.r_D, z.city).std()]
pd.Series(SEN, name="來自買賣率%").to_csv(OUT/"township_decomp_sensitivity.csv", encoding="utf-8-sig"); CHK.to_csv(OUT/"proxy_vs_real_decomp.csv", encoding="utf-8-sig"); print(pd.Series(SEN).round(1).to_string()); print(CHK.round(2).to_string())
boot["對log人口的斜率_存量log勝算"] = float(sl(main.odds_stock)); boot["對log人口的斜率_流量log比"] = float(sl(lm - lbx))
# 人口五等分
main = main.copy(); main["pq"] = pd.qcut(main["pop"], 5, labels=["最少20%","次少","中間","次多","最多20%"])
Q = main.groupby("pq", observed=True).apply(lambda g: pd.Series({"鄉鎮數": len(g), "每百筆持有每年屋主死亡釋出": g.rel.sum()/g.n_rec.sum()*100, "簡單版(死亡×全國持有率÷存量)": g.ED.sum()/g.S_all.sum()*100, "既有建物買賣率%": g.X_ex.sum()/g.S_ex.sum()*100,
     "釋出÷買賣": (g.rel.sum()/g.n_rec.sum())/(g.X_ex.sum()/g.S_ex.sum()), "存量 繼承÷買賣": g.n_D.sum()/g.n_B.sum(), "65歲以上人口%": g.pop65_ris.sum()/g.pop_ris.sum()*100, "每人建物棟數": g.S_all.sum()/g.pop_ris.sum(), "標準化死亡比": g.deaths.sum()/(g.deaths/g.smr_death).sum(), "80歲以上屋主持有%": (g.owner_80*g.n_rec).sum()/g.n_rec.sum()*100}), include_groups=False)
Q.to_csv(OUT/"township_supply_sales_quintiles.csv", encoding="utf-8-sig"); print(Q.round(3).to_string())
json.dump(boot, open(OUT/"township_supply_sales_boot.json","w"), ensure_ascii=False, indent=1); print(json.dumps(boot, ensure_ascii=False, indent=1))
# ---------- 圖 7（原圖 9） ----------
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
a = ax[0]; x_, y_ = main.m/main.bx, np.exp(main.odds_stock); a.scatter(x_, y_, s=np.sqrt(main.S_all)/10, color=C_INH, alpha=0.45, lw=0)
c0 = np.mean(np.log(y_) - np.log(x_)); xs = np.array([0.2, 4.5]); a.plot(xs, np.exp(c0)*xs, color="#333", lw=1.1, label="斜率 1 的參考線")
a.set_xscale("log"); a.set_yscale("log"); a.minorticks_off(); a.set_xticks([0.25, 0.5, 1, 2, 4]); a.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); a.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); a.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
a.set_xlabel("屋主期望死亡率 ÷ 既有建物買賣率"); a.set_ylabel("模擬存量：繼承棟數 ÷ 買賣棟數"); a.legend(frameon=False, loc="upper left")
a.set_title(f"每點一個鄉鎮（{len(main)} 個，相關 {boot['存量log勝算 對 log(釋出÷買賣) 的相關']:.2f}，對數尺度）", loc="left", fontsize=12)
a = ax[1]; x = np.arange(5); a.bar(x - 0.2, Q["每百筆持有每年屋主死亡釋出"], 0.38, color=C_INH, label="屋主期望死亡率（每百筆持有）"); a.bar(x + 0.2, Q["既有建物買賣率%"], 0.38, color=C_BUY, label="既有建物買賣（每百棟）")
for i in range(5):
    for dx, col in [(-0.2, "每百筆持有每年屋主死亡釋出"), (0.2, "既有建物買賣率%")]: a.text(x[i] + dx, Q[col].iloc[i] + 0.04, f"{Q[col].iloc[i]:.2f}", ha="center", va="bottom", rotation=90, fontsize=11)
a.set_xticks(x); a.set_xticklabels(["最少\n20%", "次少", "中間", "次多", "最多\n20%"]); a.set_xlabel("鄉鎮市區依戶籍人口分五組"); a.set_ylabel("每年發生數（每百）"); a.set_ylim(0, 2.75); a.legend(frameon=False, loc="upper left")
a.set_title("人口越少：期望死亡率高四成，買賣率低兩成", loc="left", fontsize=12)
fig.suptitle("真實的死亡與買賣流量，與模擬存量裡「繼承對買賣」的比例高度相關", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig07_township_flows.png"); plt.close(); print("fig07 ok")
