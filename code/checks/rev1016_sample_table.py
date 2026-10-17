"""rev1016 審稿建議 15：分析樣本對照表＋「建物棟數 × 10」的依據核對。
每一列的鄉鎮數都由資料或既有輸出依原程式的篩選條件重算（不是抄報告）。
輸出 out/rev1016_sample_table.csv、out/rev1016_sample_table.md；×10 的核對數字存到 scratchpad，由 rev1016_bootstrap_summary.py 併入 out/rev1016_bootstrap.json 的 "scale_x10" 欄。"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from common import *
SCR = Path("./scratch/r1016")
b = pd.read_parquet(DATA/"bld.parquet"); town = pd.read_csv(DATA/"town.csv")
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet")
own_n = pd.read_parquet(DATA/"own.parquet", columns=["year", "building_sn", "countyntownship_cd", "county_cd"])
chk = {}
# ---------------- 鄉鎮數重算 ----------------
n_y = {y: d[d.year == y].groupby("house_cd").size() for y in [110, 111, 112]}
N = {}
N["codes_112"] = int(d[d.year == 112].house_cd.nunique()); N["codes_all_bld_112"] = int(b[b.year == 112].countyntownship_cd.nunique())
for y in [110, 111, 112]: N[f"main_{y}"] = int((n_y[y] >= 50).sum()); N[f"bld_{y}"] = int(n_y[y][n_y[y] >= 50].sum())
N["bld_112_all_BCD_ageknown"] = int((d.year == 112).sum())
m3 = pd.concat([n_y[y].rename(y) for y in [110, 111, 112]], axis=1).dropna(); N["cross_year_min300"] = int((m3.min(axis=1) >= 300).sum())
N["oos_112_min300"] = int((n_y[112] >= 300).sum())
y112 = d[d.year == 112].merge(town, on="house_cd")
N["rb_no_islands"] = int((y112[~y112.county.isin(ISLANDS)].groupby("house_cd").size() >= 50).sum())
N["rb_BD_only"] = int((y112[y112.reason.isin(["B", "D"])].groupby("house_cd").size() >= 50).sum())
N["rb_old_lowrise"] = int((y112[(y112.age >= 40) & (y112.floor_group_cd == "1")].groupby("house_cd").size() >= 50).sum())
N["rb_under40"] = int((y112[y112.age < 40].groupby("house_cd").size() >= 50).sum())
N["rb_incl_unknown_age"] = int(pd.read_csv(OUT/"robustness_include_unknown_age.csv")["鄉鎮數"].iloc[0]); N["rb_incl_unknown_age_bld"] = int(pd.read_csv(OUT/"robustness_include_unknown_age.csv")["棟數"].iloc[0])
seg = pd.read_csv(OUT/"place_effect_by_segment.csv"); N["segments"] = seg.set_index("比較")["鄉鎮數(兩類皆≥300棟)"].astype(int).to_dict()
tm = pd.read_csv(OUT/"township_master_112.csv"); N["master_112"] = len(tm)
X = ["lp", "owner_65", "res_65", "res_local", "res_move"]; N["correlates_multivar"] = int(tm.assign(lp=np.log2(tm["pop"])).dropna(subset=X + ["u"]).shape[0])
N["moran_islands"] = json.load(open(OUT/"moran.json"))["孤島數"]
S = pd.read_csv(OUT/"township_supply_sales.csv"); N["supply_all"] = len(S)
ok = S[(S.S_ex >= 2000) & (S.X_ex > 0)]; main = ok[~ok.county.isin(ISLANDS)]; N["flows_272"] = len(main); N["flows_ok_incl_islands"] = len(ok)
N["flows_272_bld_share_of_all_112"] = float(main.S_all.sum()/S.S_all.sum())
N["flows_272_share_of_national_sim_bld_112"] = float(main.S_all.sum()/10/(b.year == 112).sum())
for k in [500, 2000, 5000, 10000, 20000]: N[f"flows_Sex_ge_{k}"] = int(((S.X_ex > 0) & ~S.county.isin(ISLANDS) & (S.S_ex >= k)).sum())
L = pd.read_csv(OUT/"lvr_township.csv"); N["lvr_township"] = len(L)
N["lvr_old_244"] = int(L[(L.lv_stock_old >= 1000)][["u", "turn_old"]].dropna().shape[0])
N["lvr_old_244_turn_pos"] = int(L[(L.lv_stock_old >= 1000) & (L.turn_old > 0)].dropna(subset=["u", "turn_old", "lp"]).shape[0])
N["lvr_old_noKinmen"] = int(L[(L.lv_stock_old >= 1000) & (L.county != "金門縣")][["u", "turn_old"]].dropna().shape[0])
s = b[b.year == 112]; sb = s[(s.materials_group_cd == "B") & (s.floor_group_cd == "1") & (s.age >= 40)].groupby("countyntownship_cd").size()*10
N["brick_262"] = int(L.house_cd.map(sb).ge(500).sum())
N["townhouse_146"] = int(pd.read_csv(OUT/"lvr_townhouse_only.csv").iloc[-1]["鄉鎮數"])
N["typology_308"] = int((L.county != "金門縣").sum()); N["typology_kinmen"] = int((L.county == "金門縣").sum())
G = pd.read_csv(OUT/"typology.csv"); N["typology_sum"] = int(G["鄉鎮數"].sum())
gv = pd.read_csv(OUT/"gift_vs_inherit_place_effect.csv"); N["gift_years"] = gv.set_index("年度")["鄉鎮數"].astype(int).to_dict()
CM = pd.read_csv(OUT/"real_rates_county_mean.csv"); N["county_21"] = len(CM); N["county_21_names_missing"] = sorted(set(pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str).county_name) - set(CM.county_name))
DZ = pd.read_csv(OUT/"real_rates_district_112.csv"); N["districts"] = len(DZ); N["districts_by_city"] = DZ.groupby("city").size().to_dict(); N["districts_with_u"] = int(DZ.u.notna().sum())
N["districts_units"] = DZ.groupby("city").unit.first().to_dict()
N["cohort_counties"] = json.load(open(OUT/"cohort_county_fail.json"))["n_county"]
N["popq_312"] = int(pd.read_csv(OUT/"population_quintiles.csv")["鄉鎮數"].sum()); N["popq_272"] = int(pd.read_csv(OUT/"township_supply_sales_quintiles.csv")["鄉鎮數"].sum()); N["popq_lvr"] = int(pd.read_csv(OUT/"lvr_population_quintiles.csv")["鄉鎮數"].sum())
N["oos_eval"] = int(pd.read_csv(OUT/"oos_prediction_112.csv")["評估鄉鎮數(≥300棟)"].iloc[0])
chk["counts"] = N
print(json.dumps(N, ensure_ascii=False, indent=1, default=float))

# ---------------- ×10 的依據 ----------------
X10 = {}
rows = own_n.groupby("year").size(); blds = b.groupby("year").size()
X10["手冊"] = {"屋主資料母體(111年6月底)筆數": 10766461, "模擬筆數(手冊)": 1076646, "本檔111年筆數": int(rows[111]), "筆數比": 1076646/10766461,
              "說明": "手冊只說『依 111 年屋主資料之整體結構，模擬約 10% 之樣本資料』，並列出母體與模擬的筆數（持有紀錄）；沒有母體建物總棟數。表 29（以建物數計算）只列各類別的占比。"}
X10["每年_持有紀錄÷去重建物"] = {int(y): {"持有紀錄": int(rows[y]), "去重建物": int(blds[y]), "紀錄÷建物": float(rows[y]/blds[y])} for y in rows.index}
# 依分組的紀錄÷建物（112）：若母體的共有程度與模擬不同，建物的倍率就不會等於 10
o112 = own_n[own_n.year == 112].groupby("building_sn").size().rename("k")
bb = s.merge(o112, left_on="building_sn", right_index=True, how="left")
bb["ab"] = pd.cut(bb.age, [-1, 9, 19, 29, 39, 200], labels=["0-9", "10-19", "20-29", "30-39", "40+"]).astype(str).replace("nan", "不明")
X10["112_紀錄÷建物_依登記原因"] = bb.groupby("reason").k.mean().round(3).to_dict()
X10["112_紀錄÷建物_依屋齡"] = bb.groupby("ab").k.mean().round(3).to_dict()
X10["112_紀錄÷建物_依縣市"] = bb.merge(town[["house_cd", "county"]], left_on="countyntownship_cd", right_on="house_cd").groupby("county").k.mean().round(3).to_dict()
tq = tm[["house_cd", "pop"]].copy(); tq["pq"] = pd.qcut(tq["pop"], 5, labels=["最少20%", "次少", "中間", "次多", "最多20%"])
X10["112_紀錄÷建物_依鄉鎮人口五等分"] = bb.merge(tq, left_on="countyntownship_cd", right_on="house_cd").groupby("pq", observed=True).k.mean().round(3).to_dict()
# 手冊表 27（以筆數）與表 29（以建物數）的母體／模擬占比：
#  建物倍率_j ∝ 母體建物占比_j ÷ 模擬建物占比_j （表 29）；母體紀錄÷建物_j ∝ 表27占比_j ÷ 表29占比_j
txt = open(DATA/"manual.txt", encoding="utf-8").read().splitlines()
def parse(a, b_):
    out, field = [], None
    for line in txt[a:b_]:
        m = re.match(r"^\s*(\S*?[一-龥]+\S*)?\s+(-1\(NULL\)|\d+\(NULL\)|[A-Z]\d\d|009|[A-Z]|\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if not m: continue
        if m[1]: field = m[1]
        out.append((field, m[2], float(m[3]), float(m[4])))
    return pd.DataFrame(out, columns=["field", "cat", "orig", "sim"])
i27 = next(i for i, l in enumerate(txt) if "表 27 原始資料與模擬資料之比較" in l and i > 2000); i28 = next(i for i, l in enumerate(txt) if "表 28 原始資料與模擬資料之比較" in l and i > 2000)
i29 = next(i for i, l in enumerate(txt) if "表 29 原始資料與模擬資料之比較" in l and i > 2000); i30 = next(i for i, l in enumerate(txt) if "表 30 原始資料與模擬資料之比較" in l and i > 2000)
T27, T29 = parse(i27, i28), parse(i29, i30)
tw29 = T29[T29.field.str.contains("鄉鎮")].set_index("cat"); tw27 = T27[T27.field.str.contains("鄉鎮")].set_index("cat")
X10["表29_鄉鎮列數"] = len(tw29); X10["表29_占比合計"] = [float(tw29.orig.sum()), float(tw29.sim.sum())]
# 先核對：表 29 的「模擬占比」是否就是本檔 111 年去重建物的鄉鎮占比
sim111 = b[b.year == 111].countyntownship_cd.value_counts(normalize=True)*100
j = tw29.join(sim111.rename("ours"), how="inner"); X10["表29模擬占比_對_本檔111去重建物占比"] = {"鄉鎮數": len(j), "最大絕對差(百分點)": float((j.sim - j.ours).abs().max()), "相關": float(np.corrcoef(j.sim, j.ours)[0, 1])}
simrec111 = own_n[own_n.year == 111].countyntownship_cd.value_counts(normalize=True)*100
j2 = tw27.join(simrec111.rename("ours"), how="inner"); X10["表27模擬占比_對_本檔111持有紀錄占比"] = {"鄉鎮數": len(j2), "最大絕對差(百分點)": float((j2.sim - j2.ours).abs().max())}
big = tw29[tw29.orig >= 0.3]                                                 # 占比到小數第二位，只看占比 ≥0.3% 的鄉鎮（捨入誤差 ≤ ±1.7%）
rr = (big.orig/big.sim)
X10["表29_母體÷模擬建物占比(相對倍率)_占比≥0.3%的鄉鎮"] = {"鄉鎮數": len(big), "P10": float(rr.quantile(.1)), "中位數": float(rr.median()), "P90": float(rr.quantile(.9)), "最小": float(rr.min()), "最大": float(rr.max())}
cty = tw29.assign(c=tw29.index.str[0]).groupby("c")[["orig", "sim"]].sum(); cty["相對倍率"] = cty.orig/cty.sim
cc = pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str).set_index("county_cd").county_name
X10["表29_縣市相對倍率(母體占比÷模擬占比)"] = {cc.get(k, k): round(float(v), 3) for k, v in cty["相對倍率"].items()}
rF = float(cty.loc["F", "相對倍率"]); X10["由新北9.40倍推回全國建物倍率"] = {"新北相對倍率(表29)": rF, "新北實際/模擬(112年6月月報)": 9.3997, "推得全國倍率≈9.40/相對倍率": 9.3997/rF,
     "註": "表 29 是 111 年版、月報是 112 年 6 月；月報總棟數可能含公有與公設建號，所以只是量級。"}
# 母體的紀錄÷建物（相對）＝ 表27占比 ÷ 表29占比；模擬同理；若兩者一致，相對倍率就只來自建物分布
jj = tw27.join(tw29, lsuffix="27", rsuffix="29", how="inner"); jj = jj[jj.orig29 >= 0.3]
X10["鄉鎮_紀錄÷建物(相對值)_母體對模擬"] = {"鄉鎮數": len(jj), "相關": float(np.corrcoef(jj.orig27/jj.orig29, jj.sim27/jj.sim29)[0, 1]),
     "母體相對值的P10-P90": [float((jj.orig27/jj.orig29).quantile(.1)), float((jj.orig27/jj.orig29).quantile(.9))], "模擬相對值的P10-P90": [float((jj.sim27/jj.sim29).quantile(.1)), float((jj.sim27/jj.sim29).quantile(.9))]}
age29 = T29[T29.field.str.contains("屋齡")].copy(); nul = age29[age29.cat.str.contains("NULL")]
age29 = age29[~age29.cat.str.contains("NULL")].copy(); age29["a"] = age29.cat.astype(int)
X10["表29_屋齡不明占比(母體,模擬)"] = [float(nul.orig.iloc[0]), float(nul.sim.iloc[0])]
def grp_ratio(lo, hi):
    g = age29[(age29.a >= lo) & (age29.a <= hi)]; return float((g.orig.sum()/age29.orig.sum())/(g.sim.sum()/age29.sim.sum()))
X10["表29_屋齡已知中各屋齡帶的相對倍率(母體占比÷模擬占比，已排除不明)"] = {"0-9": grp_ratio(0, 9), "10-19": grp_ratio(10, 19), "20-29": grp_ratio(20, 29), "30-39": grp_ratio(30, 39), "40-49": grp_ratio(40, 49), "50+": grp_ratio(50, 200), "10+（既有建物）": grp_ratio(10, 200)}
X10["表29_屋齡不明在母體占4.13%而模擬為0：若母體的不明主要是老屋，老屋的實際倍率還要再高"] = True
for f in ["建材", "樓層", "用途"]:
    g = T29[T29.field.str.contains(f)]; X10[f"表29_{f}_相對倍率"] = {r.cat: round(r.orig/r.sim, 3) for r in g.itertuples() if r.sim > 0}
chk["scale_x10"] = X10
print(json.dumps(X10, ensure_ascii=False, indent=1, default=float))
json.dump(chk, open(SCR/"sample_table_checks.json", "w"), ensure_ascii=False, indent=1, default=float)

# ---------------- 樣本對照表 ----------------
R = []
def add(**k): R.append(k)
add(分析="地方效果 u（問題一主分析；分型、§3.3、§3.5 的 u 都來自這裡）", 鄉鎮數=f"{N['main_112']}（110 年 {N['main_110']}、111 年 {N['main_111']}）",
    資料期間="模擬 112 年（組成模型以 110–111 年訓練）", 建物條件=f"登記原因為買賣/贈與/繼承、屋齡已知、每棟一筆；鄉鎮該類建物 ≥50 棟（112 年 {N['bld_112']:,} 棟；未篩前 {N['bld_112_all_BCD_ageknown']:,} 棟、{N['codes_112']} 個鄉鎮碼）",
    分子="模擬：繼承棟數 O_j（期望 E_j 由 LightGBM 預測加總）", 分母="模擬：買賣+贈與+繼承棟數 n_j", 主要限制="屋齡不明（112 年 4.2%，繼承類 10.6%，小鄉鎮達 25%）被排除；新竹市、嘉義市各算一單位；鄉鎮平均 offset 低估 τ",
    章節="§2.2、§3.2、A.1、B.1", 程式="12_composition_vs_place.py、glmm.py", 核對來源="data/bcd_with_pcomp.parquet 依 n≥50 重算；out/composition_vs_place_summary.csv")
add(分析="跨年度穩定（log(O/E) 兩兩相關 0.85–0.87）", 鄉鎮數=str(N["cross_year_min300"]), 資料期間="模擬 110、111、112 年", 建物條件="同上，且三個年度都 ≥300 棟", 分子="模擬：繼承棟數", 分母="模擬：B+C+D 棟數",
    主要限制="只含大鄉鎮；三年共用生成規則，不是三次獨立抽樣", 章節="§3.2", 程式="12_composition_vs_place.py", 核對來源="重算；out/audit_composition.json n_town_min300")
add(分析="樣本外預測（110–111 估計、預測 112）", 鄉鎮數=str(N["oos_112_min300"]), 資料期間="模擬 110–111 → 112", 建物條件="112 年該類建物 ≥300 棟的鄉鎮（等權）", 分子="模擬：繼承棟數", 分母="模擬：B+C+D 棟數",
    主要限制="只含大鄉鎮；誤差以百分點計", 章節="§3.2、B.1（只留 ≥300 棟）", 程式="12_composition_vs_place.py、16_robustness_and_replication.py", 核對來源="重算；out/oos_prediction_112.csv 評估鄉鎮數")
add(分析="穩健性：分母只留買賣＋繼承／排除離島／40 年以上 1–5 層／未滿 40 年／納入屋齡不明", 鄉鎮數=f"{N['rb_BD_only']}／{N['rb_no_islands']}／{N['rb_old_lowrise']}／{N['rb_under40']}／{N['rb_incl_unknown_age']}",
    資料期間="模擬 112 年", 建物條件=f"各子樣本內 ≥50 棟（納入屋齡不明者 {N['rb_incl_unknown_age_bld']:,} 棟）", 分子="模擬：繼承棟數", 分母="模擬：子樣本的棟數", 主要限制="子樣本不同，鄉鎮集合不同；與主設定比較時只取交集",
    章節="B.1、§4.3/附錄 E（317）", 程式="16_robustness_and_replication.py", 核對來源="重算；out/robustness_place_effect.csv、robustness_include_unknown_age.csv")
add(分析="地方效果是否各類房屋一致", 鄉鎮數="／".join(f"{v}" for v in N["segments"].values()) + "（" + "、".join(N["segments"].keys()) + "）", 資料期間="模擬 112 年", 建物條件="兩類都 ≥300 棟", 分子="模擬：繼承棟數", 分母="模擬：B+C+D 棟數",
    主要限制="小樣本雜訊壓低相關", 章節="B.2", 程式="16_robustness_and_replication.py", 核對來源="out/place_effect_by_segment.csv")
add(分析="地方效果與鄉鎮特徵、人口五等分、Moran's I", 鄉鎮數=f"{N['master_112']}（多變數迴歸 {N['correlates_multivar']}；地圖只繪本島，Moran 鄰接矩陣孤島 {N['moran_islands']} 個）", 資料期間="模擬 112 年屋主檔、人建地檔；112/6 戶籍人口",
    建物條件="同主分析", 分子="—（u 為應變數）", 分母="—", 主要限制="生態相關；屋主特徵只在鄉鎮層級有效", 章節="§3.3、B.8", 程式="13a_township_covariates.py、13b_place_correlates.py", 核對來源="out/township_master_112.csv、place_effect_models.csv、moran.json")
add(分析="問題二主迴歸：u ~ 買賣率＋釋出率（含／不含 u_C）、流量拆解、圖 7–8", 鄉鎮數=f"{N['flows_272']}（含離島時 {N['flows_ok_incl_islands']}）", 資料期間="實價登錄 111S3–112S4（年化 ÷1.5）；死亡 110–112 平均；人口 112/6；模擬 112 年存量與屋主檔",
    建物條件=f"本島（排除澎湖、金門、連江）；模擬既有建物（屋齡 ≥10 年）×10 ≥2,000 棟，且實價登錄既有建物買賣 >0；涵蓋 312 個分析鄉鎮的模擬建物 {N['flows_272_bld_share_of_all_112']*100:.1f}%（全國模擬建物 {N['flows_272_share_of_national_sim_bld_112']*100:.1f}%）",
    分子="真實：實價登錄屋齡 ≥10 年建物買賣棟數；真實：死亡人數（轉成年齡別死亡率×SMR）", 分母="模擬：屋齡 ≥10 年建物 ×10（買賣率）；模擬：持有紀錄（屋主年齡比例，法人與年齡不明計為不會死亡）",
    主要限制="買賣率分母是模擬存量×10；釋出率是代理指標（死亡按戶籍地、屋主按建物所在地）；兩率的單位（棟、筆）不同", 章節="§3.4、B.4、B.9；5 頁版 §3", 程式="17_real_sales_lvr.py、21_township_supply_vs_sales.py、25_gift_place_effect.py",
    核對來源="out/township_supply_sales.csv 依篩選重算")
add(分析="流量拆解的門檻敏感度", 鄉鎮數=f"{N['flows_Sex_ge_500']}／{N['flows_Sex_ge_2000']}／{N['flows_Sex_ge_5000']}／{N['flows_Sex_ge_10000']}／{N['flows_Sex_ge_20000']}（既有建物 ≥500／2,000／5,000／10,000／20,000）",
    資料期間="同上", 建物條件="本島、買賣 >0、依門檻", 分子="同上", 分母="同上", 主要限制="門檻越高越偏大鄉鎮", 章節="B.4", 程式="21_township_supply_vs_sales.py", 核對來源="重算；out/township_decomp_sensitivity.csv")
add(分析="贈與 u_C、買賣 u_B 的地方效果", 鄉鎮數="／".join(str(v) for v in N["gift_years"].values()) + "（110／111／112）；對流量迴歸 272", 資料期間="模擬 110–112 年",
    建物條件="同主分析（結果換成『是贈與』『是買賣』）", 分子="模擬：贈與／買賣棟數", 分母="模擬：B+C+D 棟數", 主要限制="u_B 與 u_D 共用分母，相關幾乎是機械的", 章節="§3.4、B.9", 程式="25_gift_place_effect.py", 核對來源="out/gift_vs_inherit_place_effect.csv")
add(分析="真實老屋買賣率 vs u（全國帶屋齡驗證）", 鄉鎮數=f"{N['lvr_old_244']}（不含金門 {N['lvr_old_noKinmen']}）", 資料期間="實價登錄 111S3–112S4；模擬 112 年存量",
    建物條件="老屋＝屋齡 ≥40 年且 1–5 層；模擬老屋存量 ×10 ≥1,000 棟（即模擬 ≥100 棟）", 分子="真實：實價登錄老屋買賣棟數（年化）", 分母="模擬：同鄉鎮×屋齡組×樓層組的建物 ×10（全部登記原因）",
    主要限制="產品不同（小鄉鎮透天、大鄉鎮公寓）；分母×10 未經母體驗證；約 4% 為未登記建物；與 §3.4 共用分子", 章節="§3.5、B.6；5 頁版 §3", 程式="17_real_sales_lvr.py", 核對來源="out/lvr_township.csv 依篩選重算")
add(分析="限磚造老屋", 鄉鎮數=str(N["brick_262"]), 資料期間="同上", 建物條件="模擬磚造（B）、1–5 層、≥40 年 ×10 ≥500 棟", 分子="真實：主要建材含『磚』的老屋買賣", 分母="模擬：磚造老屋 ×10",
    主要限制="模擬『其他』建材在小鄉鎮占老屋近三成、真實買賣只占 3%（登記口徑可能不同）", 章節="§3.5、B.6", 程式="17_real_sales_lvr.py", 核對來源="重算；out/lvr_brick_only.csv")
add(分析="限老屋買賣幾乎都是透天厝的鄉鎮", 鄉鎮數=str(N["townhouse_146"]), 資料期間="同上", 建物條件="老屋買賣中透天 ≥85% 且老屋存量 ×10 ≥1,000", 分子="真實：老屋買賣", 分母="模擬：老屋 ×10",
    主要限制="以結果（買賣型態）選樣", 章節="§3.5", 程式="17_real_sales_lvr.py", 核對來源="out/lvr_townhouse_only.csv")
add(分析="鄉鎮分型（甲–戊）", 鄉鎮數=f"{N['typology_308']}（312 扣除金門 {N['typology_kinmen']}）", 資料期間="模擬 112 年；老屋買賣率為事後對照", 建物條件="同主分析（≥50 棟）；組成預期占比中位數在 312 個上計算",
    分子="同主分析", 分母="同主分析", 主要限制="分型依 95% 後驗區間是否含 0，邊界鄉鎮對重抽敏感（見 rev1016_bootstrap）", 章節="§4.2、5 頁版 §3 表", 程式="18_typology.py、23_typology_map.py、24_typology_stability.py", 核對來源=f"out/lvr_township.csv；typology.csv 合計 {N['typology_sum']}")
add(分析="縣市真實登記（買賣率、繼承率、存量對流量）", 鄉鎮數=f"{N['county_21']} 縣市（不含連江縣）", 資料期間="內政部統計處 108–112 年平均；模擬 108–112 年存量", 建物條件="全部建物（含屋齡不明、全部登記原因）",
    分子="真實：縣市買賣、贈與、繼承登記棟數；既有建物版用實價登錄屋齡 ≥10 年", 分母="模擬：同年同縣市去重建物 ×10", 主要限制="只有 21 個點；含新成屋（全部買賣版）", 章節="§3.4、B.5、E.1", 程式="14_real_rates.py、17_real_sales_lvr.py", 核對來源="out/real_rates_county_mean.csv")
add(分析="四縣市分區真實登記", 鄉鎮數=f"{N['districts']} 區（{'、'.join(f'{k}{v}' for k, v in N['districts_by_city'].items())}）；有 u 的 {N['districts_with_u']} 區", 資料期間="112 年各區登記統計；模擬 112 年",
    建物條件="B+C+D ≥30、模擬建物 ≥100、繼承與買賣 >0", 分子="真實：臺中/臺南/彰化為住宅『筆』、新北為建物『棟』", 分母="模擬：住宅用途（A/F/G/L）建物 ×10（筆的城市）或全部建物 ×10（新北）",
    主要限制="筆與棟不同口徑，只看縣市內排序；買賣含新成屋", 章節="§3.4、B.4、B.5", 程式="14_real_rates.py", 核對來源="out/real_rates_district_112.csv")
add(分析="合成世代（只用模擬兩快照反推全國流量）", 鄉鎮數=f"全國 1 個（縣市版 {N['cohort_counties']} 縣市，方法失效）", 資料期間="模擬 110、112 年；真實 110/7–112/6 登記", 建物條件="屋齡已知、建造年 ≤2020（新建物另計）",
    分子="模擬：登記原因組成的變化", 分母="模擬：同一建造年的建物（不明者按比例補回）×10", 主要限制="繼承高估約四成；縣市以下不可用", 章節="A.4、A.5", 程式="15_cohort_flows.py", 核對來源="out/cohort_vs_real_national.csv、cohort_county_fail.json")
T = pd.DataFrame(R); T.to_csv(OUT/"rev1016_sample_table.csv", index=False, encoding="utf-8-sig")
cols = ["分析", "鄉鎮數", "資料期間", "建物條件", "分子", "分母", "主要限制", "章節", "程式"]
esc = lambda x: str(x).replace("|", "／").replace("\n", " ")
md = ["# 分析樣本對照表（rev1016，審稿建議 15）", "", "鄉鎮數由 `code/checks/rev1016_sample_table.py` 依各程式的篩選條件從資料或既有輸出重算。報告裡的『鄉鎮』是屋主檔的鄉鎮碼，新竹市、嘉義市各為全市一碼。", "",
      "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
md += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in R]
md += ["", "## 人口五等分用的是三個不同的鄉鎮集合", "",
       f"- §3.3、附錄 E（`population_quintiles.csv`）：{N['popq_312']} 個；§3.4 圖 7 右（`township_supply_sales_quintiles.csv`）：{N['popq_272']} 個；§3.5 老屋（`lvr_population_quintiles.csv`）：{N['popq_lvr']} 個（老屋周轉率以各組合計計算，組內含老屋很少的鄉鎮）。", "",
       "## 「建物棟數 × 10」的依據", ""]
X = X10
md += [f"- 手冊只寫「依 111 年屋主資料之整體結構，模擬約 10% 之樣本資料」，並給**持有紀錄**筆數：母體 10,766,461、模擬 1,076,646（本檔 111 年 {X['手冊']['本檔111年筆數']:,}），比值恰為 10.0%。手冊沒有母體建物總棟數；表 29（以建物數計算）只給占比。所以「建物 ×10」是由紀錄的 10% 推論而來，不是手冊直接給的。",
       "- 本檔每年的持有紀錄 ÷ 去重建物：" + "、".join(f"{y} 年 {v['紀錄÷建物']:.3f}（{v['持有紀錄']:,}／{v['去重建物']:,}）" for y, v in X["每年_持有紀錄÷去重建物"].items()) + "。只有在母體的紀錄÷建物也是同一數值時，建物倍率才會等於 10。",
       f"- 模擬資料內的紀錄÷建物並不均勻：依登記原因 {X['112_紀錄÷建物_依登記原因']}；依屋齡 {X['112_紀錄÷建物_依屋齡']}；依人口五等分 {X['112_紀錄÷建物_依鄉鎮人口五等分']}。",
       f"- 表 29 的『模擬占比』就是本檔 111 年去重建物的鄉鎮占比（{X['表29模擬占比_對_本檔111去重建物占比']['鄉鎮數']} 個鄉鎮，最大差 {X['表29模擬占比_對_本檔111去重建物占比']['最大絕對差(百分點)']:.3f} 個百分點），所以可用表 29 的『母體占比 ÷ 模擬占比』看各地建物倍率的**相對**高低。占比 ≥0.3% 的 {X['表29_母體÷模擬建物占比(相對倍率)_占比≥0.3%的鄉鎮']['鄉鎮數']} 個鄉鎮：中位數 {X['表29_母體÷模擬建物占比(相對倍率)_占比≥0.3%的鄉鎮']['中位數']:.3f}、P10–P90 {X['表29_母體÷模擬建物占比(相對倍率)_占比≥0.3%的鄉鎮']['P10']:.3f}–{X['表29_母體÷模擬建物占比(相對倍率)_占比≥0.3%的鄉鎮']['P90']:.3f}。",
       f"- 新北市相對倍率 {X['由新北9.40倍推回全國建物倍率']['新北相對倍率(表29)']:.3f}；以新北月報 9.40 倍推回，全國建物倍率約 {X['由新北9.40倍推回全國建物倍率']['推得全國倍率≈9.40/相對倍率']:.2f}（年份與建物範圍不同，只是量級）。",
       f"- 屋齡帶的相對倍率（表 29，已排除母體 {X['表29_屋齡不明占比(母體,模擬)'][0]}% 的屋齡不明；模擬為 {X['表29_屋齡不明占比(母體,模擬)'][1]}%）：" + "、".join(f"{k} {v:.3f}" for k, v in X["表29_屋齡已知中各屋齡帶的相對倍率(母體占比÷模擬占比，已排除不明)"].items()) + "。",
       f"- 鄉鎮的紀錄÷建物（相對值，表 27 占比 ÷ 表 29 占比）母體與模擬的相關 {X['鄉鎮_紀錄÷建物(相對值)_母體對模擬']['相關']:.2f}（{X['鄉鎮_紀錄÷建物(相對值)_母體對模擬']['鄉鎮數']} 個鄉鎮）。"]
(OUT/"rev1016_sample_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
