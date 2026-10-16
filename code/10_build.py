"""建立分析底表。
輸出 data/bld.parquet：108–112 年每年每棟一筆（屋主檔的建物端欄位在同一棟內完全一致，先驗證再去重）
輸出 data/town.csv：屋主檔鄉鎮碼 ↔ 名稱 ↔ 縣市 ↔ 112 年 6 月底真實戶籍人口
"""
import pandas as pd, numpy as np, json
from common import *
own = pd.read_parquet(DATA/"own.parquet")
bcols = ["county_cd","countyntownship_cd","materials_group_cd","floor_group_cd","purpose_group_cd","register_reason_group_cd","b_age"]
audit = {"rows_by_year": own.groupby("year").size().to_dict()}
nun = own.groupby(["year","building_sn"])[bcols].nunique(dropna=False).max()
assert nun.max() == 1, nun     # 同年同棟的建物端欄位唯一
b = own.drop_duplicates(["year","building_sn"])[["year","building_sn"]+bcols].copy()
b["reason"] = b.register_reason_group_cd.fillna("U")
b["age"] = b.b_age.where(b.b_age >= 0)            # 110 年以 -1 表示不明，其餘年以 NA
b["built"] = b.year + 1911 - b.age
audit["buildings_by_year"] = b.groupby("year").size().to_dict()
audit["age_unknown_rate"] = b.age.isna().groupby(b.year).mean().round(4).to_dict()
# 鄉鎮對照：沿用先前研究對照表，補手冊表 28 的兩個異體字（巿）
cw = pd.read_csv(HANDOFF/"analysis/township_crosswalk.csv", dtype=str).drop_duplicates("house_cd")
cw = pd.concat([cw, pd.DataFrame([{"house_cd":"K10","name":"苗栗縣頭份市"},{"house_cd":"T01","name":"屏東縣屏東市"}])]).drop_duplicates("house_cd")
pop = pd.read_csv(REAL/"raw/all_townships_population_11206.csv")
pop["name"] = pop["縣市"] + pop["區名"].str.replace("※","",regex=False)
codes = pd.DataFrame({"house_cd": sorted(b.countyntownship_cd.dropna().unique())})
t = codes.merge(cw[["house_cd","name"]], on="house_cd", how="left").merge(pop[["name","縣市","人口數"]], on="name", how="left")
# 新竹市、嘉義市在屋主檔是整市一碼 → 用全市人口
for cd, city in [("O01","新竹市"),("I01","嘉義市"),("O77","新竹市")]:
    m = t.house_cd == cd
    if m.any():
        t.loc[m, "name"] = city + "（全市）"; t.loc[m, "縣市"] = city; t.loc[m, "人口數"] = pop.loc[pop["縣市"]==city, "人口數"].sum()
cc = pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str)
t["county_cd"] = t.house_cd.str[0]
t = t.merge(cc[["county_cd","county_name","region"]], on="county_cd", how="left")
t["county"] = t["縣市"].fillna(t.county_name)
t = t.rename(columns={"人口數":"pop"})[["house_cd","name","county","region","pop"]]
audit["township_codes"] = len(t); audit["township_no_name"] = t[t.name.isna()].house_cd.tolist(); audit["township_no_pop"] = t[t["pop"].isna()].house_cd.tolist()
t.to_csv(DATA/"town.csv", index=False, encoding="utf-8-sig")
b.to_parquet(DATA/"bld.parquet", index=False)
(OUT/"audit_build.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=int))
print(json.dumps(audit, ensure_ascii=False, default=int)); print(t.head()); print(b.groupby("year").countyntownship_cd.nunique())
