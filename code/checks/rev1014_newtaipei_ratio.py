"""獨立重算：新北市 真實已登記建物總棟數（112/6 地政月報）÷ 模擬屋主檔 112 年去重建物數。
全市與 9 個地政事務所轄區。只讀先前研究，不改任何既有檔案。
輸出 out/rev1014_newtaipei_ratio.json
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, xlrd
from common import ROOT, OUT, DATA, HANDOFF

XLS = REAL/"raw/newtaipei_11206_land_building_registration_original.xls"
wb = xlrd.open_workbook(str(XLS))

# ---- 1. 真實值：讀每張工作表的標題與第 3 列 ----
sheets = []
for s in wb.sheets():
    title = str(s.cell_value(0, 5)).strip()          # F1 = 報表標題
    a3, e3, f3 = str(s.cell_value(2, 0)), str(s.cell_value(2, 4)), s.cell_value(2, 5)
    m = re.match(r"新北市(.+?)(地政事務所|區)?土地及建物登記管理", title)
    sheets.append({"sheet": s.name, "title": title, "unit": m.group(1) if m else "全市",
                   "kind": {"地政事務所": "office", "區": "district"}.get(m.group(2) if m else None, "city"),
                   "A3": a3, "E3": e3, "F3": float(f3) if f3 != "" else None,
                   "period": str(s.cell_value(0, 6)).strip()})
assert all("建物總棟數" in x["E3"] for x in sheets), "E3 標題不是建物總棟數"
assert all(x["A3"].startswith("截至本月底已登記") for x in sheets)

# 由工作表順序推出「事務所 → 其後各區」的轄區（檔案本身的結構）
file_juris, cur = {}, None
for x in sheets:
    if x["kind"] == "office": cur = x["unit"]; file_juris[cur] = []
    elif x["kind"] == "district": file_juris[cur].append(x["unit"])

REVIEWER = {"板橋": ["板橋","土城"], "新莊": ["新莊","五股","泰山","林口"],
            "新店": ["新店","深坑","石碇","坪林","烏來"], "淡水": ["淡水","三芝","石門","八里"],
            "汐止": ["汐止","金山","萬里"], "瑞芳": ["瑞芳","平溪","雙溪","貢寮"],
            "三重": ["三重","蘆洲"], "中和": ["中和","永和"], "樹林": ["樹林","鶯歌","三峽"]}
alld = [d for v in REVIEWER.values() for d in v]
juris_ok = {"n_districts": len(alld), "unique": len(set(alld)) == len(alld),
            "matches_xls_sheet_structure": {k: sorted(v) for k, v in file_juris.items()} == {k: sorted(v) for k, v in REVIEWER.items()}}

real_city = next(x["F3"] for x in sheets if x["kind"] == "city")
real_office = {x["unit"]: x["F3"] for x in sheets if x["kind"] == "office"}
district_F3 = {x["unit"]: x["F3"] for x in sheets if x["kind"] == "district"}

# ---- 2. 模擬值：直接從 own.parquet（逐筆屋主×建物）去重，不經 bld.parquet ----
own = pd.read_parquet(DATA/"own.parquet", columns=["year","building_sn","county_cd","countyntownship_cd",
                                                   "gender_cd","purpose_group_cd","register_reason_group_cd","b_age"])
o = own[(own.year == 112) & (own.county_cd == "F")]
town = pd.read_csv(DATA/"town.csv", dtype=str)
town = town[town.house_cd.str.startswith("F")]
town["dist"] = town.name.str.replace("新北市", "").str.replace("區", "")
assert set(o.countyntownship_cd.unique()) <= set(town.house_cd), set(o.countyntownship_cd.unique()) - set(town.house_cd)
assert set(town.dist) == set(alld), set(town.dist) ^ set(alld)
# 同一 building_sn 是否跨鄉鎮
assert o.groupby("building_sn").countyntownship_cd.nunique().max() == 1
b = o.drop_duplicates("building_sn").merge(town[["house_cd","dist"]], left_on="countyntownship_cd", right_on="house_cd")
d2o = {d: k for k, v in REVIEWER.items() for d in v}
b["office"] = b.dist.map(d2o)
sim_city = int(b.building_sn.nunique())
sim_office = b.groupby("office").building_sn.nunique().to_dict()

# 口徑分解：建物上至少一名「不詳或非自然人」(gender 3)、用途分布、屋齡不明
g3 = o.assign(g3=o.gender_cd.astype(str) == "3").groupby("building_sn").g3.agg(["any","all"])
comp = {"rows_112_F": int(len(o)), "buildings_112_F": sim_city,
        "share_bld_any_owner_gender3": round(float(g3["any"].mean()), 4),
        "share_bld_all_owners_gender3": round(float(g3["all"].mean()), 4),
        "purpose_share": b.purpose_group_cd.fillna("NA").value_counts(normalize=True).round(4).to_dict(),
        "reason_share": b.register_reason_group_cd.fillna("U").value_counts(normalize=True).round(4).to_dict(),
        "age_unknown_share": round(float((b.b_age.isna() | (b.b_age < 0)).mean()), 4)}

# 對照 bld.parquet（專案既有底表）
bld = pd.read_parquet(DATA/"bld.parquet", columns=["year","building_sn","county_cd"])
comp["bld_parquet_112_F"] = int(((bld.year == 112) & (bld.county_cd == "F")).sum())

rows = [{"unit": "全市", "real": int(real_city), "sim": sim_city, "ratio": round(real_city/sim_city, 4)}]
for k in REVIEWER:
    rows.append({"unit": k, "districts": REVIEWER[k], "real": int(real_office[k]), "sim": int(sim_office[k]),
                 "ratio": round(real_office[k]/sim_office[k], 4)})
res = {"source_xls": str(XLS), "period": sheets[0]["period"],
       "cell_check": {"A3": sheets[0]["A3"], "E3": sheets[0]["E3"], "F3_101": real_city},
       "sheet_map": [{k: x[k] for k in ("sheet","unit","kind","F3")} for x in sheets],
       "district_sheets_F3_all_zero": all(v == 0 for v in district_F3.values()),
       "sum_offices_real": int(sum(real_office.values())), "sum_offices_sim": int(sum(sim_office.values())),
       "jurisdiction_check": juris_ok, "composition": comp, "table": rows,
       "reviewer": {"city": [1856231, 197478, 9.40], "office_range": [8.56, 9.56]}}
(OUT/"rev1014_newtaipei_ratio.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print(json.dumps({k: res[k] for k in ("period","cell_check","district_sheets_F3_all_zero","sum_offices_real","sum_offices_sim","jurisdiction_check","composition")}, ensure_ascii=False, indent=1, default=float))
print(pd.DataFrame(rows).to_string(index=False))
