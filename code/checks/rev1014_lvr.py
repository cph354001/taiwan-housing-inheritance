"""rev1014 審稿核對 1、2：實價登錄的時間窗、重複、屋齡基準；分子分母範圍。只讀，輸出 out/rev1014_lvr.json"""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from common import *
SEASONS = ["111S3","111S4","112S1","112S2","112S3","112S4"]
parts = []
for s in SEASONS:
    for f in glob.glob(str(ROOT/f"external/lvr/{s}/*_lvr_land_a.csv")):
        x = pd.read_csv(f, dtype=str, skiprows=[1], on_bad_lines="skip", encoding="utf-8-sig")
        x["cl"] = f.split("/")[-1][0]; x["season"] = s; parts.append(x)
L0 = pd.concat(parts, ignore_index=True)
R = {"rows_all": len(L0)}
L0["n_bld"] = pd.to_numeric(L0["交易筆棟數"].str.extract(r"建物(\d+)")[0], errors="coerce").fillna(0)
L = L0[(L0.n_bld > 0) & L0["交易標的"].fillna("").str.contains("房地|建物")].copy()
R["rows_used"] = len(L); R["bld_used"] = int(L.n_bld.sum())
# --- 交易日期 ---
d = L["交易年月日"].str.strip(); y = pd.to_numeric(d.str[:-4], errors="coerce"); m = pd.to_numeric(d.str[-4:-2], errors="coerce")
L["ty"], L["tm"] = y, m; L["tq"] = y.astype("Int64").astype(str) + "Q" + ((m-1)//3+1).astype("Int64").astype(str)
L["ym"] = y*12 + m
w = L.n_bld
inwin = (L.ym >= 111*12+7) & (L.ym <= 112*12+12)
R["trade_in_window_share_rows"] = float(inwin.mean()); R["trade_in_window_share_bld"] = float(np.average(inwin, weights=w))
R["trade_before_window_bld_share"] = float(np.average(L.ym < 111*12+7, weights=w)); R["trade_after_window_bld_share"] = float(np.average(L.ym > 112*12+12, weights=w)); R["trade_date_bad"] = int(L.ym.isna().sum())
R["trade_year_bld"] = L.groupby("ty").n_bld.sum().astype(int).to_dict()
R["trade_quarter_bld_111Q1_113Q1"] = {k: int(v) for k, v in L.groupby("tq").n_bld.sum().items() if k[:3] in ("111","112","113")}
ct = pd.crosstab(L.season, L.tq, values=L.n_bld, aggfunc="sum").fillna(0).astype(int)
keep = [c for c in ct.columns if c[:3] in ("111","112","113")]; ct2 = ct[keep].copy(); ct2["其他(110以前)"] = ct.drop(columns=keep).sum(1)
R["season_by_tradequarter_bld"] = ct2.to_dict("index")
sq = {"111S3":(111,3),"111S4":(111,4),"112S1":(112,1),"112S2":(112,2),"112S3":(112,3),"112S4":(112,4)}
qidx = L.ty*4 + ((L.tm-1)//3); sidx = L.season.map(lambda s: sq[s][0]*4 + sq[s][1]-1)
lag = sidx - qidx
R["lag_quarters_bld_share"] = {str(int(k)): float(v) for k, v in (L.groupby(lag.clip(-1, 5)).n_bld.sum()/w.sum()).items()}
# 窗內交易日（111/7–112/12）的棟數，與「按交易日」年化
R["bld_trade_in_window"] = int(w[inwin].sum()); R["annual_by_file_div1.5"] = float(w.sum()/1.5); R["annual_trade_in_window_div1.5"] = float(w[inwin].sum()/1.5)
R["bld_trade_in_cal112"] = int(w[L.ty == 112].sum())
# --- 重複 ---
R["id_null"] = int(L["編號"].isna().sum()); R["id_unique"] = int(L["編號"].nunique())
dup = L.duplicated("編號", keep=False); R["rows_with_dup_id"] = int(dup.sum()); R["extra_rows_dup_id"] = int(L.duplicated("編號").sum()); R["extra_bld_dup_id"] = int(L[L.duplicated("編號")].n_bld.sum())
R["dup_id_across_seasons_ids"] = int((L.groupby("編號").season.nunique() > 1).sum()); R["dup_id_within_file_ids"] = int((L.groupby(["編號","season","cl"]).size() > 1).sum())
key = ["cl","鄉鎮市區","土地位置建物門牌","交易年月日","總價元","建物移轉總面積平方公尺","移轉層次","交易筆棟數"]
dk = L.duplicated(key); R["extra_rows_same_addr_date_price_area"] = int(dk.sum()); R["extra_bld_same_key"] = int(L[dk].n_bld.sum()); R["extra_rows_same_key_share"] = float(dk.mean())
dk_cross = L[L.duplicated(key, keep=False)].groupby(key, dropna=False).season.nunique(); R["same_key_groups"] = int(len(dk_cross)); R["same_key_groups_cross_season"] = int((dk_cross > 1).sum())
key2 = ["cl","鄉鎮市區","土地位置建物門牌","移轉層次","建築完成年月"]   # 同一門牌同層在六季內出現多次（含真實轉手、更正重登）
d2 = L.duplicated(key2); R["extra_rows_same_address_floor"] = int(d2.sum()); R["extra_rows_same_address_floor_share"] = float(d2.mean())
note = L["備註"].fillna("")
for k in ["解除","撤銷","更正","重複","解約","親友|特殊關係","預售","增建","未登記建物|未辦保存|無建物登記"]:
    R[f"note_contains_{k}_rows"] = int(note.str.contains(k).sum())
R["移轉編號_dist"] = L["移轉編號"].value_counts(dropna=False).head(6).to_dict()
# --- 屋齡 ---
cy = L["建築完成年月"].str.strip(); yr = pd.to_numeric(cy.str[:-4], errors="coerce")
R["built_len_dist"] = cy.str.len().value_counts(dropna=False).to_dict()
R["built_missing_rows"] = int(yr.isna().sum()); R["built_yr_out_1_113"] = int((yr.notna() & ~yr.between(1,113)).sum()); R["built_yr_113_bld"] = int(w[yr == 113].sum()); R["built_yr_112_bld"] = int(w[yr == 112].sum())
age112 = np.where(yr.between(1,113), 112 - yr, np.nan); ageT = np.where(yr.between(1,113), L.ty - yr, np.nan)
R["age_fixed112_minus_age_at_trade_bld"] = {str(k): int(v) for k, v in L.assign(dif=age112-ageT).groupby("dif").n_bld.sum().items() if abs(k) <= 3}
bins = [-1,9,19,29,39,49,200]; labs = ["0-9","10-19","20-29","30-39","40-49","50+"]
a1 = pd.cut(age112, bins, labels=labs).astype(str); a2 = pd.cut(ageT, bins, labels=labs).astype(str)
R["age_group_changes_if_at_trade_bld_share"] = float(np.average(a1 != a2, weights=w)); R["agegroup_fixed112_bld"] = L.assign(a=a1).groupby("a").n_bld.sum().astype(int).to_dict(); R["agegroup_at_trade_bld"] = L.assign(a=a2).groupby("a").n_bld.sum().astype(int).to_dict()
R["age_minus1_bld (yr=113, 落入NA)"] = int(w[age112 == -1].sum())
# --- 分子的組成 ---
R["交易標的_bld"] = L.groupby("交易標的").n_bld.sum().astype(int).to_dict(); R["交易標的_all_rows"] = L0["交易標的"].value_counts().to_dict()
R["excluded_rows_with_building_but_target"] = L0[(L0.n_bld > 0) & ~L0["交易標的"].fillna("").str.contains("房地|建物")]["交易標的"].value_counts().to_dict()
R["主要用途_bld_top"] = L.groupby(L["主要用途"].fillna("（空白）")).n_bld.sum().sort_values(ascending=False).head(12).astype(int).to_dict()
R["建物型態_bld"] = L.groupby(L["建物型態"].fillna("（空白）")).n_bld.sum().sort_values(ascending=False).astype(int).to_dict()
R["n_bld_dist_rows"] = L.n_bld.clip(upper=5).value_counts().sort_index().to_dict(); R["bld_from_multi_rows"] = int(w[L.n_bld > 1].sum())
fl = L["總樓層數"]; R["floors_missing_rows"] = int(fl.isna().sum()); R["floors_missing_bld"] = int(w[fl.isna()].sum())
old = (age112 >= 40)
R["old_bld_all_floors"] = int(w[old].sum())
json.dump(R, open(OUT/"rev1014_lvr.json","w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
