"""rev1014 核對 1–2（老屋部分）：老屋（≥40 年、1–5 層）分子的交易日分布、重複、未登記建物；改按交易日選窗時的老屋買賣率與對 u 的 Spearman。輸出 out/rev1014_lvr_old.json"""
import sys, glob, json, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from common import *
LET = {"a":"臺北市","b":"臺中市","c":"基隆市","d":"臺南市","e":"高雄市","f":"新北市","g":"宜蘭縣","h":"桃園市","i":"嘉義市","j":"新竹縣","k":"苗栗縣","m":"南投縣","n":"彰化縣","o":"新竹市","p":"雲林縣","q":"嘉義縣","t":"屏東縣","u":"花蓮縣","v":"臺東縣","w":"金門縣","x":"澎湖縣","z":"連江縣"}
SEASONS = ["111S3","111S4","112S1","112S2","112S3","112S4"]; SQ = {s: int(s[:3])*4 + int(s[-1]) - 1 for s in SEASONS}
CN = {c: i for i, c in enumerate("零一二三四五六七八九")}
def cn2int(s):          # 與 17_real_sales_lvr.py 相同
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
        x = pd.read_csv(f, dtype=str, skiprows=[1], on_bad_lines="skip", encoding="utf-8-sig"); x["county"] = LET[f.split("/")[-1][0]]; x["season"] = s; parts.append(x)
L = pd.concat(parts, ignore_index=True)
L["n_bld"] = pd.to_numeric(L["交易筆棟數"].str.extract(r"建物(\d+)")[0], errors="coerce").fillna(0)
L = L[(L.n_bld > 0) & L["交易標的"].fillna("").str.contains("房地|建物")].copy()
yr = pd.to_numeric(L["建築完成年月"].str.strip().str[:-4], errors="coerce"); L["age"] = np.where(yr.between(1, 113), 112 - yr, np.nan)
L["floors"] = L["總樓層數"].map(cn2int)
town = pd.read_csv(DATA/"town.csv"); nm = dict(zip(town.name, town.house_cd))
L["house_cd"] = (L.county + L["鄉鎮市區"].fillna("").str.replace("台","臺")).map(nm); L.loc[L.county == "新竹市", "house_cd"] = "O01"; L.loc[L.county == "嘉義市", "house_cd"] = "I01"
d = L["交易年月日"].str.strip(); ty = pd.to_numeric(d.str[:-4], errors="coerce"); tm = pd.to_numeric(d.str[-4:-2], errors="coerce")
L["tq"] = ty*4 + (tm-1)//3; L["lag"] = L.season.map(SQ) - L.tq; L["ym"] = ty*12 + tm
O = L[(L.age >= 40) & L.floors.between(1,5)].dropna(subset=["house_cd"]).copy(); w = O.n_bld
R = {"old_rows": len(O), "old_bld": int(w.sum()), "old_bld_annual_by_file": float(w.sum()/1.5)}
R["old_trade_in_111Jul_112Dec_share"] = float(np.average(O.ym.between(111*12+7, 112*12+12), weights=w))
R["old_trade_before_111Jul_share"] = float(np.average(O.ym < 111*12+7, weights=w))
R["old_trade_before_111Apr_share"] = float(np.average(O.ym < 111*12+4, weights=w))
R["old_lag_share"] = {str(int(k)): float(v) for k, v in (O.groupby(O.lag.clip(-1, 6)).n_bld.sum()/w.sum()).items()}
R["old_by_trade_quarter"] = {f"{int(k)//4}Q{int(k)%4+1}": int(v) for k, v in O.groupby("tq").n_bld.sum().items() if k >= 110*4}
R["old_note_未登記_bld_share"] = float(np.average(O["備註"].fillna("").str.contains("未登記建物|未辦保存|無建物登記|未保存登記"), weights=w))
R["old_note_增建_bld_share"] = float(np.average(O["備註"].fillna("").str.contains("增建"), weights=w))
R["old_n_bld_gt1_bld_share"] = float(w[O.n_bld > 1].sum()/w.sum())
R["old_dup_id_extra_rows"] = int(O.duplicated("編號").sum())
key = ["county","鄉鎮市區","土地位置建物門牌","交易年月日","總價元","建物移轉總面積平方公尺","移轉層次","交易筆棟數"]
R["old_dup_fullkey_extra_rows"] = int(O.duplicated(key).sum())
R["old_same_address_floor_extra_rows"] = int(O.duplicated(["county","鄉鎮市區","土地位置建物門牌","移轉層次"]).sum())
R["old_主要用途_share"] = (O.groupby(O["主要用途"].fillna("（空白）")).n_bld.sum()/w.sum()).sort_values(ascending=False).head(8).round(4).to_dict()
R["old_建物型態_share"] = (O.groupby(O["建物型態"].fillna("（空白）")).n_bld.sum()/w.sum()).sort_values(ascending=False).head(6).round(4).to_dict()
# 替代分子：(a) 依交易日 111/7–112/6 一整年（各季檔可見的延遲 0–5 季都已涵蓋，缺很少）(b) 主設定去掉完全重複列
T = pd.read_csv(OUT/"lvr_township.csv")
alt = {"main(季檔/1.5)": O.groupby("house_cd").n_bld.sum()/1.5,
       "交易日111/7-112/6(÷1)": O[O.ym.between(111*12+7, 112*12+6)].groupby("house_cd").n_bld.sum(),
       "交易日111/7-112/12(÷1.5,112Q4不完整)": O[O.ym.between(111*12+7, 112*12+12)].groupby("house_cd").n_bld.sum()/1.5,
       "去完全重複列": O.drop_duplicates(key).groupby("house_cd").n_bld.sum()/1.5}
# 主設定的老屋存量用 1–5 層 & 40+（模擬 floor_group_cd=1），即 T.lv_stock_old；注意主設定分子的 fl=='1' 與此相同
big = T.lv_stock_old >= 1000; res = {}
for k, x in alt.items():
    t = T[["house_cd","u","lv_stock_old","turn_old","county"]].merge(x.rename("x"), left_on="house_cd", right_index=True, how="left").fillna({"x": 0}); t["r"] = t.x/t.lv_stock_old*100; tb = t[big]
    res[k] = {"national_old_rate%": float(t.x.sum()/t.lv_stock_old.sum()*100), "spearman_u_244": float(spearmanr(tb.u, tb.r)[0]), "spearman_with_main_turn_old": float(spearmanr(tb.turn_old, tb.r)[0]), "n": int(len(tb))}
R["alt_numerators"] = res
R["check_main_reproduces_spearman"] = float(spearmanr(T[big].u, T[big].turn_old)[0])
json.dump(R, open(OUT/"rev1014_lvr_old.json","w"), ensure_ascii=False, indent=1, default=float); print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
