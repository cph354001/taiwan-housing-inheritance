"""鄉鎮層級的共變數（全部是鄉鎮彙總，不涉及個人對接）。
 屋主端（屋主檔，自然人）：65 歲以上比例、平均年齡、大學以上比例
 居民端（人建地檔 112 年）：居民數、65 歲以上比例、出生地＝現居縣市比例、遷徙次數平均、遷入現址年數中位數
 真實：112 年 6 月底戶籍人口（data/town.csv）"""
import pandas as pd, numpy as np
from common import *
own = pd.read_parquet(DATA/"own.parquet", columns=["year","gender_cd","age","education_cd","countyntownship_cd"])
o = own[(own.year == 112) & (own.gender_cd != 3) & own.age.notna()]
A = o.groupby("countyntownship_cd").agg(owner_n=("age","size"), owner_age=("age","mean"), owner_65=("age", lambda s: (s >= 65).mean()),
                                         owner_univ=("education_cd", lambda s: (s[s < 6] >= 4).mean())).reset_index().rename(columns={"countyntownship_cd":"house_cd"})
parts = []
use = ["year","coun_cd","addr_city_cd","birthday_date","birth_place_cd","move_cnt","movein_date"]
for c in pd.read_csv(RAW_PPL, usecols=use, dtype=str, chunksize=1_000_000):
    parts.append(c[c.year == "112"])
p = pd.concat(parts); print("people 112 rows", len(p))
ref = pd.Timestamp("2023-06-30")
p["age"] = (ref - pd.to_datetime(p.birthday_date, errors="coerce")).dt.days/365.25
p["stay"] = (ref - pd.to_datetime(p.movein_date, errors="coerce")).dt.days/365.25
p["local"] = (p.birth_place_cd == p.coun_cd).astype(float)
p["move"] = pd.to_numeric(p.move_cnt, errors="coerce")
R = p.groupby("addr_city_cd").agg(res_n=("age","size"), res_65=("age", lambda s: (s >= 65).mean()), res_local=("local","mean"),
                                  res_move=("move","mean"), res_stay=("stay", lambda s: s[s >= 0].median())).reset_index()
cw = pd.read_csv(HANDOFF/"analysis/township_crosswalk.csv", dtype=str)          # 屋主檔碼 ↔ 人建地檔碼（新竹市、嘉義市一對多）
R = R.merge(cw[["house_cd","addr_city_cd"]], on="addr_city_cd", how="left")
# 先前研究漏掉的兩個異體字鄉鎮（頭份、屏東市）：由手冊表 30 的代碼補上
R.loc[R.addr_city_cd == "10005050", "house_cd"] = "K10"; R.loc[R.addr_city_cd == "10013010", "house_cd"] = "T01"
print("people-file townships without house_cd:", R[R.house_cd.isna()].addr_city_cd.tolist())
wm = lambda g, c: np.average(g[c], weights=g.res_n)
R2 = R.dropna(subset=["house_cd"]).groupby("house_cd").apply(lambda g: pd.Series({"res_n": g.res_n.sum(), **{c: wm(g, c) for c in ["res_65","res_local","res_move","res_stay"]}})).reset_index()
T = pd.read_csv(DATA/"town.csv").merge(A, on="house_cd", how="left").merge(R2, on="house_cd", how="left")
T.to_csv(DATA/"town_cov.csv", index=False, encoding="utf-8-sig")
v = T.dropna(subset=["pop","res_n"])
print("模擬居民數×10 vs 真實戶籍人口：Pearson(log) %.4f, 比值中位數 %.3f"%(np.corrcoef(np.log(v.res_n), np.log(v["pop"]))[0,1], (v.res_n*10/v["pop"]).median()))
print(T.describe().T.round(3))
