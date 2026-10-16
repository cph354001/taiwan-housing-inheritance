import pandas as pd, numpy as np
R = "./"
b = pd.read_parquet(R+"data/bld.parquet"); b = b[b.year.isin([110,111,112])]
print(b.groupby("year").building_sn.agg(["size","nunique","min","max"]))
S = {y: set(b.loc[b.year==y,"building_sn"]) for y in (110,111,112)}
U = S[110]|S[111]|S[112]; I3 = S[110]&S[111]&S[112]
print("union", len(U), "三年共同", len(I3), "占聯集", len(I3)/len(U), "占112", len(I3)/len(S[112]))
for a,c in [(110,111),(111,112),(110,112)]:
    i = len(S[a]&S[c]); print(f"{a}&{c}: {i}  占{a} {i/len(S[a]):.4f} 占{c} {i/len(S[c]):.4f} Jaccard {i/len(S[a]|S[c]):.4f}")
# 隨機基準：若各年 ID 從 [min,max] 中隨機抽
lo, hi = b.building_sn.min(), b.building_sn.max(); N = hi-lo+1
f = {y: len(S[y])/N for y in S}; print("ID 空間", N, "各年填充率", {y: round(v,4) for y,v in f.items()})
print("隨機預期 兩年重疊占一年:", {f"{a}&{c}": round(f[c],4) for a,c in [(110,111),(111,112),(110,112)]}, "三年共同占112:", round(f[110]*f[111],4))
# 屬性一致率（共同 ID；若 ID 在某年重複，取第一筆）
cols = ["countyntownship_cd","built","materials_group_cd","floor_group_cd","purpose_group_cd","reason"]
g = {y: b[b.year==y].drop_duplicates("building_sn").set_index("building_sn")[cols] for y in S}
for a,c in [(110,111),(111,112),(110,112)]:
    ids = list(S[a]&S[c]); x, z = g[a].loc[ids], g[c].loc[ids]
    eq = {k: float((x[k].astype(str).values==z[k].astype(str).values).mean()) for k in cols}
    allk = (np.column_stack([x[k].astype(str).values==z[k].astype(str).values for k in cols[:4]])).all(1).mean()
    # 隨機配對基準：同一年度任意兩棟
    xs, zs = g[a].sample(200000, random_state=0), g[c].sample(200000, random_state=1)
    base = {k: float((xs[k].astype(str).values==zs[k].astype(str).values).mean()) for k in cols}
    ball = (np.column_stack([xs[k].astype(str).values==zs[k].astype(str).values for k in cols[:4]])).all(1).mean()
    print(f"\n{a}->{c} n={len(ids)} 鄉鎮/建造年/建材/樓層全同 {allk:.4f}（隨機兩棟 {ball:.4f}）")
    print("  逐欄:", {k: round(v,4) for k,v in eq.items()}); print("  隨機:", {k: round(v,4) for k,v in base.items()})
    if (a,c)==(111,112):
        t = pd.crosstab(x.reason.values, z.reason.values, normalize="index").round(3); print(t)
        print("  112 邊際:", z.reason.value_counts(normalize=True).round(3).to_dict())
        ct = pd.crosstab(x.reason.values, z.reason.values)
        from scipy.stats import chi2_contingency; chi = chi2_contingency(ct); n = ct.values.sum()
        print("  卡方 p", chi[1], "Cramer V", np.sqrt(chi[0]/n/(min(ct.shape)-1)))
I3l = list(I3); allsame = np.ones(len(I3l), bool)
for k in cols[:4]:
    v = [g[y].loc[I3l, k].astype(str).values for y in (110,111,112)]; allsame &= (v[0]==v[1])&(v[1]==v[2])
print("\n三年共同 ID 中四項屬性三年全同:", allsame.mean(), allsame.sum())
