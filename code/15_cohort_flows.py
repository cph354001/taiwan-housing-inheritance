"""輔助分析：只用模擬資料的年度變化，能不能反推每年的買賣與繼承發生率？
想法（合成世代）：同一批建造年的建物，在 110 與 112 兩個快照之間，登記原因的組成會改變。
 假設每棟每年以 λ 的機率發生一次新登記（與目前狀態無關），新登記的原因為 k 的機率是 h_k/λ，則
   dp_A/dt = -λ p_A                （第一次登記只會減少，不會再出現）
   dp_k/dt = h_k - λ p_k           （k = 買賣、贈與、繼承…）
 所以 λ = -Δlog p_A / Δt，h_k = Δp_k/Δt + λ·p̄_k。
只用 110–112 年（屋齡到每一歲才可對齊世代）。不確定性：對每年的「建造年×原因」次數表做多項重抽。"""
import pandas as pd, numpy as np, re, json
from common import *
b = pd.read_parquet(DATA/"bld.parquet"); K = ["A","B","C","D","E","F","U"]
v = b[b.year.isin([110,111,112]) & b.age.notna()].copy(); v["built"] = v.built.astype(int)
N = v.groupby(["year","built","reason"]).size().unstack(fill_value=0).reindex(columns=K, fill_value=0)
def est(N, y0=110, y1=112, lo=1900, hi=2020):
    n0, n1 = N.loc[y0], N.loc[y1]; idx = [c for c in n0.index.intersection(n1.index) if lo <= c <= hi]; n0, n1 = n0.loc[idx], n1.loc[idx]
    w = (n0.sum(1) + n1.sum(1))/2; P0 = n0.div(n0.sum(1), axis=0).mul(w, axis=0).sum()/w.sum(); P1 = n1.div(n1.sum(1), axis=0).mul(w, axis=0).sum()/w.sum()
    dt = y1 - y0; lam = -np.log(P1.A/P0.A)/dt; h = (P1 - P0)/dt + lam*(P0 + P1)/2
    return pd.Series({"λ": lam, "買賣": h.B, "贈與": h.C, "繼承": h.D, "pA0": P0.A, "pA1": P1.A, "棟數": w.sum()})
def boot(N, R=400, **kw):
    rng = np.random.default_rng(7); out = []
    for _ in range(R):
        Nb = N.copy()
        for y in N.index.get_level_values(0).unique():
            blk = N.loc[y].values; Nb.loc[y] = rng.multinomial(blk.sum(), (blk/blk.sum()).ravel()).reshape(blk.shape)
        out.append(est(Nb, **kw))
    return pd.DataFrame(out)
rows = []
for lab, kw in [("全部（建造年 ≤2020）", {}), ("屋齡 0–19 年", {"lo":2002}), ("屋齡 20–39 年", {"lo":1982,"hi":2001}), ("屋齡 40 年以上", {"hi":1981})]:
    e = est(N, **kw); B = boot(N, **kw)
    rows.append({"範圍": lab, **{k: "%.2f [%.2f, %.2f]"%(e[k]*100, *(np.percentile(B[k], [2.5,97.5])*100)) for k in ["λ","買賣","贈與","繼承"]}, "棟數": int(e["棟數"])})
T = pd.DataFrame(rows); T.to_csv(OUT/"cohort_hazards_national.csv", index=False, encoding="utf-8-sig"); print(T.to_string())
# 與真實流量比較：快照為 110/6 與 112/6，期間 = 110 下半年 + 111 全年 + 112 上半年
real = {}
for y in [110,111,112]:
    for line in open(REAL/f"raw/moi{y}.csv", encoding="utf-8-sig"):
        m = re.match(r'"(\d+)年/ 區域別總計/ (.+?)",([\d\-]+),', line)
        if m: real[(y, m[2])] = int(m[3])
avg = lambda k: (0.5*real[(110,k)] + real[(111,k)] + 0.5*real[(112,k)])/2
stock = b[b.year.isin([110,111,112])].groupby("year").size().mean()*10
e = est(N); known = v[v.year.isin([110,112])].groupby("year").size().mean(); allb = b[b.year.isin([110,112])].groupby("year").size().mean()
scale_up = allb/known                                    # 屋齡不明的建物按比例補回
old_n = e["棟數"]*scale_up*10
new = v[(v.year == 112) & (v.built >= 2021)].reason.value_counts().reindex(K, fill_value=0) - v[(v.year == 110) & (v.built >= 2021)].reason.value_counts().reindex(K, fill_value=0)
cmp_rows = []
for k, kk, rk in [("買賣","B",["買賣"]), ("繼承","D",["繼承"]), ("贈與","C",["贈與","夫妻贈與"])]:
    sim_old = e[k]*old_n; sim_new = new[kk]/2*10*scale_up; rl = sum(avg(x) for x in rk)
    cmp_rows.append({"登記原因": k, "既有建物每百棟(合成世代)": round(e[k]*100, 2), "模擬反推_既有建物(棟/年)": round(sim_old), "模擬_期間新建物(棟/年,下界)": round(sim_new), "模擬合計": round(sim_old + sim_new), "真實登記(棟/年)": round(rl), "模擬/真實": round((sim_old + sim_new)/rl, 2),
                     "模擬每百棟": round((sim_old + sim_new)/stock*100, 2), "真實每百棟": round(rl/stock*100, 2)})
Cm = pd.DataFrame(cmp_rows); Cm.to_csv(OUT/"cohort_vs_real_national.csv", index=False, encoding="utf-8-sig"); print(Cm.to_string())
# 繼承高估的來源：h_D ＝ 存量淨增 Δp_D/Δt ＋ 流出補回 λ·p̄_D。若「繼承狀態」建物的再移轉率不是 λ 而是 λ_D，
# 要對上真實繼承率所需的 λ_D ＝ (h_D_real − Δp_D/Δt) / p̄_D
n0, n1 = N.loc[110], N.loc[112]; idx = [c for c in n0.index.intersection(n1.index) if 1900 <= c <= 2020]; n0, n1 = n0.loc[idx], n1.loc[idx]
w = (n0.sum(1) + n1.sum(1))/2; P0 = n0.div(n0.sum(1), axis=0).mul(w, axis=0).sum()/w.sum(); P1 = n1.div(n1.sum(1), axis=0).mul(w, axis=0).sum()/w.sum()
net_D = (P1.D - P0.D)/2; pbar_D = (P0.D + P1.D)/2; real_D = avg("繼承")/old_n     # 真實繼承率以既有建物為分母（新建物的繼承可忽略）
dec = {"h_D_合成世代": float(e["繼承"]*100), "其中_存量淨增": float(net_D*100), "其中_流出補回(λ·p̄_D)": float(e["λ"]*pbar_D*100), "真實繼承率_每百棟既有建物": float(real_D*100),
       "λ_全狀態共用(由第一次登記衰減)": float(e["λ"]*100), "要對上真實所需的_λ_D": float((real_D - net_D)/pbar_D*100), "繼承建物平均屋齡": float(v[(v.year == 112) & (v.reason == "D")].age.mean())}
json.dump(dec, open(OUT/"cohort_inherit_gap.json","w"), ensure_ascii=False, indent=1); print(dec)
# 縣市層級：方法失效的證據
cc = pd.read_csv(HANDOFF/"analysis/county_crosswalk.csv", dtype=str); R = pd.read_csv(OUT/"real_rates_county_year.csv"); R = R[R.year.isin([110,111,112])].groupby("county")[["r_B","r_D"]].mean()
cr = []
for cd, g in v.groupby("county_cd"):
    Nc = g.groupby(["year","built","reason"]).size().unstack(fill_value=0).reindex(columns=K, fill_value=0)
    if {110,112} <= set(Nc.index.get_level_values(0)) and g.building_sn.size > 30000: cr.append({"county_cd": cd, **est(Nc).to_dict()})
cr = pd.DataFrame(cr).merge(cc[["county_cd","county_name"]], on="county_cd").merge(R, left_on="county_name", right_index=True)
from scipy.stats import spearmanr
fail = {"n_county": len(cr), "spearman_買賣": spearmanr(cr["買賣"], cr.r_B)[0], "spearman_繼承": spearmanr(cr["繼承"], cr.r_D)[0], "λ範圍%": [cr["λ"].min()*100, cr["λ"].max()*100]}
print(fail); cr.to_csv(OUT/"cohort_hazards_county.csv", index=False, encoding="utf-8-sig"); json.dump(fail, open(OUT/"cohort_county_fail.json","w"), ensure_ascii=False, default=float)
