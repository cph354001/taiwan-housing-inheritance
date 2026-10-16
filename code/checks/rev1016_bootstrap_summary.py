"""彙整 rev1016_bootstrap.py 的每次重抽結果（scratchpad/r1016/rep_*.json）→ out/rev1016_bootstrap.json。
也把 rev1016_sample_table.py 的「×10」核對（scratchpad/r1016/sample_table_checks.json）併入同一檔。"""
import sys, json, glob, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from common import *
SCR = Path("./scratch/r1016")
orig = json.load(open(SCR/"rep_orig.json"))
reps = [json.load(open(f)) for f in sorted(glob.glob(str(SCR/"rep_0*.json")))]
B = len(reps); R = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in reps])
q = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
out = {"B": B, "seed": 20261016, "設計": {
    "重抽單位": "建物（帶著該棟全部持有紀錄），在鄉鎮內有放回重抽；112 年對全部建物重抽，110、111 年對買賣/贈與/繼承且屋齡已知的建物在『年度×鄉鎮』內重抽",
    "每次重估": ["LightGBM 組成模型（繼承、贈與各一；與 12、25 號相同超參數、400 棵樹；以重抽的 110+111 年訓練、預測 112 年）",
                "GLMM 的 u（繼承）與 u_C（贈與），含全國校準", "釋出率中的屋主年齡比例（死亡率為真實資料，固定）", "買賣率分母（模擬既有建物 ×10）；分子（實價登錄）固定",
                "272 鄉鎮的主要迴歸（含與不含 u_C；標準化係數）", "18_typology.py 的分型規則（組成預期占比中位數也重算）"],
    "鄉鎮集合": "固定為原本的 272（迴歸）與 312（分型，金門 4 個不分型）",
    "合併分布(_cl)": "每次重抽另對 272 鄉鎮的縣市做一次群集重抽，近似『流程誤差＋最後一步的抽樣誤差』",
    "B=0 重現檢查": "不重抽時 u、u_C 與原輸出最大差 5e-6，m、bx 相對差 <1e-6，迴歸係數與分型（甲 44、乙 22、丙 13、丁 80、戊 149）完全重現",
    "不涵蓋": "模擬資料與母體間的系統性落差（生成規則、邏輯調整、建物 ×10 的倍率），實價登錄與死亡資料本身的誤差，以及代理指標的定義誤差。重抽把模擬資料當成母體的隨機 10% 樣本；它實際上是依階層機率分布產製的，所以這個區間是『如果它是隨機樣本』的流程不確定性，不是模擬與母體差距的界限。"}}
coef = {}
for key, se_key, lab in [("b_lb_noC", "se_lb_noC", "買賣率（不含 u_C）"), ("b_lm_noC", "se_lm_noC", "釋出率（不含 u_C）"), ("R2_noC", None, "R2（不含 u_C）"),
                         ("b_lb_C", "se_lb_C", "買賣率（含 u_C）"), ("b_lm_C", "se_lm_C", "釋出率（含 u_C）"), ("b_uC_C", "se_uC_C", "u_C（含 u_C）"), ("R2_C", None, "R2（含 u_C）"),
                         ("tau_D", None, "τ（繼承）"), ("tau_C", None, "τ（贈與）")]:
    s = R[key]; qq = q(s)
    c = {"原估計": orig[key], "重抽平均": float(s.mean()), "重抽偏差(平均−原)": float(s.mean() - orig[key]), "重抽SD（流程）": float(s.std()), "重抽2.5–97.5%（流程，鄉鎮固定）": qq,
         "基本(反射)區間 2θ−q": [2*orig[key] - qq[1], 2*orig[key] - qq[0]]}
    if se_key:
        c["原縣市群集SE"] = orig[se_key]; c["原估計±1.96SE"] = [orig[key] - 1.96*orig[se_key], orig[key] + 1.96*orig[se_key]]
        c["流程SD÷原SE"] = float(s.std()/orig[se_key]); c["合併SE≈sqrt(SE²+流程SD²)"] = float(math.sqrt(orig[se_key]**2 + s.var()))
        c["重抽平均的群集SE"] = float(R[se_key].mean())
        if key + "_cl" in R:
            qc = q(R[key + "_cl"]); c["合併分布2.5–97.5%（流程＋縣市群集重抽）"] = qc; c["合併分布SD"] = float(R[key + "_cl"].std())
            c["合併基本(反射)區間"] = [2*orig[key] - qc[1], 2*orig[key] - qc[0]]
    coef[lab] = c
out["係數"] = coef
P = {"買賣率係數<0（不含 u_C）": (R.b_lb_noC < 0).mean(), "釋出率係數>0（不含 u_C）": (R.b_lm_noC > 0).mean(),
     "|買賣率係數|>|釋出率係數|（不含 u_C）": (R.b_lb_noC.abs() > R.b_lm_noC.abs()).mean(), "加入 u_C 後買賣率係數<0": (R.b_lb_C < 0).mean(),
     "|買賣率係數|>|釋出率係數|（含 u_C）": (R.b_lb_C.abs() > R.b_lm_C.abs()).mean(), "u_C 係數>0": (R.b_uC_C > 0).mean(),
     "合併分布：|買賣率|>|釋出率|（不含 u_C）": (R.b_lb_noC_cl.abs() > R.b_lm_noC_cl.abs()).mean(), "合併分布：加入 u_C 後買賣率係數<0": (R.b_lb_C_cl < 0).mean(),
     "合併分布：|買賣率|>|釋出率|（含 u_C）": (R.b_lb_C_cl.abs() > R.b_lm_C_cl.abs()).mean()}
out["比例"] = {k: float(v) for k, v in P.items()}
d_lb_lm = R.b_lb_noC.abs() - R.b_lm_noC.abs(); out["|買賣|−|釋出| 差（不含 u_C）"] = {"原": abs(orig["b_lb_noC"]) - abs(orig["b_lm_noC"]), "重抽2.5–97.5%": q(d_lb_lm), "合併2.5–97.5%": q(R.b_lb_noC_cl.abs() - R.b_lm_noC_cl.abs())}
# u 本身：流程重抽 SD 對 GLMM 後驗 SD
U = pd.DataFrame([r["u312"] for r in reps]); L = pd.read_csv(OUT/"lvr_township.csv").set_index("house_cd")
usd = U.std().reindex(L.index); ratio = (usd/L.u_sd).dropna()
out["u：重抽SD÷後驗SD"] = {"中位數": float(ratio.median()), "P10": float(ratio.quantile(.1)), "P90": float(ratio.quantile(.9)), "重抽u與原u相關的中位數": float(np.median([np.corrcoef(U.iloc[i].reindex(L.index).dropna(), L.u.reindex(U.iloc[i].reindex(L.index).dropna().index))[0, 1] for i in range(B)]))}
# 分型
TY = pd.DataFrame([r["type"] for r in reps]); orig_ty = pd.Series(orig["type"]); keep = orig_ty[orig_ty != "金門"].index
TY = TY[keep]; fr = pd.DataFrame({t: (TY == t).mean() for t in ["甲", "乙", "丙", "丁", "戊", "未納入"]})
fa = fr["甲"]
bins = {"≥90%": int((fa >= .9).sum()), "50–90%": int(((fa >= .5) & (fa < .9)).sum()), "10–50%": int(((fa > .1) & (fa < .5)).sum()), "≤10%（但 >0）": int(((fa > 0) & (fa <= .1)).sum()), "0%": int((fa == 0).sum())}
o_jia = orig_ty[keep] == "甲"
out["分型"] = {"鄉鎮數": len(keep), "原分型各型數": orig_ty[keep].value_counts().to_dict(),
    "各次重抽的型數(平均, 2.5–97.5%)": {t: [float(TY.eq(t).sum(axis=1).mean()), *q(TY.eq(t).sum(axis=1))] for t in ["甲", "乙", "丙", "丁", "戊"]},
    "被分為甲型的頻率分布（308 鄉鎮）": bins, "10–90% 之間": int(((fa > .1) & (fa < .9)).sum()),
    "原甲型 44 個中：≥90% 仍為甲": int((fa[o_jia] >= .9).sum()), "原甲型：50–90%": int(((fa[o_jia] >= .5) & (fa[o_jia] < .9)).sum()), "原甲型：<50%": int((fa[o_jia] < .5).sum()),
    "原非甲型中：甲型頻率 ≥50%": int((fa[~o_jia] >= .5).sum()), "原非甲型中：甲型頻率 >10%": int((fa[~o_jia] > .1).sum()),
    "每個鄉鎮分到最常見型的頻率 ≥90% 的鄉鎮數": int((fr.max(axis=1) >= .9).sum()), "最常見型＝原分型的鄉鎮數": int((fr.idxmax(axis=1) == orig_ty[keep]).sum()),
    "高於↔低於一般鄉鎮翻轉（同一鄉鎮曾被分到甲/丙且曾被分到乙/丁）": int((((fr["甲"] + fr["丙"]) > 0) & ((fr["乙"] + fr["丁"]) > 0)).sum()),
    "原甲型各鄉鎮的甲型頻率": {L.loc[h, "name"]: round(float(fa[h]), 3) for h in fa[o_jia].sort_values().index},
    "原非甲型但甲型頻率 >10% 的鄉鎮": {f"{L.loc[h, 'name']}（原{orig_ty[h]}）": round(float(fa[h]), 3) for h in fa[~o_jia & (fa > .1)].sort_values(ascending=False).index}}
out["時間"] = {"每次重抽秒數中位數(4 個程序並行、每程序 LightGBM 2 執行緒)": float(R.sec_total.median())}
sc = SCR/"sample_table_checks.json"
if sc.exists():
    x = json.loads(sc.read_text().replace("NaN", "null")); out["scale_x10"] = x["scale_x10"]; out["sample_counts"] = x["counts"]
json.dump(out, open(OUT/"rev1016_bootstrap.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ("scale_x10", "sample_counts")}, ensure_ascii=False, indent=1, default=float))
