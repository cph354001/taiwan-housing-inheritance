"""分型的跨年穩定性：用 110、111、112 年各自獨立估計的地方效果與期望占比，分別套用 18_typology.py 的規則，看鄉鎮的分型是否一致。"""
import pandas as pd, numpy as np, json
from common import *
def classify(d):
    hi = d.exp >= d.exp.median(); pos, neg = (d.u - 1.96*d.u_sd) > 0, (d.u + 1.96*d.u_sd) < 0
    return pd.Series(np.select([hi & pos, hi & neg, ~hi & pos, ~hi & neg], list("甲乙丙丁"), default="戊"), index=d.house_cd)
E = {y: pd.read_csv(OUT/f"township_effects_{y}.csv") for y in (110, 111, 112)}
E = {y: d[d.county != "金門縣"] for y, d in E.items()}
C = pd.DataFrame({y: classify(d) for y, d in E.items()}).dropna()
res = {"鄉鎮數": int(len(C))}
for a, b in [(110, 111), (111, 112), (110, 112)]:
    res[f"{a}對{b}_分型相同比例"] = float((C[a] == C[b]).mean())
    res[f"{a}對{b}_甲型一致（{a}年甲型在{b}年仍為甲型）"] = float((C.loc[C[a] == "甲", b] == "甲").mean())
    res[f"{a}對{b}_不互相矛盾（偏繼承↔偏買賣）比例"] = float(1 - (C[a].isin(list("甲丙")) & C[b].isin(list("乙丁")) | C[a].isin(list("乙丁")) & C[b].isin(list("甲丙"))).mean())
res["三年都為甲型"] = int(((C[110] == "甲") & (C[111] == "甲") & (C[112] == "甲")).sum()); res["112年甲型數"] = int((C[112] == "甲").sum())
res["112年甲型中三年都偏繼承（甲或丙）"] = float(C.loc[C[112] == "甲"].isin(list("甲丙")).all(axis=1).mean())
res["112年甲型中其他年度曾被歸為偏買賣（乙或丁）"] = int(C.loc[C[112] == "甲", [110, 111]].isin(list("乙丁")).any(axis=1).sum())
res["各年甲型數"] = {int(y): int((C[y] == "甲").sum()) for y in (110, 111, 112)}
# 用三年合併（u 取三年平均、標準差除以 √3 近似）的分型
M = pd.concat([d.set_index("house_cd")[["u", "u_sd", "exp"]] for d in E.values()], axis=1, keys=[110, 111, 112]).dropna()
avg = pd.DataFrame({"house_cd": M.index, "u": M.xs("u", axis=1, level=1).mean(axis=1).values, "u_sd": (M.xs("u_sd", axis=1, level=1).mean(axis=1)/np.sqrt(3)).values, "exp": M.xs("exp", axis=1, level=1).mean(axis=1).values})
C["三年合併"] = classify(avg).reindex(C.index)
res["三年合併_甲型數"] = int((C["三年合併"] == "甲").sum()); res["112年甲型在三年合併中仍為甲型"] = float((C.loc[C[112] == "甲", "三年合併"] == "甲").mean())
json.dump(res, open(OUT/"typology_stability.json", "w"), ensure_ascii=False, indent=1); C.to_csv(OUT/"typology_by_year.csv", encoding="utf-8-sig"); print(json.dumps(res, ensure_ascii=False, indent=1))
