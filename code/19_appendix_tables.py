"""附錄 E 的兩張表（輸出 Markdown，供報告直接引用）。"""
import pandas as pd, numpy as np
from common import *
C = pd.read_csv(OUT/"real_rates_county_mean.csv").merge(pd.read_csv(OUT/"county_raw_vs_expected.csv"), left_on="county_name", right_on="county")
C = C.sort_values("r_B")
L = ["| 縣市 | 真實買賣率 | 真實繼承率 | 真實流量繼承占比 | 模擬存量繼承占比 | 只看房屋的期望 | 實際÷期望 | 屋主 65 歲以上 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in C.iterrows():
    L.append(f"| {r.county_name} | {r.r_B:.2f} | {r.r_D:.2f} | {r.fD*100:.1f}% | {r['原始%']:.1f}% | {r['只看房屋的期望%']:.1f}% | {r['O/E']:.2f} | {r.owner_65*100:.1f}% |")
T = pd.read_csv(OUT/"lvr_township.csv"); T = T[T.n >= 300].sort_values("u")
M = ["| 鄉鎮市區 | 戶籍人口 | 實際繼承占比 | 只看房屋的期望 | 地方效果 u | 真實老屋買賣周轉率 |", "|---|---:|---:|---:|---:|---:|"]
fmt = lambda r: f"| {r['name']} | {r['pop']:,.0f} | {r.raw*100:.1f}% | {r.exp*100:.1f}% | {r.u:+.2f} | {'—' if not np.isfinite(r.turn_old) or r.lv_stock_old < 1000 else format(r.turn_old, '.2f')} |"
for _, r in T.head(12).iterrows(): M.append(fmt(r))
M.append("| … | | | | | |")
for _, r in T.tail(12).iterrows(): M.append(fmt(r))
(OUT/"appendix_E.md").write_text("## E.1 21 縣市\n\n" + "\n".join(L) + "\n\n（真實買賣率、繼承率：每 100 棟每年登記棟數，108–112 年平均，依買賣率由低到高排列；模擬存量占比、期望值與實際÷期望為 112 年、屋齡已知的建物；屋主 65 歲以上為五年平均。）\n\n## E.2 地方效果最低與最高的鄉鎮市區（至少 300 棟）\n\n" + "\n".join(M) + "\n\n（真實老屋買賣周轉率：每 100 棟每年買賣棟數；老屋存量不足 1,000 棟者不列。）\n", encoding="utf-8")
print((OUT/"appendix_E.md").read_text())
