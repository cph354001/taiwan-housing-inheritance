"""圖 1：總結圖（三格）。所有數字由 out/ 的結果檔讀出。"""
import pandas as pd, numpy as np
from common import *
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
FS = 11
# ---- 讀數字
V = pd.read_csv(OUT/"variance_decomposition.csv"); V = V[(V.year == 112) & (V["權重"] == "鄉鎮等權")].iloc[0]
comp, place, both = V["只來自房屋組成"]*100, V["只來自地方"]*100, V["兩者重疊"]*100
DC = pd.read_csv(OUT/"township_supply_sales_decomp.csv", index_col=0); main = DC.loc["不含離島三縣"]
share_B, n_town = main["來自買賣率%"], int(main["鄉鎮數"])
SN = pd.read_csv(OUT/"township_decomp_sensitivity.csv", index_col=0)["來自買賣率%"]
# 敏感度範圍：存量門檻、加權、人口斜率（sensitivity 檔）＋主代理指標的變體（decomp 檔）；
# 不含「65 歲以上死亡人數」與「簡單版」兩種通不過縣市檢查的代理指標
alt = DC.loc[[i for i in DC.index if ("65" not in i) and ("簡單版" not in i)], "來自買賣率%"]
lo, hi = min(SN.min(), alt.min()), max(SN.max(), alt.max())
LS = pd.read_csv(OUT/"lvr_vs_sim.csv"); rho = LS[(LS["變數"] == "模擬：繼承的地方效果 u") & (LS["真實指標"] == "老屋買賣周轉率")]["全國Spearman"].iloc[0]
TY = pd.read_csv(OUT/"typology.csv"); jia = TY[TY["型"].str.startswith("甲")].iloc[0]
n_jia, pop_jia = int(jia["鄉鎮數"]), jia["戶籍人口合計(萬)"]
C_OVL, C_COMP = "#d8c3b5", "#b9bec6"
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1, 1.05, 1.25]})
# ---- 格 1：變異分解（直立堆疊條）
a = ax[0]; segs = [("房屋組成", comp, C_COMP), ("兩者共變", both, C_OVL), ("鄉鎮偏離", place, C_INH)]; y0 = 0
for lab, v, c in segs:
    a.bar(0, v, bottom=y0, width=0.55, color=c, edgecolor="white", lw=1.5)
    a.text(0.38, y0 + v/2, f"{lab}\n{v:.0f}%", ha="left", va="center", fontsize=12, color="#222" if lab != "鄉鎮偏離" else C_INH, fontweight="bold" if lab == "地方偏離" else "normal")
    y0 += v
a.set_xlim(-0.4, 1.35); a.set_ylim(0, 100); a.axis("off")
a.set_title("① 繼承占比的鄉鎮差異", loc="left", fontsize=13, fontweight="bold")
# ---- 格 2：示意
a = ax[1]; a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
a.set_title("② 偏離量到什麼？", loc="left", fontsize=13, fontweight="bold")
a.text(0.5, 0.93, "存量繼承占比 ≈ $h_D$ ÷ 全部移轉", ha="center", va="center", fontsize=FS, color="#333")
box = dict(boxstyle="round,pad=0.4", lw=0)
a.text(0.5, 0.69, "繼承多？ $h_D$ 高\n（屋主死亡多）", ha="center", va="center", fontsize=12, color="white", bbox={**box, "fc": C_INH})
a.text(0.5, 0.505, "還是", ha="center", va="center", fontsize=FS, color="#444")
a.text(0.5, 0.32, "買賣少？ $h_B$ 低\n（房屋較少經買賣移轉）", ha="center", va="center", fontsize=12, color="white", bbox={**box, "fc": C_BUY})
a.text(0.5, 0.1, "兩者都會讓繼承占比變高；\n存量本身分不出是哪一個", ha="center", va="center", fontsize=FS, color="#333")
# ---- 格 3：答案
a = ax[2]; a.set_xlim(0, 106); a.set_ylim(0, 1); a.axis("off")
SF = pd.read_csv(OUT/"township_stock_vs_flows.csv", index_col=0); bu, br = SF.loc["地方效果 u（扣除房屋組成）", ["標準化係數_買賣率", "標準化係數_釋出率"]]
a.set_title("③ 對照外部流量：兩種關聯都在", loc="left", fontsize=13, fontweight="bold")
a.text(0, 0.86, "調整後的鄉鎮偏離 u 對外部指標（標準化係數）", ha="left", va="center", fontsize=11.5, color="#222")
for yy, lab, v, c in [(0.68, "買賣率（實價登錄 ÷ 模擬存量）", bu, C_BUY), (0.50, "屋主期望死亡率（代理指標）", br, C_INH)]:
    a.barh(yy, abs(v)*60, height=0.10, left=0, color=c, alpha=0.9); a.text(abs(v)*60 + 2, yy, f"{v:+.2f}　{lab}".replace("-", "−"), ha="left", va="center", fontsize=11, color="#222")
a.text(0, 0.30, f"u 越高的鄉鎮，老屋買賣率越低（Spearman {rho:.2f}）".replace("-", "−"), ha="left", va="center", fontsize=11.5, color="#222")
a.text(0, 0.08, f"→ 應用示範：組成預期高、繼承又偏高的 {n_jia} 個鄉鎮", ha="left", va="center", fontsize=12, color=C_INH, fontweight="bold")
fig.tight_layout(w_pad=2.5); fig.savefig(FIG/"fig01_summary.png", pad_inches=0.12); plt.close(); print("fig01 ok")
