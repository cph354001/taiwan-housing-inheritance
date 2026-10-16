"""報告用圖：圖 4、5、6、8、9、10、12、13（圖 1 在 22_summary_figure，圖 2 在 11_generator_forensics，
圖 3 在 13b_place_correlates，圖 7 在 21_township_supply_vs_sales）。檔名序＝報告圖號。"""
import os
import pandas as pd, numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr
from common import *
# 印刷寬 165 mm：雙欄圖傳入寬 9.8（common.py 再乘 0.8 → 7.84 吋），最小字 11 pt → 印出約 9 pt
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
FS = 11   # 圖內註記的最小字級
T = pd.read_csv(OUT/"township_master_112.csv"); L = pd.read_csv(OUT/"lvr_township.csv")
# 其他腳本產生、只需改檔名的圖（11 號產生圖 2、13b 號產生圖 3）
for old, new in [("fig1_generator_change.png", "fig02_generator_change.png"), ("fig4_map.png", "fig03_map.png")]:
    if (FIG/old).exists(): os.replace(FIG/old, FIG/new)
# ---------- 圖 4：房屋組成能解釋多少 ----------
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
a = ax[0]; isl = T.county.isin(ISLANDS)
a.scatter(T.exp[~isl]*100, T.raw[~isl]*100, s=np.sqrt(T.n[~isl])*0.8, color=C_BUY, alpha=0.45, lw=0, label="本島鄉鎮市區")
a.scatter(T.exp[isl]*100, T.raw[isl]*100, s=np.sqrt(T.n[isl])*0.8+8, color=C_INH, alpha=0.8, lw=0, label="澎湖、金門")
a.plot([0, 70], [0, 70], color="#444", lw=1); a.text(42, 47, "實際＝期望", fontsize=FS, rotation=37, color="#444")
a.set_xlim(0, 72); a.set_ylim(0, 95); a.set_xlabel("只看房屋條件的期望繼承占比（%）"); a.set_ylabel("實際繼承占比（%）")
a.set_title("每點一個鄉鎮市區（面積∝建物數）", loc="left", fontsize=12); a.legend(frameon=False, loc="upper left")
r = np.corrcoef(T.exp, T.raw)[0, 1]; a.text(33, 4, f"相關係數 {r:.2f}\n線上方：比房屋條件\n預期的更常繼承", fontsize=FS, color="#333")
a = ax[1]; S = T[T.n >= 300].sort_values("u").reset_index(drop=True)
a.vlines(S.index, S.u - 1.96*S.u_sd, S.u + 1.96*S.u_sd, color="#c9cdd4", lw=0.8)
a.scatter(S.index, S.u, s=7, color=np.where(S.u > 0, C_INH, C_BUY), zorder=3); a.axhline(0, color="#444", lw=0.8)
a.annotate(S.name[0], (0, S.u[0]), xytext=(10, -16), textcoords="offset points", fontsize=FS, ha="left", va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
a.annotate(S.name[len(S)-1], (len(S)-1, S.u[len(S)-1]), xytext=(-14, 6), textcoords="offset points", fontsize=FS, ha="right", va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
a.annotate(S.name[len(S)-2], (len(S)-2, S.u[len(S)-2]), xytext=(-22, 2), textcoords="offset points", fontsize=FS, ha="right", va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
a.set_xlabel(f"鄉鎮市區（≥300 棟，{len(S)} 個，依 u 排序）"); a.set_ylabel("鄉鎮偏離 u（logit）")
a.set_title("調整房屋組成後的鄉鎮偏離與 95% 區間", loc="left", fontsize=12); a.set_xticks([])
AL = T.alpha.iloc[0]
sec = a.secondary_yaxis("right", functions=(lambda u: expit(logit(0.25) + AL + u)*100, lambda p: logit(np.clip(p, 1e-3, 99.9)/100) - logit(0.25) - AL)); sec.set_ylabel("換算：參考房屋在當地的占比（%）", fontsize=FS); sec.set_yticks([20, 25, 30, 35, 40, 45])
fig.suptitle("繼承占比大半跟著房屋組成走，但扣掉房屋後仍有穩定的地方差異（112 年）", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig04_composition_vs_place.png"); plt.close()
# ---------- 圖 5：地方效果是不是雜訊？ ----------
e = {y: pd.read_csv(OUT/f"township_effects_{y}.csv") for y in [110, 111, 112]}
m = e[111][["house_cd","n","O","E"]].merge(e[112][["house_cd","n","O","E","name"]], on="house_cd", suffixes=("_a","_b")); m = m[(m.n_a >= 300) & (m.n_b >= 300)]
m["x"] = np.log(m.O_a/m.E_a); m["y"] = np.log(m.O_b/m.E_b)
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
a = ax[0]; a.scatter(m.x, m.y, s=14, color=C_GIFT, alpha=0.6, lw=0); lim = [-0.75, 0.95]; a.plot(lim, lim, color="#444", lw=1); a.set_xlim(lim); a.set_ylim(lim)
a.set_xticks([-0.5, 0, 0.5]); a.set_yticks([-0.5, 0, 0.5])
a.set_xlabel("111 年：log(實際 ÷ 期望)"); a.set_ylabel("112 年：log(實際 ÷ 期望)")
a.set_title(f"兩年各自產製（r = {np.corrcoef(m.x, m.y)[0,1]:.2f}）", loc="left", fontsize=12)
a.text(0.97, 0.03, f"兩年都 ≥300 棟的\n{len(m)} 個鄉鎮市區", transform=a.transAxes, ha="right", va="bottom", fontsize=FS, color="#333")
P = pd.read_csv(OUT/"oos_prediction_112.csv").set_index("模型"); a = ax[1]
mods = [("只用全國平均", "全國\n平均", C_GREY), ("只知道鄉鎮（不看房子）", "只知道\n鄉鎮過去\n占比", "#eea47f"), ("房屋條件", "房屋\n條件", C_BUY), ("房屋條件＋縣市", "房屋\n＋縣市", C_BUY), ("房屋條件＋鄉鎮", "房屋\n＋鄉鎮", C_INH)]
v = P.loc[[k for k, _, _ in mods], "鄉鎮層級RMSE_pp"].values; bars = a.bar([l for _, l, _ in mods], v, color=[c for _, _, c in mods], width=0.62)
for bb, val in zip(bars, v): a.text(bb.get_x() + bb.get_width()/2, val + 0.1, f"{val:.1f}", ha="center", fontsize=FS)
a.set_ylabel("預測誤差（RMSE，百分點）"); a.set_ylim(0, v.max()*1.12); a.set_title(f"110–111 年訓練、預測 112 年（{int(P['評估鄉鎮數(≥300棟)'].iloc[0])} 個鄉鎮）", loc="left", fontsize=12)
fig.suptitle("鄉鎮偏離跨年度穩定，並具樣本外預測力", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig05_reliability_oos.png"); plt.close()
# ---------- 圖 6：人口梯度（模擬的繼承占比） ----------
Q = pd.read_csv(OUT/"population_quintiles.csv"); LQ = pd.read_csv(OUT/"lvr_population_quintiles.csv")
xl = [f"{q.replace('20%', '')}\n{int(round(p/1000))} 千人" for q, p in zip(Q.pop_q, Q["人口中位數"])]
x = np.arange(5)
fig, a = plt.subplots(figsize=(6, 4.4))
a.plot(x, Q["原始繼承占比"]*100, "-o", color=C_INH, lw=2, label="實際繼承占比"); a.plot(x, Q["只看房屋的期望"]*100, "--o", color=C_GREY, lw=2, label="只看房屋條件的期望")
a.plot(x, Q["參考老屋繼承占比"]*100, "-s", color=C_GIFT, lw=2, label="參考房屋（模型預測 25%）\n在當地的占比")
for col, c, pos in [("原始繼承占比", C_INH, {0: (0.05, -3.4), 4: (0, -3.2)}), ("只看房屋的期望", C_GREY, {0: (0, -3.2), 4: (0.05, 1.2)}), ("參考老屋繼承占比", C_GIFT, {0: (0, 1.3), 4: (0, 1.3)})]:
    for i in [0, 4]: a.text(x[i] + pos[i][0], Q[col][i]*100 + pos[i][1], f"{Q[col][i]*100:.1f}", ha="center", fontsize=FS, color=c)
a.set_xticks(x); a.set_xticklabels(xl); a.set_xlim(-0.35, 4.35); a.set_ylabel("繼承占比（%）"); a.set_ylim(6, 48); a.legend(frameon=False, loc="upper right")
a.set_xlabel(f"{int(Q['鄉鎮數'].sum())} 個鄉鎮市區依戶籍人口分五組（人口中位數）")
a.set_title("模擬資料：人口越少的鄉鎮，繼承占比越高", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig06_population_gradient_sim.png"); plt.close()
# ---------- 圖 8：問題二的主答案——地方效果 u 對真實買賣率與釋出率 ----------
S8 = pd.read_csv(OUT/"township_supply_sales.csv"); S8 = S8[(S8.S_ex >= 2000) & (S8.X_ex > 0) & ~S8.county.isin(ISLANDS)].dropna(subset=["u","bx","m"])
from matplotlib.ticker import NullFormatter
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
for a, col, lab, c in [(ax[0], "bx", "買賣率（實價登錄 ÷ 模擬存量）\n每 100 棟每年，對數尺度", C_BUY), (ax[1], "m", "屋主期望死亡率（代理指標）\n每 100 筆持有每年，對數尺度", C_INH)]:
    a.scatter(S8[col], S8.u, s=np.sqrt(S8.S_all)/9, color=c, alpha=0.45, lw=0); a.set_xscale("log")
    z = np.polyfit(np.log(S8[col]), S8.u, 1); xs = np.linspace(S8[col].min(), S8[col].max(), 50); a.plot(xs, np.polyval(z, np.log(xs)), color="#333", lw=1.2)
    a.xaxis.set_minor_formatter(NullFormatter()); a.set_xlabel(lab, fontsize=11); a.set_title(f"Spearman {spearmanr(S8[col], S8.u)[0]:.2f}（{len(S8)} 個鄉鎮市區）".replace("-", "−"), loc="left", fontsize=11.5)
ax[0].set_ylabel("調整房屋組成後的偏離 u"); ax[0].set_xticks([0.5, 1, 2, 4]); ax[0].set_xticklabels(["0.5", "1", "2", "4"]); ax[1].set_xticks([0.7, 1, 1.5, 2]); ax[1].set_xticklabels(["0.7", "1", "1.5", "2"])
fig.suptitle("調整房屋組成後的鄉鎮偏離：買賣率低的地方偏高，屋主死亡代理指標高的地方也偏高", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(w_pad=1.5); fig.savefig(FIG/"fig08_u_vs_rates.png"); plt.close()
# ---------- 圖 9：縣市的買賣率與繼承率（真實流量） ----------
C = pd.read_csv(OUT/"real_rates_county_mean.csv"); LC = pd.read_csv(OUT/"lvr_county.csv")
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.5))
a = ax[0]; a.scatter(LC.r_B_exist, LC.r_D, s=np.sqrt(LC.stock)/9, color=C_BUY, alpha=0.55, lw=0)
off = {"澎湖縣": (7, 0), "金門縣": (7, 0), "臺北市": (-10, 10), "桃園市": (10, -2), "新竹縣": (-9, -2), "基隆市": (0, 13), "臺中市": (6, 11), "彰化縣": (4, -13), "新北市": (-12, -9), "屏東縣": (0, 13)}
ha_ = {"基隆市": "center", "屏東縣": "center", "彰化縣": "center"}
for _, r_ in LC.iterrows():
    if r_.county_name in off:
        o = off[r_.county_name]; a.annotate(r_.county_name, (r_.r_B_exist, r_.r_D), xytext=o, textcoords="offset points", fontsize=FS, va="center", ha=ha_.get(r_.county_name, "left" if o[0] > 0 else "right"))
a.set_xlim(0.35, 2.3); a.set_ylim(0.4, 1.32); a.set_xlabel("每 100 棟每年既有建物買賣（棟）"); a.set_ylabel("每 100 棟每年繼承登記（棟）")
a.set_title(f"既有建物買賣多，繼承較少（r = {np.corrcoef(np.log(LC.r_B_exist), np.log(LC.r_D))[0,1]:.2f}）".replace("-", "−"), loc="left", fontsize=12)
a = ax[1]; a.scatter(C.sD*100, C.fD*100, s=28, color=C_INH, alpha=0.8, lw=0); a.plot([5, 62], [5, 62], color="#444", lw=1)
for _, r_ in C.iterrows():
    off2 = {"澎湖縣": (-7, -3), "新竹縣": (7, -8), "臺北市": (-7, 4), "金門縣": (7, -3), "桃園市": (8, -2)}
    if r_.county_name in off2: a.annotate(r_.county_name, (r_.sD*100, r_.fD*100), xytext=off2[r_.county_name], textcoords="offset points", fontsize=FS, ha="left" if off2[r_.county_name][0] > 0 else "right")
a.set_xlabel("模擬資料：存量中的繼承占比（%）"); a.set_ylabel("真實登記：流量中的繼承占比（%）")
a.set_title(f"存量占比與流量占比（Spearman {spearmanr(C.sD, C.fD)[0]:.2f}）", loc="left", fontsize=12)
fig.suptitle(f"繼承占比高的縣市：買賣少，繼承登記也多（{len(C)} 縣市，108–112 年平均）", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig09_county_rates.png"); plt.close()
# ---------- 圖 9：模擬的地方效果 vs 真實的老屋買賣率（全國鄉鎮） ----------
B = L[L.lv_stock_old >= 1000].copy(); U = pd.read_csv(OUT/"lvr_by_place_effect.csv")
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
a = ax[0]; a.scatter(B.u, B.turn_old, s=np.sqrt(B.lv_stock_old)/5, color=C_BUY, alpha=0.45, lw=0)
z = np.polyfit(B.u, np.log(B.turn_old.clip(lower=0.05)), 1); xs = np.linspace(-0.6, 0.8, 50); a.plot(xs, np.exp(np.polyval(z, xs)), color="#333", lw=1.2)
a.set_yscale("log"); a.set_ylim(0.22, 4.2); a.set_yticks([0.3, 0.5, 1, 2, 3]); a.set_yticklabels(["0.3", "0.5", "1", "2", "3"]); a.minorticks_off()
a.set_xlabel("模擬：鄉鎮偏離 u（越右繼承占比越高於預期）"); a.set_ylabel("老屋買賣率（%，對數）")
a.set_title(f"每點一個鄉鎮（{len(B)} 個，Spearman {spearmanr(B.u, B.turn_old)[0]:.2f}）".replace("-", "−"), loc="left", fontsize=12)
a = ax[1]; bars = a.bar(range(5), U["老屋周轉率%"], color=[C_BUY]*2 + [C_GREY] + [C_INH]*2, width=0.62)
for bb, val in zip(bars, U["老屋周轉率%"]): a.text(bb.get_x() + bb.get_width()/2, val + 0.03, f"{val:.2f}", ha="center", fontsize=FS)
a.set_xticks(range(5)); a.set_xticklabels(["最低\n20%", "次低", "中間", "次高", "最高\n20%"]); a.set_xlabel("鄉鎮偏離 u 分五組（←低於預期　高於預期→）"); a.set_ylabel("真實：老屋買賣率（%）"); a.set_ylim(0, U["老屋周轉率%"].max()*1.12)
a.set_title("五組的老屋買賣率", loc="left", fontsize=12)
fig.suptitle("繼承占比高於預期的鄉鎮，實價登錄中的老屋較少出售", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig10_sim_vs_real_turnover.png"); plt.close()
# ---------- 圖 10：真實老屋買賣率依人口五等分 ----------
BRK = pd.read_csv(OUT/"lvr_brick_only.csv")["磚造老屋周轉率%"].iloc[:5].values
fig, a = plt.subplots(figsize=(6, 4.4))
a.plot(x, LQ["老屋周轉率%"], "-o", color=C_BUY, lw=2, label="全部老舊低樓層（40 年以上、1–5 層）"); a.plot(x, BRK, ":s", color=C_INH, lw=2, label="其中磚造（產品較一致）")
for i in [0, 4]:
    a.text(x[i], BRK[i] + (0.07 if i == 0 else -0.13), f"{BRK[i]:.2f}", ha="center", va="center", fontsize=FS, color=C_INH)
    a.text(x[i], LQ["老屋周轉率%"][i] + (-0.1 if i == 0 else 0.08), f"{LQ['老屋周轉率%'][i]:.2f}", ha="center", va="center", fontsize=FS, color=C_BUY)
a.set_xticks(x); a.set_xticklabels(xl); a.set_xlim(-0.35, 4.35); a.set_ylabel("每 100 棟每年被買賣的棟數"); a.set_ylim(0.6, 1.9); a.legend(frameon=False, loc="upper left")
a.set_xlabel(f"{int(LQ['鄉鎮數'].sum())} 個鄉鎮市區依戶籍人口分五組（人口中位數）")
a.set_title("小鄉鎮的老屋整體較少被買賣，\n但磚造老屋差異不大", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig13_old_house_sales_by_population.png"); plt.close()
# ---------- 圖 11：合成世代反推的全國流量 vs 真實 ----------
Cm = pd.read_csv(OUT/"cohort_vs_real_national.csv"); fig, a = plt.subplots(figsize=(6.4, 4))
x = np.arange(3); a.bar(x - 0.2, Cm["模擬合計"]/1e4, 0.38, color=C_GIFT, label="模擬資料兩個快照反推"); a.bar(x + 0.2, Cm["真實登記(棟/年)"]/1e4, 0.38, color=C_GREY, label="真實登記")
for i in range(3): a.text(x[i] - 0.2, Cm["模擬合計"][i]/1e4 + 0.4, f"{Cm['模擬合計'][i]/1e4:.1f}", ha="center", fontsize=FS); a.text(x[i] + 0.2, Cm["真實登記(棟/年)"][i]/1e4 + 0.4, f"{Cm['真實登記(棟/年)'][i]/1e4:.1f}", ha="center", fontsize=FS)
a.set_xticks(x); a.set_xticklabels(["買賣", "繼承", "贈與\n（含夫妻贈與）"]); a.set_ylabel("萬棟／年"); a.set_ylim(0, Cm["真實登記(棟/年)"].max()/1e4*1.1); a.legend(frameon=False)
a.set_title("由 110、112 年兩個存量快照反推的全國年登記量", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig12_cohort_vs_real.png"); plt.close(); print("figures ok")
