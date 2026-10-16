"""資料鑑識：五個年度的「登記原因 × 屋齡」是用同一種規則生成的嗎？
作法：各年度以精確屋齡（每一歲）計算繼承占比 D/(B+C+D)。
若生成時只以 10 年屋齡組為條件、組內歲數另外抽，則同一屋齡組內占比應為水平線（階梯狀）。
"""
import pandas as pd, numpy as np
from common import *
b = pd.read_parquet(DATA/"bld.parquet")
d = b[b.reason.isin(list("BCD")) & b.age.notna() & (b.age <= 69)].copy()
d["D"] = (d.reason == "D").astype(float)
d["cls"] = np.minimum(d.age // 10, 4).astype(int)          # 手冊：屋齡以 10 年切 5 類
t = d.groupby(["year","age"]).agg(n=("D","size"), share=("D","mean")).reset_index()
t.to_csv(OUT/"forensic_inherit_by_exact_age.csv", index=False)
rows = []
for y, g in d.groupby("year"):
    # 組內斜率：在 10 年屋齡組內（扣除組平均後）繼承占比對歲數的線性斜率，單位 pp/歲
    for sub, lab in [(g[g.cls < 4], "0–39 年（四個 10 年組內）"), (g[(g.cls == 4)], "40–69 年（最高組內）")]:
        x = sub.age - sub.groupby("cls").age.transform("mean"); yv = sub.D - sub.groupby("cls").D.transform("mean")
        slope = (x*yv).sum()/(x*x).sum()*100
        # 置換檢定：組內打亂歲數
        rng = np.random.default_rng(0); null = []
        for _ in range(200):
            xp = sub.groupby("cls").age.transform(lambda s: rng.permutation(s.values)); xp = xp - sub.groupby("cls").age.transform("mean")
            null.append((xp*yv).sum()/(xp*xp).sum()*100)
        rows.append({"year": y, "範圍": lab, "組內斜率_pp每歲": slope, "置換虛無SD": np.std(null), "z": slope/np.std(null), "n": len(sub)})
R = pd.DataFrame(rows); R.to_csv(OUT/"forensic_within_class_slope.csv", index=False, encoding="utf-8-sig"); print(R.round(4).to_string())
# 同樣檢查建材（磚造占比）是否也只在屋齡組層級相依
d2 = b[b.age.notna() & (b.age <= 69)].copy(); d2["brick"] = (d2.materials_group_cd == "B").astype(float)
m = d2.groupby(["year","age"]).brick.mean().unstack(0); m.to_csv(OUT/"forensic_brick_by_exact_age.csv")
print(m.loc[[38,39,40,41,45,50,55]].round(3))
# 圖
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
for ax, ys, title in [(axes[0], [108,109], "108、109 年：階梯狀"), (axes[1], [110,111,112], "110–112 年：連續上升")]:
    for y, ls in zip(ys, ["-","--",":"]):
        g = t[t.year == y]; ax.plot(g.age, g.share*100, ls, lw=1.8, label=f"{y} 年", color=C_INH if y >= 110 else C_GREY if y == 108 else "#3d4350")
    for x in [10,20,30,40]: ax.axvline(x, color="#cfd3da", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=12); ax.set_xlabel("屋齡（歲）"); ax.legend(frameon=False)
axes[0].set_ylabel("繼承占比（%）")
fig.suptitle("同一個欄位、兩種生成規則：各年度繼承占比隨屋齡的變化", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG/"fig02_generator_change.png"); print("ok")
