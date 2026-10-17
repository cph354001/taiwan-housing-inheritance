"""應用：把鄉鎮依「房屋組成（期望繼承占比）」與「地方效果」分成四型，並附上真實老屋買賣周轉率。"""
import pandas as pd, numpy as np
from common import *
L = pd.read_csv(OUT/"lvr_township.csv")
L["老屋占比"] = L.lv_stock_old/L.lv_stock
hi_c = L.exp >= L.exp.median()
pos, neg = (L.u - 1.96*L.u_sd) > 0, (L.u + 1.96*L.u_sd) < 0          # 地方效果的 95% 後驗區間不含 0 才歸類
L["型"] = np.select([hi_c & pos, hi_c & neg, ~hi_c & pos, ~hi_c & neg], ["甲 組成預期占比高、繼承又高於一般鄉鎮", "乙 組成預期占比高、繼承低於一般鄉鎮", "丙 組成預期占比低、繼承高於一般鄉鎮", "丁 組成預期占比低、繼承低於一般鄉鎮"], default="戊 與一般鄉鎮無法區分")
L.loc[L.county == "金門縣", "型"] = "（金門不納入分型，見附錄 E）"      # 金門的模擬地方效果與真實登記不一致
G = L[L.county != "金門縣"].groupby("型").apply(lambda g: pd.Series({"鄉鎮數": len(g), "戶籍人口合計(萬)": g["pop"].sum()/1e4, "人口中位數": g["pop"].median(), "實際繼承占比%": g.O.sum()/g.n.sum()*100, "期望繼承占比%": g.E.sum()/g.n.sum()*100, "實際÷期望": g.O.sum()/g.E.sum(),
     "老屋真實買賣周轉率%": g.lv_sales_old.sum()/g.lv_stock_old.sum()*100, "老屋平均幾年買賣一次": g.lv_stock_old.sum()/g.lv_sales_old.sum(), "居民65歲以上%": np.average(g.res_65, weights=g.res_n)*100,
     "例": "、".join(g.sort_values("n", ascending=False).name.head(4).str[3:])}), include_groups=False)
G.to_csv(OUT/"typology.csv", encoding="utf-8-sig"); pd.set_option("display.width", 250); print(G.round(2).to_string())
cols = ["name","county","pop","n","raw","exp","u","u_sd","p_ref","turn_old","型","res_65","owner_65"]
A = L[cols].rename(columns={"name":"鄉鎮市區","county":"縣市","pop":"戶籍人口","n":"建物數(買賣+贈與+繼承)","raw":"實際繼承占比","exp":"只看房屋的期望占比","u":"地方效果u","u_sd":"u的後驗SD","p_ref":"參考房屋(25%)在當地的占比","turn_old":"真實老屋買賣周轉率%","res_65":"居民65+比例","owner_65":"屋主65+比例"})
A.sort_values("地方效果u", ascending=False).round(4).to_csv(OUT/"appendix_township_table.csv", index=False, encoding="utf-8-sig")
Q = pd.read_csv(OUT/"lvr_population_quintiles.csv"); print((100/Q["老屋周轉率%"]).round(0).tolist())
