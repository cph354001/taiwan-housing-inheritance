"""圖 11：鄉鎮分型地圖（§4.2）。資料來自 18_typology.py 的 appendix_township_table.csv；地圖處理與 13b 相同。"""
import pandas as pd, numpy as np, json, geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Patch
from common import *
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11})
def _geom(gj):
    polys = gj["coordinates"] if gj["type"] == "MultiPolygon" else [gj["coordinates"]]; out = []
    for rings in polys:
        rings = [r for r in rings if len(r) >= 4]
        if rings: out.append(Polygon(rings[0], rings[1:]).buffer(0))
    return MultiPolygon([q for p in out for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]) if not q.is_empty])
_j = json.load(open(ROOT/"external/shp/twtown2010.json"))
g = gpd.GeoDataFrame([f["properties"] for f in _j["features"]], geometry=[_geom(f["geometry"]) for f in _j["features"]], crs=4326)
fix = {"臺北縣":"新北市","台北縣":"新北市","桃園縣":"桃園市","臺中縣":"臺中市","台中縣":"臺中市","臺南縣":"臺南市","台南縣":"臺南市","高雄縣":"高雄市"}
g["county"] = g.county.str.replace("台","臺").replace(fix); g["stem"] = g.town.str.replace("台","臺").str[:-1]
g.loc[g.county.isin(["新竹市","嘉義市"]), "stem"] = "（全市"
A = pd.read_csv(OUT/"appendix_township_table.csv"); A["county"] = A["縣市"]; A["stem"] = A.apply(lambda r: r["鄉鎮市區"].replace(r["縣市"], "", 1)[:-1], axis=1); A["型"] = A["型"].str[0]
g = g.dissolve(by=["county","stem"], as_index=False).merge(A[["county","stem","型","真實老屋買賣周轉率%"]], on=["county","stem"], how="left")
T = pd.read_csv(OUT/"typology.csv"); T["k"] = T["型"].str[0]; T = T.set_index("k")
COL = {"甲": "#c0392b", "乙": "#5dade2", "丙": "#f0a86b", "丁": "#2a78d6", "戊": "#c9ccd1"}
LAB = {"甲": "甲　組成預期占比高、繼承又高於一般鄉鎮", "乙": "乙　組成預期占比高、繼承低於一般鄉鎮", "丙": "丙　組成預期占比低、繼承高於一般鄉鎮", "丁": "丁　組成預期占比低、繼承低於一般鄉鎮", "戊": "戊　與一般鄉鎮無法區分"}
fig, ax = plt.subplots(figsize=(11.5, 7.2))
main = g.cx[119.9:122.1, 21.8:25.4]
main.plot(ax=ax, facecolor="white", edgecolor="#b0b4ba", linewidth=0.3, hatch="////")
for k, c in COL.items():
    sub = main[main["型"] == k]
    if len(sub): sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.2)
ax.set_axis_off()
handles = [Patch(color=COL[k], label=f"{LAB[k]}（{int(T.loc[k,'鄉鎮數'])} 個，老屋買賣率 {T.loc[k,'老屋真實買賣周轉率%']:.2f}）") for k in "甲乙丙丁戊"] + [Patch(facecolor="white", edgecolor="#b0b4ba", hatch="////", label="未納入分型（三類建物不足 50 棟）")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.665, 0.22), frameon=False, fontsize=10, title="分型（老屋買賣率＝實價登錄老屋買賣 ÷ 模擬老屋存量，每 100 棟每年；\n澎湖 6 個鄉鎮有分型但未繪出，金門不納入）", title_fontsize=10, alignment="left")
ax.set_title(f"甲型 {int(T.loc['甲','鄉鎮數'])} 個鄉鎮市區（{T.loc['甲','戶籍人口合計(萬)']:.0f} 萬人）：組成預期占比高、繼承又偏高，老屋買賣率也最低", loc="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig11_typology_map.png"); plt.close(); print("fig11 ok", main["型"].value_counts().to_dict())
