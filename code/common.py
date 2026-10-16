"""共用設定：路徑、字型、代碼標籤。"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT/"data", ROOT/"out", ROOT/"fig"
HANDOFF = ROOT/"ref"                      # 鄉鎮、縣市代碼對照表（ref/analysis/）
RAW_DIR = Path(os.environ.get("HOUSING_RAW_DIR", ROOT/"raw"))   # 原始模擬資料（需向 SRDA 申請，不隨附）
PKG2 = RAW_DIR
RAW_OWN = RAW_DIR/"108-112_house_own_subset_sumy_rawdata_simulation.csv"
RAW_PPL = RAW_DIR/"108-112_moi_people_house_dtl_simulation.csv"
REAL = ROOT/"external/external_validation"   # 外部公開資料（見 README）
plt.rcParams["font.family"] = ["PingFang HK", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
C_BUY, C_INH, C_GIFT, C_GREY = "#2a78d6", "#d9622b", "#7a5fb5", "#8a8f98"
REASON = {"A": "第一次登記", "B": "買賣", "C": "贈與", "D": "繼承", "E": "拍賣", "F": "其他", "U": "不明"}
ISLANDS = ["澎湖縣", "金門縣", "連江縣"]

# 報告版面寬 16.5 cm：把圖的尺寸統一縮為 0.8 倍，讓字在印刷時夠大
_orig_subplots = plt.subplots
def _subplots(*a, **k):
    if "figsize" in k: k["figsize"] = (k["figsize"][0]*0.8, k["figsize"][1]*0.8)
    return _orig_subplots(*a, **k)
plt.subplots = _subplots
