"""rev1014 核對 6 附帶：逐棟 offset 模型（B=50）下的變異分解與人口斜率三項，與舊版並列。需先跑 rev1014_glmm_bldoffset.py。輸出 out/rev1014_bldoffset_decomp.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy.special import logit
from common import *
t = pd.read_csv(OUT/"township_master_112.csv").merge(pd.read_csv(OUT/"rev1014_u_compare_112.csv")[["house_cd","u_new"]], on="house_cd")
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet"); d = d[(d.year == 112) & d.house_cd.isin(t.house_cd)]
t = t.merge(d.assign(lo=logit(np.clip(d.p_comp, 1e-6, 1-1e-6))).groupby("house_cd").lo.mean().rename("off_new"), on="house_cd")
def dec(off, u):
    cv = np.cov(np.c_[off, u].T, bias=True); tot = cv[0,0] + cv[1,1] + 2*cv[0,1]
    return {"組成": cv[0,0]/tot, "地方": cv[1,1]/tot, "重疊": 2*cv[0,1]/tot, "相關": cv[0,1]/np.sqrt(cv[0,0]*cv[1,1]), "總變異": tot}
x = t.lp - t.lp.mean(); sl = lambda v: float((x*(v - v.mean())).sum()/(x*x).sum())
R = {"舊_鄉鎮等權": dec(t.off, t.u), "新_鄉鎮等權": dec(t.off_new, t.u_new),
     "人口斜率_舊": {"raw": sl(t.logit_raw), "off": sl(t.off), "u": sl(t.u)}, "人口斜率_新": {"raw": sl(t.logit_raw), "off": sl(t.off_new), "u": sl(t.u_new), "resid": sl(t.logit_raw - t.off_new - t.u_new)}}
R["人口斜率_新"]["share_off_of_raw"] = R["人口斜率_新"]["off"]/R["人口斜率_新"]["raw"]; R["人口斜率_新"]["exp_u"] = float(np.exp(R["人口斜率_新"]["u"])); R["人口斜率_新"]["exp_off"] = float(np.exp(R["人口斜率_新"]["off"]))
json.dump(R, open(OUT/"rev1014_bldoffset_decomp.json","w"), ensure_ascii=False, indent=1, default=float); print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
