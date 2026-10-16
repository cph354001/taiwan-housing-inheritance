"""rev1014 核對 4、5：人口斜率的三項分解，以及「參考房屋 25%」的換算。輸出 out/rev1014_slopes_ref.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy.special import expit, logit
from scipy.stats import norm
from common import *
t = pd.read_csv(OUT/"township_master_112.csv"); R = {"n_town": len(t), "n_bld": int(t.n.sum())}
t["lr0"] = logit(t.O/t.n).replace([np.inf,-np.inf], np.nan)
t["resid"] = t.logit_raw - t.off - t.u          # 原始 − 組成 − 地方 = α + 未被收縮吸收的部分 + 連續性校正
x = t.lp - t.lp.mean(); sl = lambda v: float((x*(v - v.mean())).sum()/(x*x).sum())
S = {k: sl(t[k]) for k in ["logit_raw","off","u","resid"]}
S["sum_off_u_resid"] = S["off"] + S["u"] + S["resid"]
S["share_off_of_raw"] = S["off"]/S["logit_raw"]; S["share_u_of_raw"] = S["u"]/S["logit_raw"]; S["share_resid_of_raw"] = S["resid"]/S["logit_raw"]
S["share_off_of_off_plus_u"] = S["off"]/(S["off"] + S["u"])
S["exp"] = {k: float(np.exp(S[k])) for k in ["logit_raw","off","u","resid"]}
m = t.lr0.notna(); xm = t.lp[m] - t.lp[m].mean(); S["slope_logit(O/n)_no_continuity_n"] = [float((xm*(t.lr0[m]-t.lr0[m].mean())).sum()/(xm*xm).sum()), int(m.sum())]
# 建物數加權版本
w = t.n.values; xw = t.lp - np.average(t.lp, weights=w); slw = lambda v: float((w*xw*(v - np.average(v, weights=w))).sum()/(w*xw*xw).sum())
SW = {k: slw(t[k]) for k in ["logit_raw","off","u","resid"]}; SW["share_off_of_raw"] = SW["off"]/SW["logit_raw"]; SW["exp"] = {k: float(np.exp(SW[k])) for k in SW if k in ["logit_raw","off","u"]}
R["slopes_equal_weight"] = S; R["slopes_bld_weight"] = SW
R["sd_resid"] = float(t.resid.std()); R["mean_resid"] = float(t.resid.mean()); R["alpha"] = float(t.alpha.iloc[0])
# ---- 5. 參考房屋 ----
AL = float(t.alpha.iloc[0]); tau = float(pd.read_csv(OUT/"composition_vs_place_summary.csv", index_col=0).loc[112, "tau_place"])
q = t.u.quantile([.05,.95]); R["ref"] = {"alpha": AL, "tau": tau, "u_P5_posterior_mean": float(q.iloc[0]), "u_P95_posterior_mean": float(q.iloc[1]),
  "with_alpha_P5_P95": [float(expit(logit(.25)+AL+q.iloc[0])), float(expit(logit(.25)+AL+q.iloc[1]))],
  "without_alpha_P5_P95": [float(expit(logit(.25)+q.iloc[0])), float(expit(logit(.25)+q.iloc[1]))],
  "with_alpha_u0": float(expit(logit(.25)+AL)),
  "N(0,tau2)_P5_P95_with_alpha": [float(expit(logit(.25)+AL+norm.ppf(.05)*tau)), float(expit(logit(.25)+AL+norm.ppf(.95)*tau))],
  "N(0,tau2)_P5_P95_without_alpha": [float(expit(logit(.25)+norm.ppf(.05)*tau)), float(expit(logit(.25)+norm.ppf(.95)*tau))],
  "sd_posterior_mean_u": float(t.u.std()), "P5_P95_township_names": [t.loc[(t.u - q.iloc[0]).abs().idxmin(), "name"], t.loc[(t.u - q.iloc[1]).abs().idxmin(), "name"]],
  "quantile_method": "pandas 線性內插，312 個鄉鎮（112 年、≥50 棟）的後驗平均 u，等權"}
# 組成模型校準常數：E 被乘上 k（12_composition_vs_place.py:30），所以「模型預測 25%」與 offset 的尺度差 k
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet"); d = d[(d.year == 112) & d.house_cd.isin(t.house_cd)]
R["ref"]["k_scale_E_112"] = float(d.D.sum()/d.p_comp.sum()); R["ref"]["mean_p_comp_112"] = float(d.p_comp.mean()); R["ref"]["mean_D_112"] = float(d.D.mean())
a4 = d[(d.age >= 40) & (d.age < 50)]; R["ref"]["mean_p_comp_age40_49"] = float(a4.p_comp.mean()); R["ref"]["actual_D_age40_49"] = float(a4.D.mean())
json.dump(R, open(OUT/"rev1014_slopes_ref.json","w"), ensure_ascii=False, indent=1, default=float); print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
