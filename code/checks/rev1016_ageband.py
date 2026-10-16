"""rev1016 審稿建議 07：地方效果是否隨屋齡／樓層改變。輸出 out/rev1016_ageband.json、out/rev1016_ageband_u112.csv
 1. 分帶（屋齡 <30、30–39、40+；樓層 1–5、6+；40+ 且 1–5 層）各自估計 u（同 glmm.py：offset＝組成模型期望，帶內校準 ΣE＝ΣO，n≥50）。
    帶間相關：收縮後 u 的 Pearson／Spearman、兩帶皆 ≥300 棟者、以及雙變量二項隨機效果模型（精確積分）估計的潛在相關（disattenuated）。
 2. 巢狀模型：w_jb = μ_b + a_j + v_jb（鄉鎮截距＋鄉鎮×屋齡帶偏離），比較 τ_v²/τ_a²；與「單一 u」「各帶獨立」比較 112 年概似，
    並以 110–111 年估計、預測 112 年鄉鎮×屋齡帶占比（組成模型只用 110–111 年重新訓練）。
 3. 各帶 u 對 log 買賣率＋log 屋主死亡釋出率（272 鄉鎮，標準化，縣市群集 SE；同 21_township_supply_vs_sales.py 的 reg）；老屋帶 u 對老屋買賣率 Spearman。
 4. 分型（18_typology.py）改用老屋帶 u：甲型名單 Jaccard。
積分：潛在 logit 偏離 x 在 [-4,4] 取 801 點（間距 0.01）的格點；大鄉鎮後驗 SD 約 0.02，梯形法在此間距已足夠（並與 glmm.fit 核對 τ）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln, logsumexp
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
from common import *
import glmm
np.random.seed(20261016)
X = np.linspace(-4, 4, 801); H = X[1] - X[0]
OUTJ = {}
d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet")
d["ab"] = pd.cut(d.age, [-1, 29, 39, 500], labels=["<30", "30-39", "40+"]).astype(str)
fg = d.floor_group_cd.astype(str)
d["fb"] = np.select([fg == "1", fg.isin(["2","3","4","5"])], ["1-5", "6+"], default="NA")
d["old_low"] = np.where((d.age >= 40) & (fg == "1"), "40+&1-5", "other")
town = pd.read_csv(DATA/"town.csv")

def cells(g, col, keep_min=50, pcol="p_comp"):
    """鄉鎮×帶的 O、n、E；E 在每帶內校準到 ΣE＝ΣO（只算 n≥keep_min 的格子）。"""
    t = g.groupby(["house_cd", col]).agg(n=("D","size"), O=("D","sum"), E=(pcol,"sum")).reset_index()
    t = t[t.n >= keep_min].copy()
    t["cal"] = t.groupby(col).O.transform("sum")/t.groupby(col).E.transform("sum"); t["E"] *= t.cal
    t["off"] = logit(t.E/t.n); return t

def logL(O, n, off, shift=0.0):
    """各格子在潛在偏離格點 X 上的二項 log 概似（含組合數）。"""
    eta = off[:, None] + X[None, :] + shift
    return (gammaln(n+1) - gammaln(O+1) - gammaln(n-O+1))[:, None] + O[:, None]*(-np.logaddexp(0, -eta)) + (n-O)[:, None]*(-np.logaddexp(0, eta))

# ---------- 1. 單帶：glmm.fit ----------
def band_fit(g, col, lev):
    t = cells(g, col); t = t[t[col] == lev]
    f = glmm.fit(t.O, t.n, t.off); t = t.assign(u=f["u"], u_sd=f["u_sd"], alpha_=f["beta"][0], tau_=f["tau"])
    return t.set_index("house_cd"), f
SPEC = {"age": ("ab", ["<30","30-39","40+"]), "floor": ("fb", ["1-5","6+"]), "oldlow": ("old_low", ["40+&1-5"])}
U = {}; TAU = {}
for y in [110, 111, 112]:
    g = d[d.year == y]
    for k, (col, levs) in SPEC.items():
        for lev in levs:
            t, f = band_fit(g, col, lev); TAU[(y, lev)] = {"tau": float(f["tau"]), "alpha": float(f["beta"][0]), "n_town": int(len(t)), "n_bld": int(t.n.sum()), "national_share": float(t.O.sum()/t.n.sum())}
            if y == 112: U[lev] = t
main_u = pd.read_csv(OUT/"township_effects_112.csv").set_index("house_cd")
TAU.update({(y, "全部(主分析)"): {"tau": float(pd.read_csv(OUT/"composition_vs_place_summary.csv", index_col=0).loc[y, "tau_place"])} for y in [110,111,112]})
OUTJ["1_各帶tau"] = {f"{y}_{lev}": v for (y, lev), v in TAU.items()}
U112 = pd.DataFrame({lev: t.u for lev, t in U.items()}).join(pd.DataFrame({f"n_{lev}": t.n for lev, t in U.items()})).join(main_u.u.rename("u_all"))
U112.join(pd.DataFrame({f"usd_{lev}": t.u_sd for lev, t in U.items()})).to_csv(OUT/"rev1016_ageband_u112.csv", encoding="utf-8-sig")

# ---------- 1b. 雙變量二項隨機效果（精確格點積分）：潛在相關 ----------
def biv_fit(t1, t2):
    j = t1.index.intersection(t2.index); a, b = t1.loc[j], t2.loc[j]
    L1, L2 = logL(a.O.values, a.n.values, a.off.values), logL(b.O.values, b.n.values, b.off.values)
    m1, m2 = L1.max(1, keepdims=True), L2.max(1, keepdims=True); E1, E2 = np.exp(L1 - m1), np.exp(L2 - m2)
    def nll(p):
        mu1, mu2, s1, s2, r = p[0], p[1], np.exp(p[2]), np.exp(p[3]), np.tanh(p[4])
        z1, z2 = (X - mu1)/s1, (X - mu2)/s2
        Phi = np.exp(-(z1[:, None]**2 - 2*r*z1[:, None]*z2[None, :] + z2[None, :]**2)/(2*(1 - r*r)))/(2*np.pi*s1*s2*np.sqrt(1 - r*r))
        lik = ((E1 @ Phi)*E2).sum(1)*H*H
        v = -(np.log(lik) + m1[:, 0] + m2[:, 0]).sum()
        return v if np.isfinite(v) else 1e12
    # 起點：各帶單獨的 α、τ 與收縮 u 的相關；Nelder–Mead（L-BFGS 的線搜尋會踩到下溢點）
    p0 = np.r_[t1.alpha_.iloc[0], t2.alpha_.iloc[0], np.log(t1.tau_.iloc[0]), np.log(t2.tau_.iloc[0]), np.arctanh(np.corrcoef(a.u, b.u)[0,1])]
    r = minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000, "maxfev": 8000})
    # 數值 Hessian 求 atanh(ρ) 的 SE
    e = 1e-3; k = len(r.x); Hs = np.zeros((k, k)); f0 = nll(r.x)
    for i in range(k):
        for l in range(i, k):
            ei, el = np.eye(k)[i]*e, np.eye(k)[l]*e
            Hs[i, l] = Hs[l, i] = (nll(r.x+ei+el) - nll(r.x+ei-el) - nll(r.x-ei+el) + nll(r.x-ei-el))/(4*e*e)
    try: se_z = float(np.sqrt(np.linalg.inv(Hs)[4, 4]))
    except Exception: se_z = np.nan
    z = r.x[4]
    return {"converged": bool(r.success), "nll": float(r.fun), "nll_at_start": float(nll(p0)), "n_town": int(len(j)), "tau1": float(np.exp(r.x[2])), "tau2": float(np.exp(r.x[3])), "latent_rho": float(np.tanh(z)), "rho_95CI": [float(np.tanh(z - 1.96*se_z)), float(np.tanh(z + 1.96*se_z))]}
PAIRS = [("<30","30-39"), ("30-39","40+"), ("<30","40+"), ("1-5","6+"), ("40+&1-5","<30")]
C = {}
for a_, b_ in PAIRS:
    t1, t2 = U[a_], U[b_]; j = t1.index.intersection(t2.index); jb = j[(t1.loc[j].n >= 300) & (t2.loc[j].n >= 300)]
    lr = lambda t: np.log((t.O + .5)/(t.E + .5))
    C[f"{a_} vs {b_}"] = {"n_town(both n>=50)": int(len(j)), "Pearson_shrunk_u": float(np.corrcoef(t1.loc[j].u, t2.loc[j].u)[0,1]), "Spearman_shrunk_u": float(spearmanr(t1.loc[j].u, t2.loc[j].u)[0]),
        "n_town(both n>=300)": int(len(jb)), "Pearson_shrunk_u_both300": float(np.corrcoef(t1.loc[jb].u, t2.loc[jb].u)[0,1]), "Pearson_logOE_both300": float(np.corrcoef(lr(t1.loc[jb]), lr(t2.loc[jb]))[0,1]),
        "exact_bivariate": biv_fit(t1, t2)}
    print(a_, b_, C[f"{a_} vs {b_}"])
C["各帶u與主分析u的Pearson"] = {lev: float(U112[[lev, "u_all"]].dropna().corr().iloc[0,1]) for lev in U}
OUTJ["1_帶間相關_112"] = C

# ---------- 2. 巢狀模型：鄉鎮截距＋鄉鎮×屋齡帶 ----------
BANDS = ["<30","30-39","40+"]
def prep(t, col="ab"):
    hc = sorted(t.house_cd.unique()); idx = {h: i for i, h in enumerate(hc)}
    Ls, Ms, Ps = [], [], []
    for b in BANDS:
        s = t[t[col] == b]; L = np.zeros((len(hc), len(X))); P = np.zeros(len(hc), bool)
        ii = s.house_cd.map(idx).values; L[ii] = logL(s.O.values, s.n.values, s.off.values); P[ii] = True
        m = np.where(P, L.max(1), 0.0); Ls.append(np.where(P[:, None], np.exp(L - m[:, None]), 1.0)); Ms.append(m); Ps.append(P)
    return hc, Ls, Ms, Ps
def nested(p, D, post=False):
    hc, Ls, Ms, Ps = D; mu = p[:3]; ta, tv = np.exp(p[3]), np.exp(p[4])
    pa = norm.pdf(X, 0, ta)*H                                     # a 的先驗權重（同一格點）
    G, NUM = [], []
    for b in range(3):
        K = norm.pdf(X[:, None] - mu[b] - X[None, :], 0, tv)*H        # K[x, a]
        g = Ls[b] @ K; g = np.where(Ps[b][:, None], g, 1.0); G.append(g)
        if post: NUM.append(np.where(Ps[b][:, None], (Ls[b]*X[None, :]) @ K, (mu[b] + X)[None, :]))
    prod = pa[None, :]*G[0]*G[1]*G[2]; lik = prod.sum(1)
    ll = np.log(lik) + sum(Ms)
    if not post: return -ll.sum()
    W = prod/lik[:, None]
    return {b: (W*NUM[i]/G[i]).sum(1) for i, b in enumerate(BANDS)}, (W*X[None, :]).sum(1)
def single(p, D, post=False):
    """單一 u：w_jb = μ_b + a_j（帶內相同的鄉鎮偏離）。直接在 a 格點上計算二項概似。"""
    hc, raw = D; mu = p[:3]; ta = np.exp(p[3]); pa = np.log(norm.pdf(X, 0, ta)*H)
    S = np.zeros((len(hc), len(X)))
    for i, b in enumerate(BANDS):
        s = raw[b]; S[s[0]] += logL(s[1], s[2], s[3], mu[i])
    M = S + pa[None, :]; ll = logsumexp(M, axis=1)
    if not post: return -ll.sum()
    W = np.exp(M - ll[:, None]); ea = (W*X[None, :]).sum(1); return {b: mu[i] + ea for i, b in enumerate(BANDS)}, ea
def prep_single(t, col="ab"):
    hc = sorted(t.house_cd.unique()); idx = {h: i for i, h in enumerate(hc)}
    return hc, {b: (t[t[col] == b].house_cd.map(idx).values, t[t[col] == b].O.values, t[t[col] == b].n.values, t[t[col] == b].off.values) for b in BANDS}
def fit_models(t):
    Dn, Ds = prep(t), prep_single(t)
    rn = minimize(nested, np.r_[0,0,0,np.log(.25),np.log(.15)], args=(Dn,), method="L-BFGS-B", bounds=[(-1,1)]*3 + [(np.log(.02),np.log(1.5))]*2)
    rs = minimize(single, np.r_[0,0,0,np.log(.25)], args=(Ds,), method="L-BFGS-B", bounds=[(-1,1)]*3 + [(np.log(.02),np.log(1.5))])
    sep = {}; llsep = 0.0
    for b in BANDS:
        s = t[t.ab == b]; f = glmm.fit(s.O, s.n, s.off); sep[b] = (s.house_cd.values, f["beta"][0] + f["u"], f["tau"], f["beta"][0]); llsep += f["ll"]
    return {"nested": (rn, Dn), "single": (rs, Ds), "sep": sep, "ll": {"single(單一u)": -rs.fun, "nested(截距+帶偏離)": -rn.fun, "separate(各帶獨立)": llsep}}
t112 = cells(d[d.year == 112], "ab")
F = fit_models(t112)
rn = F["nested"][0]; ta, tv = np.exp(rn.x[3]), np.exp(rn.x[4]); rs = F["single"][0]
k_par = {"single(單一u)": 4, "nested(截距+帶偏離)": 5, "separate(各帶獨立)": 6}
OUTJ["2_巢狀模型_112"] = {"n_cells": int(len(t112)), "n_town": int(t112.house_cd.nunique()), "tau_a(鄉鎮截距)": float(ta), "tau_v(鄉鎮×屋齡帶)": float(tv), "tau_v2/tau_a2": float(tv**2/ta**2),
    "帶偏離占總變異": float(tv**2/(ta**2 + tv**2)), "隱含的帶間潛在相關": float(ta**2/(ta**2 + tv**2)), "tau(單一u模型)": float(np.exp(rs.x[3])),
    "loglik": F["ll"], "AIC": {k: float(2*k_par[k] - 2*v) for k, v in F["ll"].items()}, "LR(nested vs single)": float(2*(F["ll"]["nested(截距+帶偏離)"] - F["ll"]["single(單一u)"])),
    "mu_b(nested)": dict(zip(BANDS, rn.x[:3].round(4).tolist()))}
print(OUTJ["2_巢狀模型_112"])
# 110 與 111 年也估一次 τ_a、τ_v（穩定性）
for y in [110, 111]:
    ty = cells(d[d.year == y], "ab"); Dn = prep(ty)
    r = minimize(nested, np.r_[0,0,0,np.log(.25),np.log(.15)], args=(Dn,), method="L-BFGS-B", bounds=[(-1,1)]*3 + [(np.log(.02),np.log(1.5))]*2)
    OUTJ["2_巢狀模型_112"][f"{y}年 tau_a, tau_v"] = [float(np.exp(r.x[3])), float(np.exp(r.x[4]))]

# ---------- 2b. 樣本外：110–111 年估計，預測 112 年鄉鎮×屋齡帶占比 ----------
FEATS = ["age","materials_group_cd","floor_group_cd","purpose_group_cd"]
dd = d.copy()
for c in FEATS[1:]: dd[c] = dd[c].astype(str).astype("category")
tr, te = dd[dd.year < 112].copy(), dd[dd.year == 112].copy()
gm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, random_state=1)
gm.fit(tr[FEATS], tr.D); tr["p_tr"] = gm.predict_proba(tr[FEATS])[:, 1]; te["p_tr"] = gm.predict_proba(te[FEATS])[:, 1]
ttr = cells(tr, "ab", pcol="p_tr")                                             # 110+111 合併
calf = ttr.groupby("ab").cal.first()
Ft = fit_models(ttr)
Wn, _ = nested(Ft["nested"][0].x, Ft["nested"][1], post=True); hcn = Ft["nested"][1][0]
Ws, _ = single(Ft["single"][0].x, Ft["single"][1], post=True); hcs = Ft["single"][1][0]
mu_n, mu_s = Ft["nested"][0].x[:3], Ft["single"][0].x[:3]
g12 = te.groupby(["house_cd","ab"]).agg(n=("D","size"), O=("D","sum"), E=("p_tr","sum")).reset_index()
g12 = g12[g12.n >= 50].copy(); g12["E"] *= g12.ab.map(calf).values; g12["off"] = logit(g12.E/g12.n)
raw_tr = ttr.set_index(["house_cd","ab"]); g12["raw_tr"] = [raw_tr.O.get((h, b), np.nan)/raw_tr.n.get((h, b), np.nan) if (h, b) in raw_tr.index else np.nan for h, b in zip(g12.house_cd, g12.ab)]
def look(hc, Wd, mu):
    ix = {h: i for i, h in enumerate(hc)}
    return np.array([Wd[b][ix[h]] if h in ix else mu[BANDS.index(b)] for h, b in zip(g12.house_cd, g12.ab)])
sepmap = {b: dict(zip(*Ft["sep"][b][:2])) for b in BANDS}
W_sep = np.array([sepmap[b].get(h, Ft["sep"][b][3]) for h, b in zip(g12.house_cd, g12.ab)])
PRED = {"只有房屋組成": expit(g12.off + g12.ab.map(dict(zip(BANDS, mu_n))).values),
        "組成＋單一u": expit(g12.off + look(hcs, Ws, mu_s)), "組成＋巢狀(截距+帶偏離)": expit(g12.off + look(hcn, Wn, mu_n)),
        "組成＋各帶獨立u": expit(g12.off + W_sep), "110–111年原始占比(不看組成)": g12.raw_tr.fillna(ttr.O.sum()/ttr.n.sum()).values}
def score(p, sub):
    p = np.clip(np.asarray(p, float)[sub], 1e-6, 1-1e-6); O, n = g12.O.values[sub], g12.n.values[sub]
    ll = (O*np.log(p) + (n-O)*np.log(1-p)).sum(); big = n >= 300
    return {"binom_loglik": float(ll), "RMSE_pp(全部格子)": float(np.sqrt(((O/n - p)**2).mean())*100), "加權RMSE_pp(棟數)": float(np.sqrt(np.average((O/n - p)**2, weights=n))*100),
            "RMSE_pp(格子n>=300)": float(np.sqrt(((O/n - p)[big]**2).mean())*100), "n_cells": int(sub.sum()), "n_cells_300": int(big.sum())}
OOS = {}
for name, p in PRED.items():
    OOS[name] = {"全部帶": score(p, np.ones(len(g12), bool)), **{b: score(p, (g12.ab == b).values) for b in BANDS}}
base = OOS["組成＋單一u"]["全部帶"]["binom_loglik"]
for name in OOS: OOS[name]["全部帶"]["loglik相對單一u"] = OOS[name]["全部帶"]["binom_loglik"] - base
OUTJ["2b_樣本外預測112"] = {"說明": "組成模型只用 110–111 年訓練；110–111 合併估計；112 年格子 n≥50；E 用訓練期各帶校準係數", "訓練期tau_a,tau_v": [float(np.exp(Ft["nested"][0].x[3])), float(np.exp(Ft["nested"][0].x[4]))],
    "訓練期loglik": Ft["ll"], **OOS}
print(json.dumps(OUTJ["2b_樣本外預測112"], ensure_ascii=False, indent=1))

# ---------- 3. 各帶 u 的主要迴歸 ----------
S = pd.read_csv(OUT/"township_supply_sales.csv").set_index("house_cd"); MAIN = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)].copy()
def reg(df, yv):
    Xd = pd.DataFrame({"log買賣率": np.log(df.bx), "log釋出率": np.log(df.m)}); Z = (Xd - Xd.mean())/Xd.std(); yy = (yv - yv.mean())/yv.std()
    f = sm.OLS(yy, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": df.county})
    return {"n": int(len(df)), "b_買賣率": float(f.params["log買賣率"]), "se_買賣率": float(f.bse["log買賣率"]), "b_釋出率": float(f.params["log釋出率"]), "se_釋出率": float(f.bse["log釋出率"]), "R2": float(f.rsquared)}
R3 = {"主分析u（核對 −0.48/+0.38）": reg(MAIN, MAIN.u)}
for lev in U:
    m = MAIN.join(U[lev].u.rename("ub"), how="inner").dropna(subset=["ub"]); R3[lev] = reg(m, m.ub)
    mm = MAIN.join(U[lev][["u","n"]].rename(columns={"u":"ub","n":"nb"}), how="inner"); mm = mm[mm.nb >= 300]; R3[lev + "（該帶≥300棟）"] = reg(mm, mm.ub)
    R3[lev + "：主分析u限同一批鄉鎮"] = reg(m, m.u)
OUTJ["3_各帶主要迴歸"] = R3; print(pd.DataFrame(R3).T.round(3).to_string())
LT = pd.read_csv(OUT/"lvr_township.csv").set_index("house_cd"); big = LT[LT.lv_stock_old >= 1000]
SP = {}
for lev in ["u_all"] + list(U):
    m = big.join(U112[[lev]], how="inner").dropna(subset=[lev, "turn_old"]) if lev != "u_all" else big.dropna(subset=["u","turn_old"]).assign(u_all=lambda x: x.u)
    SP[lev] = {"Spearman(u_band, 老屋買賣率)": float(spearmanr(m[lev], m.turn_old)[0]), "n": int(len(m)), "同批鄉鎮主分析u": float(spearmanr(m.u, m.turn_old)[0]),
               "不含金門": float(spearmanr(m[m.county != "金門縣"][lev], m[m.county != "金門縣"].turn_old)[0])}
OUTJ["3b_u對老屋買賣率Spearman(老屋存量≥1000)"] = SP; print(SP)

# ---------- 4. 分型改用老屋帶 u ----------
def typ(Lx, u, usd):
    hi_c = Lx.exp >= Lx.exp.median(); pos, neg = (u - 1.96*usd) > 0, (u + 1.96*usd) < 0
    t = pd.Series(np.select([hi_c & pos, hi_c & neg, ~hi_c & pos, ~hi_c & neg], list("甲乙丙丁"), default="戊"), index=Lx.index)
    t[Lx.county == "金門縣"] = "金門"; return t
L = pd.read_csv(OUT/"lvr_township.csv").set_index("house_cd")
T0 = typ(L, L.u, L.u_sd)
T4 = {}
for lev in ["40+", "40+&1-5", "<30", "30-39"]:
    ub = U[lev].u.reindex(L.index); sb = U[lev].u_sd.reindex(L.index)
    T1 = typ(L, ub, sb)
    a, b = set(T0[T0 == "甲"].index), set(T1[T1 == "甲"].index)
    T4[lev] = {"原甲型數": len(a), "改用該帶u甲型數": len(b), "交集": len(a & b), "Jaccard_甲": len(a & b)/len(a | b) if a | b else np.nan,
               "原甲型中該帶u缺值(<50棟)": int(ub.reindex(list(a)).isna().sum()),
               "原甲型在該帶的歸類": T1.reindex(sorted(a)).value_counts().to_dict(), "新增甲型": [L.name[h] for h in sorted(b - a)], "掉出甲型": [L.name[h] for h in sorted(a - b)],
               "Jaccard_丁": (lambda a_, b_: len(a_ & b_)/len(a_ | b_))(set(T0[T0 == "丁"].index), set(T1[T1 == "丁"].index)),
               "原甲型的該帶u>0比例": float((ub.reindex(list(a)) > 0).mean()), "原甲型的該帶u中位數": float(ub.reindex(list(a)).median())}
    # 只用點估計（不要求區間排除 0）的寬鬆版：u_band>0 且組成高
    T4[lev]["寬鬆版(u>0)甲型Jaccard"] = (lambda a_, b_: len(a_ & b_)/len(a_ | b_))(a, set(L.index[(L.exp >= L.exp.median()) & (ub > 0) & (L.county != "金門縣")]))
OUTJ["4_分型"] = T4; print(json.dumps(T4, ensure_ascii=False, indent=1))
json.dump(OUTJ, open(OUT/"rev1016_ageband.json", "w"), ensure_ascii=False, indent=1, default=float)
print("done")
