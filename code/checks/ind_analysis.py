import sys, json, pandas as pd, numpy as np, statsmodels.api as sm
from scipy.special import logit, expit
from scipy.stats import spearmanr, pearsonr
R = "./"
SP = "./scratch/"
sys.path.insert(0, R+"code"); import glmm
P = pd.read_parquet(SP+"pred_BCD.parquet"); town = pd.read_csv(R+"data/town.csv")[["house_cd","county"]]
def effects(df, tgt, pcol):
    g = df.assign(y=(df.reason==tgt).astype(int)).groupby("house_cd").agg(n=("y","size"), O=("y","sum"), E=(pcol,"sum"))
    g = g[g.n>=50].copy(); g["E"] *= g.O.sum()/g.E.sum(); f = glmm.fit(g.O, g.n, logit(g.E/g.n))
    g["u"], g["u_sd"] = f["u"], f["u_sd"]; g["lr"] = np.log((g.O+.5)/(g.E+.5)); return g, f["tau"]
U, TAU = {}, {}
for yr in (110,111,112):
    py = P[P.year==yr]
    for t in "BCD": U[(t,yr)], TAU[(t,yr)] = effects(py, t, "p_"+t)
print("tau", {f"{k[0]}{k[1]}": round(v,4) for k,v in TAU.items()})
rows = []
for yr in (110,111,112):
    ref = pd.read_csv(R+f"out/township_effects_{yr}.csv").set_index("house_cd")
    m = pd.DataFrame({t: U[(t,yr)].u for t in "BCD"}).join(town.set_index("house_cd")).dropna()
    dm = lambda x: x - x.groupby(m.county).transform("mean")
    rows.append({"yr":yr, "n":len(m), "tau_C":TAU[("C",yr)], "tau_B":TAU[("B",yr)], "tau_D(mine)":TAU[("D",yr)],
                 "maxdiff uD vs 12": float((m.D-ref.u.reindex(m.index)).abs().max()),
                 "r(D,C)":m.D.corr(m.C), "rho(D,C)":spearmanr(m.D,m.C)[0], "within r(D,C)":dm(m.D).corr(dm(m.C)),
                 "r(D,B)":m.D.corr(m.B), "r(C,B)":m.C.corr(m.B),
                 "R2 uB~uD+uC": sm.OLS(m.B, sm.add_constant(m[["D","C"]])).fit().rsquared})
print(pd.DataFrame(rows).round(4).to_string())
# 272 鄉鎮
S = pd.read_csv(R+"out/township_supply_sales.csv").set_index("house_cd")
S = S[(S.S_ex>=2000)&(S.X_ex>0)&~S.county.isin(["澎湖縣","金門縣","連江縣"])]
M = S[["county","lp","bx","m","owner_65"]].copy()
for t in "BCD": M["u"+t] = U[(t,112)].u
M = M.dropna(); M["lb"], M["lm"] = np.log(M.bx), np.log(M.m); print("N272 =", len(M))
z = lambda s: (s-s.mean())/s.std()
def reg(yv):
    f = sm.OLS(z(yv), sm.add_constant(pd.DataFrame({"lb":z(M.lb),"lm":z(M.lm)}))).fit(cov_type="cluster", cov_kwds={"groups":pd.factorize(M.county)[0]})
    return [round(f.params.lb,3), round(f.bse.lb,3), round(f.params.lm,3), round(f.bse.lm,3), round(f.rsquared,3), round(spearmanr(yv,M.lb)[0],3), round(spearmanr(yv,M.lm)[0],3)]
for t in "DCB": print("reg", t, reg(M["u"+t]))
b1 = np.polyfit(M.uC, M.uD, 1); com = np.polyval(b1, M.uC); res = M.uD - com
print("corr(uD,uC)272", round(M.uD.corr(M.uC),4), "R2", round(M.uD.corr(M.uC)**2,4))
print("common rho(lb,lm)", round(spearmanr(com,M.lb)[0],3), round(spearmanr(com,M.lm)[0],3), " resid", round(spearmanr(res,M.lb)[0],3), round(spearmanr(res,M.lm)[0],3))
rz = lambda a: sm.OLS(a, sm.add_constant(M.lp)).fit().resid
print("uB rho bx", round(spearmanr(M.uB,M.bx)[0],4), " partial r (|lp)", round(np.corrcoef(rz(M.uB), rz(M.lb))[0,1],4), " partial Spearman", round(spearmanr(rz(M.uB), rz(M.lb))[0],4))
print("corr(lb,lm)", round(M.lb.corr(M.lm),3), " owner_65: rho uD", round(spearmanr(M.uD,M.owner_65)[0],3), "uC", round(spearmanr(M.uC,M.owner_65)[0],3), "uB", round(spearmanr(M.uB,M.owner_65)[0],3))
# ---- 2(b) 檢驗
print("\n== 2(b)")
print("uD itself rho(lb,lm)", round(spearmanr(M.uD,M.lb)[0],3), round(spearmanr(M.uD,M.lm)[0],3))
rng = np.random.default_rng(0)
def split(Z):
    c = np.polyval(np.polyfit(Z, M.uD, 1), Z); r = M.uD - c
    return [spearmanr(c,M.lb)[0], spearmanr(c,M.lm)[0], spearmanr(r,M.lb)[0], spearmanr(r,M.lm)[0], np.corrcoef(Z,M.uD)[0,1]]
# (i) 純雜訊 placebo
A = np.array([split(rng.normal(size=len(M))) for _ in range(500)]); print("noise placebo mean [com_lb,com_lm,res_lb,res_lm,r]", A.mean(0).round(3))
# (ii) 釋出率代理 placebo：只含釋出率資訊 + 雜訊，調成與 uD 相關約 0.44
for s in (1.0, 1.5, 2.0):
    A = np.array([split(z(M.lm).values + s*rng.normal(size=len(M))) for _ in range(500)]); print(f"lm-proxy placebo s={s} mean", A.mean(0).round(3))
# (iii) 對稱 placebo：買賣率代理 → 應出現「共同跟買賣、殘差跟釋出」
A = np.array([split(-z(M.lb).values + 1.5*rng.normal(size=len(M))) for _ in range(500)]); print("lb-proxy placebo mean", A.mean(0).round(3))
# (iv) 偏回歸：控制兩個真實發生率後，uC 對 uD 是否仍有關？uC 加入後 uD 對兩率的係數如何變
X0 = pd.DataFrame({"lb":z(M.lb),"lm":z(M.lm)})
cl = dict(cov_type="cluster", cov_kwds={"groups":pd.factorize(M.county)[0]})
f0 = sm.OLS(z(M.uD), sm.add_constant(X0)).fit(**cl); f1 = sm.OLS(z(M.uD), sm.add_constant(X0.assign(uC=z(M.uC)))).fit(**cl)
print("uD~lb+lm", f0.params.round(3).to_dict(), "  +uC", f1.params.round(3).to_dict(), "se", f1.bse.round(3).to_dict(), "R2", round(f0.rsquared,3), round(f1.rsquared,3))
fc = sm.OLS(z(M.uC), sm.add_constant(X0)).fit(); rC = fc.resid; rD = sm.OLS(z(M.uD), sm.add_constant(X0)).fit().resid
print("partial corr(uD,uC | lb,lm)", round(np.corrcoef(rC,rD)[0,1],3))
# residual vs uB
print("corr(resid, uB)", round(np.corrcoef(res, M.uB)[0,1],3), " corr(uD,uB)", round(M.uD.corr(M.uB),3), " corr(common,uB)", round(np.corrcoef(com,M.uB)[0,1],3))
# ---- 2(a) 虛無模擬（112 年）
print("\n== 2(a)")
py = P[P.year==112].copy(); py = py[py.house_cd.isin(U[("D",112)].index)]
pD, pC, pB = py.p_D.values, py.p_C.values, py.p_B.values; s = pD+pC+pB; pD, pC = pD/s, pC/s
hc = py.house_cd.values
obs = pd.DataFrame({t: U[(t,112)].u for t in "BCD"}); obs_lr = pd.DataFrame({t: U[(t,112)].lr for t in "BCD"})
print("observed r(uD,uB)", round(obs.D.corr(obs.B),3), " observed r(lrD,lrB) raw log O/E", round(obs_lr.D.corr(obs_lr.B),3))
codes, uniq = pd.factorize(hc)
def sim(aD=None, aC=None):
    qD, qC = pD, pC
    if aD is not None:
        qD = expit(logit(pD) + aD[codes]); qC = expit(logit(pC) + aC[codes]); tot = qD+qC; over = tot>0.98; qD[over] *= .98/tot[over]; qC[over] *= .98/tot[over]
    r = rng.random(len(pD)); reason = np.where(r<qD, "D", np.where(r<qD+qC, "C", "B"))
    df = pd.DataFrame({"house_cd":hc, "reason":reason, "p_B":py.p_B.values, "p_C":py.p_C.values, "p_D":py.p_D.values})
    e = {t: effects(df, t, "p_"+t) for t in "BD"}
    return [e["D"][0].u.corr(e["B"][0].u), e["D"][0].lr.corr(e["B"][0].lr), e["D"][1], e["B"][1]]
A = np.array([sim() for _ in range(30)]); print("null (no place effect) mean [r_u, r_rawlogOE, tauD, tauB]", A.mean(0).round(3), " sd", A.std(0).round(3))
nt = len(uniq)
for rho in (0.0, 0.4):
    out = []
    for _ in range(30):
        zz = rng.multivariate_normal([0,0], [[1,rho],[rho,1]], nt); out.append(sim(0.29*zz[:,0], 0.36*zz[:,1]))
    A = np.array(out); print(f"D、C 有地方效果(τ .29/.36, corr {rho})、B 無自身效果：mean [r_u, r_raw, tauD, tauB]", A.mean(0).round(3))
