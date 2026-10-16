import pandas as pd, numpy as np, re
REAL="./external/external_validation/raw/"
real={}
for y in [108,109,110,111,112]:
    for line in open(REAL+f"moi{y}.csv",encoding="utf-8-sig"):
        m=re.match(r'"(\d+)年/ 區域別總計/ (.+?)",([\d\-]+),',line)
        if m: real[(y,m[2])]=int(m[3])
for k in ["買賣","繼承","贈與","第一次登記"]:
    print(k,[real.get((y,k)) for y in [108,109,110,111,112]])
P="./data/"
b=pd.read_parquet(P+"bld.parquet"); b=b[b.year.isin([110,111,112])]
v=b[b.age.notna()].copy(); v["built"]=v.built.astype(int)
K=["A","B","C","D","E","F","U"]
N=v.groupby(["year","built","reason"]).size().unstack(fill_value=0).reindex(columns=K,fill_value=0).astype(float)
def est(N,y0,y1,lo=1900,hi=None,muD=None):
    hi=hi or y0+1911-1
    n0,n1=N.loc[y0],N.loc[y1]; idx=[c for c in n0.index.intersection(n1.index) if lo<=c<=hi]; n0,n1=n0.loc[idx],n1.loc[idx]
    w=(n0.sum(1)+n1.sum(1))/2; P0=n0.div(n0.sum(1),axis=0).mul(w,axis=0).sum()/w.sum(); P1=n1.div(n1.sum(1),axis=0).mul(w,axis=0).sum()/w.sum()
    dt=y1-y0; lam=-np.log(P1.A/P0.A)/dt; pbar=(P0+P1)/2
    return pd.Series({"λ":lam*100,"hB":((P1.B-P0.B)/dt+lam*pbar.B)*100,"hD":((P1.D-P0.D)/dt+lam*pbar.D)*100,"dpD":(P1.D-P0.D)/dt*100,"pD":pbar.D*100,"dpA":(P1.A-P0.A)/dt*100,"n":w.sum()})
for a,c in [(110,111),(111,112),(110,112)]:
    print(a,c,est(N,a,c,hi=2020).round(3).to_dict())
stock=b.groupby("year").size().mean()*10
print("stock",stock)
for y in [110,111,112]: print(y,"real D per100 %.3f B per100 %.3f"%(real[(y,'繼承')]/stock*100, real[(y,'買賣')]/stock*100))
# required mu_D to match real
e=est(N,110,112,hi=2020); rD=(0.5*real[(110,'繼承')]+real[(111,'繼承')]+0.5*real[(112,'繼承')])/2/stock*100
print("real D per100 (period) %.3f"%rD, "implied exit rate from D: %.2f%% vs λ %.2f%%"%((rD-e.dpD)/e.pD*100, e["λ"]))
# heterogeneity check by age bands
for lo,hi in [(1900,1961),(1962,1981),(1982,2001),(2002,2020)]:
    print(lo,hi,est(N,110,112,lo=lo,hi=hi).round(3).to_dict())
