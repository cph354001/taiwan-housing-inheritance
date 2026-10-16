import pandas as pd, numpy as np
P="./data/"
b=pd.read_parquet(P+"bld.parquet"); b=b[b.year.isin([110,111,112])]
K=["A","B","C","D","E","F","U"]
print(b.reason.value_counts().to_dict())
# unknown-age share by reason/year
t=b.assign(unk=b.age.isna()).groupby(["year","reason"]).unk.mean().unstack().round(4); print(t*100)
print("overall unk", b.groupby("year").apply(lambda g:g.age.isna().mean()).round(4).to_dict())
# reason shares among all vs known
for y in [110,111,112]:
    g=b[b.year==y]; print(y, "all", (g.reason.value_counts(normalize=True)[["A","B","C","D"]]*100).round(2).to_dict(), "known", (g[g.age.notna()].reason.value_counts(normalize=True)[["A","B","C","D"]]*100).round(2).to_dict())
v=b[b.age.notna()].copy(); v["built"]=v.built.astype(int)
def est(N, y0=110, y1=112, lo=1900, hi=2020, mu=None):
    n0,n1=N.loc[y0],N.loc[y1]; idx=[c for c in n0.index.intersection(n1.index) if lo<=c<=hi]; n0,n1=n0.loc[idx],n1.loc[idx]
    w=(n0.sum(1)+n1.sum(1))/2; P0=n0.div(n0.sum(1),axis=0).mul(w,axis=0).sum()/w.sum(); P1=n1.div(n1.sum(1),axis=0).mul(w,axis=0).sum()/w.sum()
    dt=y1-y0; lam=-np.log(P1.A/P0.A)/dt; pbar=(P0+P1)/2
    h=(P1-P0)/dt+lam*pbar
    out={"λ":lam,"B":h.B,"C":h.C,"D":h.D,"dpD/dt":(P1.D-P0.D)/dt,"λpD":lam*pbar.D,"pD0":P0.D,"pD1":P1.D,"n":w.sum()}
    return pd.Series(out)
N=v.groupby(["year","built","reason"]).size().unstack(fill_value=0).reindex(columns=K,fill_value=0).astype(float)
base=est(N); print("BASE\n",(base*100).round(3))
# H1: impute unknown-age per (year, reason) proportionally to that year-reason's known built distribution
U=b[b.age.isna()].groupby(["year","reason"]).size()
N1=N.copy()
for y in [110,111,112]:
    blk=N1.loc[y]
    for k in K:
        u=U.get((y,k),0)
        if u and blk[k].sum()>0: N1.loc[y,k]=(blk[k]+u*blk[k]/blk[k].sum()).values
h1=est(N1); print("H1 impute by reason\n",(h1*100).round(3))
# H1b: impute unknown within township x reason (more local)
U2=b[b.age.isna()].groupby(["year","countyntownship_cd","reason"]).size().rename("u").reset_index()
kn=v.groupby(["year","countyntownship_cd","reason","built"]).size().rename("c").reset_index()
kn["sh"]=kn.c/kn.groupby(["year","countyntownship_cd","reason"]).c.transform("sum")
m=kn.merge(U2,on=["year","countyntownship_cd","reason"],how="left").fillna({"u":0}); m["c2"]=m.c+m.u*m.sh
N2=m.groupby(["year","built","reason"]).c2.sum().unstack(fill_value=0).reindex(columns=K,fill_value=0)
h2=est(N2); print("H1b impute by town x reason\n",(h2*100).round(3))
# convert to national annual counts like code: scale
stock_all=b[b.year.isin([110,112])].groupby("year").size().mean()
for lab,e,sc in [("base",base,stock_all/v[v.year.isin([110,112])].groupby("year").size().mean()),("H1",h1,1.0),("H1b",h2,1.0)]:
    nn=e["n"]*sc*10
    print(lab,"D existing 萬/yr", round(e["D"]*nn/1e4,2), "ratio to real 7.0156:", round((e["D"]*nn+184)/70156,3), "B", round(e["B"]*nn/1e4,2), "C", round(e["C"]*nn/1e4,2))
