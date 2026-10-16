import pandas as pd, numpy as np
P="./data/"
o=pd.read_parquet(P+"own.parquet"); o=o[o.year.isin([110,111,112])]
o["reason"]=o.register_reason_group_cd
npb=o.groupby(["year","building_sn","reason"]).size().rename("k").reset_index()
t=npb.groupby(["year","reason"]).k.agg(["mean",lambda s:(s>1).mean()]).unstack(0).round(3); print(t)
# records-based vs building-based D share
for y in [110,111,112]:
    g=o[o.year==y]; nb=npb[npb.year==y]
    print(y,"rec share D %.3f  bld share D %.3f  rec n %d bld n %d"%((g.reason=="D").mean()*100,(nb.reason=="D").mean()*100,len(g),len(nb)))
# est with record counts, by building cohort
b=pd.read_parquet(P+"bld.parquet",columns=["year","building_sn","built","age"]); b=b[b.year.isin([110,111,112])]
x=npb.merge(b,on=["year","building_sn"]); x=x[x.age.notna()]; x["built"]=x.built.astype(int)
K=["A","B","C","D","E","F","U"]
def est(N, y0=110, y1=112, lo=1900, hi=2020):
    n0,n1=N.loc[y0],N.loc[y1]; idx=[c for c in n0.index.intersection(n1.index) if lo<=c<=hi]; n0,n1=n0.loc[idx],n1.loc[idx]
    w=(n0.sum(1)+n1.sum(1))/2; P0=n0.div(n0.sum(1),axis=0).mul(w,axis=0).sum()/w.sum(); P1=n1.div(n1.sum(1),axis=0).mul(w,axis=0).sum()/w.sum()
    lam=-np.log(P1.A/P0.A)/2; pbar=(P0+P1)/2; h=(P1-P0)/2+lam*pbar
    return pd.Series({"λ":lam,"B":h.B,"C":h.C,"D":h.D,"dpD":(P1.D-P0.D)/2,"pD0":P0.D,"pD1":P1.D,"n":w.sum()})
Nb=x.groupby(["year","built","reason"]).size().unstack(fill_value=0).reindex(columns=K,fill_value=0).astype(float)
Nr=x.groupby(["year","built","reason"]).k.sum().unstack(fill_value=0).reindex(columns=K,fill_value=0).astype(float)
eb,er=est(Nb),est(Nr); print(pd.DataFrame({"bld":eb*100,"rec":er*100}).round(3))
# raw increments 110->112 of D counts
for lab,col in [("bld",None),("rec","k")]:
    c=x.groupby(["year","reason"]).size() if col is None else x.groupby(["year","reason"]).k.sum()
    print(lab,"D 110,111,112:",[int(c[(y,"D")]) for y in [110,111,112]],"total:",[int(c.xs(y).sum()) for y in [110,111,112]], "D growth %:", round((c[(112,'D')]/c[(110,'D')]-1)*100,2), "total growth %:", round((c.xs(112).sum()/c.xs(110).sum()-1)*100,2))
