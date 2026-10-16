import pandas as pd, numpy as np
P="./data/"
o=pd.read_parquet(P+"own.parquet"); o=o[o.year==112]; o["reason"]=o.register_reason_group_cd
print(o[["own_yr","first_own_house_yr","age","b_age"]].describe().round(2))
print(o.groupby("reason").agg(own_yr=("own_yr","mean"),own_le1=("own_yr",lambda s:(s<=1).mean()),fohy=("first_own_house_yr","mean"),age=("age","mean"),bage=("b_age","mean")).round(3))
k=o.dropna(subset=["own_yr","b_age"]); print("share own_yr > b_age: %.3f"%((k.own_yr>k.b_age+1).mean()))
print("corr own_yr,b_age %.3f; corr own_yr,age %.3f"%(k.own_yr.corr(k.b_age),k.own_yr.corr(k.age)))
# same person: is own_yr constant across their buildings?
g=o.groupby("person_sn").own_yr.agg(["nunique","size"]); g=g[g["size"]>1]; print("multi-bld persons:",len(g),"share with constant own_yr: %.3f"%((g["nunique"]==1).mean()))
# within-building: own_yr constant across co-owners?
h=o.groupby("building_sn").own_yr.agg(["nunique","size"]); h=h[h["size"]>1]; print("multi-owner bld:",len(h),"share constant own_yr: %.3f"%((h["nunique"]==1).mean()))
# owner age by reason (report: D 56.0, B 55.4)
n=o[o.gender_cd!=3]; print("owner age by reason (natural persons):", n.groupby("reason").age.mean().round(2).to_dict())
print("owner age by reason (all):", o.groupby("reason").age.mean().round(2).to_dict())
