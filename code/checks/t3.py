import pandas as pd, numpy as np, glob
R="./"
LET = {"a":"臺北市","b":"臺中市","c":"基隆市","d":"臺南市","e":"高雄市","f":"新北市","g":"宜蘭縣","h":"桃園市","i":"嘉義市","j":"新竹縣","k":"苗栗縣","m":"南投縣","n":"彰化縣","o":"新竹市","p":"雲林縣","q":"嘉義縣","t":"屏東縣","u":"花蓮縣","v":"臺東縣","w":"金門縣","x":"澎湖縣","z":"連江縣"}
CN={c:i for i,c in enumerate("零一二三四五六七八九")}
def cn2int(s):
    if not isinstance(s,str) or not s.endswith("層"): return np.nan
    s=s[:-1]
    if s.isdigit(): return int(s)
    try:
        if "十" in s: a,_,c=s.partition("十"); return (CN[a] if a else 1)*10+(CN[c] if c else 0)
        return CN[s]
    except KeyError: return np.nan
parts=[]
for se in ["111S3","111S4","112S1","112S2","112S3","112S4"]:
    for f in glob.glob(R+f"external/lvr/{se}/*_lvr_land_a.csv"):
        x=pd.read_csv(f,dtype=str,skiprows=[1],usecols=["鄉鎮市區","交易標的","交易筆棟數","總樓層數","建物型態","主要用途","主要建材","建築完成年月","備註"],on_bad_lines="skip",encoding="utf-8-sig")
        x["county"]=LET[f.split("/")[-1][0]]; parts.append(x)
L=pd.concat(parts,ignore_index=True)
L["n_bld"]=pd.to_numeric(L["交易筆棟數"].str.extract(r"建物(\d+)")[0],errors="coerce").fillna(0)
L=L[(L.n_bld>0)&L["交易標的"].fillna("").str.contains("房地|建物")].copy()
yr=pd.to_numeric(L["建築完成年月"].str.strip().str[:-4],errors="coerce"); L["age"]=np.where(yr.between(1,113),112-yr,np.nan)
L["floors"]=L["總樓層數"].map(cn2int)
town=pd.read_csv(R+"data/town.csv"); nm=dict(zip(town.name,town.house_cd))
L["house_cd"]=(L.county+L["鄉鎮市區"].fillna("").str.replace("台","臺")).map(nm); L.loc[L.county=="新竹市","house_cd"]="O01"; L.loc[L.county=="嘉義市","house_cd"]="I01"
def mat(s):
    s=s if isinstance(s,str) else ""
    if "磚" in s: return "B磚"
    if "鋼筋" in s or "鋼骨" in s or "混凝土" in s: return "A RC/SRC"
    if any(k in s for k in ["鐵","鋼","金屬"]): return "C金屬"
    return "D其他"
L["m"]=L["主要建材"].map(mat)
YEARS=1.5
TM=pd.read_csv(R+"out/township_master_112.csv")[["house_cd","pop"]].dropna()
TM["pq"]=pd.qcut(TM["pop"],5,labels=["Q1少","Q2","Q3","Q4","Q5多"])
old=L[(L.age>=40)&L.floors.between(1,5)].merge(TM,on="house_cd")
b=pd.read_parquet(R+"data/bld.parquet"); s=b[(b.year==112)].merge(TM,left_on="countyntownship_cd",right_on="house_cd")
so=s[(s.age>=40)&(s.floor_group_cd=="1")]
mm={"A":"A RC/SRC","B":"B磚","C":"C金屬","D":"D其他"}
so=so.assign(m=so.materials_group_cd.map(mm))
st=so.groupby(["pq","m"],observed=True).size().unstack(fill_value=0)*10
sa=old.groupby(["pq","m"],observed=True).n_bld.sum().unstack(fill_value=0)/YEARS
print("SIM old low-rise stock share %:\n",(st.div(st.sum(1),axis=0)*100).round(1))
print("REAL old low-rise sales share %:\n",(sa.div(sa.sum(1),axis=0)*100).round(1))
print("turnover per 100 by material:\n",(sa/st*100).round(2))
q1=old[old.pq=="Q1少"]
print("Q1 real old sales: 主要建材 top:\n",q1.groupby("主要建材").n_bld.sum().sort_values(ascending=False).head(10)/YEARS)
for mt in ["A RC/SRC","C金屬","D其他"]:
    g=q1[q1.m==mt]; print(mt,"建物型態:",(g.groupby("建物型態").n_bld.sum()/g.n_bld.sum()*100).round(1).sort_values(ascending=False).head(4).to_dict(),"用途:",(g.groupby("主要用途").n_bld.sum()/g.n_bld.sum()*100).round(1).sort_values(ascending=False).head(4).to_dict())
print("Q1 sim old purpose by material:\n",(so[so.pq=="Q1少"].groupby(["m","purpose_group_cd"]).size().unstack(fill_value=0).pipe(lambda d:d.div(d.sum(1),axis=0))*100).round(1))
# all-age, all-floor shares for manual check
for y in [110,111,112]:
    g=b[b.year==y]; print(y,"bld material %:",(g.materials_group_cd.value_counts(normalize=True)*100).round(2).to_dict())
o=pd.read_parquet(R+"data/own.parquet",columns=["year","materials_group_cd"])
for y in [111,112]:
    print(y,"own-record material %:",(o[o.year==y].materials_group_cd.value_counts(normalize=True)*100).round(2).to_dict())
# real all-sales material share nationally
print("REAL all sales material %:",(L.groupby("m").n_bld.sum()/L.n_bld.sum()*100).round(1).to_dict())
print("=== county level old low-rise: sim stock share vs real sales share, D other turnover")
st=so.groupby(["county_cd","m"]).size().unstack(fill_value=0)*10
old["cc"]=old.house_cd.str[0]
sa=old.groupby(["cc","m"]).n_bld.sum().unstack(fill_value=0)/YEARS
C=pd.DataFrame({"simB%":st["B磚"]/st.sum(1)*100,"simD%":st["D其他"]/st.sum(1)*100,"realB%":sa["B磚"]/sa.sum(1)*100,"realD%":sa["D其他"]/sa.sum(1)*100,"turnB":sa["B磚"]/st["B磚"]*100,"turnD":sa["D其他"]/st["D其他"]*100,"turnA":sa["A RC/SRC"]/st["A RC/SRC"]*100}).round(2)
print(C.sort_values("simD%"))
print("corr simD% vs turnB:", C[["simD%","turnB"]].corr(method="spearman").iloc[0,1].round(2))
q1=old[(old.pq=="Q1少")&(old.m=="D其他")]
print("Q1 D-other materials:", (q1.groupby(q1["主要建材"].fillna("<空>")).n_bld.sum()/YEARS).sort_values(ascending=False).head(8).round(1).to_dict())
