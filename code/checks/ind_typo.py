import pandas as pd, numpy as np, itertools
R = "./out/"
def lab(df, median_over=None):
    med = (median_over if median_over is not None else df).exp.median()
    hi = df.exp >= med; lo_ci, hi_ci = df.u - 1.96*df.u_sd, df.u + 1.96*df.u_sd
    s = pd.Series("戊", index=df.house_cd)
    s[(hi & (lo_ci > 0)).values] = "甲"; s[(hi & (hi_ci < 0)).values] = "乙"
    s[(~hi & (lo_ci > 0)).values] = "丙"; s[(~hi & (hi_ci < 0)).values] = "丁"
    return s
E = {y: pd.read_csv(R+f"township_effects_{y}.csv") for y in (110,111,112)}
for y in E: print(y, len(E[y]), "金門", (E[y].county=="金門縣").sum())
# 版本1：與 24 相同（先排除金門再取中位數）
for variant in ("排除金門後取中位數", "含金門取中位數(同18)"):
    L = {}
    for y, e in E.items():
        L[y] = lab(e[e.county!="金門縣"], None if variant.startswith("排除") else e)
    C = pd.DataFrame(L).dropna(); print("\n==", variant, "共同鄉鎮", len(C))
    inh, sal = set("甲丙"), set("乙丁")
    for a, b in itertools.combinations((110,111,112), 2):
        same = (C[a]==C[b]).mean(); A = C[C[a]=="甲"]; keep = (A[b]=="甲").mean()
        flip = ((C[a].isin(inh)&C[b].isin(sal))|(C[a].isin(sal)&C[b].isin(inh))).sum()
        print(f"{a}-{b}: 相同 {same:.4f}  甲持續 {keep:.4f} ({(A[b]=='甲').sum()}/{len(A)})  翻面 {flip}")
    print("各年甲型", {y:int((C[y]=="甲").sum()) for y in C}, "三年皆甲", int((C=="甲").all(axis=1).sum()))
    # 甲型持續的對稱版（反方向）
    for a, b in [(111,110),(112,111),(112,110)]:
        A = C[C[a]=="甲"]; print(f"  反向 {a}甲在{b}仍甲 {(A[b]=='甲').mean():.4f}")
    # 跳動是否都經過戊
    ch = pd.concat([C[[a,b]].set_axis(["x","y"],axis=1) for a,b in itertools.combinations((110,111,112),2)])
    ch = ch[ch.x!=ch.y]; print("  改變中涉及戊的比例", ((ch.x=="戊")|(ch.y=="戊")).mean().round(4), "改變數", len(ch))
    print("  非戊之間改變：", ch[(ch.x!="戊")&(ch.y!="戊")].value_counts().to_dict())
    # 三年合併：逆變異加權（另一種合併）與 24 的作法
    k = [e[e.county!="金門縣"].set_index("house_cd")[["u","u_sd","exp"]] for e in E.values()]
    M = pd.concat(k, axis=1, keys=[110,111,112]).dropna()
    u = M.xs("u",axis=1,level=1); s = M.xs("u_sd",axis=1,level=1); ex = M.xs("exp",axis=1,level=1).mean(axis=1)
    A1 = pd.DataFrame({"house_cd":M.index, "u":u.mean(axis=1).values, "u_sd":(s.mean(axis=1)/np.sqrt(3)).values, "exp":ex.values})
    w = 1/s**2; A2 = pd.DataFrame({"house_cd":M.index, "u":((u*w).sum(1)/w.sum(1)).values, "u_sd":(1/np.sqrt(w.sum(1))).values, "exp":ex.values})
    for nm, A in (("24作法 mean(sd)/√3", A1), ("逆變異加權", A2)):
        c = lab(A).reindex(C.index); j112 = C[C[112]=="甲"].index
        print(f"  三年合併[{nm}] 甲 {int((c=='甲').sum())}，112甲在內 {(c[j112]=='甲').mean():.3f}")
# 與 18 的 112 年分型比對
T = pd.read_csv(R+"typology.csv"); print(T.iloc[:, :2])
