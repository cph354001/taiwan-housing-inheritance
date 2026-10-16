"""rev1016 審稿建議 09：整套估計流程的不確定性（鄉鎮內重抽建物，重跑組成模型→GLMM→釋出率→主要迴歸→分型）。

重抽單位：建物（同棟的所有持有紀錄一起抽）。理由：
  - u、u_C 的分析單位是建物（每棟一筆，O_j、n_j 都是棟數）；同棟多位所有權人的建物欄位完全相同、彼此不獨立，
    對持有紀錄重抽會把同一棟拆開，並把多人共有的建物放大權重。
  - 釋出率中的屋主年齡比例以持有紀錄計（Σ 各年齡組持有筆數 ÷ 全部持有筆數）；以建物為群集重抽、帶著它的全部持有紀錄，
    可同時更新 u、u_C、釋出率、買賣率分母（模擬存量），彼此一致。
各年度：
  - 112 年：鄉鎮內重抽「全部建物」（959,237 棟；含第一次登記、屋齡不明等），其中買賣/贈與/繼承且屋齡已知者進入 u、u_C；
    全部建物進入既有建物存量 S_ex（買賣率分母）；其持有紀錄進入屋主年齡比例。
  - 110、111 年：只當組成模型的訓練資料，各自在「年度 × 鄉鎮」內重抽買賣/贈與/繼承且屋齡已知的建物。
每次重抽：
  (1) 重抽建物；(2) 以重抽後的 110+111 年重訓 LightGBM（與 12、25 號程式完全相同的超參數，400 棵樹；繼承、贈與各一個），
  預測 112 年（112 年的組成模型本來就只用 110–111 年訓練，所以只需一個模型）；(3) 重估 GLMM 的 u（繼承）與 u_C（贈與）；
  (4) 以重抽後的持有紀錄重算鄉鎮屋主年齡比例 → 釋出率 m（死亡率為真實戶政資料，固定）；以重抽後的存量重算買賣率分母；
  (5) 重跑 272 鄉鎮的主要迴歸（含與不含 u_C）與 18_typology.py 的分型規則。
鄉鎮樣本固定為原本的 272（迴歸）與 312（分型；金門 4 個不分型）。
另外每次加一個縣市群集重抽（對 272 鄉鎮所屬縣市），得到「流程誤差＋最後一步抽樣誤差」的合併分布。
不涵蓋：模擬資料與母體之間的系統性落差（生成規則、邏輯調整），也不涵蓋實價登錄與死亡資料本身的誤差。
用法：python rev1016_bootstrap.py B [workers] [lgb_threads]   （B=0 只做重現檢查）
中間結果寫在 scratchpad；最終只寫 out/rev1016_bootstrap.json。"""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, lightgbm as lgb, statsmodels.api as sm
from scipy.special import logit
from common import *
import glmm
SEED = 20261016
SCR = Path("./scratch/r1016")
FEATS = ["age", "materials_group_cd", "floor_group_cd", "purpose_group_cd"]
BL = ["0-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100"]
LGB_THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# ------------------------------------------------------------------ 準備（只做一次）
def prepare():
    d = pd.read_parquet(DATA/"bcd_with_pcomp.parquet")                      # 12 號程式的分析樣本（110–112，B/C/D、屋齡已知、對得到人口）
    for c in FEATS[1:]: d[c] = d[c].astype(str).astype("category")
    d["C"] = (d.reason == "C").astype(int)
    town = pd.read_csv(DATA/"town.csv")
    b = pd.read_parquet(DATA/"bld.parquet", columns=["year", "building_sn", "countyntownship_cd", "age"]); b = b[b.year == 112].reset_index(drop=True)
    own = pd.read_parquet(DATA/"own.parquet", columns=["year", "building_sn", "gender_cd", "age", "countyntownship_cd"]); own = own[own.year == 112]
    own["band"] = pd.cut(own.age.clip(upper=100), [-1, 39, 49, 59, 69, 79, 89, 100], labels=BL)
    nrec = own.groupby("building_sn").size()
    W = own[own.gender_cd != 3].groupby(["building_sn", "band"], observed=False).size().unstack(fill_value=0)
    b["nrec"] = b.building_sn.map(nrec).fillna(0).values
    BW = W.reindex(b.building_sn).fillna(0).values                          # 每棟各年齡組（自然人）持有筆數
    b["ex"] = (b.age >= 10).astype(float)                                   # 17 號：既有建物＝屋齡已知且 ≥10 年（ab 不是 0-9 或 NA）
    # 死亡率（與 21 號程式相同）
    RIS = ROOT/"external/ris"; AGES = [f"{a:03d}" for a in range(100)] + ["100up"]
    t2 = town.dropna(subset=["name"]); nm = dict(zip(t2.name, t2.house_cd))
    def to_cd(site):
        s = site.replace("台", "臺").replace("　", "").strip()
        s = {"高雄市三民一": "高雄市三民區", "高雄市三民二": "高雄市三民區", "高雄市鳳山一": "高雄市鳳山區", "高雄市鳳山二": "高雄市鳳山區"}.get(s, s)
        if s.startswith("新竹市"): return "O01"
        if s.startswith("嘉義市"): return "I01"
        return nm.get(s)
    D = {}
    for y in [110, 111, 112]:
        r = pd.DataFrame(json.load(open(RIS/f"death_{y}.json"))["responseData"]); r = r[r.according == "按登記日期分"].copy(); r["house_cd"] = r.site_id.map(to_cd)
        M = pd.DataFrame({a: r[f"death_age_{a}_m"].astype(int) + r[f"death_age_{a}_f"].astype(int) for a in AGES}); M["house_cd"] = r.house_cd.values
        D[y] = M.dropna(subset=["house_cd"]).groupby("house_cd").sum()
    P = pd.concat([pd.DataFrame(json.load(open(RIS/f"pop_11206_p{p}.json"))["responseData"]) for p in range(1, 5)]); P["house_cd"] = P.site_id.map(to_cd)
    PM = pd.DataFrame({a: P[f"people_age_{a}_m"].astype(int) + P[f"people_age_{a}_f"].astype(int) for a in AGES}); PM["house_cd"] = P.house_cd.values
    PM = PM.dropna(subset=["house_cd"]).groupby("house_cd").sum()
    BANDS = [(0, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 100)]
    band = lambda M: pd.DataFrame({l: M[AGES[a:b_+1]].sum(axis=1) for l, (a, b_) in zip(BL, BANDS)})
    D3 = band((D[110] + D[111] + D[112])/3); PB = band(PM); q_nat = D3.sum()/PB.sum()
    smr = D3.sum(axis=1)/(PB*q_nat).sum(axis=1); q_smr = pd.DataFrame(np.outer(smr, q_nat), index=smr.index, columns=BL)
    # 原始輸出
    S = pd.read_csv(OUT/"township_supply_sales.csv").set_index("house_cd")
    main = S[(S.S_ex >= 2000) & (S.X_ex > 0) & ~S.county.isin(ISLANDS)]
    L = pd.read_csv(OUT/"lvr_township.csv").set_index("house_cd")
    ref = {"main": main, "L": L, "uC": pd.read_csv(OUT/"place_effects_BCD_112.csv").set_index("house_cd").u_C}
    return dict(d=d, town=town, b=b, BW=BW, q_smr=q_smr, ref=ref)

def group_draw(keys, rng):
    """在每個群內有放回重抽，回傳每列被抽中的次數（群內總數不變）。"""
    codes, _ = pd.factorize(keys); order = np.argsort(codes, kind="stable"); cs = codes[order]
    n_g = np.bincount(cs); start = np.r_[0, np.cumsum(n_g)[:-1]]
    pick = start[cs] + np.floor(rng.random(len(cs))*n_g[cs]).astype(np.int64)
    w = np.zeros(len(keys), np.int64); np.add.at(w, order[pick], 1)
    return w

def lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, random_state=1, n_jobs=LGB_THREADS)

def place(t112, y, p, w, town):
    """t112：112 年分析列；y：結果 0/1；p：組成模型預測；w：重抽次數。回傳 312 個鄉鎮（n≥50）的 GLMM 結果。"""
    g = pd.DataFrame({"house_cd": t112.house_cd.values, "n": w, "O": w*y, "E": w*p}).groupby("house_cd").sum().reset_index().merge(town, on="house_cd")
    g = g[g.n >= 50].copy(); g["E"] *= g.O.sum()/g.E.sum(); g["exp"] = g.E/g.n
    f = glmm.fit(g.O, g.n, logit(g.exp)); g["u"], g["u_sd"] = f["u"], f["u_sd"]; g["tau"] = f["tau"]; g["alpha"] = f["beta"][0]
    return g.set_index("house_cd")

def reg(M, with_uC, groups):
    z = lambda s: (s - s.mean())/s.std()
    X = pd.DataFrame({"lb": z(np.log(M.bx)), "lm": z(np.log(M.m))}, index=M.index)
    if with_uC: X["uC"] = z(M.uC)
    f = sm.OLS(z(M.u), sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = {"b_lb": f.params.lb, "b_lm": f.params.lm, "se_lb": f.bse.lb, "se_lm": f.bse.lm, "R2": f.rsquared}
    if with_uC: out.update({"b_uC": f.params.uC, "se_uC": f.bse.uC})
    return out

def typology(L):
    hi = L.exp >= L.exp.median(); pos, neg = (L.u - 1.96*L.u_sd) > 0, (L.u + 1.96*L.u_sd) < 0
    ty = pd.Series(np.select([hi & pos, hi & neg, ~hi & pos, ~hi & neg], ["甲", "乙", "丙", "丁"], default="戊"), index=L.index)
    ty[L.county == "金門縣"] = "金門"
    return ty

def one(P, rep, rng=None):
    d, b, town = P["d"], P["b"], P["town"]
    if rng is None:                                                              # 重現檢查：不重抽
        wb = np.ones(len(b), np.int64); wtr = np.ones((d.year < 112).sum(), np.int64)
    else:
        wb = group_draw(b.countyntownship_cd.values, rng)
        tr0 = d[d.year < 112]; wtr = group_draw((tr0.year.astype(str) + tr0.house_cd).values, rng)
    t0 = time.time()
    tr = d[d.year < 112]; tr = tr.iloc[np.repeat(np.arange(len(tr)), wtr)]
    t112 = d[d.year == 112]
    w112 = pd.Series(wb, index=b.building_sn.values).reindex(t112.building_sn.values).values
    assert not np.isnan(w112).any()
    # 112 年各列只有少數幾種特徵組合；預測只需在唯一組合上做
    uq = t112[FEATS].drop_duplicates(); key = pd.MultiIndex.from_frame(t112[FEATS].astype(str)); ukey = pd.MultiIndex.from_frame(uq.astype(str))
    pos = ukey.get_indexer(key)
    res = {}
    for tgt in ["D", "C"]:
        m = lgbm(); m.fit(tr[FEATS], tr[tgt]); res["p"+tgt] = m.predict_proba(uq)[:, 1][pos]
    t_lgb = time.time() - t0
    gD = place(t112, t112.D.values, res["pD"], w112, town); gC = place(t112, t112.C.values, res["pC"], w112, town)
    # 釋出率與買賣率分母
    hc = b.countyntownship_cd.values
    Wb = pd.DataFrame(P["BW"]*wb[:, None], columns=BL).groupby(hc).sum(); nrec = pd.Series(b.nrec.values*wb).groupby(hc).sum()
    Wn = Wb.div(nrec, axis=0); m = (Wn*P["q_smr"].reindex(Wn.index)).sum(axis=1)*100
    S_ex = pd.Series(b.ex.values*wb).groupby(hc).sum()*10
    main = P["ref"]["main"]
    M = pd.DataFrame({"county": main.county, "X_ex": main.X_ex}).copy()
    M["S_ex"] = S_ex.reindex(M.index); M["bx"] = M.X_ex/M.S_ex*100; M["m"] = m.reindex(M.index); M["u"] = gD.u.reindex(M.index); M["uC"] = gC.u.reindex(M.index)
    nmiss = int(M[["bx", "m", "u", "uC"]].isna().any(axis=1).sum()); M = M.dropna()
    grp = pd.factorize(M.county)[0]
    out = {"rep": rep, "n_reg": len(M), "n_missing": nmiss, "tau_D": float(gD.tau.iloc[0]), "tau_C": float(gC.tau.iloc[0]), "alpha_D": float(gD.alpha.iloc[0]),
           "n_town_D": len(gD), "sec_lgb": t_lgb}
    out.update({k+"_noC": v for k, v in reg(M, False, grp).items()}); out.update({k+"_C": v for k, v in reg(M, True, grp).items()})
    if rng is not None:                                                          # 合併：再對 272 鄉鎮做一次縣市群集重抽
        cts = M.county.unique(); pick = rng.choice(cts, len(cts)); Mc = pd.concat([M[M.county == c].assign(cl=i) for i, c in enumerate(pick)])
        out.update({k+"_noC_cl": v for k, v in reg(Mc, False, Mc.cl.values).items() if k[:2] == "b_"}); out.update({k+"_C_cl": v for k, v in reg(Mc, True, Mc.cl.values).items() if k[:2] == "b_"})
    # 分型（18 號規則；鄉鎮集合＝原本 312 個）
    L0 = P["ref"]["L"]; L = pd.DataFrame({"county": L0.county, "exp": gD.exp.reindex(L0.index), "u": gD.u.reindex(L0.index), "u_sd": gD.u_sd.reindex(L0.index)})
    ty = typology(L.dropna()).reindex(L0.index).fillna("未納入")
    out["type"] = ty.to_dict(); out["type_counts"] = ty.value_counts().to_dict()
    out["u"] = M.u.round(5).to_dict(); out["m"] = M.m.round(6).to_dict(); out["bx"] = M.bx.round(6).to_dict(); out["uC"] = M.uC.round(5).to_dict()
    out["u312"] = gD.u.round(5).to_dict()
    out["sec_total"] = time.time() - t0
    return out

_P = None
def _init():
    global _P; _P = prepare()
def _work(rep):
    rng = np.random.default_rng([SEED, rep]); o = one(_P, rep, rng)
    json.dump(o, open(SCR/f"rep_{rep:04d}.json", "w"), ensure_ascii=False, default=float); return rep, o["sec_total"]

if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    B = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if B == 0:
        P = prepare(); o = one(P, -1)
        main, L = P["ref"]["main"], P["ref"]["L"]
        chk = {"max|u-u_orig|_272": float(np.abs(pd.Series(o["u"]) - main.u).max()), "max|uC-uC_orig|_272": float(np.abs(pd.Series(o["uC"]) - P["ref"]["uC"].reindex(main.index)).max()),
               "max|m-m_orig|/m": float((np.abs(pd.Series(o["m"]) - main.m)/main.m).max()), "max|bx-bx_orig|/bx": float((np.abs(pd.Series(o["bx"]) - main.bx)/main.bx).max()),
               "max|u312-u_orig|": float(np.abs(pd.Series(o["u312"]) - L.u).max()),
               "type_counts": o["type_counts"], "type_agree_with_typology_csv": None,
               **{k: o[k] for k in o if k.startswith(("b_", "se_", "R2", "tau", "n_", "sec"))}}
        lab = {"甲": "甲 組成預期占比高、繼承又高於一般鄉鎮", "乙": "乙 組成預期占比高、繼承低於一般鄉鎮", "丙": "丙 組成預期占比低、繼承高於一般鄉鎮", "丁": "丁 組成預期占比低、繼承低於一般鄉鎮", "戊": "戊 與一般鄉鎮無法區分"}
        G = pd.read_csv(OUT/"typology.csv").set_index("型")["鄉鎮數"]
        chk["type_agree_with_typology_csv"] = all(int(G[lab[k]]) == v for k, v in o["type_counts"].items() if k in lab)
        json.dump(o, open(SCR/"rep_orig.json", "w"), ensure_ascii=False, default=float)
        print(json.dumps(chk, ensure_ascii=False, indent=1, default=float))
    else:
        reps = [r for r in range(B) if not (SCR/f"rep_{r:04d}.json").exists()]
        t0 = time.time()
        if workers == 1:
            _init()
            for r in reps: print(_work(r), round(time.time() - t0, 1), flush=True)
        else:
            from multiprocessing import Pool
            with Pool(workers, initializer=_init) as pool:
                for r, s in pool.imap_unordered(_work, reps): print(r, round(s, 1), round(time.time() - t0, 1), flush=True)
