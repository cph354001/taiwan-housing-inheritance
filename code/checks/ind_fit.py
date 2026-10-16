# 獨立實作：對 B、C、D 三個結果各做「留一年度」LightGBM 組成模型，存逐棟預測
import sys, pandas as pd, numpy as np, lightgbm as lgb
R = "./"
SP = "./scratch/"
d = pd.read_parquet(R + "data/bcd_with_pcomp.parquet")
X = d[["age", "materials_group_cd", "floor_group_cd", "purpose_group_cd"]]
out = d[["year", "house_cd", "reason", "p_comp"]].copy()
for tgt in "BCD":
    y = (d.reason.values == tgt).astype(int); p = np.full(len(d), np.nan)
    for yr in (110, 111, 112):
        te = (d.year == yr).values
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, random_state=1, verbose=-1)
        clf.fit(X[~te], y[~te]); p[te] = clf.predict_proba(X[te])[:, 1]
        print(tgt, yr, flush=True)
    out["p_" + tgt] = p
out.to_parquet(SP + "pred_BCD.parquet")
print("D pred vs pipeline p_comp max abs diff:", np.abs(out.p_D - out.p_comp).max())
