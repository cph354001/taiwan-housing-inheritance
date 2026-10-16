"""把兩個原始模擬 CSV 轉成 parquet（只做型別整理，不改值）。"""
import pandas as pd
from common import ROOT, RAW_OWN, RAW_PPL
own = pd.read_csv(RAW_OWN, dtype={"county_cd": str, "countyntownship_cd": str, "floor_group_cd": str,
                                  "purpose_group_cd": str, "materials_group_cd": str,
                                  "register_reason_group_cd": str}, low_memory=False)
own = own.drop(columns=["source_file"])
own.to_parquet(ROOT / "data/own.parquet", index=False)
print("own", own.shape, own.dtypes.to_dict())
it = pd.read_csv(RAW_PPL, dtype=str, chunksize=1_000_000)
first = next(it); print("ppl cols", list(first.columns)); print(first.head(3).T)
