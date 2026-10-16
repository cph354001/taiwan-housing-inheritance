#!/bin/bash
# 依序重跑全部分析（約 10–15 分鐘）。原始模擬資料放在 raw/（或設定環境變數 HOUSING_RAW_DIR），外部資料放在 external/，見 README。
set -e
cd "$(dirname "$0")"
python3 00_to_parquet.py
python3 10_build.py
python3 11_generator_forensics.py
python3 12_composition_vs_place.py
python3 13a_township_covariates.py
python3 13b_place_correlates.py
python3 14_real_rates.py
python3 15_cohort_flows.py
python3 16_robustness_and_replication.py
python3 17_real_sales_lvr.py
python3 21_township_supply_vs_sales.py
python3 18_typology.py
python3 19_appendix_tables.py
python3 20_figures.py
python3 22_summary_figure.py
python3 23_typology_map.py
python3 24_typology_stability.py
python3 25_gift_place_effect.py
