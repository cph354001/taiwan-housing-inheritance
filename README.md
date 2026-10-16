# How Do Houses Change Hands? What the Inheritance Share Measures

Analysis code for a study of inherited housing in Taiwan, at the township level.

**Authors:** 陳柏翰, 林雨錡, 陳冠佑

**Technical report (Chinese, 33 pages):** [`report.pdf`](report.pdf) has the derivations, robustness checks, and township-level tables.

This repository contains code, two code-lookup tables, and the report. **It contains no data.**

## The question

In an ownership snapshot, a township's *inheritance share* is the share of buildings whose most recent transfer was an inheritance. That share can be high for two reasons: owners die more often, or houses are rarely sold. A stock snapshot cannot tell these apart.

We use simulated owner records from Taiwan's Ministry of the Interior (2021–2023) and work in three steps:

1. Adjust for what the houses are: building age, material, number of floors, and use. A LightGBM composition model gives each building an expected probability of having been inherited.
2. Estimate a shrunken township effect `u` with a random-intercept binomial model (empirical Bayes).
3. Link `u` to external flow data:
   - actual sales, from the national real-price registration data;
   - an owner-mortality proxy, built from age-specific death statistics.

**Main finding.** After adjusting for housing composition, a township's deviation `u` has two associations at once:

- it is lower where sales are less frequent;
- it is higher where owner mortality is higher.

Both associations hold across:

- age bands;
- alternative mortality proxies;
- spatial models;
- denominator error;
- a 200-replicate bootstrap of the full pipeline.

So the inheritance share does not simply measure how many inheritances happen locally.

## Layout

```
code/            main analysis, run in the order given in code/run_all.sh
code/checks/     robustness and verification scripts (cited as code/checks/… in the report)
ref/analysis/    township and county code crosswalks (owner-file codes ↔ household-registration codes)
report.pdf       technical report
```

Running the code creates three folders:

- `data/` for intermediate files;
- `out/` for numeric results;
- `fig/` for figures.

Comments and output labels in the code are in Chinese.

## Data you need

**1. Simulated owner data.** This is the Ministry of the Interior's simulated "house-owner" and "people–building" files, years 108–112 of the ROC calendar (2019–2023). Apply for access through the Survey Research Data Archive (SRDA), Academia Sinica. Put the two files in `raw/`, or set `HOUSING_RAW_DIR` to the folder that holds them:

- `108-112_house_own_subset_sumy_rawdata_simulation.csv`
- `108-112_moi_people_house_dtl_simulation.csv`

**2. Public external data**, placed under `external/`:

| Path | Data | Source |
|---|---|---|
| `external/external_validation/raw/moi108.csv`–`moi112.csv` | Building ownership registrations by region and registration type | Department of Statistics, Ministry of the Interior, https://statis.moi.gov.tw/micst/ (open-data set 13898) |
| `external/external_validation/raw/all_townships_population_11206.csv` | Township population, June 2023 | Department of Household Registration, https://www.ris.gov.tw/ |
| `external/external_validation/districts_real_112.csv` | District-level housing transfers by registration type, 2023 (Taichung, Tainan, Changhua, New Taipei) | City and county government statistics; New Taipei Land Administration Department |
| `external/external_validation/raw/newtaipei_112xx_land_building_registration_original.xls` | New Taipei monthly land and building registration reports, 2023 | New Taipei Land Administration Department |
| `external/lvr/<quarter>/*_lvr_land_a.csv` | Real-price registration of property sales, 2022 Q3 – 2023 Q4 | https://plvr.land.moi.gov.tw/DownloadOpenData |
| `external/ris/death_108.json`–`death_112.json`, `pop_11206_p1.json`–`p4.json` | Deaths by single year of age and township, 2019–2023 (ODRP031); village population by single year of age, June 2023 (ODRP014); both are raw API responses | https://www.ris.gov.tw/rs-opendata/api/Main/docs/v1 |
| `external/shp/twtown2010.json` | Township boundaries (2010 districts) | ronnywang/twgeojson |

## Running

```bash
pip install pandas==2.3.3 numpy==2.4.1 scipy==1.17.0 lightgbm==4.7.0 scikit-learn==1.8.0 \
            statsmodels==0.14.6 geopandas==1.2.0 matplotlib==3.10.8 pyarrow esda libpysal spreg
bash code/run_all.sh            # full main analysis, about 10–15 minutes
python3 code/checks/<script>.py  # individual checks; run from the repository root
```

**Environment**

- Developed with Python 3.11 on macOS.
- Figures use the PingFang font for Chinese text. On other systems, change `font.family` in `code/common.py`.
- Random seeds are set inside each script.
- `code/checks/rev1016_bootstrap.py` (the 200-replicate full-pipeline bootstrap) takes about 40 seconds per replicate. It writes intermediate results to `scratch/`.

## Scripts

| Script | What it does |
|---|---|
| `00_to_parquet.py`, `10_build.py` | Read raw files, keep one record per building, match townships |
| `11_generator_forensics.py` | Check how each year of the simulated data was generated |
| `12_composition_vs_place.py`, `glmm.py` | Composition model (LightGBM), township effects (random-intercept binomial model), variance decomposition |
| `13a_township_covariates.py`, `13b_place_correlates.py` | Township covariates, Moran's I |
| `14_real_rates.py`, `17_real_sales_lvr.py`, `21_township_supply_vs_sales.py` | Sales rates from registrations and real-price data; owner-mortality proxy |
| `15_cohort_flows.py` | Synthetic-cohort flow estimates |
| `16_robustness_and_replication.py` | Robustness checks and simulation checks |
| `18_typology.py`, `23_typology_map.py`, `24_typology_stability.py` | Township typology |
| `25_gift_place_effect.py` | Township effects for gifts and sales |
| `19_appendix_tables.py`, `20_figures.py`, `22_summary_figure.py` | Appendix tables and figures |
| `checks/rev1014_*.py`, `checks/rev1016_*.py` | Out-of-sample prediction, calibration, residual spatial dependence, age bands, mortality proxies, denominator error, full-pipeline bootstrap, sample table |
