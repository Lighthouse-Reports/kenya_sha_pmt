# Kenya PMT Analysis

**Authors:** Purity Mukami & Gabriel Geiger

Reconstruction and evaluation of a **Proxy Means Testing (PMT)** model for determining public healthcare premiums in Kenya, using household survey microdata from the Kenya National Bureau of Statistics (KNBS).

Two survey datasets are used:
- **IBS 2015** — Integrated Budget Survey (~21.7k households)
- **KCHS 2021** — Kenya Continuous Household Survey (~17k households)

Separate OLS regression models are trained for **urban** and **rural** subpopulations, predicting log per-capita consumption from observable household characteristics. The models are then evaluated for predictive accuracy, fairness across demographic groups, and robustness.

## Data Sources

Raw microdata is from KNBS surveys and is not included in the repository. They can be obtained by submitting a short application on the [KNBS's data catalogue](https://statistics.knbs.or.ke/nada/index.php/catalog/123). The data on the catalogue does contain assets, only purchases. If you would like to obtain assets data please email gabriel@lighthousereports.com. Once you have the data, place the original `.dta` files in `data/01_raw/` following the existing subdirectory structure, then run the pipeline from step 1.

---

## Directory Structure

```
kenya_analysis/
├── code/
│   ├── 00_load/        # Load raw KNBS Stata (.dta) files → CSV
│   ├── 01_process/     # Feature engineering & training data preparation
│   ├── 02_train/       # OLS model training (urban/rural splits)
│   └── 03_analysis/    # Model evaluation, fairness tests, robustness
├── config/
│   ├── paths.py        # Central path definitions for the project
│   ├── pmt_variables.xlsx       # Survey column → PMT variable mapping
│   ├── fairness_variables.xlsx  # Demographic variables for fairness tests
├── data/
│   ├── 01_raw/         # Raw KNBS survey microdata (.dta)
│   ├── 02_intermediate/# Per-table CSVs extracted from raw data
│   ├── 03_processed/   # Final training datasets & validation sets
│   ├── 04_models/      # Pickled OLS model objects
│   └── 05_results/     # Performance metrics, fairness reports, charts
├── util/
│   ├── normalize_column_name.py  # Column name standardization
│   ├── charts.py                 # Plotting utilities
│   └── county_regions.csv        # County → region mapping (47 counties, 8 regions)
└── requirements.txt
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. LOAD  (code/00_load/)                                   │
│     load_knbs.py / load_kchs_2021.py / load_ibs_2015.py    │
│     Raw .dta files  →  per-table CSVs in 02_intermediate/   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PROCESS  (code/01_process/)                             │
│     prep_ibs_data.ipynb / prep_kchs_data.ipynb              │
│     - Aggregate individual data to household level          │
│     - Extract household head characteristics                │
│     - Map occupations, education levels, assets             │
│     - Dummify categoricals, normalize column names          │
│     - Create log consumption label                          │
│     Output: td_*_filtered.csv, td_*_all.csv                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TRAIN  (code/02_train/)                                 │
│     training.py / train_model.py                            │
│     - Split by urban/rural                                  │
│     - 80/20 train-test split                                │
│     - Fit OLS regression (target: log consumption)          │
│     Output: *.pickle models, *_validation.csv               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. ANALYZE  (code/03_analysis/)                            │
│                                                             │
│  analyze_model.ipynb                                        │
│     RMSE, R², confusion matrices, exclusion/inclusion rates │
│                                                             │
│  fairness_tests.ipynb                                       │
│     Group-level fairness by county, sex, education,         │
│     household size, age, urban/rural, etc.                  │
│     Disparate impact detection, Fisher's exact test         │
│                                                             │
│  individual_analysis.ipynb                                  │
│     Per-household variable contributions, "overcharged"     │
│     (incorrectly excluded) poor household identification    │
│                                                             │
│  robustness_test.ipynb                                      │
│     Logistic regression on exclusion errors to identify     │
│     features that predict model failures                    │
│                                                             │
│  Output: data/05_results/ (CSV, Excel, PNG)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
pip install -r requirements.txt
```

---


