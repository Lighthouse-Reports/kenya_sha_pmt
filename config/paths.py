from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config path
CONFIG = ROOT / "config"
UTIL = ROOT / "util"

# Data paths 
DATA = ROOT / "data"
KCHS_2021 = DATA / "01_raw/knbs_kchs_2021/"
IBS_2015 = DATA / "01_raw/knbs_ibs_2015/"
KCHS_2022 = DATA / "01_raw/knbs_kchs_2022/"
INTERMEDIATE_DATA = DATA / "02_intermediate/"
IBS_2015_LABELED = INTERMEDIATE_DATA / 'knbs_ibs_2015/'
MODELS = DATA / "04_models/"
IMPUTATION_MODELS = DATA / "04_models/imputation/"
PROCESSED_DATA = DATA / "03_processed/"

# Output paths
OUTPUTS = ROOT / "outputs"
RESULTS = DATA / "05_results/"
