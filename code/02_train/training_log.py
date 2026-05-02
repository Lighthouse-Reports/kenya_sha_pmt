"""
Training pipeline for hierarchical ordinal logistic models on KCHS 2021 data.

Households are first segmented into three coarse income bands (low / middle / high)
using a parent logit model, then each band is scored independently with a fine-grained
band model.  All models are serialized to pickle and their summary statistics are
written to Excel.

Usage
-----
    python training_log.py
"""
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from train_log_model import LogModel, train_logit_model

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import paths 

PARENT_BINS = [0, 15000, 100000, float('inf')]
FINE_BINS = [0, 5000, 10000, 15000, 30000, 50000, 100000, 200000, 300000, float('inf')]
FINE_BINS_LABELS = {'0-5k' : 0, '5k-10k' : 1, '10k-15k' : 2, '15k-30k' : 0, '30k-50k' : 1, '50k-100k' : 2, '100k-200k' : 0, '200k-300k' : 1, '300k+' : 2}

VARS_TO_DROP = [
    "own_tablet_yes",
    "own_landline_yes",
    "own_phone_yes",
    "own_hometheatre_yes",
    "own_radio_yes",
    "own_mtcycle_yes",
    "household_size",
    "mm_age",
    "prop_over65",
    "prop_under5",
    "prop_m1864",
    "prop_w1864",
    'household_size',
    'ownership_dwelling_owns',
    'ownership_dwelling_no_rent_with_consent_of_owner',
    'ownership_dwelling_pays_rent_lease'
]

def load_training_data() -> pd.DataFrame:
    """
    Load and prepare the KCHS 2021 training dataset.

    Reads the filtered training set and the full survey file, computes an
    inflation-adjusted total expenditure column, then joins it onto the filtered
    data.

    Returns
    -------
    pd.DataFrame
        Filtered training data
    """
    td_filtered = pd.read_csv(paths.PROCESSED_DATA / 'td_kchs_2021_filtered.csv', index_col = 'hhid')
    td_all = pd.read_csv(paths.PROCESSED_DATA / 'td_kchs_2021_all.csv', index_col = 'hhid')

    # Calculate total inflation-adjusted KES expenditure 
    inflation_factor = 1.211

    td_all['total_expenditure_adjusted'] = (
        td_all['padqexp']
        * td_all['pdeflator'] # Deflate consumption back to nominal 
        * td_all['adq_scale_consumption'] # Mulity by adult equivalent scale factor
        * inflation_factor # Adjust for inflation from 2021 to 2024
    )

    # Join with our filtered trianing data, drop consumption, and dummify urban_rural 
    td_filtered = (
        td_filtered
        .join(td_all['total_expenditure_adjusted'], how = 'left')
        .drop(columns = 'actual_consumption_log')
        .pipe(pd.get_dummies, columns=['urban_rural_classification'], drop_first=True)
        .assign(
            urban_rural_classification_Urban=lambda df: df['urban_rural_classification_Urban'].astype(int)
        )
    )

    return td_filtered 

def calculate_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign coarse and fine expenditure band labels to training households.

    Two label columns are added in-place:

    - ``coarse_band`` — three buckets (0 = low <15k KES, 1 = middle 15k–100k,
      2 = high >100k) used to train the parent model and route households to
      the appropriate child model.
    - ``fine_band`` — nine buckets collapsed to three relative labels (0/1/2)
      within each coarse band, used as the target for the child models.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing ``total_expenditure_adjusted``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``coarse_band`` and ``fine_band`` columns appended.
    """
    # Calculate coarse bins
    train_df['coarse_band'] = pd.cut(
        train_df['total_expenditure_adjusted'],
        bins=PARENT_BINS,
        labels=[0, 1, 2],
        right=True
    )

    # Calculate specific bins  
    train_df['fine_band'] = pd.cut(
        train_df['total_expenditure_adjusted'],
        bins= FINE_BINS,
        labels= list(FINE_BINS_LABELS.keys()),
        right=True
    ).map(FINE_BINS_LABELS)

    return train_df 

def global_split(train_df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Create one stratified holdout set shared across all models.

    Splitting here — before any model sees the data — ensures no household
    can appear in both a model's training data and the evaluation set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Fully labelled dataset (output of :func:`calculate_labels`).
    test_size : float
        Fraction to hold out (default 0.2).

    Returns
    -------
    tuple
        df_train, df_test — both retain all columns.
    """
    df_train, df_test = train_test_split(
        train_df,
        test_size=test_size,
        random_state=42,
        stratify=train_df['coarse_band'],
    )
    print(f"Global split: {len(df_train):,} train / {len(df_test):,} test")
    print(f"  Coarse band distribution (test):\n{df_test['coarse_band'].value_counts().sort_index()}\n")
    return df_train, df_test


def train_models(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict[str, LogModel]:
    """
    Train the full set of hierarchical logistic models.

    All models receive slices of the same df_train and df_test so the holdout
    set is never contaminated by training data from any model.

    Trains five models in total:

    - ``parent_model_all_vars`` — ordinal logit on all features, target ``coarse_band``.
    - ``parent_model_adjusted`` — same as above but with ``VARS_TO_DROP`` removed
      (asset-ownership and demographic variables that reduce out-of-sample performance).
    - ``low_income_model`` — fine-band model fit only on low-band households.
    - ``middle_income_model`` — fine-band model fit only on middle-band households.
    - ``high_income_model`` — fine-band model fit only on high-band households.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training portion (output of :func:`global_split`).
    df_test : pd.DataFrame
        Held-out test portion (output of :func:`global_split`).

    Returns
    -------
    dict[str, LogModel]
        Mapping of model name to fitted :class:`LogModel` instance.
    """
    DROP_ALWAYS = ['total_expenditure_adjusted', 'fine_band', 'coarse_band']

    def _prepare(df_tr, df_te, label, extra_drop=None):
        """Extract features and label from already-split train and test dataframes."""
        drop = DROP_ALWAYS + (extra_drop or [])
        feature_cols = [c for c in df_tr.columns if c not in drop]
        return (
            df_tr[feature_cols],
            df_te[feature_cols],
            df_tr[label].astype(int),
            df_te[label].astype(int),
        )

    # Parent model — all variables
    parent_model_all_vars = train_logit_model(
        *_prepare(df_train, df_test, label='coarse_band'),
        model_name='parent_model_all_vars',
    )

    # Parent model — asset / demographic variables removed
    parent_model_adjusted = train_logit_model(
        *_prepare(df_train, df_test, label='coarse_band', extra_drop=VARS_TO_DROP),
        model_name='parent_model_adjusted',
    )

    # Sub-models for low, middle and high income. 
    for band, name in [(0, 'low_income_model'), (1, 'middle_income_model'), (2, 'high_income_model')]:
        tr = df_train[df_train['coarse_band'] == band]
        te = df_test[df_test['coarse_band'] == band]
        print(f"\n{name}: {len(tr):,} train / {len(te):,} test households")

    model_low = train_logit_model(
        *_prepare(df_train[df_train['coarse_band'] == 0], df_test[df_test['coarse_band'] == 0], label='fine_band'),
        model_name='low_income_model',
    )

    model_middle = train_logit_model(
        *_prepare(df_train[df_train['coarse_band'] == 1], df_test[df_test['coarse_band'] == 1], label='fine_band'),
        model_name='middle_income_model',
    )

    model_high = train_logit_model(
        *_prepare(df_train[df_train['coarse_band'] == 2], df_test[df_test['coarse_band'] == 2], label='fine_band'),
        model_name='high_income_model',
    )

    return {
        parent_model_all_vars.name : parent_model_all_vars,
        parent_model_adjusted.name : parent_model_adjusted,
        model_low.name : model_low,
        model_middle.name : model_middle,
        model_high.name : model_high,
    }

def get_model_stats(submodel) -> tuple[pd.DataFrame] :
    """
    Get model stats and coefficients for an ordinal logistic model
    """
    coefs_df = (
        pd.DataFrame({
                "Coefficient": submodel.params,
                "Std Err": submodel.bse,
                "z-value": submodel.tvalues,
                "p-value": submodel.pvalues
        })
        .reset_index()
        .rename(columns={"index": "term"})
    )

    coefs_df['sig'] = coefs_df['p-value'].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "insig"
    )

    summary_stats = pd.DataFrame({
        "Statistic": [
            "Log-Likelihood",
            "Null Log-Likelihood",
            "McFadden Pseudo R-squared",
            "LLR Statistic",
            "LLR p-value",
            "AIC",
            "BIC",
            "N observations",
        ],
        "Value": [
            submodel.llf,
            submodel.llnull,
            submodel.prsquared,
            submodel.llr,
            submodel.llr_pvalue,
            submodel.aic,
            submodel.bic,
            submodel.nobs,
        ]
    })

    return summary_stats, coefs_df

def save_models(
        models : dict[str, LogModel],
        train_df : pd.DataFrame, 
        model_path : Path = paths.MODELS,
        stats_path : Path = paths.RESULTS / 'model_stats') -> None :
    """
    Save all trained models to pickle files.

    Parameters
    ----------
    models : dict[str, LogModel]
        Flat dictionary of LogModel objects keyed by model name.
    model_path : Path, optional
        Directory path for saving models (default: paths.MODELS).
    stats_path : Path, optional
        Directory path for saving model stats (default: paths.RESULTS / 'model_stats').
    """
    for submodel in models.values() :
        # Save model files
        with open(model_path / f'{submodel.name}.pickle', 'wb') as f:
            pickle.dump(submodel, f)

        # Save model stats
        model_stats, coefs = get_model_stats(submodel.model)
        with pd.ExcelWriter(stats_path / f'{submodel.name}_stats.xlsx', engine="openpyxl") as writer:
            model_stats.to_excel(writer, sheet_name="Summary Stats", index=False)
            coefs.to_excel(writer, sheet_name="Coefficients", index=False)

    # Save train_df 
    train_df.to_csv(paths.PROCESSED_DATA / 'td_kchs_2021_log.csv')
    

def main() :
    """
    Execute full training pipeline: train models, score validation sets, and save outputs.
    """
    # Load training data
    train_df = load_training_data()

    # Calculate labels / bins
    train_df = calculate_labels(train_df)

    # Global stratified split — one shared holdout for all models
    df_train, df_test = global_split(train_df)

    # Train models
    print("\nTraining models...")
    models = train_models(df_train, df_test)

    # Save models
    save_models(models, train_df)
    

if __name__ == "__main__" : 
    main()