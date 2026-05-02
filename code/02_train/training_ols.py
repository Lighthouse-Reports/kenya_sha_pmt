"""
Training pipeline for PMT models from KCHS and IBS survey data.

Orchestrates model training, validation scoring, and model persistence for
both KCHS 2021 and IBS 2015 datasets. Generates validation sets with predictions
and distribution metrics (deciles, percentiles).

Authors: Purity Mukami & Gabriel Geiger
"""

import sys 
import pickle 
import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from pathlib import Path
from train_ols_model import OLSModel, train_ols_models

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import paths 

def train_models() -> dict[str, dict] : 
    """
    Train PMT models for KCHS 2021 and IBS 2015 datasets.
    
    Returns
    -------
    dict[str, dict]
        Dictionary with 'kchs' and 'ibs' keys, each containing urban/rural model pairs.
    """
    # Train KCHS model  
    kchs_training = pd.read_csv(paths.PROCESSED_DATA / 'td_kchs_2021_filtered.csv', index_col = 'hhid')
    kchs_models = train_ols_models(kchs_training, 'kchs')

    # Train IBS models. 
    ibs_training = pd.read_csv(paths.PROCESSED_DATA / 'td_ibs_2015_filtered.csv', index_col = 'unique_id')
    ibs_models = train_ols_models(ibs_training, 'ibs')

    return {'kchs' : kchs_models, 'ibs' : ibs_models}

def score_val_set(submodel : OLSModel) -> pd.DataFrame : 
    """
    Generate predictions on test set for a single model.
    
    Parameters
    ----------
    submodel : Model
        Model object with fitted estimator and test data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Actual' and 'Predicted' log consumption values.
    """
    # Extract key parts from our named tuple 
    model_file = submodel.model 
    X_test = submodel.X_test 
    y_test = submodel.y_test

    # Predict on our test set 
    preds = model_file.predict(X_test)

    # Make a dataframe 
    results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})

    return results 


def calculate_poverty_classification(df : pd.DataFrame) -> pd.DataFrame :
    """
    Classify households as actually or predicted poor using urban/rural poverty lines.

    Derives separate poverty lines for rural and urban households as the maximum
    actual consumption among households flagged as absolutely poor, then applies
    those thresholds to both actual and predicted consumption values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'urban_rural_classification', 'abs_poor', 'Actual',
        and 'Predicted' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two added boolean columns: 'Actually_Poor' and
        'Predicted_Poor'.
    """

    # Calculate the Rural Poverty line 
    rural_poverty_line = (
        df[
            (df['urban_rural_classification'] == 'Rural')
            & (df['abs_poor'] == 1)
        ]
        ['Actual_KES'].max()
    )

    # Calculate the urban poverty line 
    urban_poverty_line = (
        df[
            (df['urban_rural_classification'] == 'Urban')
            & (df['abs_poor'] == 1)
        ]
        ['Actual_KES'].max()
    )

    # Calculate actually poor and predicted poor 
    df['Actually_Poor'] = df.apply(
        lambda row: row['Actual_KES'] <= rural_poverty_line if row['urban_rural_classification'] == 'Rural' else row['Actual_KES'] <= urban_poverty_line, axis=1
    )

    df['Predicted_Poor'] = df.apply(
        lambda row: row['Predicted_KES'] <= rural_poverty_line if row['urban_rural_classification'] == 'Rural' else row['Predicted_KES'] <= urban_poverty_line, axis=1
    )

    return df 

def score_validation_sets(model_pair : dict[str, OLSModel]) -> pd.DataFrame : 
    """
    Score validation sets for urban/rural model pair and compute distribution metrics.
    
    Parameters
    ----------
    model_pair : dict[str, Model]
        Dictionary with urban and rural Model objects.
    
    Returns
    -------
    pd.DataFrame
        Validation results with predictions, KES values, and decile/percentile rankings.
    """
    scored_val_sets = []

    for _, submodel in model_pair.items() : 
        scored_val_set = score_val_set(submodel)
        scored_val_sets.append(scored_val_set)
    
    validation_df = pd.concat(scored_val_sets).reset_index()

    # Add KES calcuations 
    validation_df['Actual_KES'] = np.exp(validation_df['Actual']) 
    validation_df['Predicted_KES'] = np.exp(validation_df['Predicted'])

    # Add decile and perecntile 
    validation_df['Actual_Decile'] = pd.qcut(validation_df['Actual_KES'], 10, labels=False) + 1
    validation_df['Actual_Percentile'] = pd.qcut(validation_df['Actual_KES'], 100, labels=False) + 1
    decile_edges = pd.qcut(validation_df['Actual_KES'], 10, retbins=True)[1]
    percentile_edges = pd.qcut(validation_df['Actual_KES'], 100, retbins=True)[1]

    # Get the predicted decile and percentile 
    validation_df['Predicted_Decile'] = np.digitize(validation_df['Predicted_KES'], decile_edges)
    validation_df['Predicted_Percentile'] = np.digitize(validation_df['Predicted_KES'], percentile_edges)

    # Get log error 
    validation_df['log_error'] = validation_df['Predicted'] - validation_df['Actual']

    return validation_df

def get_model_stats(submodel : sm.OLS) -> tuple[pd.DataFrame] : 
    """
    Get model stats and coefficients for one model 
    """
    coefs_df = (
        pd.DataFrame({
                "Coefficient": submodel.model.params,
                "Std Err": submodel.model.bse,
                "t-value": submodel.model.tvalues,
                "p-value": submodel.model.pvalues
        })
        .reset_index()
        .rename(columns={"index": "term"})
    )
    
    coefs_df['sig'] = coefs_df['p-value'].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "insig"
    )

    summary_stats = pd.DataFrame({
        "Statistic": ["R-squared", "Adj. R-squared", "F-statistic", "Prob(F-statistic)"],
        "Value": [submodel.model.rsquared, submodel.model.rsquared_adj, submodel.model.fvalue, submodel.model.f_pvalue]
    })

    return summary_stats, coefs_df


def save_models(
        models : dict[str, dict], 
        model_path : Path = paths.MODELS, 
        stats_path : Path = paths.RESULTS / 'model_stats') -> None : 
    """
    Save all trained models to pickle files.
    
    Parameters
    ----------
    models : dict[str, dict]
        Dictionary of model pairs (KCHS and IBS).
    path : Path, optional
        Directory path for saving models (default: paths.MODELS).
    """
    for model_pair in  models.values() : 
        for submodel in model_pair.values() : 
            # Save model files 
            with open(model_path / f'{submodel.name}.pickle', 'wb') as f: 
                pickle.dump(submodel, f)

            # Save model stats 
            model_stats, coefs = get_model_stats(submodel)
            with pd.ExcelWriter(stats_path / f'{submodel.name}_stats.xlsx', engine="openpyxl") as writer:
                model_stats.to_excel(writer, sheet_name="Summary Stats", index=False)
                coefs.to_excel(writer, sheet_name="Coefficients", index=False)
                
def main() : 
    """
    Execute full training pipeline: train models, score validation sets, and save outputs.
    """
    # Train models 
    print("\nTraining models...")
    models = train_models() 

    # Get KCHS validation set 
    print("Scoring validation sets...")
    kchs_validation = score_validation_sets(models['kchs'])
    ibs_validation = score_validation_sets(models['ibs'])

    # Merge with full training data 
    print("Merging with training data...")
    kchs_td_all = pd.read_csv(paths.PROCESSED_DATA / 'td_kchs_2021_all.csv', low_memory = False)
    ibs_td_all = pd.read_csv(paths.PROCESSED_DATA / 'td_ibs_2015_all.csv', low_memory = False)
    kchs_validation = kchs_validation.merge(kchs_td_all, on = 'hhid', how='left')
    ibs_validation = ibs_validation.merge(ibs_td_all, on = 'unique_id', how='left')

    # Add Poverty classification to the full training data as well
    kchs_validation = calculate_poverty_classification(kchs_validation)
    ibs_validation = calculate_poverty_classification(ibs_validation)

    # Save models 
    print("Saving models...")
    save_models(models)

    # Save validation data 
    print("Saving validation sets...")
    kchs_validation.to_csv(paths.PROCESSED_DATA / 'kchs_2021_validation.csv', index=False)
    ibs_validation.to_csv(paths.PROCESSED_DATA / 'ibs_2015_validation.csv', index=False)

    print("Done.")

if __name__ == "__main__" : 
    main()