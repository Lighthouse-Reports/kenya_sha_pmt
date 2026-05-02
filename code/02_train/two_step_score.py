"""
Two-step income classification scoring pipeline.

Runs a hierarchical prediction: a parent model first assigns households to a
coarse income band (low/middle/high), then a band-specific submodel produces
a fine-grained income range within that band. Results for both parent model
variants (all-vars and adjusted) are joined and exported for validation.
"""
import sys
import pickle
import os
import pandas as pd
import numpy as np
from pathlib import Path
from train_log_model import LogModel

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from config import paths

COARSE_LABELS = {0 : "Low income", 1 : "Middle income", 2 : "High income"}
LOW_LABELS = {0: '0-5k', 1: '5k-10k',   2: '10k-15k'}
MEDIUM_LABELS = {0: '15k-30k', 1: '30k-50k',  2: '50k-100k'}
HIGH_LABELS = {0: '100k-200k', 1: '200k-300k', 2: '300k+'}

def load_models() -> dict[str : LogModel] :
    """Load all trained LogModel objects from the models directory, excluding IBS and KCHS variants."""
    models = {}
    for model in os.listdir(paths.MODELS) :
        if ('ibs' not in model) and ('kchs' not in model) and ('DS_Store' not in model) :
            with open(paths.MODELS / model, 'rb') as f :
                model_file = pickle.load(f)

            print(f"Loaded {model_file.name}.")

            models[model_file.name] = model_file

    return models


def predict(model : LogModel, df : pd.DataFrame) -> pd.DataFrame :
    """
    Generate class predictions for df using a trained LogModel.

    Scales df using only the features selected after elastic net regularization,
    aligning indices to the model's original scaler, then returns the predicted class index.
    """
    # First get all columns in the scaler to get correct indices.
    all_cols = model.scaler.feature_names_in_.tolist()

    # Then get specific columns and indices used in the model after elastic net
    selected_cols = model.X_train.columns.to_list()
    indices = [all_cols.index(c) for c in selected_cols]

    # Appropriately scale the test set for the model.
    X_scaled = pd.DataFrame(
        (df[selected_cols].values - model.scaler.mean_[indices]) / model.scaler.scale_[indices],
        columns=selected_cols,
        index=df.index,
    )

    probs = model.model.predict(X_scaled)
    return probs.idxmax(axis=1).astype(int)

def rename_values(row : pd.Series, coarse_col : str, fine_col : str) -> str :
    """Map a numeric fine-grained prediction to its income range label based on the coarse band."""
    coarse = row[coarse_col]

    value = row[fine_col]
    if coarse == 'Low income':
        return LOW_LABELS[value]

    elif coarse == 'Middle income':
        return MEDIUM_LABELS[value]

    else:
        return HIGH_LABELS[value]

def two_step_prediction(
        parent_model : LogModel,
        submodels : dict[int, LogModel],
        test_df : pd.DataFrame
) -> pd.DataFrame :
    """
    Run the two-step hierarchical prediction on test_df.

    First predicts a coarse income band with parent_model, then routes each
    household to the appropriate submodel for a fine-grained prediction within
    that band. Both predictions are mapped to human-readable labels.
    """
    print(f"\nRunning two step prediction with parent model {parent_model.name} on test set with shape {test_df.shape}")
    result_df = test_df.copy()

    # Run coarse prediction
    result_df['coarse_pred'] = predict(parent_model, test_df)

    # Run fine prediction
    fine_pred = pd.Series(index = test_df.index, dtype = float)

    for band, submodel in submodels.items() :
        mask = result_df['coarse_pred'] == band
        fine_pred[mask] = predict(submodel, test_df[mask])

    result_df['fine_pred'] = fine_pred.astype(int)

    # Rename values
    result_df['coarse_pred'] = result_df['coarse_pred'].map(COARSE_LABELS)
    result_df['coarse_band'] = result_df['coarse_band'].map(COARSE_LABELS)
    result_df['fine_pred'] = result_df.apply(lambda row: rename_values(row, 'coarse_pred', 'fine_pred'), axis=1)
    result_df['fine_band'] = result_df.apply(lambda row: rename_values(row, 'coarse_band', 'fine_band'), axis=1)

    return result_df

def main() :
    """
    Score the KCHS 2021 test set with both parent model variants and export results.

    Loads all trained models, runs two-step predictions for the all-vars and
    adjusted parent models, joins the outputs, and saves a validation CSV.
    """
    # Load models
    print("\nLoading models...")
    models = load_models()
    submodels = {
        0 : models['low_income_model'],
        1 : models['middle_income_model'],
        2 : models['high_income_model'],
    }

    # Load training data and filter to only test set
    print("\nLoading test set...")
    test_ids = models['parent_model_all_vars'].X_test.index
    test_set = pd.read_csv(paths.PROCESSED_DATA / 'td_kchs_2021_log.csv', index_col='hhid').loc[test_ids]

    # Run two step predictions on both parent models
    all_vars_two_step = two_step_prediction(models['parent_model_all_vars'], submodels, test_set)
    adjusted_two_step = (
        two_step_prediction(models['parent_model_adjusted'], submodels, test_set)
        [['coarse_pred', 'fine_pred']]
        .rename(
            columns = {'coarse_pred' : 'coarse_pred_adjusted', 'fine_pred' : 'fine_pred_adjusted'}
        )
    )

    validation_df = all_vars_two_step.rename(columns={
        'coarse_pred' : 'coarse_pred_all_vars',
        'fine_pred'   : 'fine_pred_all_vars',
    }).join(adjusted_two_step)

    # Export
    print("Saving validation dataframe...")
    validation_df.to_csv(paths.PROCESSED_DATA / 'kchs_2021_validation_log.csv')

    print("\nDone.")

    
if __name__ == "__main__" : 
    main()