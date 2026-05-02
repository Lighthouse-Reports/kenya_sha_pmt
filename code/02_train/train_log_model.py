"""
Ordinal logit model training pipeline.

Selects features via elastic net cross-validation (LogisticRegressionCV),
then fits a final OrderedModel (ordinal logit) on the selected variables.

Expects a pre-split train/test set — the global stratified holdout split is
handled upstream in training_log.py so that all models share the same test set
and no household leaks between training and evaluation.

Exports
-------
LogModel : dataclass
    Immutable container for a fitted model, its scaler, and train/test splits.
train_logit_model : function
    End-to-end training function — scale, regularize, select, fit, return LogModel.
"""
import warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass(frozen=True)
class LogModel:
    """Immutable container for a fitted ordinal logit model and its train/test splits."""
    name    : str
    model   : object
    scaler  : object        
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series


def train_logit_model(
        X_train : pd.DataFrame,
        X_test  : pd.DataFrame,
        y_train : pd.Series,
        y_test  : pd.Series,
        model_name : str = '',
) -> LogModel:
    """
    Scale features, select via elastic net CV, then fit a final ordinal logit model.

    The train/test split is done upstream (training_log.py) so that all models
    share the same global holdout set.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices — unscaled, same columns.
    y_train, y_test : pd.Series
        Integer labels (0 / 1 / 2).
    model_name : str
        Label used for logging and stored in the returned LogModel.

    Returns
    -------
    LogModel
        Fitted ordinal logit result alongside scaler and train/test splits.
    """
    print("\nTraining model", model_name)

    # Scale
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    # Turn labels into ordered categoricals
    y_train = pd.Categorical(y_train, categories=[0, 1, 2], ordered=True)
    y_test  = pd.Categorical(y_test,  categories=[0, 1, 2], ordered=True)

    # Cross-validated elastic net regularization for feature selection
    print("Running regularization...")
    selector = LogisticRegressionCV(
        Cs=np.logspace(-2, 1, 10),
        cv=5,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0],
        random_state=42,
        max_iter=1000,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)

    # Fit final OrderedModel on selected variables only
    print("Fitting final model...")
    selected_vars = X_train.columns[(selector.coef_ != 0).any(axis=0)].tolist()
    final_model   = OrderedModel(y_train, X_train[selected_vars], distr='logit')
    result        = final_model.fit(method='bfgs', maxiter=1000)

    print(f"Final model fitted with {len(selected_vars)}/{len(X_train.columns)} columns.")

    return LogModel(
        name=model_name,
        model=result,
        scaler=scaler,
        X_train=X_train[selected_vars],
        X_test=X_test[selected_vars],
        y_train=y_train,
        y_test=y_test,
    )


def main() : 
    pass 


if __name__ == "__main__" : 
    main()