"""
Training utilities for training proxy means testing models based on various KNBS Surveys. 

This module provides functions to train separate OLS regression models for urban
and rural populations. The PMT differentiates between rural and urban population, 
so a submodel for each is trained and then returned as named tuples. 

Author: Purity Mukami 
"""
import pandas as pd 
import statsmodels.api as sm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class OLSModel:
    name : str
    model : sm.OLS
    X_test : pd.DataFrame
    y_test : pd.Series 

def split_rural_urban(df : pd.DataFrame) -> tuple[pd.DataFrame] :
    """Split dataframe into urban and rural subsets based on classification column.""" 
    urban_data = df[df['urban_rural_classification'] == "Urban"].copy()
    rural_data = df[df['urban_rural_classification'] == "Rural"].copy()

    return (urban_data, rural_data)

def prepare_train_test(df : pd.DataFrame, label : str) -> tuple:
    """
    Prepare train/test split and add intercept term for OLS regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    label : str
        Column name for the target variable.
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test with intercept terms added.
    """
    y = df[label]
    
    # Drop IDs and Target columns
    X = df.drop(columns=[label, 'urban_rural_classification'])

    #  80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add the intercept (constant) 
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    return X_train, X_test, y_train, y_test

def train_ols_models(
        df : pd.DataFrame, 
        model_name : str = '', 
        label: str = 'actual_consumption_log', 
        filter_sig : None  | float = None
    ) -> dict[str, OLSModel] :
    """
    Train separate OLS models for urban and rural populations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with features and target variable.
    model_name : str, optional
        Name prefix for the trained models e.g. IBS or KCHS 
    label : str, optional
        Target variable column name (default: 'actual_consumption_log').
    filter_sig : float, optional
        Unused parameter (reserved for future significance filtering).
    
    Returns
    -------
    tuple[namedtuple]
        Urban and rural model objects containing fitted model, test data, and metadata.
    """ 

    # Drop any rows that do not have the label 
    df = df.dropna(subset = [label])

    # Split into urban and rural. 
    urban_df, rural_df = split_rural_urban(df)

    # Get train and test sets for each model 
    X_train_r, X_test_r, y_train_r, y_test_r = prepare_train_test(rural_df, label)
    X_train_u, X_test_u, y_train_u, y_test_u = prepare_train_test(urban_df, label)

    # Train submodels 
    urban_estimator = sm.OLS(y_train_u, X_train_u).fit()
    rural_estimator = sm.OLS(y_train_r, X_train_r).fit()

    # Create named tuple object 
    urban_model = OLSModel(
        name = f"{model_name}_urban",
        model = urban_estimator, 
        X_test = X_test_u, 
        y_test = y_test_u
    )

    rural_model = OLSModel(
        name = f"{model_name}_rural",
        model = rural_estimator, 
        X_test = X_test_r, 
        y_test = y_test_r
    )
    
    return {urban_model.name : urban_model, rural_model.name : rural_model}

def main() : 
    pass 

if __name__ == "__main__" : 
    main()