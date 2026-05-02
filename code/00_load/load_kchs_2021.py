"""
Kenya Continious Households Survey 2021 Loader and Preprocessing Module

This module provides utility functions for loading and preprocessing datasets
from KNBS (Kenya National Bureau of Statistics) 2021 KCHS Survey. 

This module relies on key functions from the knbs_core module. 

Currently processes household, consumption, non-food expenditure,
assets, and individual member datasets. 

Authors: Gabriel Geiger & Purity Mukami
Date: 2025-12-22
"""

import sys 
import pandas as pd 
from pathlib import Path

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import paths 
from knbs_core import *
from util.normalize_column_name import normalize_cols

DATASETS = [
    'individuals', 
    'households',
    'consumption',
    'nonfood', 
    'livestock',
    'assets',
]

def all_columns_match(df : pd.DataFrame, mapping_keys : list) -> bool : 
    # Columns in mapping_dict that are missing in df
    existing_cols = set(df.columns)
    mapped_cols = set(mapping_keys)

    missing_cols = mapped_cols - existing_cols

    # If there is a mismatch between columns, return False 
    if len(missing_cols) > 0 :  
        print("\nFollowing columns were not matched: ", ' '.join(missing_cols))
        print('Dataframe columns: ', ' '.join(df.columns))

        return False 
    
    return True 


def pivot_expenditure(nf_exp_long : pd.DataFrame, col : str) -> pd.DataFrame : 
    """
    Pivots the long-format expenditure data to a wide format based on the specified column. 
    In effect, column is either nonfood item code (specific purchases) or purchase categories. 

    Args: 
        nf_exp_long (pd.DataFrame): The long dataframe to be put into wide format
        col (str) : The col to pivot on. 
    
    Returns : 
        nfp_wide: The dataframe in a wide format 
    """

    nfp_wide = (
        nf_exp_long
        .pivot_table(
            index = 'hhid',
            columns = col, 
            values = 'nfcons',
            fill_value = 0, 
            aggfunc='sum', 
            observed=False
        )
        .fillna(0)
        .reset_index()
        .pipe(normalize_cols)
    )

    return nfp_wide

def load_nf_expenditure(path : Path, dd : dict[str : str]) -> pd.DataFrame : 
    """
    Load and reshape expenditure datafrom long to wide format.

    The expenditure dataset is pivoted so that each row represents a household and
    columns purchases of various items. 
    Args:
        path (Path): Path to the asset Stata (.dta) file.
        dd (dict): Data dictionary used to relabel survey variables.

    Returns:
        pd.DataFrame: Wide-format household asset dataset.
    """

    nf_exp_long = load_kchs_dataset(path, dd)

    # Create a wide version of specific expenditure data e.g. toilet paper
    nfp_exp_items_wide = pivot_expenditure(nf_exp_long, 'nonfooditem_code')

    # Create a wide version of category-level expenditure data e.g. housing
    nfp_exp_cat_wide = pivot_expenditure(nf_exp_long, 'coicopcode') 
    
    # Merge two together and drop duplicates cols
    nfp_exp = pd.merge(nfp_exp_items_wide, nfp_exp_cat_wide, how = 'left', on='hhid', suffixes=('', '_drop'))
    nfp_exp = nfp_exp.loc[:, ~nfp_exp.columns.str.endswith('_drop')]

    return nfp_exp

def load_assets_data(path : Path, dd : dict[str : str]) -> pd.DataFrame : 
    """
    Load and reshape household asset data from long to wide format.

    The asset dataset is pivoted so that each row represents a household and
    columns represent asset-level attributes (e.g. ownership, value, age)
    by asset item code. 
    Args:
        path (Path): Path to the asset Stata (.dta) file.
        dd (dict): Data dictionary used to relabel survey variables.

    Returns:
        pd.DataFrame: Wide-format household asset dataset.
    """
    df = load_kchs_dataset(path, dd)

    df = (
        df 
        .pivot(
            index='hhid',
            columns='item_code',
            values=['type_of_transaction','amount_paid', 'value_aquired', 'num_items_owned', 'item_age', 'item_resell_value']
        )
        .reset_index()
        .pipe(flatten_multiindex_cols, col_name = 'type_of_transaction')
        .pipe(normalize_cols)
        .rename(columns = {'hhid_' : 'hhid'}) # For some reason hhid gets an extra underscore 
    )

    return df 

def load_kchs_dds(path : Path = paths.DATA) -> dict[str : pd.DataFrame] : 
    """
    Loads the corresponding data dictionary for each of the KCHS datasets. 
    Each data dictionary is a sheet in the GSheets Kenya Data Dictionary file on the Drive. 

    Args: 
        path (Path): Path to data dictionaries 
    
    Returns:
        dictionaries: A dictionary where each key is name of dataset and each value is a data dictionary as a dataframe
    """

    xlsx_path = path / "KCHS Data Dictionary v1.7.xlsx"

    return {
        name : pd.read_excel(xlsx_path, sheet_name = name)
        for name in DATASETS
    }
 
def load_kchs_dataset(path : Path, dd : pd.DataFrame) -> pd.DataFrame : 
    """Loads a KNBS .data file from KCHS and its correpsonding data dictionary. 
    Runs a series of processing sterps using the the generic process_knbs_dataset. 
    Args:
        path (Path): Path to the KNBS dataset
        dd (pd.DataFrame): A data dictionary for the dataset
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    # Read the Stata file
    df = read_stata(path)[0]

     # Create mapping between question code and variable 
    mapping_dict = dict(zip(dd['Question Code'], dd['Variable']))

    df = process_knbs_dataset(df, mapping_dict)
    
    return df 
    
def load_kchs_2021(path : Path = paths.KCHS_2021, export : bool = True) -> dict[str : pd.DataFrame] : 
    # Load data dictionaries 
    dds = load_kchs_dds()

    # Load household level 
    hh_data = load_kchs_dataset(path / 'households_microdata.dta', dds['households'])
    print("\nLoaded household data with shape", hh_data.shape)

    # Load consumption data 
    con_data = load_kchs_dataset(path / 'consagg_microdata.dta', dds['consumption'])
    print("Loaded consumption data with shape", con_data.shape)

    # Non-food expenditure 
    exp_data = load_nf_expenditure(path / 'nonfood_items_microdata.dta', dds['nonfood'])
    print("Loaded non-food expenditure data with shape", exp_data.shape)

    # Assets data 
    assets_data = load_assets_data(path / 'nfitemsKNP.dta', dds['assets'])
    print("Loaded assets data with shape", assets_data.shape)

    # Agriculture data 
    ag_data = load_kchs_dataset(path / 'livestock.dta', dds['livestock'])

    # Individual data 
    individual_data = load_kchs_dataset(path / 'individuals_microdata.dta', dds['individuals'])
    print("\nLoaded individual household member data with shape", individual_data.shape)

    datasets = {
        'households' : hh_data, 
        'consumption' : con_data, 
        'nonfood' : exp_data, 
        'assets' : assets_data, 
        'agriculture' : ag_data,
        'individuals' : individual_data
    }

    if export : 
        export_knbs_datasets(paths.INTERMEDIATE_DATA / 'knbs_kchs_2021', datasets, dataset_name='kchs_2021')

    return datasets 

def main() : 
    load_kchs_2021() 

if __name__ == "__main__" : 
    main()