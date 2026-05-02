"""
KNBS Data Utilities

This module provides core and generic functions to read, clean, and transform KNBS (Kenyan Statistics Bureau) survey datasets
stored in Stata (.dta) format. Functions from this module are imported into the loading modules for specific datasets 
e.g. IBS 2015 or KCHS 2021. 

Author: Gabriel Geiger 
""" 

import sys 

from pandas.io.stata import StataReader
import pandas as pd 
from pathlib import Path

# Import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from util.normalize_column_name import normalize_column_name, normalize_cols

def drop_duplicate_columns(df : pd.DataFrame) -> pd.DataFrame : 
    """Drop duplicate multi-level column names, keep the first"""

    return df.loc[:, ~df.columns.duplicated()]

def flatten_multiindex_cols(df : pd.DataFrame, col_name : str) -> pd.DataFrame : 
    """
    Flattens a MultiIndex columns of a pandas DataFrame into single-level columns.
    """

    df.columns.name = None
    df.columns = [f"{str(val).replace(col_name, 'owns')}_{str(col).strip("  ")}" for val, col in df.columns]

    return df 

def add_unique_id(df : pd.DataFrame) -> pd.DataFrame : 
    """
    Adds a unique id column to a KNBS dataframe by combining cluster_id and household_id
    """

    clid = 'cluster_id' if 'cluster_id' in df.columns else 'clid'
    hhid = 'household_id' if 'household_id' in df.columns else 'hhid'

    df["unique_id"] = (
        df[clid].astype(str)
        + "_"
        + df[hhid].astype(str)
    ).str.replace(".0", "", regex=False)

    return df 

def all_columns_match(df : pd.DataFrame, mapping_keys : list) -> bool : 
    """
    Checks that all columns in a dataframe have a label in a mapping dictionary
    """
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

def relabel_columns(df : pd.DataFrame, col_labels : dict[str : str]) -> pd.DataFrame : 
    """
    Relabel the columns of a KNBS dataset from non-descriptive codes (e.g., 'c6', 'd8')
    to descriptive names.

    Args:
        df (pd.DataFrame):
            A pandas DataFrame representing a raw KNBS dataset.
        col_labels (dict[str, str]):
            A dictionary mapping original column codes to descriptive labels,
            e.g., {'c6': 'Number of people in household'}. Can come from the Stata
            file's variable labels or a manual data dictionary.

    Returns:
        pd.DataFrame:
            A DataFrame with renamed and normalized columns, e.g.,
            'number_of_people_in_household'.
    """
    col_mapping = {
        normalize_column_name(col) : normalize_column_name(label)
        for col, label in col_labels.items()
    }

    if all_columns_match(df, col_mapping.keys()) == False : 
        raise Exception(f"Fatal error: Columns not matched between raw KCHS dataframe and data dictionary.")

    renamed_df = df.rename(columns = col_mapping)

    return renamed_df

def read_stata(path : Path, 
               convert_categoricals : bool = True,
               return_value_labels : bool = False, 
    ) -> tuple[pd.DataFrame, dict[str : str]] : 
    """
    Read a Stata (.dta) file into a pandas DataFrame.

    Stata files store variable labels (column descriptions) and value labels
    (category mappings) as metadata. In KNBS survey data, columns are typically
    named using short survey codes (e.g. ``c6``, ``c7``), and categorical
    variables may be stored as numeric codes with associated value labels.

    By default, ``StataReader`` converts categorical codes to their string
    labels. In some cases this conversion can be problematic, so this function
    allows disabling categorical conversion and optionally returning the raw
    value labels for manual mapping.

    Args:
        path (Path):
            Path to the Stata (.dta) file.
        convert_categoricals (bool):
            Whether to convert categorical codes to labeled strings.
        return_value_labels (bool):
            Whether to return value labels in addition to variable labels.

    Returns:
        tuple:
            If ``return_value_labels`` is False:
                (DataFrame, variable_labels)
            If ``return_value_labels`` is True:
                (DataFrame, variable_labels, value_labels)
    """
    with StataReader(path) as reader:
        df = reader.read(convert_categoricals = convert_categoricals)
        variable_labels = reader.variable_labels()
        value_labels = reader.value_labels()

    if return_value_labels : 
        return (df, variable_labels, value_labels)
    else : 
        return (df, variable_labels)

def process_knbs_dataset(df : pd.DataFrame, var_labels : dict[str : str]) -> pd.DataFrame : 
    """
    A generic function to run a series of simple preprocessing steps on a raw KNBS dataframe. 
    First normalizes column names (removing any punctiation, lowering etc.). 
    Then relables columns from survey question names (c6) to more intrperetable labels
    Finally adds a unique id for each row and drops a small number of duplicate column names

    Args:
        df (pd.DataFrame): The raw unprocessed dataframe 
        var_labels (dict): Column labels, either from stata file or manual 
    
    Returns:
        df (pd.DataFrame): The processed dataframe
    """
    df = (
        df
        .pipe(normalize_cols)
        .pipe(relabel_columns, col_labels = var_labels)
        .pipe(drop_duplicate_columns) 
        .pipe(add_unique_id)
    )

    return df 

def load_knbs_dataset(path : Path) -> pd.DataFrame : 
    """A generic function to load a KNBS .dta file and run a series of simple proccessing steps. 
    Args:
        path (Path): Path to the KNBS dataset
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    # Read the Stata file
    df, var_labels = read_stata(path)

    # Run cleaning pipeline 
    df = process_knbs_dataset(df, var_labels)
    
    return df 

def export_knbs_datasets(path : Path, datasets : dict, dataset_name : str = '') -> None : 

    for d_name, df in datasets.items() : 
        filename = f'{dataset_name}_{d_name}.csv'
        print(f"Saving {filename} to {path}")
        
        df.to_csv(path / filename, index=False)

    
