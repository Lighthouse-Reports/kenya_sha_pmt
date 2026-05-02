import re 
import pandas as pd 

def normalize_column_name(col_name: str) -> str :
    """Normalize column names by converting to lowercase and replacing spaces with underscores.

    Args:
        col_name (str): Original column name.

    Returns:
        str: Normalized column name.
    """    
    # remove zeros between letters and numbers
    col_name = re.sub(r'([A-Za-z])0+(\d)', r'\1\2', col_name)
    
    # lowercase
    col_name = col_name.lower()

    # Replace spaces with underscores 
    col_name = col_name.replace(' ','_')

    # Replace dashes 
    col_name = col_name.replace('-',"_")

    # Replace slashes 
    col_name = col_name.replace('/','_')

    # Replace parantheses 
    col_name = col_name.replace('(','').replace(')','')

    # Replace brackets 
    col_name = col_name.replace('[', '').replace(']','')

    # Replace ? 
    col_name = col_name.replace('?','')

    # Replace column name 
    col_name = col_name.replace(',','')

    # Replace any /
    col_name = col_name.replace('\'','')

    # Replace __ with _ 
    col_name = col_name.replace('__','_')

    # Strip 
    col_name = col_name.strip()
    
    return col_name

def normalize_cols(df : pd.DataFrame) -> pd.DataFrame : 
    """
    Renames columns to normalize column names 
    """
    df.columns = df.columns.map(normalize_column_name)
    return df