"""
KNBS Integrated Budget Survey 2015/2016 Loader and Preprocessing Module

This module provides utility functions for loading and preprocessing datasets
from KNBS (Kenya National Bureau of Statistics) 2015 Integrated Budget Household survey. 

This module relies on key functions from the knbs_core module. 

Currently processes household, consumption, non-food expenditure,
agricultural holdings, and individual member datasets. 

Author: Gabriel Geiger
Date: 2025-12-21
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

def map_item_codes(exp_df : pd.DataFrame, value_labels : dict[int : str]) -> pd.DataFrame : 
    """
    Maps item codes in expenditure data to item names (e.g. item code 801 becomes 'toilet paper')

    Args: 
        exp_df (pd.DataFrame): Dataframe containing the IBS nonfood expenditure data
        value_labels (dict): A dictionary mapping item codes to their string item names 
    Returns: 
        exp_df (pd.DataFrame): The expenditure dataframe with items 
    """
    exp_df['item'] = exp_df['item'].map(value_labels['nf2'])
    return exp_df 

def aggregate_land(df : pd.DataFrame) -> pd.DataFrame : 
    """
    Aggregates land parcel data to create aggregate land metrics. The agricultural holdings dataset is in long format
    where every row is a household's land parcel. This function produces a wide format with aggregated values. 

    Args: 
        df (pd.DataFrame): Long format agricultural holdings dataset 
    
    Returns: 
        df (pd.DataFrame): Dataset in a wide format with aggregated metrics. 
    """

    df = (
        df.groupby('unique_id')
        .agg(
            engaged_in_agriculture = pd.NamedAgg(
                column='did_any_member_of_the_household_engage_in_farming_in_the_last_12_months_whethe',
                aggfunc=lambda x: 1 if x.iloc[0] == 'Yes' else 0
            ),
            total_parcels = pd.NamedAgg(
                column='total_cultivated_parcels_of_land_by_household_members', 
                aggfunc='first'
            ),
            land_area_owned = pd.NamedAgg(
                column='what_is_the_area_of_the_parcel', 
                aggfunc=lambda x: x[df.loc[x.index, 'does_hh_own_this_parcel'] == 'Yes'].sum()
            ),
            productive_land = pd.NamedAgg(
                column='what_is_the_area_of_the_parcel', 
                aggfunc= 'sum'
            ),
            land_value_sell_owned = pd.NamedAgg(
                column='if_hh_were_to_sell_this_parcel_today_how_much_could_it_fetch', 
                aggfunc=lambda x: x[df.loc[x.index, 'does_hh_own_this_parcel'] == 'Yes'].sum()
            ),

        )
        .fillna(0)
        .reset_index()
    )

    return df 


def load_nf_expenditure(path : Path) -> pd.DataFrame : 
    """
    Loads the non-food expenditure dataset. 
    Automatic categorical conversion in StataReader fails for this file, so value labels are mapped manually. 
    The data are then reshaped from long to wide format, with one row per household and one column per 
    expenditure item (e.g. toilet_paper).

    Args: 
        path (Path): Path to the expenditure data
    
    Returns:
        df (pd.DataFrame): Expenditure dataframe in a wide format. 
    """

    df, var_labels, val_labels = read_stata(path, convert_categoricals=False, return_value_labels=True)

    df_wide = (
        df
        .rename(
            columns = {
                'hhid' : 'household_id', 'clid' : 'cluster_id', 'nf01' : 'item', 'nf04_amt' : 'item_amt',
            }
        )
        .pipe(add_unique_id)
        .pipe(map_item_codes, value_labels = val_labels)
        .drop_duplicates(subset=['unique_id', 'item'], keep='first') # Drop small number of duplicate purchases (around 40 rows) 
        .pivot(index='unique_id', columns='item', values='item_amt') # Pivot to wide format 
        .reset_index()
    )

    # Run formatting on our wide df 
    df_wide = (
        df_wide
        .loc[:, ~df_wide.columns.isna()] # Remove one 'NA' column
        .pipe(normalize_cols)
        .fillna(0)
    )

    return df_wide

def load_asset_dataset(path : Path) -> pd.DataFrame : 
    df = load_knbs_dataset(path) 
    
    df = (
        df
        .drop_duplicates(subset=['unique_id', 'item_code'], keep='first') # Drop small number of duplicate purchases
        .pivot(
            index='unique_id',
            columns='item_code',
            values = ['does_your_household_own_this_item', 
                      'number_of_items_the_household_owns', 
                      'what_is_the_age_of_this_item_if_>1_item_estimate_average_age',
                      'if_you_were_to_sell_this_item_taday_how_much_would_you_receive'
            ]
        )
        .reset_index()
        .pipe(flatten_multiindex_cols, col_name = 'does_your_household_own_this_item')
        .pipe(normalize_cols)
        .rename(columns = {'unique_id_' : 'unique_id'})
    )
    
    # Fill NA with "No" for owns column 
    owns_cols = [col for col in df.columns if col.startswith('owns_')]
    df[owns_cols] = df[owns_cols].fillna("No")

    return df 

def load_land_dataset(path : Path) -> pd.DataFrame : 
    """
    Loads the agricultural holding dataset, which is in a long format (each row is an individual parcel).
    Runs processing steps and then returns an aggregated version (ie. total land metrics)

    Args: 
        path (Path): Path to the agricultural holding DTA file. 
    Returns:
        df (pd.DataFrame): Processed agricultural holding data 
    """
    df, var_labels = read_stata(path)

    df = (
        df 
        .pipe(normalize_cols)
        .pipe(relabel_columns, col_labels = var_labels)
        # For some reason this .dta file uses different names for these variables 
        .rename( 
            columns = {
                'anonymized_cluster' : 'cluster_id',
                'anonymized_hh_number': 'household_id'
            }
        )
        .pipe(add_unique_id)
        .pipe(aggregate_land)
    )

    return df 

def load_agriculture_output(path : Path) -> pd.DataFrame : 
    df = load_knbs_dataset(path)

    df = (
        df
        .groupby('unique_id')
        .agg(
            engaged_in_crop_farming = pd.NamedAgg(
                column = 'did_any_member_of_the_hh_engage_in_crop_farming_in_the_last_12_months',
                aggfunc = lambda x: (x == 'Yes').any()
            ),
            sold_crops = pd.NamedAgg(
                column = 'how_much_of_the_harvest_was_sold__kgs', 
                aggfunc= lambda x: (x > 0).any()
            )
        )
        .fillna(False)
        .reset_index()
    )

    return df 

def load_livestock(path : Path) -> pd.DataFrame : 
    df = load_knbs_dataset(path) 

    # Rename wacking columns 
    df = df.rename(columns = {
        'during_the_last_twelve_months_has_any_member_of_the_household_reared_any_...' : 'livestock_reared',
        'how_many__._._._did_household_sell_during_the_last_12_months' : 'livestock_sold', 
        'how_many_...\x85_did_household_purchase_during_the_last_12_months' : 'livestock_purchased'
    })

    # Create pivot based on animals sold 
    pivot_sold_animals = (
        df
        .pivot_table(columns = 'livestock_reared', 
                     index = 'unique_id', 
                     values = 'livestock_sold', 
                     aggfunc=lambda x: (x > 0).any(),
                     observed=False
        )
        .fillna(0)
        .reset_index()
        .pipe(normalize_cols)
        .add_prefix('livestock_sold_')
        .rename(columns = {'livestock_sold_unique_id' : 'unique_id'})
    )
    pivot_sold_animals['sold_livestock'] = pivot_sold_animals.filter(like = 'livestock_sold').any(axis=1)


    # Create pivot based on animals purchased 
    pivot_purch_animals = (
        df
        .pivot_table(columns = 'livestock_reared', 
                     index = 'unique_id', 
                     values = 'livestock_purchased', 
                     aggfunc=lambda x: (x > 0).any(),
                     observed = False
        )
        .fillna(0)
        .reset_index()
        .pipe(normalize_cols)
        .add_prefix('livestock_purch_')
        .rename(columns = {'livestock_purch_unique_id' : 'unique_id'})
    )
    pivot_purch_animals['purchased_livestock'] = pivot_purch_animals.filter(like = 'livestock_purch').any(axis=1)

    merged_df = pd.merge(pivot_purch_animals, pivot_sold_animals, how = 'left', on = 'unique_id')

    return merged_df

def load_individual_data(path : Path) -> pd.DataFrame : 
    individual = load_knbs_dataset(path / 'HH_Members_Information.dta')
    work = load_knbs_dataset(path / 'labor.dta') 

    individual['individual_id'] = individual['unique_id'] + '_' + individual['line_number'].astype(str)
    work['individual_id'] = work['unique_id'] + '_' + work['line_number'].astype(str)
    work = work.drop_duplicates(subset=['individual_id'])

    merged = pd.merge(individual, work, on = 'individual_id', how='left', suffixes = (None, '_work'))

    return merged 
   

def load_ibs_2015(path : Path = paths.IBS_2015, export : bool = True) -> dict[str : pd.DataFrame] : 
    """
    Loads datasets from the 2015 Integrated Budget Survey (IBS). Merges all household level data 
    into one dataframe. Returns the individual level data seperately. 

    Args: 
        path (Path): Path to the folder containing the IBS .dta files. Defaults to path set in paths config. 
    
    Returns: 
        datasets (dict): A dictionary containing each dataset as a Pandas dataframe
    """

    # Load household data 
    hh_data = load_knbs_dataset(path / "HH_Information.dta")
    print("\nLoaded household data with shape", hh_data.shape)

    # Consumption data 
    con_data = load_knbs_dataset(path / 'Consumption_aggregate.dta')
    print("Loaded consumption data with shape", con_data.shape)

    # Non food expenditure data 
    exp_data = load_nf_expenditure(path / 'nonfood.dta')
    print("Loaded non-food expenditure data with shape", exp_data.shape)

    # Load asset data 
    assets_data = load_asset_dataset(path / 'assets.dta')
    print("Loaded assets data with shape", exp_data.shape)

    # Land data 
    land_data = load_land_dataset(path / 'Agriculture holding (K1_K19).dta')
    print("Loaded land data with shape", land_data.shape)

    # Agricultural output 
    ag_output = load_agriculture_output(path / 'Agriculture output (L1_L20).dta')
    print("Loaded ag output data with shape", ag_output.shape)

    # Livestock 
    livestock = load_livestock(path / 'Livestock (M1_M15).dta')
    print("Loaded livestock data with shape", livestock.shape)

    # Individual data 
    individual_data = load_individual_data(path)
    print("\nLoaded individual household member data with shape", individual_data.shape, '\n')

    datasets = {
        'households' : hh_data, 
        'consumption' : con_data, 
        'nonfood' : exp_data, 
        'assets' : assets_data, 
        'land' : land_data, 
        'ag_output' : ag_output, 
        'livestock' : livestock,
        'individuals' : individual_data
    }

    # Load our renaming dict to rename any columms we want to manually override 
    ibs_rename = pd.read_excel(paths.CONFIG / 'ibs_rename.xlsx')
    ibs_rename_mapping = dict(zip(ibs_rename['IBS current'], ibs_rename['Rename']))

    for key, df in datasets.items() : 
        datasets[key] = df.rename(columns = ibs_rename_mapping)

    if export : 
        export_knbs_datasets(paths.INTERMEDIATE_DATA / 'knbs_ibs_2015', 
                             datasets, 
                             dataset_name='ibs_2015'
        )

    return datasets 


def main() : 
    load_ibs_2015() 

if __name__ == "__main__" : 
    main()