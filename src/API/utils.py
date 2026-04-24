"""This script contains function that slice test data and compute corresponding score."""

from pathlib import Path
from typing import Dict
import pandas as pd
from train_model import fetch_test_score

FULLCENSUSDATAPATH = Path(__file__).parent.parent.parent / "data" / "census.csv"
FULLCENSUSDATA = pd.read_csv(FULLCENSUSDATAPATH)

TESTINGDATAPATH = Path(__file__).parent.parent.parent / "deployment" / "data_for_scoring.csv"
TESTINGDATA = pd.read_csv(TESTINGDATAPATH)

query_param_example = {"native-country": "United-States",
                    "race": "White", 
                    "education": "Masters"}

def fetch_post_query_params(census_df = FULLCENSUSDATA):
    """Return dictionary: keys are column name of dataframe,
    values are list of unique value of that columns.
    
    Args:
        PATH: path to a csv file
    Returns:
        dictionary with each colmanes as key, and list of unique
        values of that column as value.
    """
    cat_features = census_df.select_dtypes(include="object").columns.tolist()
    cat_features.remove("salary")

    post_query_params = {}
    for feature in cat_features:
        post_query_params[f"{feature}"] = census_df[feature].unique().tolist()

    return post_query_params

def slice_data(census_df: pd.DataFrame = TESTINGDATA, filters: Dict = None):
    """
    Slice census_df according to a filter.

    Args:
        census_df: data frame to filter
        filter: a dictionary of column name and column value
    
    Return:
        Subset of the data frame with filtered value according to filter
    """
    if filters is None:
        filters = {"native-country": "United-States"}

    mask = True
    for col, val in filters.items():
        mask &= (census_df[col] == val)
    try:
        return census_df[mask]
    except KeyError:
        return pd.DataFrame()

def compute_slice_score(test_data: pd.DataFrame = TESTINGDATA,
                        query_param: dict = None):
    """Perform preiction on slice of data {'column': 'column_value'},
    compute and return each model scores on that slice.

    Args:
        data: a test-like dataset
        query_param: dictionary with column name as keys,
                     and a value of thet column as value.
    Example of query_param: {"native_country": "United-States",
                             "race": "White",
                             "Education": "Masters"}
    """
    score_detail = {}

    if query_param is None:
        df = test_data
        score_detail["slice_method"] = "No slice, this is the score on all test data."

    else:
        slices = []
        for key, val in query_param.items():
            slices.append(f"(DATA['{key}'] == '{val}')")
        slice_string = " & ".join(slices)

        score_detail["slice_method"] = f"DATA[{slice_string}]"

        df = slice_data(census_df=test_data, filters=query_param)

    score_detail["slice_size"] = df.shape[0]

    scores = fetch_test_score(data=df)
    score_detail["scores"] = scores

    return score_detail

if __name__ == "__main__":
    print(compute_slice_score())
