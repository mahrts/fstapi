"""This file contains fixture for all pytest."""

import pytest
from ingest.download_data import download_census
from ml.data import process_data

@pytest.fixture(scope="session")
def census_df():
    df = download_census()
    return df

@pytest.fixture(scope="session")
def data_processor(census_df):
    cat_features = census_df.select_dtypes(include="object").columns.tolist()
    cat_features.remove("salary")

    features, targets, _, _ = process_data(df = census_df, categorical_features=cat_features,
                            label="salary", training=True, encoder=None, lb=None)

    return features, targets
