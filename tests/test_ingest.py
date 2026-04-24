import pandas as pd

def test_download_census(census_df):
    assert isinstance(census_df, pd.DataFrame), "does not return a dataframe"
    assert census_df.shape[0] > 0, "returned df has now row"
    assert census_df.shape[1] > 0, "returned df has no columns"
    assert "salary" in census_df.columns.tolist(), "There is not target column salary."
    assert len(census_df) == len(census_df.dropna()), "There are still missing values"
