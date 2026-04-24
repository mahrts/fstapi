"""This script download census data from the web and cleans_up the salary columns.

The data can be found at: https://archive.ics.uci.edu/dataset/20/census+income.

As indicated in the original url, we use the package```fetch_ucirepo``` with
 ```id=20``` to load the data directly from the web.
"""

import os
from pathlib import Path
import logging
from ucimlrepo import fetch_ucirepo

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def download_census():
    """Download and clean census_data, and save it to data/census.csv."""
    DATA_PATH = Path(__file__).parent.parent.parent / "data"
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    logger.info("Loading census data from https://archive.ics.uci.edu/dataset/20/census+income")
    census_income = fetch_ucirepo(id=20)
    df = census_income.data.features
    df["salary"] = census_income.data.targets

    logger.info("Remove space and points from salary columns.")
    df["salary"] = df["salary"].apply(lambda x: x.replace(" ", "").replace(".", ""))

    logger.info("Target column is clean, the pd.dataframe is saved")
    df[df["occupation"] != "?"].dropna().to_csv(DATA_PATH / "census.csv", index=False)

if __name__=="__main__":
    download_census()
