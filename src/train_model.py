"""This script train RandomForestClassifier on the census data."""

import logging

from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from ingest.download_data import download_census

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__=="__main__":
    logger.info("Loading census data from https://archive.ics.uci.edu/dataset/20/census+income")
    df = download_census()

    logger.info("Taking list of columns with object type")
    cat_features = df.select_dtypes(include="object").columns.tolist()
    cat_features.remove("salary")
    logger.info(f"Cat_features: {cat_features}")

    logger.info("splitting train and test data.")
    train, test = train_test_split(df.dropna(), test_size=0.20)
    logger.info(f"train shape: {train.shape}, test shape: {test.shape}")

    logger.info("Process training data: get lb and encoder from this.")
    X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True)

    logger.info("Process test data with lb and encoder.")
    X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features,
            label="salary", training=False, encoder=encoder, lb=lb)

    logger.info(f"{X_test.shape}, {y_test.shape}")
    logger.info(f"{X_train.shape}, {y_train.shape}")

    logger.info("Start training randomforest.")
    rf = train_model(X_train, y_train)
    joblib.dump(rf, Path(__file__).parent.parent / "rndfmodel.pkl")
    logger.info("Training finished, model saved.")
