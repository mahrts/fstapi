"""This script train RandomForestClassifier on the census data."""

import os
import logging
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
from ingest.download_data import download_census
import pandas as pd
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

DEPLOYMENT_PATH = Path(__file__).parent.parent / "deployment"

def training():
    logger.info("Loading census data from https://archive.ics.uci.edu/dataset/20/census+income")
    df = download_census()

    logger.info("Taking list of columns with object type")
    cat_features = df.select_dtypes(include="object").columns.tolist()
    cat_features.remove("salary")
    logger.info(f"Cat_features: {cat_features}")

    logger.info("splitting train and test data.")
    train, test = train_test_split(df.dropna(), test_size=0.20)
    
    logger.info("Saving test data to deployment path, for testing the trained model.")
    if not os.path.isdir(DEPLOYMENT_PATH):
        os.mkdir(DEPLOYMENT_PATH)
    test.to_csv(DEPLOYMENT_PATH / "data_for_scoring.csv", index=False)
    logger.info("test data saved: data_for_scoring.csv")

    logger.info("Process training data: get lb and encoder from this.")
    X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True)

    logger.info("Start training randomforest.")
    rf = train_model(X_train, y_train)
    logger.info("training finished...")

    logger.info("Save model.pkl, and save the encoder.pkl, label_binarizer.pkl")
    joblib.dump(rf, DEPLOYMENT_PATH / "rndfmodel.pkl")
    joblib.dump(encoder, DEPLOYMENT_PATH / "encoder.pkl")
    joblib.dump(lb, DEPLOYMENT_PATH / "label_binarizer.pkl")
    logger.info("Model and encoder saved to deployment.")

def load_model(model_path = DEPLOYMENT_PATH / "rndfmodel.pkl"):
    """Loading and return the trained model."""
    return joblib.load(model_path)

def load_encoder(encoder_path = DEPLOYMENT_PATH / "encoder.pkl"):
    """Loading and return the fitted encoder before training."""
    return joblib.load(encoder_path)

def load_lb(lb_path = DEPLOYMENT_PATH / "label_binarizer.pkl"):
    """Loading the fitted labelbinarizer before training."""
    return joblib.load(lb_path)

def load_data_for_scoring(data_path = DEPLOYMENT_PATH / "data_for_scoring.csv"):
    """Loading the test data: part of the data for testing, not training."""
    return pd.read_csv(data_path)


if __name__=="__main__":
    training()
