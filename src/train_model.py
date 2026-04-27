"""This script train RandomForestClassifier on the census data."""

import os
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.model import inference
from ml.model import compute_model_metrics
from ingest.download_data import download_census

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

DEPLOYMENT_PATH = Path(__file__).parent.parent / "deployment"

def training():
    """This function train random forest model with census.csv data to predict 'salary' column.
    
    A folder called 'deployment' will be created, with the following files
        - data_for_scoring.csv: Part of the data which was not used for training.
        - rndfmodel.pkl: trained random forest model.
        - label_binarizer.pkl: binarizer used for data prepreprocesing before training.
        - encoder.pkl: encoder used for data preprocessing before training.
    """
    logger.info("Loading census.csv data")
    DATAPATH = Path(__file__).parent.parent / "data" / "census.csv"
    try:
        df = pd.read_csv(DATAPATH)
    except FileNotFoundError:
        download_census()
        df = pd.read_csv(DATAPATH)

    logger.info("Taking list of columns with object type")
    cat_features = df.select_dtypes(include="object").columns.tolist()
    cat_features.remove("salary")
    logger.info("Cat_features: %s", cat_features)

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

def fetch_test_score(model_path: str = None, encoder_path: str = None,
                     lb_path: str = None, data: pd.DataFrame = None):
    """
    Loads trained model, loads preprocessing steps (encoder, lb),
    loads testing data.
    Make inference and compute scores.

    Args:
        model_path: path to the trained model
        encoder_path: path to the encoder
        lb_path: path to the saved label encoder
        data: the (test) data .csv to infer with

    Returns:
        Score dictionary: {"precision":precision,
                           "recall": recall,
                           "fbeta": fbeta}
    """
    if model_path is None:
        model_path = DEPLOYMENT_PATH / "rndfmodel.pkl"
    model = joblib.load(model_path)

    if encoder_path is None:
        encoder_path = DEPLOYMENT_PATH / "encoder.pkl"
    encoder = joblib.load(encoder_path)

    if lb_path is None:
        lb_path = DEPLOYMENT_PATH / "label_binarizer.pkl"
    lb = joblib.load(lb_path)

    if data is None:
        data = pd.read_csv(DEPLOYMENT_PATH / "data_for_scoring.csv")

    if data.shape[0] != 0:
        cat_features = data.select_dtypes(include="object").columns.tolist()
        cat_features.remove("salary")

        logger.info("Computing test features and targets")
        x_test, y_test, _, _ = process_data(
                                df=data, categorical_features=cat_features,
                                label="salary", training=False,
                                encoder=encoder, lb=lb)

        logger.info("Making inferemce")
        preds = inference(model=model, x=x_test)

        logger.info("Computing scores")
        precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

        f1 = 2 * (precision * recall)/(precision + recall)

        logger.info("Returning score dict")
        return {"precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "fbeta": round(fbeta, 3)}

    if data.shape[0] == 0:
        return {"precision": -1,
                "recall": -1,
                "f1_score": -1,
                "fbeta": -1}

def data_inference(features: dict, model_path: str = None, encoder_path: str = None,
                     lb_path: str = None):
    """Predict salary range for newly input data.
    
    Args:
        D: dictionary with keys given by census data features
           [age,workclass,fnlwgt,education,education-num,
            marital-status,occupation,relationship,race,
            sex,capital-gain,capital-loss,hours-per-week,
            native-country]
            and values are list of features.
        model_path: path to trained model
        encoder_path: path to trained encoder (used for preprocessing)
        lb_path: path to label binarizer (used for preprocessing)
    Returns:
        D: Dictionary with predicted salary range as 
           additional key.
    """
    df = pd.DataFrame(features)
    cat_features = df.select_dtypes(include="object").columns.tolist()

    if model_path is None:
        model_path = DEPLOYMENT_PATH / "rndfmodel.pkl"
    model = joblib.load(model_path)

    if encoder_path is None:
        encoder_path = DEPLOYMENT_PATH / "encoder.pkl"
    encoder = joblib.load(encoder_path)

    if lb_path is None:
        lb_path = DEPLOYMENT_PATH / "label_binarizer.pkl"
    lb = joblib.load(lb_path)

    x, _, _, _ = process_data(df=df, categorical_features=cat_features,
                                label=None, training=False,
                                encoder=encoder, lb=lb)

    preds = lb.inverse_transform(inference(model=model, x=x))

    return preds

if __name__=="__main__":
    training()
