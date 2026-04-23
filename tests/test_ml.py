import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference


def test_process_data(data_processor):
    X = data_processor[0]
    y = data_processor[1]

    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert y.shape[0] > 0

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_train_model():
    X_train = np.array([[1., 3.], [5.,6.]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    y = np.random.randint(0, 2, 50)
    preds = np.random.randint(0,2, 50)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert precision >= 0

    assert isinstance(recall, float)
    assert recall >= 0

    assert isinstance(fbeta, float)
    assert fbeta >= 0

def test_inference():
    X_train = np.array([[1., 3.], [5.,6.]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)

    x = np.random.rand(2, 2)

    preds = inference(model, x)

    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)
