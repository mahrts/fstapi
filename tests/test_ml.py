import numpy as np

def test_data(data_processor):
    X = data_processor[0]
    y = data_processor[1]

    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert y.shape[0] > 0

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
