"""This scripts contains all the fastapi endpoints for deployment."""

import sys
from typing import Dict
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pandas as pd
from .utils import fetch_post_query_params
from .utils import compute_slice_score
sys.path.append("../")

QUERYPARAMSRANGE = fetch_post_query_params()
EXAMPLEQUERYKEY = {"native-country": "United-States",
                    "race": "White", 
                    "education": "Masters"}

class SliceQueryDict(BaseModel):
    """Type hints for to compute scores on data slice."""
    data: Dict[str, str] = EXAMPLEQUERYKEY

    @field_validator("data")
    @classmethod
    def validate_dict(cls, v):
        """
        Validates if a dictionary v is valid to slice test data.
        """
        for key, value in v.items():
            if key not in QUERYPARAMSRANGE:
                raise ValueError(f"Invalid key: {key}")
            if value not in QUERYPARAMSRANGE[key]:
                raise ValueError(f"Invalid value '{value}' for key '{key}'")
        return v

app = FastAPI()

@app.get("/")
def display():
    aboutme = "Welcome. This is an API to predict salary column of census.csv data"
    model_technique = "randomforest from sklearn"
    data_columns_for_slice = list(QUERYPARAMSRANGE.keys())
    unique_val = QUERYPARAMSRANGE
    target = "salary"
    usage = {"find test scores": "got to '/scores' endpoints",
             "slicing_score": "got to '/docs', "
             "and put valid dict of kye-value pairs (see possible values below first)"}
    return {"about": aboutme,
            "model": model_technique,
            "usage": usage,
            "column_names_to_slice (all possible slice key)": data_columns_for_slice,
            "slice_unique_value (all possible values for each key)": unique_val,
            "target_column": target}

TESTINGDATAPATH = Path(__file__).parent.parent.parent / "deployment" / "data_for_scoring.csv"
TESTINGDATA = pd.read_csv(TESTINGDATAPATH)

@app.get("/scores")
def test_score():
    response = compute_slice_score(test_data=TESTINGDATA,
                                   query_param = None)
    return response

@app.post("/slice")
def slice_score(slice_method: SliceQueryDict):
    query_dict = slice_method.data

    response = compute_slice_score(test_data=TESTINGDATA,
                                   query_param = query_dict)
    return response
