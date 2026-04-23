"""This scripts contains all the fastapi endpoints for deployment."""

import sys
sys.path.append("../")
import pandas as pd
from fastapi import FastAPI, Request
from train_model import load_encoder
from train_model import load_lb
from train_model import load_data_for_scoring
from train_model import load_model
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.encoder = load_encoder()
    app.state.lb = load_lb()
    app.state.data_columns = load_data_for_scoring().columns.tolist()
    app.state.target = "salary"

    print("Hello, census api is starting!")
    yield
    print("Census api is closing, bye!")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def display(request: Request):
    model = request.app.state.model
    encoder = request.app.state.encoder
    lb = request.app.state.lb
    data_columns = request.app.state.data_columns
    target = request.app.state.target
    return {"app_type": "Logistic regression (classification model)",
            "model_features": [str(type(model)), str(type(encoder)), str(type(lb))],
            "data_columns": data_columns,
            "target": target}

# @app.get("/scores")
# def score(request: Request, data: pd.DataFrame):
#     pass

# @app.post("/slice_scores/{column}")
# def slicing(request: Rquest, data: pd.DataFrame, column: str, column_value: str):
#     pass
