from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from typing import Union, List
from pydantic import BaseModel, Field
from loguru import logger

import pandas as pd
import numpy as np
import skops.io as sio
import uuid
import uvicorn
import sys
import yaml
import os

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10
)

logger.add(
    "log.log", rotation="1 MB", level='DEBUG', compression="zip"
)
CONFIG_PATH = "./config_prod.yml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

MODEL_DIR = config['MODEL_DIR']
VERSION = config['VERSION']

trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.OneHotEncoder",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype",
]

model = sio.load(MODEL_DIR, trusted=trusted_types)


app = FastAPI(
    title='Sample API for ML Model Serving',
    version=VERSION,
    description="Based on ML with FastAPI Serving"
)

class PredictionInput(BaseModel):
    Age: int
    Sex: object
    BP: object
    Cholesterol: object
    Na_to_K: float

class ResponseModel(BaseModel):
    prediction_Id: object
    predict: object

@app.post("/predict", response_model = ResponseModel, status_code=status.HTTP_200_OK)
async def predict(input: PredictionInput):
    result = {
        'prediction_Id': str(uuid.uuid4()),
        'predict': ""
    }
    logger.info(input.dict())
    input_df = pd.DataFrame([input.dict()]) #반드시 리스트로 감싸기
    prediction = model.predict(input_df)
    logger.info(prediction)

    result['predict'] = str(prediction)

    return result

@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

@app.get("/health")
async def service_health():
    return {"ok."}