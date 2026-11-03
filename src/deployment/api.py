
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

import numpy as np

from src.deployment.predict import MOSAICPredictor


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="MOSAIC Prediction API",
    description=(
        "Short-horizon Limit Order Book direction "
        "classification using MOSAIC."
    ),
    version="1.0.0",
)


# Load model once when the API starts.
predictor = MOSAICPredictor()


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    sequence: List[List[float]] = Field(
        ...,
        description=(
            "A sequence of 100 observations, "
            "each containing 144 features."
        ),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "project": "MOSAIC",
        "status": "running",
        "model": "HistGradientBoosting",
        "sequence_length": 100,
        "input_features": 144,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_market_direction(request: PredictionRequest):

    try:

        sequence = np.asarray(
            request.sequence,
            dtype=np.float32,
        )

        result = predictor.predict(sequence)

        return {
            "project": "MOSAIC",
            "prediction": result["prediction"],
            "class_id": result["class_id"],
            "probabilities": result["probabilities"],
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error
