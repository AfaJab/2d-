import pandas as pd
from ml_sarcasme.preprocess import preprocess_text
from ml_sarcasme.registry import load_model

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

@app.get("/predict")
def predict(sentence: str):

    X_processed = preprocess_text(sentence)
    y_pred = app.state.model.predict(X_processed)

    return y_pred


@app.get("/")
def root():

    return dict(greeting="Hello")
