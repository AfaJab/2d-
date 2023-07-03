import pandas as pd
from sarcasme.ml_sarcasme.preprocess import preprocess_text
from sarcasme.ml_sarcasme.registry import load_model,load_tokenizer
from sarcasme.ml_sarcasme.data import tokenize_data

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

app.state.model = load_model("baseline")
app.state.tokenizer = load_tokenizer("tokenizer")

@app.get("/predict")
def predict(sentence: str):
    model = app.state.model
    X_processed = preprocess_text(sentence)
    df= pd.DataFrame({'comment':[X_processed], 'label':[0]})
    X,y,vocab_size = tokenize_data(df
                                   ,tokenizer=app.state.tokenizer)
    y_pred = float(model.predict(X)[0][0])
    return y_pred


@app.get("/")
def root():

    return dict(greeting="Hello")
