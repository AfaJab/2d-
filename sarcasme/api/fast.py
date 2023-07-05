import pandas as pd
from sarcasme.ml_sarcasme.preprocess import preprocess
# from sarcasme.ml_sarcasme.registry import load_model,load_tokenizer
# from sarcasme.ml_sarcasme.data import tokenize_data
from sarcasme.ml_sarcasme.model import initialize_bert_model

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

app.state.model = initialize_bert_model()
app.state.model.load_weights('models/minibert_final.h5')
# app.state.tokenizer = load_tokenizer("tokenizer")

@app.get("/predict")
def predict(sentence: str):
    model = app.state.model
    X_processed = preprocess(sentence)
    df= pd.DataFrame({'comment':[X_processed], 'label':[0]})
    df= df['comment']
    # X,y,vocab_size = tokenize_data(df
                                #    ,tokenizer=app.state.tokenizer)
    y_pred = float(model.predict(df)[0][0])
    return y_pred


@app.get("/")
def root():

    return dict(greeting="Hello")
