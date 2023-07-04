from google.cloud import storage
from sarcasme.ml_sarcasme.registry import load_weights
from sarcasme.ml_sarcasme.model import initialize_bert_model
from sarcasme.ml_sarcasme.preprocess import preprocess
import pandas as pd

model = initialize_bert_model()
model = load_weights(model, "minibert")
df = pd.read_csv("raw_data/train-balanced-sarcasm.csv").iloc[:1000]
X = df[['comment']]
X_processed = X.applymap(preprocess)
y = df['label']

print(model.evaluate(X,y))
