import pickle
from pathlib import Path
from tensorflow import keras
from keras import models
from google.cloud import storage
from sarcasme.params import BUCKET_NAME

def load_weights(model,
                model_name:str
                ):
    path_to_model = f"models/{model_name}.h5"
    if not Path(path_to_model).is_file():
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(path_to_model)
        blob.download_to_filename(path_to_model)
        print("✅  Model Downloaded from GCS")
    model.load_weights(path_to_model)
    print("✅ Model loaded sucessfully from local, that's awesome !")
    return model


def save_model(model
               ,model_name:str
               ,model_target="gcs"):
    path_to_model = f"models/{model_name}.h5"
    model.save(path_to_model)
    print("✅ Model saved in local ! YEAH !")
    if model_target == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(path_to_model)
        blob.upload_from_filename(path_to_model)
        print("✅ Model saved to GCS")

def load_tokenizer(tokenizer_name="tokenizer"):
    with open(f'models/{tokenizer_name}.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer loaded sucessfully from local, that's even better !")
    return tokenizer

def save_tokenizer(tokenizer
                   ,tokenizer_name="tokenizer"):
    with open(f'models/{tokenizer_name}.pickle', 'wb') as handle :
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer
