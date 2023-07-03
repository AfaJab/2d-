from tensorflow import keras
from keras import models
import pickle

def load_model(model_name:str):
    path_to_model = f"models/{model_name}.h5"
    new_model = models.load_model(path_to_model)
    print("✅ Model loaded sucessfully from local, that's awesome !")
    return new_model


def save_model(model
               ,model_name:str):
    path_to_model = f"models/{model_name}.h5"
    model.save(path_to_model)
    print("✅ Model saved in local ! YEAH !")

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
