from tensorflow import keras
from keras import models

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
