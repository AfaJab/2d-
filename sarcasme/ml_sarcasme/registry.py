import pickle
from pathlib import Path
from tensorflow import keras
from keras import models
from google.cloud import storage
from sarcasme.params import BUCKET_NAME


def load_model(model_name:str
               ,model_target="local"):
    path_to_model = f"models/{model_name}.h5"
    print(model_target)
    print(path_to_model)
    if model_target=="gcs":
        client = storage.Client()
        try:
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(path_to_model)
            blob.download_to_filename(path_to_model)
            latest_model = keras.models.load_model(path_to_model)
            print("✅ Latest model downloaded from cloud storage")
            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    if model_target=="local":
        new_model = models.load_model(path_to_model)
    else :
        raise ValueError("model_target must be 'local' or 'gcs'")
    print("✅ Model loaded sucessfully from local, that's awesome !")
    return new_model


def save_model(model
               ,model_name:str
               ,model_target="local"):
    '''
    Save the model in local and additionaly in GCS if model_target == "gcs"
    params:
        model: the model to save
        model_name: the name of the model
        model_target: the target where to save the model
    '''
    path_to_model = f"models/{model_name}.h5"
    model.save(path_to_model)
    print("✅ Model saved in local ! YEAH !")
    if model_target == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(path_to_model)
        blob.upload_from_filename(path_to_model)
        print("✅ Model saved to GCS")

def save_weights(model: keras.Model,
                 model_name:str,
                 gcs:bool=False
                 ) -> None:
    model_dir= Path(f'models/from_weights/{model_name}')
    model_path = model_dir.joinpath('weights.h5')
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)
    model.save_weights(model_path)
    if gcs:
        client = storage.Client()
        blob = client.bucket(BUCKET_NAME).blob(model_path.as_posix())
        blob.upload_from_filename(model_path)
        print("✅ Model weights saved to GCS")

def load_weights(model: keras.Model,
                model_name:str
                )  -> None:
    '''
    Load the weights of a pretrained bert from a local file or download it
    from GCS if it is not found locally
    '''
    weigh_path = Path(f'models/from_weights/{model_name}/weights.h5')
    if not weigh_path.is_file():
        print(f'No model found in {weigh_path} trying to download from GCS')
        client = storage.Client()
        # Make sure the directory exists
        if not weigh_path.parent.is_dir(): weigh_path.parent.mkdir(parents=True)
        try:
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(weigh_path.as_posix())
            blob.download_to_filename(weigh_path)
        except:
            raise ValueError(f"\n❌ No model weights found in GCS bucket {BUCKET_NAME} at {weigh_path}")
    model.load_weights(weigh_path)
    return model



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
