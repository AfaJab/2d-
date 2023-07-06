from sarcasme.ml_sarcasme.model import initialize_minibert_model
from sarcasme.ml_sarcasme.registry import load_weights
import pandas as pd
from pathlib import Path

path_to_local_model = Path("models/minibert_final.h5")
if path_to_local_model.is_file():
    path_to_local_model.unlink()

model = initialize_minibert_model()
model = load_weights(model,"minibert_final")
test2= pd.DataFrame({'comment':["you don't say!"], 'label':[0]})
test_X2= test2['comment']
model.predict(test_X2)
