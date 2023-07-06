from sarcasme.ml_sarcasme.model import initialize_minibert_model
from sarcasme.ml_sarcasme.registry import load_weights,save_model
import pandas as pd

model = initialize_minibert_model()
save_model(model,"test_model")
model = load_weights(model,"test_model")

test2= pd.DataFrame({'comment':["you don't say!"], 'label':[0]})
test_X2= test2['comment']
model.predict(test_X2)
