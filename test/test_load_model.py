from sarcasme.ml_sarcasme.model import initialize_bert_model
import pandas as pd

model = initialize_bert_model()
model.load_weights('models/minibert_final.h5')

test2= pd.DataFrame({'comment':["you don't say!"], 'label':[0]})
test_X2= test2['comment']
model.predict(test_X2)
