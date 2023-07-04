import pickle
from sarcasme.ml_sarcasme.data import clean_data,tokenize_data,load_data
from sarcasme.ml_sarcasme.model import initialize_model,fit_model
from sarcasme.ml_sarcasme.model import initialize_bert_model
from sarcasme.ml_sarcasme.registry import save_model,save_weights
from sarcasme.ml_sarcasme.preprocess import preprocess

from sklearn.model_selection import train_test_split

def train():
    '''
    Train the model
    '''
    df = load_data()
    df = clean_data(df)
    X,y,vocab_size = tokenize_data(df)
    model = initialize_model(vocab_size)
    history, model = fit_model(model, X, y)
    save_model(model,"baseline")

def train_bert_model():
    df = load_data()
    df = clean_data(df,subset=False)
    X = df[['comment']]
    y = df['label']
    X_processed = X.applymap(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.01, shuffle=True)
    model = initialize_bert_model()
    history, model = fit_model(model, X_train, y_train)
    res = model.evaluate(X_test,y_test)
    save_weights(model, "minibert", gcs=True)

if __name__=="__main__":
    train_bert_model()
