from sarcasme.ml_sarcasme.data import clean_data,tokenize_data,load_data
from sarcasme.ml_sarcasme.model import initialize_model,fit_model
from sarcasme.ml_sarcasme.model import initialize_bert_model
from sarcasme.ml_sarcasme.registry import save_model
from sarcasme.ml_sarcasme.preprocess import preprocess

from sklearn.model_selection import train_test_split
import json

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
    df = clean_data(df, subset = False)
    df = df
    X = df[['comment']]
    y = df['label']
    print(df.shape)
    X_processed = X.applymap(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.01, shuffle=True)
    model = initialize_bert_model()
    history, model = fit_model(model, X_train, y_train)
    save_model(model, "mediumbert")
    res = model.evaluate(X_test,y_test)
    print(res)
    res_json = json.dumps(dict(loss=res[0], Accuracy=res[1], Precision=res[2], Recall=res[3]))
    with open("models/res.json", "w") as file:
        file.write(res_json)


if __name__=="__main__":
    train_bert_model()
