import os
from pathlib import Path
from sarcasme.ml_sarcasme.data import clean_data,tokenize_data,load_data,load_tokenizer
from sarcasme.ml_sarcasme.model import initialize_model,fit_model
from sarcasme.ml_sarcasme.registry import save_model
from sklearn.model_selection import train_test_split


def train():
    '''
    Train the model
    '''
    df = load_data()
    df = clean_data(df)
    df_train, df_test = train_test_split(df, test_size=0.01)
    X_train,y_train,vocab_size_train = tokenize_data(df_train)
    # Load the saved tokenizer
    X_test,y_test,vocab_size_test = tokenize_data(df_test,load_tokenizer())
    print(vocab_size_test,vocab_size_train)
    model = initialize_model(vocab_size_train)
    history, model = fit_model(model, X_train, y_train)
    print(model.evaluate(X_test, y_test))
    save_model(model,"baseline")




if __name__=="__main__":
    train()
