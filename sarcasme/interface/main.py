from ml_sarcasme.data import clean_data,tokenize_data,load_data
from ml_sarcasme.model import initialize_model,fit_model
from ml_sarcasme.registry import save_model,load_model


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

if __name__=="__main__":
    train()
