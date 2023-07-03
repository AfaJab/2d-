import time
from sarcasme.ml_sarcasme.data import clean_data
from sklearn.model_selection import train_test_split
from colorama import Fore, Style

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, GRU, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
# from keras.preprocessing.sequence import pad_sequences

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")




def initialize_model(vocab_size) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    embedding_dim = 100
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=150))
    model.add(GRU(80))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Precision(), Recall()])

    model.summary()

    return model


def fit_model(model
              ,X_train
              ,y_train
              ,patience=3
              ,verbose=1):
    '''
    Fit the model with the training data
    params:
    - model : the model to fit
    - X_train : the training data
    - y_train : the training labels
    - patience : the number of epochs to wait before early stopping
    - verbose : the verbosity level
    '''
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience,restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        batch_size = 32,epochs=1,
                        validation_split=0.2,
                        callbacks=[early_stopping]
                        ,verbose=verbose)
    return history,model