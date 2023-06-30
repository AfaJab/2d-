import pandas as pd

from colorama import Fore, Style
from pathlib import Path

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sarcasme.params import *



def load_data(path = "raw_data/train-balanced-sarcasm.csv"):
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame
               ,subset = True) -> pd.DataFrame:
    '''
    Clean the data by removing duplicates and empty values
    '''
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)
    if subset : return df.sample(n=50000,random_state=42)
    return df

def tokenize_data(df:pd.DataFrame):
    '''
    Tokenize the data and return the X,y and vocab_size
    '''

    X = df[['comment']]
    y = df['label']

    X = X.squeeze()
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)

    X = pad_sequences(sequences, maxlen=150)

    vocab_size = len(tokenizer.word_index) + 1

    return X,y,vocab_size



def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path = "raw_data/train-balanced-sarcasm.csv",
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
