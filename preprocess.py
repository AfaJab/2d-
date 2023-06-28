import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
import string
from tqdm import tqdm

def preprocess(phrase):
    # negations
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    # general
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"(?i:AFAIK)", " As far as I know ", phrase)
    phrase = re.sub(r"(?i:AMAA)", " Ask me almost anything ", phrase)
    phrase = re.sub(r"(?i:CCW)", " Comments and criticism welcome ", phrase)
    phrase = re.sub(r"(?i:CMV)", " Change my view ", phrase)
    phrase = re.sub(r"(?i:DAE)", " Does anybody else ", phrase)
    phrase = re.sub(r"(?i:FTFY)", " fixed that for you ", phrase)
    phrase = re.sub(r"(?i:FWP)", " First world problems ", phrase)
    phrase = re.sub(r"(?i:IANAD)", " I am not a doctor ", phrase)
    phrase = re.sub(r"(?i:IANAL)", "  I am not a lawyerI am not a lawyer ", phrase)
    phrase = re.sub(r"(?i:IIRC)", "  If I recall correctly ", phrase)
    phrase = re.sub(r"(?i: IMO )", " in my opinion ", phrase)
    phrase = re.sub(r"(?i:IMHO)", " in my honest opinion ", phrase)
    phrase = re.sub(r"(?i: IRL )", " in real life ", phrase)
    phrase = re.sub(r"(?i: ITT )", " In this thread ", phrase)
    phrase = re.sub(r"(?i:MRW)", " My reaction when ", phrase)
    phrase = re.sub(r"(?i:MFW)", " my face when ", phrase)
    phrase = re.sub(r"(?i:NSFL)", " Not safe for life ", phrase)
    phrase = re.sub(r"(?i:NSFW)", " Not safe for work ", phrase)
    phrase = re.sub(r"(?i: OC )", " original content ", phrase)
    phrase = re.sub(r"(?i:OOTL)", " out of the loop ", phrase)
    phrase = re.sub(r"(?i: OP )", " original poster ", phrase)
    phrase = re.sub(r"(?i:PM)", " private message ", phrase)
    phrase = re.sub(r"(?i:TIFU)", " today I fucked up ", phrase)
    phrase = re.sub(r"(?i: TIL )", " today I learned ", phrase)
    phrase = re.sub(r"(?i:TL;DR)", " too long, didn’t read ", phrase)
    phrase = re.sub(r"(?i:SMH)", " shake my head ", phrase)
    phrase = re.sub(r"(?i: TIL)", " today I learned ", phrase)


    return phrase


def preprocess_text(text_data):
    '''Function performs decontraction and punctuation removal'''
    preprocessed_text = []
    
    
    for sentence in tqdm(text_data):
        sent = preprocess(sentence)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        
        exclude = set(string.punctuation)
        sent = ''.join(ch for ch in sent if ch not in exclude)
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        
       
        preprocessed_text.append(sent.lower().strip())
        
    return preprocessed_text



