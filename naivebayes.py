import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.covariance import log_likelihood
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import pdb


def load_data(path):
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    path = 'data/right.csv'
    data = load_data(path)
    print(data)
