import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.covariance import log_likelihood
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from eval import Evaluation
import numpy as np
import math
import pdb
import chardet


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    # df = df.rename({'type': 'raw-label'}, axis=1)
    df = df.rename({'article': 'text'}, axis=1)
    df['label'] = df['type'].map({'center': 0, 'right': 1, 'left': 2})
    print(f'text number of files: {len(df)}')
    return df[['label', 'text']]


class NaiveBayes():
    """Naive Bayes classifier."""

    def __init__(self):
        super().__init__()
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
        self.vocabulary = None

    def preprocess(self, data):
        print(f'start fitting...')
        corpus = data['text'].tolist()
        self.vectorizer = CountVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=3, max_df=0.8)
        self.feature = self.vectorizer.fit_transform(corpus)
        self.y = data['label']
        print(f'Finished fiting vectorizer')

    def fit(self):
        # TODO: store ngram counts for each category in self.ngram_count
        self.clf = MultinomialNB()
        self.clf.fit(self.feature, self.y)

    def predict(self, test):
        print('start predicting...')
        corpus = test['text'].tolist()
        X = self.vectorizer.transform(corpus)
        self.pred = self.clf.predict(X)
        self.true = test['label']
        return self.pred

    def eval(self):
        eval = Evaluation(self.pred, self.true)
        eval.all_metrics()


if __name__ == "__main__":
    path = 'data/MBIC/labeled_dataset.csv'
    data = load_data(path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    print(f'train data: {len(train)}; test data: {len(test)} ')

    nb = NaiveBayes()
    nb.preprocess(train)
    nb.fit()
    _ = nb.predict(test)
    nb.eval()
