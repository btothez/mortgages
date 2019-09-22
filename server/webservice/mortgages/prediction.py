import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class Predictor:
    class __Predictor:
        def __init__(self):
            print('Loading encoder...')
            self.encoder = pickle.load(open('./pickles/encoder.pkl', 'rb'))
            print('Loading vectorizer...')
            self.vectorizer = pickle.load(open('./pickles/vectorizer.pkl', 'rb'))
            print('Loading lsa...')
            self.lsa = pickle.load(open('./pickles/lsa.pkl', 'rb'))
            print('Loading knn...')
            self.knn_lsa = pickle.load(open('./pickles/knn_lsa.pkl', 'rb'))

    def __getattr__(self, name):
        return getattr(self.instance, name)

    instance = None

    def __init__(self):
        if not Predictor.instance:
            print('there was not an instance')
            Predictor.instance = Predictor.__Predictor()
        else:
            print('there was an instance')

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def predict(self, words):
        predict_x = pd.Series([words])
        predict_x_tfidf = self.instance.vectorizer.transform(predict_x)
        predict_x_lsa = self.instance.lsa.transform(predict_x_tfidf)

        p = self.instance.knn_lsa.predict(predict_x_lsa)
        return (
            list(self.instance.encoder.inverse_transform(p)),
            list(self.instance.knn_lsa.predict_proba(predict_x_lsa)[0][p])
        )

