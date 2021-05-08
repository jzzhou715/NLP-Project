# https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset?select=yelp.csv

import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# =============================================================================
# Clean
# =============================================================================

data = pd.read_csv('yelp.csv')

data = data[['text', 'cool', 'useful', 'funny']]

data['reaction'] = data['cool'] + data['useful'] + data['funny']

popular_threshold = np.mean(data['reaction'])

data['popular'] = (data['reaction'] > popular_threshold)

data_label = data[['text', 'cool', 'useful', 'funny', 'reaction', 'popular']]

data_X = data['text']

# =============================================================================
# Vectorize nlp
# =============================================================================

Vec = CountVectorizer(input='filename',
                      analyzer = 'word',
                      stop_words='english',
                      lowercase = True)

Vec_Bi = CountVectorizer(input='filename',
                         analyzer = 'word',
                         stop_words='english',
                         lowercase = True,
                         binary=True)

review_vec = pd.DataFrame()

path = '/Users/Zhou/Box/GitHub Repository/NLP-Project/yelp.csv'

