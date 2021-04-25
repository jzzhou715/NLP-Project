# https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset?select=yelp.csv

import pandas as pd
import matplotlib as plt
import numpy as np

data = pd.read_csv('yelp.csv')

data = data[['text', 'cool', 'useful', 'funny']]

data['reaction'] = data['cool'] + data['useful'] + data['funny']

popular_threshold = np.mean(data['reaction'])

data['popular'] = (data['reaction'] > popular_threshold)

data_label = data[['text', 'cool', 'useful', 'funny', 'reaction', 'popular']]

data_X = data['text']

