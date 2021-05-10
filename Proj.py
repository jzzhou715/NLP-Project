# https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset?select=yelp.csv

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

# =============================================================================
# Clean
# =============================================================================

data = pd.read_csv('yelp.csv')

data = data[['text', 'cool', 'useful', 'funny']]

data['reaction'] = data['cool'] + data['useful'] + data['funny']

data['reaction'] = data[['cool','useful','funny']].sum(axis = 1)

popular_threshold = np.mean(data['reaction'])

data['popular'] = (data['reaction'] > popular_threshold)

data_label = data[['text', 'cool', 'useful', 'funny', 'reaction', 'popular']]

data_X = data['text']

# =============================================================================
# Vectorize nlp
# =============================================================================

Vec = CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      lowercase = True)

Vec_Bi = CountVectorizer(analyzer = 'word',
                         stop_words='english',
                         lowercase = True,
                         binary=True)

review_vec = pd.DataFrame()
    
data_X_t = Vec.fit_transform(data_X.tolist())

features = Vec.get_feature_names()

BoW = pd.DataFrame(data_X_t.toarray(), columns = features)

BoW["Popular"] = data['popular']

BoW.sum()

BoW.sum().plot.hist()

BoW_removeTypo = BoW.loc[:, BoW.sum(axis = 0) > 1]

# =============================================================================
# Modeling
# =============================================================================

train, test = train_test_split(BoW_removeTypo, test_size = 0.2)

train_y = train["popular"]

train_X = train.drop(["popular"], axis = 1)

test_y = test["popular"]

test_X = test.drop(["popular"], axis = 1)

log = LogisticRegression()

log.fit(train_X, train_y)

prediction = log.predict(test_X)

print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))

