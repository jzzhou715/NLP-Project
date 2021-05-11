import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

class TextDataClassifier():
    def __init__(self, dir):
        '''
        Creates an instance of Classifer.

        Parameters
        ----------
        dir : str
            directory of the file.

        Returns
        -------
        None.

        '''
        self.file = pd.read_csv(dir)
    
    def preprocess(self, X, y):
        '''
        Preprocess the data to prepare for model training.

        Parameters
        ----------
        X : str
            Column names that consists of predictors.
        y : str
            Column names that consists of response variable.

        Returns
        -------
        None.

        '''
        self.data = self.file[X + y]
        
    def BoW(self, figname, ana = 'word', sw = 'english', lc = True, bi = False, typo_threshold = 1):
        '''
        Process the predictors to prepare for model training

        Parameters
        ----------
        ana : str, optional
            CountVectorizer parameter analyzer. The default is 'word'.
        sw : str, optional
            Count Vectorizer parameter stop_wrods. The default is 'english'.
        lc : str, optional
            Count Vectorizer parameter lowercase. The default is True.
        bi : TYPE, optional
            Count Vectorizer parameter binary. The default is False.
        typo_threshold : int, optional
            Set the threshold to remove typo. The default is 1.

        Returns
        -------
        None.

        '''
        Vec = CountVectorizer(analyzer = ana,
                              stop_words = sw,
                              lowercase = lc,
                              binary = bi)
        
        data_X_t = Vec.fit_transform(self.data.X.tolist())
        
        features = Vec.get_feature_names()
        
        BoW = pd.DataFrame(data_X_t.toarray(), columns = features)
        
        BoW.sum()
        
        fig = BoW.sum().plot.hist()
        fig.figure.savefig(figname, dpi = 600)
        fig.clear()
        
        BoW_removeTypo = BoW.loc[:, BoW.sum(axis = 0) > typo_threshold]
        
        BoW_removeTypo['y'] = self.data.y
        
        self.BoW_noTypo = BoW_removeTypo
        
        print(self.BoW_noTypo.head)
        
class MultiClassifier(TextDataClassifier):
    def __init__(self, dir):
        '''
        Creates an instance of MultiClassifier.

        Parameters
        ----------
        dir : str
            directory of the file.

        Returns
        -------
        None.

        '''
        super().__init__(dir)
        
    def preprocess(self, X, y):
        '''
        Preprocess the data to prepare for model training.

        Parameters
        ----------
        X : str
            Column names that consists of predictors.
        y : str
            Column names that consists of response variable.

        Returns
        -------
        None.

        '''
        self.data = pd.DataFrame()
        
        temp = self.file
        temp = self.file[X + y]
        temp['max'] = temp[y].apply(lambda x: x == x.max(), axis = 1).values.tolist()
        temp['y'] = temp['max'].apply(lambda x: y[x.index(True)]if x.count(True) == 1 else np.nan)
        temp = temp[X + ['y']]
        temp.dropna(subset = ['y'], axis = 0, inplace = True)
        temp.index = np.arange(len(temp))

        self.data['y'] = temp['y']
        self.data['X'] = temp[X]
        
        print(self.data.head)
        
    def NB(self, test_percent = 0.2):
                
        train, test = train_test_split(self.BoW_noTypo, test_size = test_percent)

        train_y = train["y"]
        
        train_X = train.drop(["y"], axis = 1)
        
        test_y = test["y"]
        
        test_X = test.drop(["y"], axis = 1)
        
        nb = MultinomialNB()
        
        nb.fit(train_X, train_y)
        
        prediction = nb.predict(test_X)
        
        self.logReport =  metrics.classification_report(test_y, prediction)
        self.confMat = metrics.confusion_matrix(test_y, prediction)
        
        print(self.logReport)
        print(self.confMat)
        
class BiClassifier(TextDataClassifier):
    def __init__(self, dir):
        '''
        Creates an instance of BiClassifier.

        Parameters
        ----------
        dir : str
            directory of the file.

        Returns
        -------
        None.

        '''
        super().__init__(dir)
        
    def preprocess(self, X, y, threshold):
        '''
        Preprocess the data to prepare for model training.

        Parameters
        ----------
        X : str
            Column names that consists of predictors.
        y : str
            Column names that consists of response variable.
        threshold : TYPE
            A threshold to turn reponse into a binary variable.

        Returns
        -------
        None.

        '''
        
        self.data = pd.DataFrame()
        
        temp = self.file[X + y]
        temp['y_sum'] = temp[y].sum(axis = 1)
        if isinstance(threshold, (int, float)):
            th = threshold
        elif isinstance(threshold, str):
            if(threshold == 'mean'):
                th = np.mean(temp['y_sum'])
            elif(threshold == 'median'):
                th = np.median(temp['y_sum'])
            else:
                raise TypeError('Undefined threshold processing type.')
        else:
            raise TypeError('Unsupported input parameter type.')
        temp['y'] = (temp['y_sum'] > th)
        self.data['y'] = temp['y']
        self.data['X'] = temp[X]

        
        print(self.data)
        
    def LogisticReg(self, test_percent = 0.2):
        '''
        Training and testing logistic regression model.

        Parameters
        ----------
        test_percent : float, optional
            Percentage split into testing set. The default is 0.2.

        Returns
        -------
        None.

        '''
        
        train, test = train_test_split(self.BoW_noTypo, test_size = test_percent)
        train_y = train["y"]
        
        train_X = train.drop(["y"], axis = 1)
        
        test_y = test["y"]
        
        test_X = test.drop(["y"], axis = 1)
        
        log = LogisticRegression()
        
        log.fit(train_X, train_y)
        
        prediction = log.predict(test_X)
        
        self.logReport =  metrics.classification_report(test_y, prediction)
        self.confMat = metrics.confusion_matrix(test_y, prediction)
        
        print(self.logReport)
        print(self.confMat)
        
if __name__ == '__main__':
    yelp_lr = BiClassifier('yelp.csv')
    yelp_lr.preprocess(X = ['text'], y = ['cool','useful','funny'], threshold = 'mean')
    yelp_lr.BoW(figname = 'lr_hist.png')
    yelp_lr.LogisticReg()
    
    # yelp_mc = MultiClassifier('yelp.csv')
    # yelp_mc.preprocess(X = ['text'], y = ['cool','useful','funny'])
    # yelp_mc.BoW(figname = 'mc_hist.png', typo_threshold = 1)
    # yelp_mc.NB()
    