import pandas as pd
import numpy as np

class Classifier():
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
        
        
        
class SVM(Classifier):
    def __init__(self, dir):
        '''
        Creates an instance of SVM.

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

        self.data['X'] = temp[X]
        self.data['y'] = temp['y']
        
        print(self.data.head)
        
class LogReg(Classifier):
    def __init__(self, dir):
        '''
        Creates an instance of SVM.

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
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
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
        self.data = temp[X + ['y']]
        
        print(self.data)
        
if __name__ == '__main__':
    yelp = SVM('yelp.csv')
    yelp.preprocess(X = ['text'], y = ['cool','useful','funny'])