# A library for streamlining ML processes
# by Matthew Mauer
# last editted 2020-05-10

'''
    EDITS TO COME:
        - more exception handling!!!
        - more Grid Parameters in SupervisedLearner
        - an UnsupervervisedLearner...
'''

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, \
                                  PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge, \
                                 LinearRegression, Lasso
# from sklearn.svm import LinearSVC, SVR
# from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge



def read(path):
    ''' 
    read in and display the data
    '''
    df = pd.read_csv(path)
    rown, coln = df.shape
    print(f'There are {rown} rows and {coln} columns in the data set.')
    return df

def clean_events(df):
    '''
    Convert NaN values in event data to 0
    '''
    events = ['battle_cnt', 'protest_cnt',
       'riot_cnt', 'explosion_cnt', 'violence_on_civs_cnt',
       'battle_fatal', 'protest_fatal', 'riot_fatal', 'explosion_fatal',
       'violence_on_civs_fatal']

    df[events] = df[events].fillna(value=0)

    return df

def impute(df):

    pass





class LongitudinalLearner():
    '''

    '''
    def __init(self, models={}):
        self.models = models

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    def train_test_split(self, data, test_years=2):
        '''
        Split the data into training and test data prior to model building
        based on number of years desired in the test sample.
        '''

        threshold = data.year.max() + 1 - test_years

        mask = data.year < threshold
        self.training_data = data[mask]
        self.test_data = data[~mask]

    def inputs_outputs(self, features=[], target=''):
        '''
        Grab label names for features and target variable.
        '''
        self.features = features
        self.target = target
    
    def grid_search(self, GRID={}, num_cvs=4):
        '''
        Run a grid search with time-series nested crossvalidation over the data
        and return a df of the results.

        NEXT STEPS: Pick and save the best model.
                    Expand functionality to other model types. HMMs?
        '''

        # Begin timer 
        start = datetime.datetime.now()

        X = self.training_data[self.features]
        Y = self.training_data[self.target]

        # genereate Nested CV masks
        nestedcv_masks = [self.nestCVMask(size) for size in range(1, num_cvs+1)]

        # Initialize results data frame 
        results_df = pd.DataFrame()

        # Loop over models 
        for model_key in self.models: 
            
            # Loop over parameters 
            for params in GRID[model_key]: 
                print(f"Training model: {model_key} | {params}")
                
                # Create model 
                model = self.models[model_key]
                model.set_params(**params)
                
                # Loop over cross validations
                cv_index = 1
                for cv in nestedcv_masks:

                    Xtrain = X[cv]
                    Xval = X[~cv]
                    Ytrain = Y[cv]
                    Yval = Y[~cv]

                    model.fit(Xtrain, Ytrain)
                    
                    Ypred = model.predict(Xtest)
                    
                    row = {'model':model_key}
                    row.update(params)

                    mae, mse, r2 = self.eval(Xtrain, Ytrain, Ytest, Ypred, 
                                             model=model, output=False)
                    row.update({'CV_years': cv_index, 
                                'MAE': mae, 
                                'MSE': mse, 
                                'R^2': r2})  

                    cv_index += 1

                    results_df = results_df.append(row, ignore_index=True)
                
        # End timer
        stop = datetime.datetime.now()
        print("Time Elapsed:", stop - start)

        self.results_GS = results_df
        return self.results_GS       


    def nestCVMask(self, val_years=1):
        '''
        Return a mask for each fold of Nest time-series cross validation.
        '''
        threshold = self.training_data.year.max() + 1 - val_years
        mask = self.training_data.year < threshold

        return mask


    def eval(Xtrain, Ytrain, Ytest, Ypred, model=None, output=False, visual=False):
        '''
        Evaluate a regression model and return metrics.
        '''
        mae = mean_absolute_error(Ypred, Ytest)
        mse = mean_squared_error(Ypred, Ytest)
        r2 = model.score(Xtrain, Ytrain)
        
        if output:
            print("Mean Absolute error: %.2f" % mae)
            print("Mean squared error: %.2f" % mse)
            print("R-squared: %.2f \n" % r2)
        
        return (mae, mse, r2)

      

