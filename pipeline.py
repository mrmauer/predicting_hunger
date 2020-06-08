'''
A library for streamlining ML processes
by Matthew Mauer
last editted 2020-05-10


    EDITS TO COME:
        - FIX TIME-SERIES NESTED CROSS validation
        - best_model functionality for learner grid search
        - more exception handling!!!
        - Unsupervised ML
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
    df = df.drop(columns=['Unnamed: 2', 'ADMIN0.1', 'ADMIN1.1'])
    rown, coln = df.shape
    print(f'There are {rown} rows and {coln} columns in the data set.')
    return df

def clean_events(df):
    '''
    Convert NaN values in event data to 0.
    Convert all event data to log scale.
    '''
    events = ['battle_cnt', 'protest_cnt',
       'riot_cnt', 'explosion_cnt', 'violence_on_civs_cnt',
       'battle_fatal', 'protest_fatal', 'riot_fatal', 'explosion_fatal',
       'violence_on_civs_fatal']

    df[events] = df[events].fillna(value=0)

    df[events] = df[events].apply(lambda x: np.log(x + 1))

    return df

def five_way_split(data, features=[], target='', year=2018, grouping=''):
    '''
    Drop rows that do not contain the target vairable.
    Split the data on the input year (input year included in test data). 
    Return test/train features, targets, and groups used for Group CV.
    '''
    data = data.dropna(axis=0, subset=[target])
    mask = data.year < 2018
    train = data[mask]
    test = data[~mask]

    groups = train[grouping]
    Xtrain = train[features]
    Xtest = test[features]
    Ytrain = train[target]
    Ytest = test[target]

    return Xtrain, Xtest, Ytrain, Ytest, groups

def _convert_repeat_regions(row, repeat_regions=[]):
    '''
    Helper function to be applied in unique_regions.
    '''
    if row['ADMIN1'] in repeat_regions:
        row['ADMIN1'] = '%s (%s)' % (row['ADMIN1'], row['ADMIN0'])
    return row

def unique_regions(data):
    '''
    Identify all region (ADMIN1) names that are repeated across countries.
    Update these region name to include the country name.
    e.g. 'Northern' -> 'Northern (Kenya)'
    '''
    max_occurrences = len(data.year.unique()) * 12

    region_counts = data.ADMIN1.value_counts()
    repeat_regions = list(region_counts[region_counts > max_occurrences].index)

    data = data.apply(lambda row: _convert_repeat_regions(row,repeat_regions),
                         axis=1)
    return data


def model_eval(model, Xtest, Ytest):
    # r2 = model.score
    Ypred = model.predict(Xtest)
    mse = mean_squared_error(Ytest, Ypred)
    mae = mean_absolute_error(Ytest, Ypred)
    target_std = Ytest.describe()['std']
    print(f'''
        MSE: {mse}
        MAE: {mae}
        For a target variable with Variance: {target_std**2}
        ''')

def feature_importance(model, labels=[], type='linear'):
    '''
    Accept a model a list of labels for the features that correspond to the 
    column order in the feature matrix used for training.
    Return a bar plot of either feature importance (for a tree-based model),
    or coefficients for a linear model.
    '''
    if type=='tree':
        importances = model.feature_importances_
    elif type=='linear':
        importances = model.coef_

    # Sort in descending order
    indices = np.argsort(importances)[::-1]

    # Sort the labels in a corresponding fashion
    names = [labels[i] for i in indices]

    plt.figure(figsize=[10,6])
    plt.title('Feature Importance/Weights')
    plt.bar(range(len(labels)), importances[indices])
    plt.xticks(range(len(labels)), names, rotation=90)
    plt.show()

class LongitudinalLearner():
    '''
    A supervised learning object designed to handle longitudinal data.
    Performs a grid search performing time-series nest cross validation.
    ERROR: (pandas) copying on a slice... May alter results of grid search...
    '''
    def __init__(self, models={}):
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

    def inputs_outputs(self, features=[], target='', dirty_features=[]):
        '''
        Grab label names for features and target variable.
        '''
        self.features = features
        self.target = target
        self.dirty_features = dirty_features
    
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

                    self.impute(Xtrain, Xval)

                    model.fit(Xtrain, Ytrain)
                    
                    Ypred = model.predict(Xval)
                    
                    row = {'model':model_key}
                    row.update(params)

                    mae, mse, r2 = self.eval(Xtrain, Ytrain, Yval, Ypred, 
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


    def eval(self, Xtrain, Ytrain, Ytest, Ypred, model=None, output=False, visual=False):
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

    def impute(self, train, test):
        '''
        Fill NaN values with the training sets column median for all 
        or select continuous variables
        '''
        fill = [train[col].median() for col in self.dirty_features]

        for i, col in enumerate(self.dirty_features):
            train[col].fillna(value=fill[i], inplace=True)
            test[col].fillna(value=fill[i], inplace=True)

        return 

      

