######################################################################
#
#                       Modeling Functions
#
######################################################################
# Matthew Muller
# 11/24/2022
#
# This file contains functions for modeling and evaluating models.

# Library Imports
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.base import clone
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.stats import kruskal, zscore

from MattTools.stats import Bootstrap

# Function to train a dictionary of models and return the models
def train_models(models, X, y, random_state=100):
    '''
    Summary: Function to train a dictionary of models and return the models

    models (dict) : dictionary of models to test
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    random_state (int) : random state to set

    output (dict) : dictionary of trained models
    '''
    # Iterate through the models
    for model_name, model in models.items():
        # Create a clone of the model
        model_clone = clone(model)
        # Set the random state
        model_clone.random_state = random_state
        # Fit the model
        model_clone.fit(X, y)
        # Replace the model with the fitted model
        models[model_name] = model_clone
    # Return the models
    return models

# Function to cross validate various models and return the metrics as a dataframe
def cross_val_models(models, X, y, cv_folds=5, scoring='roc_auc', random_state=100, from_pretrained=False):
    '''
    Summary: Function to test various models and return the metrics as a dataframe

    models (dict) : dictionary of models to test
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    cv_folds (int) : number of cross validation folds
    scoring (str) : scoring metric to use
    random_state (int) : random state to set

    output (pd.DataFrame) : dataframe of model metrics
    '''
    # Create empty dataframe to store results
    results = pd.DataFrame(columns=['model', 'mean', 'std', 'min', 'max'])
    # Iterate through models
    for model_name, model in models.items():
        # Create a clone of the model if it is not pretrained
        if not from_pretrained: 
            model = clone(model)
        # Set the random state
        model.random_state = random_state
        # Create a cross validation object
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # Get the cross validation scores
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        # Concat the results to the dataframe
        results = pd.concat([results, pd.DataFrame({'model': model_name,
                                                    'mean': scores.mean(),
                                                    'std': scores.std(),
                                                    'min': scores.min(),
                                                    'max': scores.max()}, index=[0])])
    # Return the results
    return results


# Function to test various models and return the metrics as a dataframe
def test_models(models, X, y, random_state=100):
    '''
    Summary: Function to test various models and return the accuracy of classifiers or 
    R-squared of regressors as a dataframe

    models (dict) : dictionary of models to test
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    random_state (int) : random state to set

    output (pd.DataFrame) : dataframe of model metrics
    '''
    # Create empty dataframe to store results
    results = pd.DataFrame(columns=['model', 'score'])
    # Iterate through models
    for model_name, model in models.items():
        scores = []
        # Set the random state
        model.random_state = random_state
        # Fit the model
        model.fit(X, y)
        # Get the score
        score = model.score(X, y)
        # Append the score to the list
        scores.append(score)
    # Concat the results to the dataframe
    results = pd.concat([results, pd.DataFrame({'model': model_name,
                                                'mean': scores.mean(),
                                                'std': scores.std(),
                                                'min': scores.min(),
                                                'max': scores.max()}, index=[0])])
    # Return the results
    return results