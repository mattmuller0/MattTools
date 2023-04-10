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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.base import clone
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.stats import kruskal, zscore


# Function to test various models and return the metrics as a dataframe
def test_models(models, X, y, cv_folds=5, scoring='roc_auc', random_state=100):
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
        # Create a clone of the model
        model_clone = clone(model)
        # Set the random state
        model_clone.random_state = random_state
        # Create a cross validation object
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # Fit the model
        model_clone.fit(X, y)
        # Get the cross validation scores
        scores = cross_val_score(model_clone, X, y, cv=cv, scoring=scoring)
        # Concat the results to the dataframe
        results = pd.concat([results, pd.DataFrame({'model': model_name,
                                                    'mean': scores.mean(),
                                                    'std': scores.std(),
                                                    'min': scores.min(),
                                                    'max': scores.max()}, index=[0])])
    # Return the results
    return results



# Function to plot the results of the test_models function with a boxplot
def plot_model_results(results, figsize=(10, 10)):
    '''
    Summary: Function to plot the results of the test_models function with a boxplot

    results (pd.DataFrame) : dataframe of model metrics
    figsize (tuple) : size of the plot

    output (None) : None
    '''
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the results
    sns.boxplot(x='model', y='mean', data=results, ax=ax)
    # Set the title
    ax.set_title('Model Comparison')
    # Set the x label
    ax.set_xlabel('Model')
    # Set the y label
    ax.set_ylabel('ROC AUC')
    # Return the figure
    return fig

# Function to plot the ROC curves of each model
def plot_roc_curves(models, X, y, figsize=(10, 10)):
    '''
    Summary: Function to plot the ROC curves of each model

    models (dict) : dictionary of models to test
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    figsize (tuple) : size of the plot

    output (None) : None
    '''
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    # Iterate through the models
    for model_name, model in models.items():
        # Create a clone of the model
        model_clone = clone(model)
        # Fit the model
        model_clone.fit(X, y)
        # Get the predicted probabilities
        y_pred = model_clone.predict_proba(X)[:, 1]
        # Get the ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        # Get the AUC
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:0.2f})')
    # Set the title
    ax.set_title('ROC Curves')
    # Set the x and y labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # Set the legend
    ax.legend()
    # Show the plot
    plt.show()