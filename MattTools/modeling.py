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
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.base import clone
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.stats import kruskal, zscore


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

# Function to test various models and return the metrics as a dataframe
def test_models(models, X, y, cv_folds=5, scoring='roc_auc', random_state=100, from_pretrained=False):
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
        # Fit the model only if it is not pretrained
        if not from_pretrained:
            model.fit(X, y)
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



# Function to plot the results of the test_models function with a barplot mean and error bars
def plot_model_results(results, figsize=(12, 8)):
    '''
    Summary: Function to plot the results of the test_models function with a barplot mean and error bars

    results (pd.DataFrame) : dataframe of model metrics
    figsize (tuple) : size of the plot

    output (None) : None
    '''
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the results
    sns.barplot(x='model', y='mean', data=results, yerr=results['std'], ax=ax)
    # Set the title
    ax.set_title('Model Results')
    # Set the x and y labels
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Score')
    # Show the plot
    plt.show()

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
        # Get the predicted probabilities
        y_pred = model.predict_proba(X)[:, 1]
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

# Function to plot the ROC curves of each model
def plot_prc_curves(models, X, y, figsize=(10, 10)):
    '''
    Summary: Function to plot the PRC curves of each model

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
        # Get the predicted probabilities
        y_pred = model.predict_proba(X)[:, 1]
        # Get the PRC curve
        precision, recall, _ = precision_recall_curve(y, y_pred)
        # Get the AUC
        roc_auc = auc(precision, recall)
        # Plot the ROC curve
        ax.plot(precision, recall, label=f'{model_name} (AUC = {roc_auc:0.2f})')
    # Set the title
    ax.set_title('PRC Curves')
    # Set the x and y labels
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    # Set the legend
    ax.legend()
    # Show the plot
    plt.show()


# Function to plot the confusion matrix of a dictionary of models in a grid square
def plot_confusion_matrices(models, X, y, figsize=(10, 10)):
    '''
    Summary: Function to plot the confusion matrix of a dictionary of models in a grid

    models (dict) : dictionary of models to test
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    figsize (tuple) : size of the plot

    output (None) : None
    '''
    # Calculate the number of rows and columns based on the number of models
    num_models = len(models)
    nrows = int(num_models ** 0.5)
    ncols = int(np.ceil(num_models / nrows))
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # Iterate through the models and subplots
    for i, (model_name, model) in enumerate(models.items()):
        # Calculate the row and column index of the subplot
        row_idx = i // ncols
        col_idx = i % ncols
        # Get the predicted values
        y_pred = model.predict(X)
        # Get the confusion matrix
        cm = confusion_matrix(y, y_pred)
        # Plot the confusion matrix on the appropriate subplot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=axes[row_idx, col_idx])
        # Set the title
        axes[row_idx, col_idx].set_title(model_name)
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # Show the plot
    plt.show()

