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
        # Get the PR curve
        precision, recall, _ = precision_recall_curve(y, y_pred)
        # Get the average precision
        avg_precision = average_precision_score(y, y_pred)
        # Plot the PR curve
        ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:0.2f})')
    # Set the title
    ax.set_title('PRC Curves')
    # Set the x and y labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
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


# Function to plot principal components decision boundaries
def plot_pca_decision_boundaries(model, X, y, figsize=(10, 10), n_components=2):
    '''
    Summary: Function to plot principal components decision boundaries

    model (sklearn model) : sklearn model or pipeline WITH PCA
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    figsize (tuple) : size of the plot
    n_components (int) : number of principal components to use (only supports 2 for now)

    output (None) : None
    '''
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    # Fit the model
    model.fit(X, y)
    # Get the principal components
    pca = model.named_steps['pca']
    # Get the principal components
    X_pca = pca.transform(X)
    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_
    # Get the first two principal components
    X_pca = X_pca[:, :n_components]
    # Get the minimum and maximum values for the first principal component
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    # Get the minimum and maximum values for the second principal component
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    # Get the predicted values
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Reshape the predicted values
    Z = Z.reshape(xx.shape)
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4)
    # Plot the data points
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.8)
    # Set the title
    ax.set_title('PCA Decision Boundaries')
    # Set the x and y labels
    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:0.2f})')
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:0.2f})')
    # Show the plot
    plt.show()