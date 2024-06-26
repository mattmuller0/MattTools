######################################################################
# Matthew Muller
# 11/24/2022
#
# Plotting Functions to help 
######################################################################

######################################################################
# Library Imports
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold


from sklearn.base import clone
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.stats import kruskal, zscore
from sklearn.utils import resample

from MattTools import stats

######################################################################
# Code Below
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    low, high = st.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    return m, low, high


######################################################################
#
#               Plotting Dimensionality Reduction
#
######################################################################
# Plot dimensionality reduction
def plot_reduction(
    reduction,
    X : pd.DataFrame, y : pd.Series, 
    dim_1 = 0, dim_2 = 1,
    save_path = None, 
    figsize = (10, 10), *args
    ):
    '''
    Summary: Function to plot the PCA model

    pca (sklearn.decomposition.PCA) : sklearn PCA model
    X (pd.DataFrame) : dataframe of features
    y (pd.Series) : series of labels
    components (int) : number of components to plot on y-axis
    save_path (str) : string pointing where to save image
    *args (tuple) : typle of arguments to pass to plt.scatter
    '''
    # Check if reduction is a valid sklearn model
    if not hasattr(reduction, 'fit_transform'):
        raise TypeError(f'{reduction} is not a valid sklearn model')
    
    # Check if reduction is fitted
    if not hasattr(reduction, 'components_'):
        raise ValueError(f'{reduction} must be fitted before plotting')
    
    # Get the transformed data
    X_reduced = reduction.transform(X)

    # Plot the data
    plt.figure(figsize=figsize)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, *args)
    plt.title(f'{reduction.__class__.__name__} Plot')
    plt.xlabel(f'{reduction.__class__.__name__} {dim_1} [{PCA.explained_variance_ratio_[dim_1]*100:0.4f}%]')
    plt.ylabel(f'{reduction.__class__.__name__} {dim_2} [{PCA.explained_variance_ratio_[dim_2]*100:0.4f}%]')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_scree(pca : PCA(), components = 50, save_path = None, figsize = (10, 10), *args
    ):
    '''
    Summary: Function to plot the scree plot of a PCA model

    pca (sklearn.decomposition.PCA) : sklearn PCA model
    components (int) : number of components to plot on y-axis
    save_path (str) : string pointing where to save image
    figsize (tuple) : tuple of figure size
    args (dict) :  arguments to pass to plt.plot
    '''
    plt.figure(figsize=figsize)
    plt.plot(np.arange(pca.n_components_)[:components], 
             pca.explained_variance_ratio_[:components],
             'o-', *args)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

######################################################################
#
#           Plotting Curves and Metrics of dict of models
#
######################################################################
# Function to plot the results of the test_models function with a barplot mean and error bars
def plot_model_results(results, output_path=None, figsize=(12, 8)):
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
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

# Function to plot the ROC curves of each model
def plot_roc_curves(models, X, y, output_path=None, figsize=(10, 10)):
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
        fpr, tpr, _ = roc_curve(y, y_pred)
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
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

# Function to plot the ROC curves of each model
def plot_prc_curves(models, X, y, output_path=None, figsize=(10, 10)):
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
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


# Function to plot the confusion matrix of a dictionary of models in a grid square
def plot_confusion_matrices(models, X, y, output_path=None, figsize=(10, 10)):
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
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


# Function to plot prediction probabilities of a dictionary of models


######################################################################
#
#            Plotting Curves and Metrics of single models
#
######################################################################
def plot_roc_curve(
    y_true, score, 
    save_path = None
    ):
    '''
    y_true (np.array) : numpy array of the testing labels
    score (np.array) : numpy array of the prediction values (using model.predict_proba)
    save_path (str) : string pointing where to save image
    '''
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                                  roc_auc=roc_auc).plot()
    plt.title('ROC Curve', fontsize = 14)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(
    y_true, score, 
    save_path=None, labels=None):
    '''
    y_true (np.array) : numpy array of the testing labels
    score (np.array) : predictions of model (using model.predict)
    save_path (str) : string pointing where to save image
    labels (list) : list of labels used for classes
    '''
    cm = confusion_matrix(y_true, score)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.title('Confusion Matrix', fontsize = 14)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# Function to plot ROC curve with mean and 95% confidence interval from bootstrapping
def plot_roc_curve_ci(model, X, y, bootstraps=100,
                      title="Mean ROC curve with 95% Confidence Interval",
                      save_path=None, *args):
    '''
    Plot ROC curve with mean and 95% confidence interval from bootstrapping.
    Parameters:
    -----------
    model : sklearn model
        Model to be used for cross validation.
    X : numpy array or pandas DataFrame
        Features used.
    y : numpy array
        Labels used for classes.
    cv : resampling technique, default=StratifiedKFold(n_splits=5)
        Cross validation object to be used.
    title : str, default="Mean ROC curve with 95% Confidence Interval"
        Title of plot.
    save_path : str, default=None
        String pointing where to save image.
    *args : dict
        Additional keyword arguments to pass to the plot function.
    '''
    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise ValueError("X must be convertable to numpy array")

    # Calculate ROC curve for each fold
    tprs = []
    aucs = []
    fprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(bootstraps):
        _X, _y = resample(X, y, stratify=y)
        # predict probabilities
        yhat = model.predict_proba(_X)

        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]

        # calculate roc curves
        fpr, tpr, _ = roc_curve(_y, yhat)

        # calculate AUC and interp TPR
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        # append to lists
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        fprs.append(fpr)

    # Plot chance level
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    # Plot mean ROC curve with 95% confidence interval
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Calculate confidence intervals
    mean_auc, ci_auc = stats.mean_confidence_interval(aucs, confidence=0.95)
    mean_tpr, ci_tpr = stats.mean_confidence_interval(tprs, confidence=0.95, axis=0)
    mean_tpr[-1] = 1.0

    # get the confidence intervals out of the array
    ci_auc = ci_auc[:, 1] - mean_auc
    ci_tpr = ci_tpr[:, 1] - mean_tpr

    # Plot confidence interval
    tprs_upper = np.minimum(mean_tpr + ci_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - ci_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey",
                    alpha=0.2, label="95% Confidence Interval")
    
    # Plot mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=f"ROC (AUC = {mean_auc.item():.2f} ± {ci_auc.item():.2f})",
            lw=2, alpha=0.8, *args)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",title=title, aspect='equal')
    ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to plot the PRC curve of a model
def plot_prc_curve(model, X, y, save_path=None, title="Precision-Recall Curve", *args):
    '''
    Plot the precision-recall curve of a model.
    Parameters:
    -----------
    model : sklearn model
        Model to be used for cross validation.
    X : numpy array or pandas DataFrame
        Features used.
    y : numpy array
        Labels used for classes.
    save_path : str, default=None
        String pointing where to save image.
    title : str, default="Precision-Recall Curve"
        Title of plot.
    *args : dict
        Additional keyword arguments to pass to the plot function.
    '''
    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise ValueError("X must be convertable to numpy array")

    # predict probabilities
    yhat = model.predict_proba(X)

    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]

    # calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y, yhat)

    # calculate average precision score
    avg_precision = average_precision_score(y, yhat)

    # plot the curve
    plt.plot(recall, precision, marker=".", label=f"AP = {avg_precision:.2f}", *args)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to plot the decision boundaries of a model
def plot_decision_boundaries(model, X, y, figsize=(10, 10), feature_one = 0, feature_two = 1):
    '''
    ######### WIP!!! #########
    Summary: Function to plot the decision boundaries of a model

    model (sklearn model) : sklearn model or pipeline
    X (np.array) : numpy array of feature data
    y (np.array) : numpy array of target data
    figsize (tuple) : size of the plot
    n_features (int) : number of features to use (only supports 2 for now)

    output (None) : None
    '''
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    # Fit the model
    model.fit(X, y)
    # Get the minimum and maximum values for the first feature
    x_min, x_max = X[:, feature_one].min() - 1, X[:, 0].max() + 1
    # Get the minimum and maximum values for the second feature
    y_min, y_max = X[:, feature_two].min() - 1, X[:, 1].max() + 1
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
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    # Set the title
    ax.set_title('Decision Boundaries')
    # Set the x and y labels
    ax.set_xlabel(f'Feature 1')
    ax.set_ylabel(f'Feature 2')
    # Show the plot
    plt.show()

# Function to plot ROC curve with mean and 95% confidence interval from cross-validation
def plot_cross_validation_auroc(model, X, y, cv=StratifiedKFold(n_splits=5),
                      title="Mean ROC curve with 95% Confidence Interval",
                      save_path=None, *args):
    '''
    Plot ROC curve with mean and 95% confidence interval from cross-validation.

    Parameters:
    -----------
    model : sklearn model
        Model to be used for cross validation.
    X : numpy array or pandas DataFrame
        Features used.
    y : numpy array
        Labels used for classes.
    cv : resampling technique, default=StratifiedKFold(n_splits=5)
        Cross validation object to be used.
    title : str, default="Mean ROC curve with 95% Confidence Interval"
        Title of plot.
    save_path : str, default=None
        String pointing where to save image.
    *args : dict
        Additional arguments to pass to the plot function
    '''
    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise ValueError("X must be convertable to numpy array")
    
    # Clone the model
    model = clone(model)

    # Calculate ROC curve for each fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        # Fit model
        model.fit(X[train], y[train])

        # Get roc curve
        preds = model.predict_proba(X[test])
        preds = preds[:, 1]

        # Get the accuracy
        acc = model.score(X[test], y[test])

        # Get the ROC curve
        fpr, tpr, _ = roc_curve(y[test], preds)
        roc_auc = auc(fpr, tpr)

        # Interpolate the ROC curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        # replace any values over 1 with 1
        interp_tpr[interp_tpr > 1] = 1

        # append to lists
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    # Plot chance level
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    # Plot mean ROC curve with 95% confidence interval
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Calculate confidence intervals
    mean_auc, ci_auc = stats.mean_confidence_interval(aucs, confidence=0.95)
    mean_tpr, ci_tpr = stats.mean_confidence_interval(tprs, confidence=0.95, axis=0)
    mean_tpr[-1] = 1.0

    # get the confidence intervals out of the array
    ci_auc = ci_auc[:, 1] - mean_auc
    ci_tpr = ci_tpr[:, 1] - mean_tpr

    # Plot confidence intervals
    tprs_upper = np.minimum(mean_tpr + ci_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - ci_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey",
                    alpha=0.2, label="95% Confidence Interval")
    
    # Plot mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=f"ROC (AUC = {mean_auc.item():.2f} ± {ci_auc.item():.2f})",
            lw=2, alpha=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",title=title, aspect='equal')
    ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to plot Precision-Recall curve with mean and 95% confidence interval from bootstrapping
def plot_pr_curve_ci(model, X, y, bootstraps=100,
                     title="Mean Precision-Recall curve with 95% Confidence Interval",
                     save_path=None, *args):
    '''
    Plot Precision-Recall curve with mean and 95% confidence interval from bootstrapping.
    Parameters:
    -----------
    model : sklearn model
        Model to be used for cross validation.
    X : numpy array or pandas DataFrame
        Features used.
    y : numpy array
        Labels used for classes.
    title : str, default="Mean Precision-Recall curve with 95% Confidence Interval"
        Title of plot.
    save_path : str, default=None
        String pointing where to save image.
    *args : dict
        Additional keyword arguments to pass to the plot function.
    '''
    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise ValueError("X must be convertible to numpy array")

    # Calculate PR curve for each fold
    precisions = []
    aucs = []
    recalls = []
    mean_recall = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(bootstraps):
        _X, _y = resample(X, y, stratify=y)
        # predict probabilities
        yhat = model.predict_proba(_X)

        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]

        # calculate PR curves
        precision, recall, _ = precision_recall_curve(_y, yhat)

        # calculate AUC and interp precision
        pr_auc = auc(recall, precision)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_precision[0] = 1.0

        # append to lists
        precisions.append(interp_precision)
        aucs.append(pr_auc)
        recalls.append(recall)

    # Plot chance level
    ax.plot([0, 1], [np.mean(y), np.mean(y)], "k--", label="chance level (AUC = 0.5)")

    # Plot mean PR curve with 95% confidence interval
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[-1] = 0.0

    # Calculate confidence intervals
    mean_auc, ci_auc = mean_confidence_interval(aucs, confidence=0.95)
    mean_precision, ci_precision = mean_confidence_interval(precisions, confidence=0.95, axis=0)
    mean_precision[-1] = 0.0

    # get the confidence intervals out of the array
    ci_auc = ci_auc[:, 1] - mean_auc
    ci_precision = ci_precision[:, 1] - mean_precision

    # Plot confidence interval
    precisions_upper = np.minimum(mean_precision + ci_precision, 1)
    precisions_lower = np.maximum(mean_precision - ci_precision, 0)
    ax.fill_between(mean_recall, precisions_lower, precisions_upper, color="grey",
                    alpha=0.2, label="95% Confidence Interval")

    # Plot mean PR curve
    ax.plot(mean_recall, mean_precision, color="b",
            label=f"PR (AUC = {mean_auc.item():.2f} ± {ci_auc.item():.2f})",
            lw=2, alpha=0.8, *args)
    ax.set(xlabel="Recall", ylabel="Precision", title=title, aspect='equal')
    ax.legend(loc="lower left")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



#=================================================================================================
#
#                        Plotting Training Functions
#=================================================================================================
# Function to plot ROC curve with mean and 95% confidence interval from cross-validation
def plot_training_roc_curve_ci(model, X, y, cv=StratifiedKFold(n_splits=5),
                      title="Mean ROC curve with 95% Confidence Interval",
                      save_path=None, *args):
    '''
    Plot ROC curve with mean and 95% confidence interval from cross-validation.

    Parameters:
    -----------
    model : sklearn model
        Model to be used for cross validation.
    X : numpy array or pandas DataFrame
        Features used.
    y : numpy array
        Labels used for classes.
    cv : resampling technique, default=StratifiedKFold(n_splits=5)
        Cross validation object to be used.
    title : str, default="Mean ROC curve with 95% Confidence Interval"
        Title of plot.
    save_path : str, default=None
        String pointing where to save image.
    *args : dict
        Additional arguments to pass to the plot function
    '''
    # add warning that function will go defunct
    print("This function will be defunct in the future. Please use plot_cross_validation_auroc instead.")

    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise ValueError("X must be convertable to numpy array")
    
    # Clone the model
    model = clone(model)

    # Calculate ROC curve for each fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        # Fit model
        model.fit(X[train], y[train])

        # Get roc curve
        preds = model.predict_proba(X[test])
        preds = preds[:, 1]

        # Get the accuracy
        acc = model.score(X[test], y[test])

        # Get the ROC curve
        fpr, tpr, _ = roc_curve(y[test], preds)
        roc_auc = auc(fpr, tpr)

        # Interpolate the ROC curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        # replace any values over 1 with 1
        interp_tpr[interp_tpr > 1] = 1

        # append to lists
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    # Plot chance level
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    # Plot mean ROC curve with 95% confidence interval
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Calculate confidence intervals
    mean_auc, ci_auc = stats.mean_confidence_interval(aucs, confidence=0.95)
    mean_tpr, ci_tpr = stats.mean_confidence_interval(tprs, confidence=0.95, axis=0)
    mean_tpr[-1] = 1.0

    # get the confidence intervals out of the array
    ci_auc = ci_auc[:, 1] - mean_auc
    ci_tpr = ci_tpr[:, 1] - mean_tpr

    # Plot confidence intervals
    tprs_upper = np.minimum(mean_tpr + ci_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - ci_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey",
                    alpha=0.2, label="95% Confidence Interval")
    
    # Plot mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=f"ROC (AUC = {mean_auc.item():.2f} ± {ci_auc.item():.2f})",
            lw=2, alpha=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",title=title, aspect='equal')
    ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to plot the test probabilities of a model using cross validation
def plot_training_probas(model, X, y, 
                         cv=StratifiedKFold(n_splits=5), 
                         plot=sns.boxplot, save_path=None, title=None, *args):
    """
    Summary:
    --------
    Plots the training probabilities of a model using cross validation

    Parameters:
    -----------
    model (sklearn model) : sklearn model to be used
    X (np.array or pd.DataFrame) : np array of features used
    y (np.array) : list of labels used for classes
    cv (resampler) : resampler to use for cross validation
    plot (sns plot) : seaborn plot to use
    save_path (str) : path to save plot to
    *args (dict) : *args to pass to seaborn plot
    """
    # check if X is a numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = X.to_numpy()
        except:
            raise TypeError("X must be a numpy array or convertible to one")

    # CV setup
    model = clone(model)

    # get scores
    scores = []
    labels = []
    for fold, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])

        labels += list(y[test])
        scores += list(model.predict_proba(X[test])[:, 1])

    # convert to dataframe
    df = pd.DataFrame({"preds": scores})
    df["labels"] = labels

    # set title
    if not title:
        title = "Distribution of Prediction Scores by Class"

    # plot the data
    sns.set_theme(style="whitegrid", palette="colorblind")
    ax = plot(df, x="labels", y="preds", order=np.unique(df["labels"]), *args)
    ax.set(ylabel="Prediction Score", xlabel="Labels", title=title)

    # statistical annotation
    groups = [df.loc[df["labels"] == label, "preds"] for label in np.unique(df["labels"])]
    kruskal_wallis = kruskal(*groups)
    if kruskal_wallis.pvalue < 0.05:
        x1, x2 = 0, 1
        y, h = df["preds"].max() + 0.02, 0.01
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k")
        ax.text((x1 + x2) * 0.5, y + h, f"p = {kruskal_wallis.pvalue:.4f}", ha="center", va="bottom", color="k")

    # save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    return(df)