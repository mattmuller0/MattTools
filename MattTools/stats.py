######################################################################
# 
#
# 
###################################################################### 
# Stats Functions to help 
# Matthew Muller
# 11/24/2022


##########
# Library Imports
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.metrics import roc_auc_score, roc_curve, auc

##########
# Set/Append Working directory
sys.path.append('.')

##########
# Code Below

# Function to calculate the confidence interval of a dataset
def mean_confidence_interval(data, confidence=0.95, axis=None):
    """
    Compute the mean, the confidence interval of the mean, and the tolerance
    interval. Note that the confidence interval is often misinterpreted [3].
    References:
    [1] https://en.wikipedia.org/wiki/Confidence_interval
    [2| https://en.wikipedia.org/wiki/Tolerance_interval
    [3] https://en.wikipedia.org/wiki/Confidence_interval#Meaning_and_interpretation
    """
    confidence = confidence / 100.0 if confidence > 1.0 else confidence
    assert(0 < confidence < 1)
    a = 1.0 * np.array(data)
    n = len(a)
    # Both s=std() and se=sem() use unbiased estimators (ddof=1).
    m = np.mean(a, axis=axis)
    s = np.std(a, ddof=1, axis=axis)
    se = st.sem(a, axis=axis)
    t = st.t.ppf((1 + confidence) / 2., n - 1)
    ci = np.c_[m - se * t, m + se * t]
    assert(ci.shape[1] == 2 and ci.shape[0] ==
           np.size(m, axis=None if axis is None else 0))
    return m, ci

# Function to bootstrap the auc score
def bootstrap_auc_confidence(y_pred, y_true, ci =  0.95,
                  n_bootstraps = 1000, rng_seed = 100,
                  plot_histogram = False):
    '''
    Summary: Binary target implementation of auc bootstrapping for determining a confidence interval

    y_pred (np.array) : numpy array of predicted values, usually given from the predict_proba method.
    y_true (np.array) : numpy array of true label values (1 is the presumed target)
    n_bootstraps (int) : integer value of the number of bootstraps
    rng_seed (str) : random seed to set a random state, which 
    plot_histogram (bool) : Plot a histogram of AUC values (default = False)

    output (list) : working_roc_auc, confidence_lower, confidence_upper
    '''

    working_roc_auc = roc_auc_score(y_true, y_pred)
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    if plot_histogram:
        import matplotlib.pyplot as plt
        plt.hist(bootstrapped_scores, bins=50)
        plt.title('Histogram of the bootstrapped ROC AUC scores')
        plt.show()

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    working_roc_auc, confidence_lower, confidence_upper = mean_confidence_interval(sorted_scores, confidence=ci)

    print(f"Confidence interval for the score: {working_roc_auc:0.3f} [{confidence_lower:0.3f} - {confidence_upper:0.3f}]")
    return working_roc_auc, confidence_lower, confidence_upper

# Create an object to bootstrap data
class Bootstrap:
    def __init__(self, n_bootstrap=100, stratified=True, rng_seed=None):
        '''
        Summary: Class to bootstrap a dataset
        n_bootstrap (int) : number of bootstraps to perform
        shuffle (bool) : whether to shuffle the indices before splitting
        rng_seed (int) : random seed to set a random state
        '''
        self.n_bootstrap = n_bootstrap
        self.stratified = stratified
        self.rng_seed = rng_seed

    def get_n_splits(self):
        return self.n_bootstrap

    def split(self, X, y=None):
        '''
        Summary: Split the data and labels into bootstrapped datasets
        X (np.array) : numpy array of data to bootstrap (can be n-dimensional)
        y (np.array) : numpy array of labels to bootstrap (can be n-dimensional)

        output (list) : bootstrapped_data, bootstrapped_labels
        '''
        if not self.stratified:
            Warning("Stratified is False, this may result in unbalanced classes in the bootstrapped datasets")
            rng = np.random.RandomState(self.rng_seed)
            indices = np.arange(len(X))
            for i in range(self.n_bootstrap):
                indices = rng.choice(indices, size=len(X), replace=True)
                # Only use indices with all classes
                while len(np.unique(y[indices])) < 2:
                    indices = rng.choice(indices, size=len(X), replace=True)
                yield indices, _ # returns an empty value in order to work as sklearn base folds
        else:
            rng = np.random.RandomState(self.rng_seed)
            # split data by unique labels
            labels = np.unique(y)
            # get indices for each label
            indices = [np.where(y == label)[0] for label in labels]
            # randomly sample each set of indices
            for i in range(self.n_bootstrap):
                # sample indices
                sampled_indices = [rng.choice(ind, size=len(ind), replace=True) for ind in indices]
                # concatenate sampled indices
                sampled_indices = np.concatenate(sampled_indices)
                # shuffle indices
                rng.shuffle(sampled_indices)
                # yield sampled indices
                yield sampled_indices, sampled_indices # returns an empty value in order to work as sklearn base folds

# Calculate the odds ratio of target columns in a dataframe against a list of columns
def odds_ratio(df, targets, columns, plot=False):
    '''
    Summary: Calculate the odds ratio of a column in a dataframe against a list of columns
    df (pd.DataFrame) : pandas dataframe
    targets (str) : list of column names to calculate the odds ratios for
    columns (list) : list of column names to calculate the odds ratio against
    ci (float) : confidence interval to calculate
    plot (bool) : whether to plot the odds ratio

    output (pd.DataFrame) : dataframe of odds ratio values
    '''
    # create empty dataframe
    odds_ratio_df = pd.DataFrame(columns=columns, index=targets)
    # loop through targets
    for target in targets:
        # loop through columns
        for column in columns:

            # drop values of 2 from df[target]
            idx = df[column] != 2
            df = df[idx]

            if len(np.unique(df[column])) != 2 or len(np.unique(df[target])) != 2:
                continue
            # get the odds ratio
            res = st.contingency.odds_ratio(pd.crosstab(df[target], df[column]))
            odds_ratio_df.loc[target, column] = res.statistic
            # get the pvalue
            odds_ratio_df.loc[target, column + '_pvalue'] = st.fisher_exact(pd.crosstab(df[target], df[column]))[1]
            # get the confidence interval
            odds_ratio_df.loc[target, column + '_ci_lower'] = res.confidence_interval(0.95)[0]
            odds_ratio_df.loc[target, column + '_ci_upper'] = res.confidence_interval(0.95)[1]

    # plot the odds ratio
    if plot:
        if len(columns) > 1:
            # get just the odds ratio
            odds_ratio_df_ = odds_ratio_df.drop(columns=[col for col in odds_ratio_df.columns if 'pvalue' in col or 'ci' in col])

            # set NaN or inf values to 0
            odds_ratio_df_ = odds_ratio_df_.replace([-np.inf], 0)
            odds_ratio_df_ = odds_ratio_df_.replace([np.inf], 10)

            # get aspect ratio
            aspect = odds_ratio_df_.shape[1] / odds_ratio_df_.shape[0]

            # Make a heatmap with gridlines
            plt.figure(figsize=(aspect * 5, 5))
            sns.heatmap(odds_ratio_df_, linewidths=0.5, linecolor='black', cmap='Blues')
            plt.show()
        else:
            # get just the odds ratio
            odds_ratio_df_ = odds_ratio_df.drop(columns=[col for col in odds_ratio_df.columns if 'pvalue' in col or 'ci' in col])
            odds_ratio_df_ci_ = odds_ratio_df.drop(columns=[col for col in odds_ratio_df.columns if 'ci' not in col])
            # set NaN or inf values to 0
            odds_ratio_df_ = odds_ratio_df_.replace([-np.inf], 0)
            odds_ratio_df_ = odds_ratio_df_.replace([np.inf], 10)

            # Make an error bar plot
            plt.figure()
            err = [odds_ratio_df_ci_.iloc[:, 0], odds_ratio_df_ci_.iloc[:, 1]]
            plt.errorbar(y=odds_ratio_df_.index, x=odds_ratio_df_.iloc[:, 0], xerr=err, fmt='o', capsize=5, color='black')
            plt.xticks(rotation=90)
            plt.title(f'Odds Ratio of {columns[0]}')
            plt.ylabel('Odds Ratio')
            plt.xlabel('Comparison Groups')
            plt.show()
    return odds_ratio_df