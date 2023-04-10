######################################################################
#
#                       Utility Functions
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
import random
import sys


# Funcition to plot progress bar
def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Function to set the random seed as a random integer
def set_random_seed(seed=np.random.randint(1, 10000)):
    '''
    Summary: Function to set the random seed for reproducibility

    seed (int) : integer value of the random seed
    '''
    print(f'Setting random seed to {seed} for reproducibility.')

    # Set the random seed if it is installed
    try:
        import random
        random.seed(seed)
    except:
        pass

    # Set random seed for numpy
    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass

    # Set random seed for sklearn
    try:
        import sklearn
        sklearn.random_state(seed)
    except:
        pass

    # Set the random seed for tensorflow if it is installed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

    # Set the random seed for pytorch if it is installed
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

# Hide warnings based on the warning type (default to all warnings) with an option to show warnings
def hide_warnings(warning_type='all'):
    '''
    Summary: Function to hide warnings based on the warning type (default to all warnings) with an option to show warnings

    warning_type (str) : string value of the warning type to hide
    '''
    # Hide warnings
    if warning_type == 'all':
        import warnings
        warnings.filterwarnings('ignore')
    else:
        import warnings
        warnings.filterwarnings('ignore', category=warning_type)

    # Show warnings
    if warning_type == 'none':
        import warnings
        warnings.filterwarnings('default')