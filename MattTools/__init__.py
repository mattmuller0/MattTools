"""
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~ MattTools ~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
Author: Matthew Muller
Date: 11/24/2022

This module contains functions for modeling and evaluating models as well as some utility functions.

It is intended to be used as a module for other projects. It's just a collection of functions that I use in my projects and I wanted to make them available to others.

"""

# Import necessary modules
import os
import sys
import glob
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.utils import resample
from sklearn.base import clone
import scipy.stats as st
import statsmodels.stats.api as sms
from scipy.stats import kruskal, zscore