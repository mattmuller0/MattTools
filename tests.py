# Script to test the functionality of the package

# Import packages
# from MattTools import utils, plotting, modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import file in the same directory as this script
import sys
sys.path.append('../')
from MattTools.plotting import plot_training_roc_curve_ci, plot_roc_curve_ci
from MattTools.utils import hide_warnings

# Hide warnings
hide_warnings()

# Load the data
data = load_breast_cancer()
X = data.data
y = data.target

# Test the plotting functionality of plot_training_roc_curve_ci
# Create a model
model = LogisticRegression()
plot_training_roc_curve_ci(model, X, y)