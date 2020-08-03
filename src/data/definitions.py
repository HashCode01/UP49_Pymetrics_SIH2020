import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set true if submission
SUBMISSION = True

# HAVOK to note if we are in ssh
HAVOK = False

# Warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # LightGBM warning


# Setup paths
DATA_DIR = ROOT_DIR + '/data/test'
MODELS_DIR = ROOT_DIR + '/models/test'
# else:
#     HAVOK = True
#     DATA_DIR = '/scratch/pymetrics/data'
#     MODELS_DIR = ROOT_DIR + '/models'


# Packages/functions used everywhere
sys.path.insert(1, 'your_path\src')
import helper.decorators
import helper.functions
import helper.base

