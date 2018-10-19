import numpy as np
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
import time
import gc
from tqdm import tqdm, tqdm_notebook
import xlearn as xl
from sklearn.metrics import f1_score

df = pd.read_csv('./feature/df_preprocess.csv', encoding='utf-8')

# Training task
ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
ffm_model.setTrain('./feature/ffm_train.txt')    # Set the path of training dataset
ffm_model.setValidate('./feature/ffm_val.txt')  # Set the path of validation dataset

# Parameters:
#  0. task: binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: f1
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'f1'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model/model.out')