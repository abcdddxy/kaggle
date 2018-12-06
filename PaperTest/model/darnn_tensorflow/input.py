#!/usr/bin/env python


import pandas as pd
import numpy as np

from BatchManager import BatchManager
from sklearn.preprocessing import StandardScaler


def preprocess_data(dat, col_names):
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return (feats, targs), scale


def load_data(params, is_scaler=False):
    df = pd.read_csv(params["data_file"])

    target = np.array(df.NDX)
    names = df.columns.tolist()

    train_size = params["train_size"]
    val_size = params["val_size"]
    test_size = params["test_size"]

    t_mean = np.mean(target[:train_size])
    t_max = np.max(target[:train_size])
    t_min = np.min(target[:train_size])
    # print(t_mean, t_max, t_min)

    if is_scaler:
        scale = StandardScaler().fit(df)
        proc_dat = scale.transform(df)

        mask = np.ones(proc_dat.shape[1], dtype=bool)
        dat_cols = list(df.columns)
        mask[dat_cols.index('NDX')] = False

        X = proc_dat[:, mask]
        y = np.array([i[0] for i in proc_dat[:, ~mask]])
        X_val = proc_dat[train_size:train_size + val_size - 1, mask]
        y_val = np.array([i[0] for i in proc_dat[train_size:train_size + val_size - 1, ~mask]])
        return {"data": X, "target": y}, {"data": X_val, "target": y_val}

    if params["keyword"] == "train":
        X = df.loc[:, [x for x in names if x != 'NDX']].as_matrix()
        y = np.array(df.NDX)
        y = y_normal(y, t_mean, t_max, t_min)
        X_val = df.loc[train_size:train_size + val_size - 1, [x for x in names if x != 'NDX']].as_matrix()
        y_val = np.array(df.loc[train_size:train_size + val_size - 1].NDX)
        y_val = y_normal(y_val, t_mean, t_max, t_min)
        return {"data": X, "target": y}, {"data": X_val, "target": y_val}

        # X_train = df.loc[:train_size - 1, [x for x in names if x != 'NDX']].as_matrix()
        # X_val = df.loc[train_size:train_size + val_size - 1, [x for x in names if x != 'NDX']].as_matrix()
        # y_train = np.array(df.loc[:train_size - 1].NDX)
        # y_train = y_train - np.mean(y_train)
        # y_val = np.array(df.loc[train_size:train_size + val_size - 1].NDX)
        # y_val = y_val - np.mean(y_val)
        # return {"data": X_train, "target": y_train}, {"data": X_val, "target": y_val}
    else:
        X_test = df.loc[train_size + val_size:train_size + val_size + test_size - 1, [x for x in names if x != 'NDX']].as_matrix()
        y_test = np.array(df.loc[train_size + val_size:train_size + val_size + test_size - 1].NDX)
        y_test = y_normal(y_test, t_mean, t_max, t_min)
        return {"data": X_test, "target": y_test}


def y_normal(y, t_mean, t_max, t_min):
    # return ((y - t_min) / (t_max - t_min) - 0.5) * 100
    return y - t_mean


def data_batch(data, params):
    return BatchManager(data, params)
