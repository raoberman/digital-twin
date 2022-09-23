# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Functions for training/inferencing ML models
"""
import time
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

def linreg(x_train, x_test, y_train, y_test, i_flag):
    
    # check if intel flag is on for sklearnex
    if i_flag:
        from sklearnex import patch_sklearn  # pylint: disable=C0415,E0401
        patch_sklearn()
    
    regr = linear_model.LinearRegression()
    train_time = []
    pred_time = []
    for _ in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        # training start
        regr_train_start = time.time()
        regr.fit(x_train, y_train)
        regr_train_time = time.time() - regr_train_start
        train_time.append(regr_train_time)
        
        # prediction start
        regr_pred_start = time.time()
        y_pred = regr.predict(x_test)
        regr_pred_time = time.time()-regr_pred_start
        pred_time.append(regr_pred_time)
    
    MSE = np.square(np.subtract(y_test, y_pred)).mean()
    return min(train_time), min(pred_time), MSE
    
def XGBHyper_train(x_train, y_train, params):
    
    # training start
    xgb_trainer = xgb.XGBRegressor()
    xgbh_grid = GridSearchCV(xgb_trainer, params, cv=4, n_jobs=8, verbose=True)
    xgbh_start = time.time()
    xgbh_grid.fit(x_train, y_train)
    xgbh_train = time.time()-xgbh_start
    best_params = xgbh_grid.best_params_
    best_grid = xgbh_grid.best_estimator_

    return xgbh_train, best_grid, best_params

def XGBReg_train(x_train, y_train, loop_ctr, i_flag, params=None):
    
    if params is None:
        params = {'n_estimators': 500, 'tree_method': 'hist'}
    xgb_model = xgb.XGBRegressor(**params)
    train_time = []
    for _ in list(range(loop_ctr)):
        # training start
        xgb_train_start = time.time()
        xgb_model.fit(x_train, y_train)
        xgb_train_time = time.time() - xgb_train_start
        train_time.append(xgb_train_time)
    
    return min(train_time), xgb_model
    
def XGB_predict(x_test, y_test, xgb_model, loop_ctr, i_flag):
    pred_time = []
    for _ in list(range(loop_ctr)):

        # prediction start
        xgb_pred_start = time.time()
        y_pred = xgb_model.predict(x_test)
        xgb_pred_time = time.time()-xgb_pred_start
        pred_time.append(xgb_pred_time)

    MSE = np.square(np.subtract(y_test, y_pred)).mean()
    return y_pred, min(pred_time), MSE

def XGB_predict_daal4py(x_test, y_test, xgb_model, loop_ctr, i_flag):
    pred_time = []
    import daal4py as d4p  # pylint: disable=C0415,E0401
    daal_model = d4p.get_gbt_model_from_xgboost(xgb_model.get_booster())
    for _ in list(range(loop_ctr)):

        # prediction start
        xgb_pred_start = time.time()
        y_pred = d4p.gbt_regression_prediction().compute(x_test, daal_model).prediction.reshape(-1)
        xgb_pred_time = time.time()-xgb_pred_start
        pred_time.append(xgb_pred_time)
    MSE = np.square(np.subtract(y_test, y_pred)).mean()
    return y_pred, min(pred_time), MSE

def XGB_predict_aug(x_test, xgb_model, i_flag):
    if i_flag:
        import daal4py as d4p  # pylint: disable=C0415,E0401
        daal_model = d4p.get_gbt_model_from_xgboost(xgb_model.get_booster())
    
    # prediction start
    # xgb_pred_start = time.time()
    if i_flag:
        y_pred = d4p.gbt_regression_prediction().compute(x_test, daal_model).prediction.reshape(-1)
    else:
        y_pred = xgb_model.predict(x_test)
    # xgb_pred_time = time.time()-xgb_pred_start
    return y_pred
