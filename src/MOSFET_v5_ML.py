# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Main executable code
"""
import argparse
import logging
import numpy as np
import pandas as pd
import joblib, pathlib
from utils.synthetic_datagen import data_gen, data_gen_aug
from utils.prepare_data import prepare_data
from utils.training import linreg, XGBHyper_train, XGBReg_train, XGB_predict, XGB_predict_daal4py, XGB_predict_aug


def main(FLAGS):

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)    
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)

    logger = logging.getLogger()

    if FLAGS.n_data_len:
        linspace_factor = int(FLAGS.n_data_len)
    else:
        linspace_factor = 1

    i_flag = FLAGS.intel
    model_file = FLAGS.modelfile
    logger.info("\n")
    if i_flag:
        logger.info("===== Running benchmarks for oneAPI tech =====")
    else:
        logger.info("===== Running benchmarks for stock =====")

    logger.info("===== Generating Synthetic Data =====")
    synth_data = data_gen(linspace_factor)
    logger.info("Synthetic data shape %s", ' '.join(
        map(str, list(synth_data.shape))))

    if FLAGS.model == 'lr':
        # Linear Regression for reference

        input_cols = ['w_l', 'vgs', 'vth', 'eta',
                      'temp', 'w_l_bins', 'vgs_bins', 'vth_bins']
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var='log-leakage')

        logger.info("===== Running Benchmarks for Linear Regression =====")
        train_time, pred_time, MSE = linreg(
            x_train, x_test, y_train, y_test, i_flag)
        logger.info("Training time = %s", train_time)
        logger.info("Prediction time = %s", pred_time)
        logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))

    if FLAGS.model == 'xgbh':
        # XGB Regression Hyperparameter training

        input_cols = ['w_l', 'vgs', 'vth', 'eta',
                      'temp', 'w_l_bins', 'vgs_bins', 'vth_bins']
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var='log-leakage')
        loop_ctr = 5
        parameters = {'nthread': [1],
                      'learning_rate': [0.02],  # so called `eta` value
                      'max_depth': [3, 5],
                      'min_child_weight': [6, 7],
                      'n_estimators': [750, 1000],
                      'tree_method': ['hist']}
        logger.info(
            "===== Running Benchmarks for XGB Hyperparameter Training =====")
        train_time, trained_model, model_params = XGBHyper_train(
            x_train, y_train, parameters)
        logger.info("Training time = %s", train_time)
        if i_flag:
            prediction, pred_time, MSE = XGB_predict(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            prediction, pred_time_daal4py, MSE_daal4py = XGB_predict_daal4py(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            logger.info("Prediction time = %s", pred_time)
            logger.info("daal4py Prediction time = %s", pred_time_daal4py)
            logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))
            logger.info('daal4py Mean SQ Error: %s',
                        str(round(np.mean(MSE_daal4py), 3)))
        else:
            prediction, pred_time, MSE = XGB_predict(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            logger.info("Prediction time = %s", pred_time)
            logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))

        if model_file != "":
            joblib.dump(trained_model, model_file)
        else:
            joblib.dump(trained_model, "model.pkl")

    if FLAGS.model == 'xgb':
        # XGB Regression

        input_cols = ['w_l', 'vgs', 'vth', 'eta',
                      'temp', 'w_l_bins', 'vgs_bins', 'vth_bins']
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var='log-leakage')

        logger.info("===== Running Benchmarks for XGB Regression =====")
        loop_ctr = 5
        train_time, trained_model = XGBReg_train(
            x_train, y_train, loop_ctr, i_flag)
        logger.info("Training time = %s", train_time)
        if i_flag:
            prediction, pred_time, MSE = XGB_predict(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            prediction, pred_time_daal4py, MSE_daal4py = XGB_predict_daal4py(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            logger.info("Prediction time = %s", pred_time)
            logger.info("daal4py Prediction time = %s", pred_time_daal4py)
            logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))
            logger.info('daal4py Mean SQ Error: %s',
                        str(round(np.mean(MSE_daal4py), 3)))
        else:
            prediction, pred_time, MSE = XGB_predict(
                x_test, y_test, trained_model, loop_ctr, i_flag)
            logger.info("Prediction time = %s", pred_time)
            logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))
        if model_file != "":
            joblib.dump(trained_model, model_file)
        else:
            joblib.dump(trained_model, "model.pkl")            

    if FLAGS.model == 'xgbfull':  # this will report results for daal4py converted model only
        # XGB Regression Hyperparameter training

        input_cols = ['w_l', 'vgs', 'vth', 'eta',
                      'temp', 'w_l_bins', 'vgs_bins', 'vth_bins']
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var='log-leakage')
        loop_ctr = 1
        parameters = {'nthread': [1],
                      'learning_rate': [0.02],  # so called `eta` value
                      'max_depth': [3, 5],
                      'min_child_weight': [6, 7],
                      'n_estimators': [750, 1000],
                      'tree_method': ['hist']}
        logger.info("\n")
        logger.info(
            "===== Running Benchmarks for full pipeline execution =====")
        train_time, trained_model, model_params = XGBHyper_train(
            x_train, y_train, parameters)
        if i_flag:
            prediction, pred_time, MSE = XGB_predict_daal4py(
                x_test, y_test, trained_model, loop_ctr, i_flag)
        else:
            prediction, pred_time, MSE = XGB_predict(
                x_test, y_test, trained_model, loop_ctr, i_flag)
        logger.info('Mean SQ Error for initial train: %s',
                    str(round(np.mean(MSE), 3)))

        synth_data_0 = synth_data.drop(columns='sub-vth')
        train_trend_df = pd.DataFrame(columns=['len_data', 'train_time'])
        train_time_vals = []
        len_data_vals = []
        for counter in range(10):
            # generating new synthetic data and making a prediction on it
            synth_data_aug = data_gen_aug(linspace_factor)
            synth_data_aug = synth_data_aug.astype(float)

            y_pred = XGB_predict_aug(synth_data_aug, trained_model, i_flag)
            synth_data_aug['log-leakage'] = y_pred

            # concatenating the original dataframe to
            synth_data_0 = pd.concat(
                [synth_data_0, synth_data_aug], axis=0, ignore_index=True)
            if len(synth_data_0) > 2500000:
                synth_data_0 = synth_data_0.tail(2500000)

            # semi supervised learning on the new "augmented data"
            train_time, trained_model = XGBReg_train(
                synth_data_0[input_cols], synth_data_0['log-leakage'], loop_ctr, i_flag, model_params)
            len_data_vals.append(len(synth_data_0))
            train_time_vals.append(train_time)
            print('augmented supervised learning round finished ' + str(counter))
        train_trend_df['len_data'] = len_data_vals
        train_trend_df['train_time'] = train_time_vals
        if FLAGS.intel:
            train_trend_df.to_csv('train_time_intel.csv', index=False)
        else:
            train_trend_df.to_csv('train_time_stock.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="",
                        required=True,
                        help="type of model lr:linreg, xgb:xgboost, xgbh: \
                        xgb with hyperparameter tuning")
    parser.add_argument('-mf',
                        '--modelfile',
                        type=str,
                        default="",
                        required=False,
                        help="dump model file")
    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies")
    parser.add_argument('-n',
                        '--n_data_len',
                        type=str,
                        default="1",
                        help="Option for data length. Provide 1 2 or 3")
    FLAGS = parser.parse_args()
    main(FLAGS)
