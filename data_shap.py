#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:06
# @Author  : ZWP
# @Desc    :
# @File    : data_shap.py
import xgboost
import shap
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def gen_y(X):
    y = []
    for i in range(X.shape[0]):
        x = X.iloc[i,]
        if x[0] + x[1] > 7:
            y.append(False)
        else:
            y.append(True)
    return y


def normal_data():
    f1_1 = [random.uniform(0, 5) for _ in range(90)]
    f1_2 = [random.uniform(5, 10) for _ in range(810)]
    f2 = [random.uniform(0, 10) for _ in range(900)]
    f3 = [random.uniform(0, 10) for _ in range(900)]
    f1 = f1_2 + f1_1
    random.shuffle(f1)
    tmp = list(zip(f1, f2, f3))
    x = [list(i) for i in tmp]
    df = pd.DataFrame(x)
    y = gen_y(df)
    df['y'] = y
    return df


def drift_data():
    drift_f1_1 = [random.uniform(0, 5) for _ in range(90)]
    drift_f1_2 = [random.uniform(5, 10) for _ in range(10)]
    drift_f2 = [random.uniform(0, 10) for _ in range(100)]
    drift_f3_1 = [random.uniform(0, 5) for _ in range(10)]
    drift_f3_2 = [random.uniform(5, 10) for _ in range(90)]
    drift_f1 = drift_f1_2 + drift_f1_1
    random.shuffle(drift_f1)
    drift_f3 = drift_f3_2 + drift_f3_1
    random.shuffle(drift_f3)
    drift_temp = list(zip(drift_f1, drift_f2, drift_f3))
    drift_x = [list(i) for i in drift_temp]
    drift_Data = pd.DataFrame(drift_x)
    drift_y = gen_y(drift_Data)
    drift_Data['y'] = drift_y
    return drift_Data


class object_model:
    def __init__(self, windows_size, test=0.2, type="SEA", classification=True, shap_class=0):
        shap.initjs()  # notebook环境下，加载用于可视化的JS代码
        self.classification = classification
        # 我们先训练好一个XGBoost model
        self.windows_size = windows_size
        self.shap_class = shap_class
        if type == "XGBoost":
            X, y = shap.datasets.boston()
            data_size = X.shape[0]
            train_size = int(data_size * (1 - test))
            self.X_train = X[:train_size]
            self.X_test = X[train_size:]
            self.y_train = y[:train_size]
            self.y_test = y[train_size:]
            test_size = data_size - train_size
            if windows_size * 2 >= test_size:
                raise Exception("窗口太大")
            model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(self.X_train, label=self.y_train), 100)
            self.explainer = shap.TreeExplainer(model)
        else:
            train_data = normal_data()
            drift = drift_data()
            data = pd.concat([train_data, drift], axis=0, ignore_index=True)
            X = data.iloc[:, 0:-1]
            y = data.iloc[:, -1]
            data_size = X.shape[0]
            train_size = int(data_size * (1 - test))
            self.X_train = X[:train_size]
            self.X_test = X[train_size:]
            self.y_train = y[:train_size]
            self.y_test = y[train_size:]
            test_size = data_size - train_size
            if windows_size * 2 > test_size:
                raise Exception("窗口太大")
            log_reg = RandomForestClassifier(n_estimators=20)
            log_reg.fit(self.X_train, self.y_train)
            self.explainer = shap.KernelExplainer(log_reg.predict_proba, self.X_train)

    def getRefWindows(self):
        X_ref = self.X_test[:self.windows_size]
        y_ref = self.y_test[:self.windows_size]
        shap_values = self.explainer.shap_values(X_ref)
        if self.classification:
            return X_ref, shap_values[self.shap_class]
        else:
            return X_ref, shap_values

    def getDetectWindows(self):
        X_detect = self.X_test[self.windows_size: self.windows_size * 2]
        y_detect = self.y_test[self.windows_size:self.windows_size * 2]
        shap_values = self.explainer.shap_values(X_detect)
        if self.classification:
            return X_detect, shap_values[self.shap_class]
        else:
            return X_detect, shap_values

    def getTrain(self):
        shap_values = self.explainer.shap_values(self.X_train)
        if self.classification:
            return self.X_train, shap_values[self.shap_class]
        else:
            return self.X_train, shap_values
