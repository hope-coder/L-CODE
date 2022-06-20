#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:06
# @Author  : ZWP
# @Desc    :
# @File    : data_shap.py
import xgboost
import shap

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import dataset_generate


class object_model:
    def __init__(self, windows_size, test=0.2, dataset="SEA", classification=True, shap_class=0):
        shap.initjs()  # notebook环境下，加载用于可视化的JS代码
        self.classification = classification
        # 我们先训练好一个XGBoost model
        self.windows_size = windows_size
        self.shap_class = shap_class

        # 数据准备
        X, y = dataset_generate.getDateset(dataset)
        data_size = X.shape[0]
        train_size = int(data_size * (1 - test))
        self.X_train = X[:train_size]
        self.X_test = X[train_size:]
        self.y_train = y[:train_size]
        self.y_test = y[train_size:]

        self.windows_number = 0
        self.windows_max_number = int((data_size - train_size) / self.windows_size)

        test_size = data_size - train_size
        if windows_size * 2 >= test_size:
            raise Exception("窗口太大")
        if dataset == "SEA":
            log_reg = RandomForestClassifier(n_estimators=20)
            log_reg.fit(self.X_train, self.y_train)
            self.explainer = shap.KernelExplainer(log_reg.predict_proba, self.X_train)
            print("测试效果" + str(log_reg.score(self.X_train, self.y_train)))
        elif dataset == "XGBoost":
            model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(self.X_train, label=self.y_train), 100)
            self.explainer = shap.TreeExplainer(model)
            # print("测试效果" + str(xgboost.score))
        else:
            log_reg = RandomForestClassifier(n_estimators=20)
            log_reg.fit(self.X_train, self.y_train)
            print("测试效果" + str(log_reg.score(self.X_train, self.y_train)))
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

    def getNextWindows(self):
        if self.windows_number < self.windows_max_number:
            X_detect = self.X_test[(self.windows_size * self.windows_number): self.windows_size * (self.windows_number + 1)]
            y_detect = self.y_test[(self.windows_size * self.windows_number): self.windows_size * (self.windows_number + 1)]
            shap_values = self.explainer.shap_values(X_detect)
            self.windows_number = self.windows_number + 1
            if self.classification:
                return X_detect, shap_values[self.shap_class]
            else:
                return X_detect, shap_values
        else:
            raise Exception("数据取完了，别拿了")

    def getTrain(self):
        shap_values = self.explainer.shap_values(self.X_train)
        if self.classification:
            return self.X_train, shap_values[self.shap_class]
        else:
            return self.X_train, shap_values
