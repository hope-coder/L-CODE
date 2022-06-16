#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:22
# @Author  : ZWP
# @Desc    : 代码问题，其中有不少的算法逻辑是通过列名索引的，因此算法要求输入的X变量必须为dataframe的格式
# @File    : main.py
from data_shap import object_model
from drift_detect import drift_detect

if __name__ == '__main__':
    window_size = 100
    test_size = 0.2
    shap_class = 1

    shap_model = object_model(window_size, test_size, shap_class=shap_class)
    X_train, shap = shap_model.getTrain()
    X_ref, shap_values = shap_model.getRefWindows()
    X_detect, shap_detect_values = shap_model.getDetectWindows()

    detector = drift_detect(X_train=shap_model.X_train, window_size=window_size, alpha=0.01, threshold=0.99)
    ref_stats, ref_stats_table = detector.statistic_dist(X_ref, shap_values)
    detect_stats, detect_stats_table = detector.statistic_dist(X_detect, shap_detect_values)

    feature_select = detector.feat_selection(shap)
    temp_detect_stats = detect_stats.set_index("feature")
    temp_ref_stats = ref_stats.set_index("feature")
    for feature_index, feature in enumerate(feature_select):
        expected_mean, expected_std = detector.expected_shap_dist(ref_stats_table[feature_select[feature_index]],
                                                                  detect_stats_table[feature_select[feature_index]])
        print(temp_ref_stats.loc[feature]["mean"], temp_ref_stats.loc[feature]["std"])
        print(expected_mean, expected_std, temp_detect_stats.loc[feature]["mean"],
              temp_detect_stats.loc[feature]["std"])
        if detector.t_test(expected_mean, expected_std, temp_detect_stats.loc[feature]["mean"],
                           temp_detect_stats.loc[feature]["std"], ):
            print(str(feature) + "列发生了漂移")
        else:
            print(str(feature) + "列未发生漂移")
