#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:22
# @Author  : ZWP
# @Desc    : 代码问题，其中有不少的算法逻辑是通过列名索引的，因此算法要求输入的X变量必须为dataframe的格式
# @File    : main.py
from data_shap import object_model
from drift_detect import drift_detect

if __name__ == '__main__':

    # 初始化相关参数
    window_size = 40
    test_size = 0.8
    shap_class = 0
    dataset = "SEA"
    end = False
    alpha = 0.1
    threshold = 0.8

    # 构造可解释模型以及漂移检测器
    shap_model = object_model(window_size, test_size, shap_class=shap_class, dataset=dataset)
    detector = drift_detect(X_train=shap_model.X_train, window_size=window_size, alpha=alpha, threshold=threshold)

    # 进行特征的过滤
    X_train, shap = shap_model.getTrain()
    feature_select = detector.feat_selection(shap)

    # 获取第一个时间窗口得值
    X_ref, shap_values = shap_model.getNextWindows()
    ref_stats, ref_stats_table = detector.statistic_dist(X_ref, shap_values)

    # 循环检测数据流中的漂移
    while not end:
        # 获取最新窗口数据
        X_detect, shap_detect_values = shap_model.getNextWindows()
        detect_stats, detect_stats_table = detector.statistic_dist(X_detect, shap_detect_values)

        # 设置一下索引列为特证名，后面好找
        temp_detect_stats = detect_stats.set_index("feature")
        temp_ref_stats = ref_stats.set_index("feature")
        if_drift = False

        # 对每一个特征进行循环，找到漂移
        for feature_index, feature in enumerate(feature_select):
            expected_mean, expected_std = detector.expected_shap_dist(ref_stats_table[feature],
                                                                      detect_stats_table[feature])

            print("\t 参考窗口统计值：", temp_ref_stats.loc[feature]["mean"], temp_ref_stats.loc[feature]["std"])
            print("\t 期望数据统计值", expected_mean, expected_std)
            print("\t 检测窗口统计值", temp_detect_stats.loc[feature]["mean"], temp_detect_stats.loc[feature]["std"])

            if detector.t_test(expected_mean, expected_std, temp_detect_stats.loc[feature]["mean"],
                               temp_detect_stats.loc[feature]["std"]):
                print(str(feature) + "列发生了漂移")
                if_drift = True
            else:
                print(str(feature) + "列未发生漂移")

        if if_drift:
            break
        else:
            # 对单个特征的表格更新操作
            for feature in feature_select:
                ref_stats_table[feature] = detector.updated_ref_dist(ref_stats_table[feature], detect_stats_table[feature])
