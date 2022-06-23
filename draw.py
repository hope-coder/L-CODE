#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 16:37
# @Author  : ZWP
# @Desc    : 
# @File    : draw.py
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid", {"font.sans-serif": ['KaiTi', 'Arial']})


class drift_visualization():
    def __init__(self, feature_select, alpha):
        self.alpha = alpha
        self.drift = []  # 记录实际漂移的位置
        self.warning = {}  # 记录检测出漂移的位置
        self.all_warning = []
        self.accuracy = []  # 记录运行过程中准确率的变化
        self.p_value = {}  # 记录判别标准p值的变化趋势
        self.windows_number = []

        for feature in feature_select:
            self.p_value[feature] = []
            self.warning[feature] = []

    def add_acc(self, windows_number, acc):
        self.accuracy.append(acc)
        self.windows_number.append(windows_number)

    def add_p_value(self, windows_number, feature_name, p_value, is_drift):
        self.p_value[feature_name].append(p_value)
        if is_drift:
            self.warning[feature_name].append(windows_number)
            self.all_warning.append(windows_number)

    def add_drift(self, windows_number):
        self.drift.append(windows_number)

    def do_draw(self, feature):
        fig, ax = plt.subplots()
        ax.set_xlabel('index')
        ax.set_ylabel('accuracy/p-value')

        x_alpha = [self.windows_number[0], self.windows_number[-1]]

        y_alpha = [self.alpha, self.alpha]

        plt.vlines(x=self.drift, ymin=0.5, ymax=1, colors='r', linestyles='-',
                   label='drift')
        plt.vlines(x=self.all_warning, ymin=0, ymax=0.5, colors='g', linestyles=':',
                   label='drift_detect')

        ax.plot(x_alpha, y_alpha, lw=0.5, color="g", label="alpha判别")
        ax.plot(self.windows_number, self.accuracy, lw=2, label='accuracy')
        for key in self.p_value.keys():
            ax.plot(self.windows_number, self.p_value[key], lw=1, color="g", label=str(key) + "_p_value")

        ax.legend()
        plt.title("漂移检测图")
        plt.show()
        # fig.savefig('./')
