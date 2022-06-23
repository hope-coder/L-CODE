#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 11:26
# @Author  : ZWP
# @Desc    : 
# @File    : test.py

from scipy.stats import t
import scipy.stats as stats
import shap

if __name__ == '__main__':
    print(stats.ttest_ind_from_stats(-0.0062015537269659434, 0.14059422500761348, 300, -0.019828943724943374, 0.16347136550783234, 300))
    X, y = shap.datasets.boston()
    print(t.sf(8, 80))
