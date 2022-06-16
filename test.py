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
    # stats.ttest_ind_from_stats()
    X, y = shap.datasets.boston()
    print(t.sf(8, 80))