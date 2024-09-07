#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import numpy as np


def gray_relational_weight(mou, sons):
    sons = np.array(sons, dtype=np.float32)
    # 计算|X0-Xi|矩阵(在这里我们把X0定义为了Y)
    absX0_Xi = np.fabs(sons - mou[:, np.newaxis])

    # 计算两级最小差a
    a = np.min(absX0_Xi)
    # 计算两级最大差b
    b = np.max(absX0_Xi)

    # 分辨系数取0.5
    rho = 0.5

    # 计算子序列中各个指标与母序列的关联系数
    gamma = (a + rho * b) / (absX0_Xi + rho * b)

    # 子序列中各个指标的灰色关联度
    return np.mean(gamma, axis=0)
