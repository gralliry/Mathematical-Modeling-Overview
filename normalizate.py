#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
import numpy as np


# 标准化处理
def normalize_l2(x, axis=0):
    # 确保X矩阵的数据类型为浮点数
    x = np.array(x, dtype=np.float32)
    # 对每一列数据进行归一化处理，即除以该列的欧几里得范数
    return x / np.sqrt(np.sum(x ** 2, axis=axis))


def normalize_mean(x, axis=0):
    x = np.array(x, dtype=np.float32)
    # 求出每一列的均值以供后续的数据预处理
    return x / np.mean(x, axis=axis)


def normalize_sum(x, axis=0):
    x = np.array(x, dtype=np.float32)
    return x / np.sum(x, axis=axis)


def normalize(x, ddof=0, axis=0):
    x = np.array(x, dtype=np.float32)
    return (x - np.mean(x, axis=axis)) / np.std(x, ddof=ddof, axis=axis)
