#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 只支持一维数组，因为每个指标都可能有不一样的正向化函数
import numpy as np


# 极小型指标转化为极大型指标的函数
def min2max(maxx, x):
    # 将输入的指标数据转换为列表
    x = np.array(x, dtype=np.float32)
    # 计算最大值与每个指标值的差，并将其放入新列表中
    ans = maxx - x
    return ans


# 中间型指标转化为极大型指标的函数
def mid2max(bestx, x):
    x = np.array(x, dtype=np.float32)
    # 计算每个指标值与最优值之间的绝对差
    h = np.fabs(x - bestx)
    # 找到最大的差值
    M = np.max(h)
    if M == 0:
        M = 1  # 防止最大差值为0的情况
    # 计算每个差值占最大差值的比例，并从1中减去，得到新指标值
    ans = 1 - h / M
    return ans


# 区间型指标转化为极大型指标的函数
def reg2max(x, lowx, highx):
    x = np.array(x, dtype=np.float32)
    # 如果指标值在区间内，则直接取为1
    y = np.copy(x)
    y[(x >= lowx) & (x <= highx)] = 1
    # 计算指标值超出区间的最大距离
    M = max(lowx - np.min(x), np.max(x) - highx)
    if M == 0:
        M = 1  # 防止最大距离为0的情况
    # 如果指标值小于下限，则计算其与下限的距离比例
    y[x < lowx] = (1 - (lowx - x) / M)[x < lowx]
    # 如果指标值大于上限，则计算其与上限的距离比例
    y[x > highx] = (1 - (x - highx) / M)[x > highx]
    return y


# 最大最小距离评价
# 根据最大正向化后的值进行得分评价
def distance(x):
    x = np.array(x, dtype=np.float32)
    # 计算标准化矩阵每列的最大值和最小值，即每个指标最大和最小的值
    # 按轴方向压缩
    x_max = np.max(x, axis=0)[np.newaxis, :]
    x_min = np.min(x, axis=0)[np.newaxis, :]

    # 计算每个参评对象与最优情况的距离d+
    d_z = np.sqrt(np.sum((x - x_max) ** 2, axis=1))
    # 计算每个参评对象与最劣情况的距离d-
    d_f = np.sqrt(np.sum((x - x_min) ** 2, axis=1))

    # 计算每个参评对象的得分
    s = d_f / (d_z + d_f)
    return s
