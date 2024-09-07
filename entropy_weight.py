#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description:
# 熵权法
import numpy as np


# 定义一个自定义的对数函数mylog，用于处理输入数组中的零元素
def mylog(x):
    x = np.array(x, dtype=np.float32)
    x[x <= 0] = 1e-10
    # 如果当前元素的值为0，则在lnp中对应位置也设置为0，因为log(0)是未定义的，这里我们规定为0
    # 如果p[i]不为0，则计算其自然对数并赋值给lnp的对应位置
    y = np.log(x)
    return y  # 返回计算后的对数数组


def entropy_weight(x):
    x = np.array(x, dtype=np.float32)
    # 对第i个指标的数据进行归一化处理，得到概率分布p
    p = x / np.sum(x, axis=0)[np.newaxis, :]
    # 对象数
    n = x.shape[0]
    e = -np.sum(p * mylog(p), axis=0) / np.log(n)
    d = 1 - e
    return d
