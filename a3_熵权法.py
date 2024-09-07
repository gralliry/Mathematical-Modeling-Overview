# 熵权法
import numpy as np

from normalizate import normalize_l2, normalize_sum

from entropy_weight import entropy_weight

if __name__ == "__main__":
    # 定义一个指标矩阵X（正向化后）
    X = np.array([
        [9, 0, 0, 0],
        [8, 3, 0.9, 0.5],
        [6, 7, 0.2, 1]
    ])

    # 对矩阵X进行标准化处理，得到标准化矩阵Z
    Z = normalize_l2(X)

    print("标准化矩阵 Z = ")
    print(Z)  # 打印标准化矩阵Z

    D = entropy_weight(Z)
    print(D)

    # 根据信息效用值计算各指标的权重
    # 将信息效用值D归一化，得到各指标的权重W
    W = normalize_sum(D)

    # 打印得到的权重数组W
    print("权重 W = ")
    print(W)
