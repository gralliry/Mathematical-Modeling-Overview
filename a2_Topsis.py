import numpy as np  # 导入numpy库，用于进行科学计算
from normalizate import normalize_l2
from topsis import distance

# 牢记指标间不能进行任何运算
# 行是对象，列是指标


if __name__ == "__main__":
    y = np.array([
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 3],
        [1, 2, 4],
    ])

    y = normalize_l2(y)
    print(y)
    y = distance(y)
    print(y)
