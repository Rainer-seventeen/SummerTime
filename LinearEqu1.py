# 方法1 高斯消元法
import numpy as np


def gaussian_elimination(A, b):

    n = len(b)

    # 生成增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        # 将对角线上的元素化为1
        Ab[i] = Ab[i] / Ab[i, i]

        # 将主元下方的元素消为0
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[i] * Ab[j, i]

    # 回代求解x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1: n] * x[i + 1: n])

    return x


# 测试数据
A = np.array([[10, 1, -1],
              [1, 10, 2],
              [-1, 2, 10]], dtype=float)
b = np.array([10, 8, 10], dtype=float)

x = gaussian_elimination(A, b)
print("解向量:", x)
