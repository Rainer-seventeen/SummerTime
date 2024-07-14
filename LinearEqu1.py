import numpy as np


def gaussian_elimination(A, b):
    n = len(b)  # 获取方程组的规模

    # 生成增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # 前向消元过程
    for i in range(n):
        # 将当前行的对角线元素化为1
        Ab[i] = Ab[i] / Ab[i, i]

        # 将当前行的主元下方的元素消为0
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[i] * Ab[j, i]

    # 回代求解x
    x = np.zeros(n)  # 初始化解向量
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1: n] * x[i + 1: n])

    return x


# 测试数据
A = np.array([[10, 1, -1],  # 系数矩阵A
              [1, 10, 2],
              [-1, 2, 10]], dtype=float)
b = np.array([10, 8, 10], dtype=float)  # 常数向量b

# 调用高斯消元法函数求解Ax = b
x = gaussian_elimination(A, b)
print("解向量:", x)
