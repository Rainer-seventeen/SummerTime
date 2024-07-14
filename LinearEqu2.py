# 迭代法求解线性方程组
# 数学方法原理通过查找资料得到
import numpy as np


def jacobi_iteration(A, b, x0, tol=1e-10, max_iterations=10000):
    # 复制初始猜测值向量x0，防止修改原始数据
    x = x0.copy()

    # 提取对角线元素D
    D = np.diag(A)
    # 计算余下的矩阵R
    R = A - np.diagflat(D)

    # 进行迭代
    for iteration in range(max_iterations):
        # 计算新的迭代值
        x_new = (b - np.dot(R, x)) / D

        # 打印当前迭代次数和新解
        print(f"Iteration {iteration + 1}: {x_new}")

        # 检查收敛条件
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        # 更新x为新计算的解
        x = x_new

    # 如果超过最大迭代次数仍未收敛，抛出异常
    raise ValueError("Jacobi 迭代法不收敛")


# 示例使用
A = np.array([[10, 1, -1],  # 系数矩阵A
              [1, 10, 2],
              [-1, 2, 10]], dtype=float)
b = np.array([10, 8, 10], dtype=float)  # 常数向量b
x0 = np.zeros_like(b)  # 初始猜测向量x0

# 调用Jacobi迭代函数求解Ax = b
x = jacobi_iteration(A, b, x0)
print("Jacobi Solution:", x)
