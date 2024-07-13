# 迭代法求解线性方程组
# 数学方法原理通过查找资料得到
import numpy as np

# Jacobi迭代法


def jacobi_iteration(A, b, x0, tol=1e-10, max_iterations=10000):
    # n = len(b)
    x = x0.copy()

    D = np.diag(A)
    R = A - np.diagflat(D)

    for iteration in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D

        print(f"Iteration {iteration + 1}: {x_new}")

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new

    raise ValueError("Jacobi method did not converge")


# Example usage
A = np.array([[10, 1, -1], [1, 10, 2], [-1, 2, 10]], dtype=float)
b = np.array([10, 8, 10], dtype=float)
x0 = np.zeros_like(b)

x = jacobi_iteration(A, b, x0)
print("Jacobi Solution:", x)
