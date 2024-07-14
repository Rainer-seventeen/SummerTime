import numpy as np


def trapezoidal_rule(f, a, b, n):
    """
    使用梯形规则进行数值积分
    f: 被积函数
    a: 积分下限
    b: 积分上限
    n: 分区数量
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral


def finite_difference(f, x, h=1e-5):
    """
    使用前向差分法进行数值微分
    f: 被微分函数
    x: 求导点
    h: 步长
    """
    derivative = (f(x) - f(x - h)) / h
    return derivative


# 示例函数
def func(x):
    return np.sin(x)


# 积分区间 [0, π]
a = 0
b = np.pi
n = 1000
result = trapezoidal_rule(func, a, b, n)
print("数值积分结果:", result)
# 求导点 x = π/4
x = np.pi / 4

result = finite_difference(func, x)
print("数值微分结果:", result)
