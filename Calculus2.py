import numpy as np


def finite_difference(f, x, h=1e-5):
    """
    使用有限差分法进行数值微分
    f: 被微分函数
    x: 求导点
    h: 步长
    """
    derivative = (f(x + h) - f(x - h)) / (2 * h)
    return derivative


# 示例函数
def func(x):
    return np.sin(x)


# 求导点 x = π/4
x = np.pi / 4

result = finite_difference(func, x)
print("数值微分结果:", result)


def simpsons_rule(f, a, b, n):
    """
    使用辛普森规则进行数值积分
    f: 被积函数
    a: 积分下限
    b: 积分上限
    n: 分区数量 (必须是偶数)
    """
    if n % 2 == 1:
        raise ValueError("分区数量 n 必须是偶数")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    integral = y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2: n - 1: 2])
    integral *= h / 3

    return integral


# 示例函数
def func(x):
    return np.sin(x)


# 积分区间 [0, π]
a = 0
b = np.pi
n = 1000  # 必须是偶数

result = simpsons_rule(func, a, b, n)
print("数值积分结果:", result)
