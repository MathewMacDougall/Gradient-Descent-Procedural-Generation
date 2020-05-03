import math


def sigmoid(x, offset, width):
    """
    A sigmoid function with a given offset from 0 and rate of change

    By default this function increases as the value of x increases.
    ie. y = 1 / (1+e^(-x)
    To flip the sigmoid so the function decreases as the value of x increases,
    subtract it from 1

    :param x: The value to evaluate over the sigmoid
    :param offset: The offset of the center of the sigmoid from 0
    :param width: The total amount of change (centered around the offset) x must undergo
    to cause the value of the sigmoid to go from 0.018 to 0.982
    :return: A value in [0, 1] that is the value of the sigmoid at x
    """
    # This is the factor that changes how quickly the sigmoid goes from 0 to 1.
    # We divide 8 by it because that is the distance a sigmoid function centered
    # about 0 takes to go from 0.018 to 0.982 (and that is what the 'width' parameter is)
    change_factor = 8 / width
    # Avoid large positive exponents to Help prevent math range errors
    # https://stackoverflow.com/a/36269186
    exponent = change_factor * (offset - x)
    if exponent < 0:
        return 1 / (1 + math.exp(exponent))
    else:
        return 1 - 1 / (1 + math.exp(-exponent))


def smooth_abs(x, eps=1e-6):
    """
    A smooth, differentiable approximation of the absolute value. 
    See http://www.cs.utep.edu/vladik/2013/tr13-44.pdf
    :param x: The value to take the absolute value of
    :param eps:
    """
    assert eps > 0
    return math.sqrt(pow(x, 2) + eps)


def double_sigmoid(x, c1, w1, c2, w2):
    """
    From http://www.roperld.com/science/doublesigmoid.pdf
    """
    v1 = math.tanh((x - c1) / w1)
    v2 = math.tanh((x - c2) / w2)
    return 0.5 * (v1 - v2)


def log_cosh(x):
    """
    https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    """
    return math.log(math.cosh(x))


def create_sigmoid_interpolation(t):
    """
    Creates and returns a function f(x) that interpolates between data points using sigmoi functions.
    For values less than the smallest value, or larger than the largest value, the value of the function
    decreases linearly. This is to help optimization functinos find a gradient even when in very non-optimal
    ranges
    :param t: A list of tuples representing the data points to interpolate. Must be in non-decreasing order
    eg. [(-3, 0), (-1, 1), (2, 1), (3, 0)]
    """
    assert len(t) >= 1

    def func(x):
        result = 0
        if x < t[0][0]:
            result -= log_cosh(x - t[0][0])
        if x > t[-1][0]:
            result -= log_cosh(x - t[-1][0])

        result += t[0][1]

        for current, next in zip(t, t[1:]):
            if (current[1] == next[1]):
                continue
            width = next[0] - current[0]
            center = current[0] + width / 2
            sign = 1 if next[1] - current[1] >= 0 else -1
            result += sign * sigmoid(x, center, width)
        return result

    return func


# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.linspace(-10, 10, num=10000)
#
#
# def f(x):
#     c1 = -3
#     w1 = 0.1
#     c2 = 3
#     w2 = 0.5
#     result = double_sigmoid(x, c1, w1, c2, w2)
#     if x < c1 - w1:
#         result -= smooth_abs(x - (c1 - w1))
#     if x > c2 + w2:
#         result -= smooth_abs(x - (c2 + w2))
#     return result
#
#
# # foo = fancy_func([(-3, 0), (-2, 1), (0, 1), (3, 0)])
# foo = fancy_func([(-3, 2), (-2, 1), (0, 1), (3, -3)])
# # bar = foo(-0.1)
# # print(bar)
#
# # plt.plot(t, abs(t), 'r-', t, [foo(x) for x in t], 'b-')
# plt.plot(t, [foo(x) for x in t], 'b-')
# plt.show()
