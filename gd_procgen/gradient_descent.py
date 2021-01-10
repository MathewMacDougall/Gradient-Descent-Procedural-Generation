import numpy as np
import copy
import itertools
from enum import Enum


class GRAD_TYPE(Enum):
    # Your standard gradient descent. The gradient is calculated for all samples at once
    BATCH = 0
    # Stochastic Gradient Descent where the gradient is calculated for a single sample at a time.
    # Samples are selected in order.
    SEQUENTIAL_STOCHASTIC = 1
    # Stochastic Gradient Descent where the gradient is calculated for a single sample at a time.
    # Samples are selected randomly.
    RANDOM_STOCHASTIC = 2


def _batch_grad(cost_fun, x: np.array):
    ret = []
    dx = 1e-3
    for val in range(len(x)):
        xpos = x.copy()
        xpos[val] += dx
        xneg = x.copy()
        xneg[val] -= dx
        diff = cost_fun(xpos) - cost_fun(xneg)
        g = diff / (2 * dx)
        ret.append(g)
    return np.array(ret)


def _create_sequential_stochastic_grad_func():
    class SSGrad:
        def __init__(self):
            self.indices = itertools.count(0)

        def grad(self, cost_fun, x):
            dx = 1e-3
            index = next(self.indices) % x.size
            xpos = x.copy()
            xpos[index] += dx
            xneg = x.copy()
            xneg[index] -= dx
            diff = cost_fun(xpos) - cost_fun(xneg)
            g = diff / (2 * dx)
            ret = np.zeros(x.size)
            ret[index] = g

            return np.asarray(ret)

    ss_grad = SSGrad()

    def grad(cost_fun, x):
        return ss_grad.grad(cost_fun, x)

    return grad


def _create_random_stochastic_grad_func(rand_seed=None):
    class RSGrad:
        def __init__(self, rand_seed):
            self.rng = np.random.RandomState(rand_seed)

        def grad(self, cost_fun, x):
            dx = 1e-3
            index = self.rng.randint(0, x.size)
            xpos = x.copy()
            xpos[index] += dx
            xneg = x.copy()
            xneg[index] -= dx
            diff = cost_fun(xpos) - cost_fun(xneg)
            g = diff / (2 * dx)
            ret = np.zeros(x.size)
            ret[index] = g
            return np.asarray(ret)

    rs_grad = RSGrad(rand_seed)

    def grad(cost_fun, x):
        return rs_grad.grad(cost_fun, x)

    return grad


def _create_grad_func(grad_type, rand_seed=None):
    if grad_type == GRAD_TYPE.BATCH:
        return _batch_grad
    elif grad_type == GRAD_TYPE.SEQUENTIAL_STOCHASTIC:
        return _create_sequential_stochastic_grad_func()
    elif grad_type == GRAD_TYPE.RANDOM_STOCHASTIC:
        return _create_random_stochastic_grad_func(rand_seed)
    else:
        raise ValueError("Invalid gradient type")


def maximize(cost_func, x0, grad_type=GRAD_TYPE.BATCH, max_iters=500, rand_seed=None):
    new_cost_func = lambda x: -cost_func(x)
    return minimize(new_cost_func, x0, grad_type, max_iters, rand_seed)


def minimize(cost_func, x0, grad_type=GRAD_TYPE.BATCH, max_iters=500, rand_seed=None):
    if not isinstance(cost_func(x0), (int, float, np.float64, np.int64)):
        raise RuntimeError(
            "Cost func outputs uneexpected type {}".format(type(cost_func(x0)))
        )

    _grad = _create_grad_func(grad_type, rand_seed)
    assert _grad

    x0 = x0.astype(np.float64)
    current_iteration = 0
    nabla = 0.1
    x = copy.deepcopy(x0)
    while current_iteration < max_iters:
        current_iteration += 1
        grad = _grad(cost_func, x)
        step = -nabla * grad
        x = x + step

    return x, cost_func(x)
