import numpy as np
import copy
import random

def _batch_grad(cost_fun, x: np.array):
    ret = []
    dx = 1e-3
    for val in range(len(x)):
        xpos = x.copy()
        xpos[val] += dx
        xneg = x.copy()
        xneg[val] -= dx
        diff = cost_fun(xpos) - cost_fun(xneg)
        g = diff / (2*dx)
        ret.append(g)
    return np.array(ret)

def _sequential_stochastic_grad(cost_fun, x: np.array):
    def _sequential_stochastic_grad_helper(cost_fun, x: np.array):
        dx = 1e-3
        index = 0
        while True:
            xpos = x.copy()
            xpos[index] += dx
            xneg = x.copy()
            xneg[index] -= dx
            diff = cost_fun(xpos) - cost_fun(xneg)
            g = diff / (2 * dx)

            index += 1 % len(x)

            yield np.asarray(g)

    return next(_sequential_stochastic_grad_helper(cost_fun, x))

def _random_stochastic_grad(cost_fun, x: np.array):
    dx = 1e-3
    index = random.randint(0, x.size-1)
    xpos = x.copy()
    xpos[index] += dx
    xneg = x.copy()
    xneg[index] -= dx
    diff = cost_fun(xpos) - cost_fun(xneg)
    g = diff / (2 * dx)
    return np.asarray(g)

GRAD_TYPE = {
    "batch": _batch_grad,
    "seq_stoc": _sequential_stochastic_grad,
    "rand_stoc": _random_stochastic_grad
}

def maximize(cost_func, x0, grad_type="batch", max_iters=500, rand_seed=None):
    new_cost_func = lambda x: -cost_func(x)
    return minimize(new_cost_func, x0, grad_type, max_iters, rand_seed)

def minimize(cost_func, x0, grad_type="batch", max_iters=500, rand_seed=None):
    random.seed(rand_seed)

    if not isinstance(cost_func(x0), (int, float, np.float64, np.int64)):
        raise RuntimeError("Cost func outputs uneexpected type {}".format(type(cost_func(x0))))

    _grad = GRAD_TYPE.get(grad_type, None)
    assert _grad

    x0 = x0.astype(np.float64)
    current_iteration = 0
    nabla = 0.1
    x = copy.deepcopy(x0)
    while current_iteration < max_iters:
        current_iteration += 1
        grad = _grad(cost_func, x)
        x = x - nabla*grad

    return x, cost_func(x)

