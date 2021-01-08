import numpy as np
import copy

def _grad(cost_fun, x: np.array):
    ret = []
    dx = 1e-3
    for val in range(len(x)):
        xpos = x.copy()
        xpos[val] += dx
        xneg = x.copy()
        xneg[val] -= dx
        diff = (cost_fun(xpos) - cost_fun(xneg))
        g = diff / (2*dx)
        ret.append(g)
    # Need asarray so can multiply by scalars (needed by projgrad)
    return np.asarray(ret)


def minimize(cost_func, x0, max_iters=500, reltol=1e-4, abstol=1e-4):
    x0 = x0.astype(np.float64)
    current_iteration = 0
    nabla = 0.1
    x = copy.deepcopy(x0)
    while current_iteration < max_iters:
        current_iteration += 1
        grad = _grad(cost_func, x)
        x = x - nabla*grad

    return x, cost_func(x)

