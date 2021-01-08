from pytest import approx
from gd_procgen.gradient_descent import minimize, maximize
import numpy as np
from gd_procgen.gradient_descent import GRAD_TYPE
from parameterized import parameterized


class TestGradientDescent:
    @parameterized.expand(list(GRAD_TYPE.keys()))
    def test_minimize_simple_func(self, grad_type):
        def cost_func(x):
            return x[0]**2
        x0 = np.array([5])
        x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type)
        assert x == approx(np.array([0]), abs=1e-3)

    @parameterized.expand(GRAD_TYPE.keys())
    def test_maximize_simple_func(self, grad_type):
        def cost_func(x):
            return -x[0]**2
        x0 = np.array([-3.4])
        x, cost = maximize(cost_func, x0, max_iters=500, grad_type=grad_type)
        assert x == approx(np.array([0]), abs=1e-3)
