import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest
from pytest import approx
from gd_procgen.gradient_descent import GRAD_TYPE, minimize, maximize
import numpy as np

class TestGradientDescent:
    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE.keys()))
    def test_minimize_single_value_function(self, grad_type):
        # f = x^2
        def cost_func(x):
            return x[0]**2
        x0 = np.array([5])
        x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type)
        assert x == approx(np.array([0]), abs=1e-3)

    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE.keys()))
    def test_maximize_single_value_function(self, grad_type):
        # f = -x^2
        def cost_func(x):
            return -x[0]**2
        x0 = np.array([-3.4])
        x, cost = maximize(cost_func, x0, max_iters=500, grad_type=grad_type)
        assert x == approx(np.array([0]), abs=1e-3)

    # @parameterized.expand(GRAD_TYPE.keys())
    # def test_minimize_single_value_function(self, grad_type):
    #     # f = x^2 + y^2 + 20
    #     def cost_func(x):
    #         return x[0]**2
    #     x0 = np.array([5, -6.5])
    #     x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type)
    #     assert x == approx(np.array([0, 0]), abs=1e-3)
    #     # TODO: you are here. Need to add more complicated tests for gradient descent
