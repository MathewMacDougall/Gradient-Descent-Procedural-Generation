import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest
from pytest import approx
from gd_procgen.gradient_descent import GRAD_TYPE, minimize, maximize
import numpy as np
import itertools

class TestGradientDescent:
    RAND_SEED = 42

    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE))
    def test_minimize_single_value_function(self, grad_type):
        # f = x^2
        def cost_func(x):
            return x[0]**2
        x0 = np.array([5])
        x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type, rand_seed=TestGradientDescent.RAND_SEED)
        assert x == approx(np.array([0]), abs=1e-3)

    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE))
    def test_maximize_single_value_function(self, grad_type):
        # f = -x^2
        def cost_func(x):
            return -(x[0]**2)
        x0 = np.array([-3.4])
        x, cost = maximize(cost_func, x0, max_iters=500, grad_type=grad_type, rand_seed=TestGradientDescent.RAND_SEED)
        assert x == approx(np.array([0]), abs=1e-3)

    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE))
    def test_minimize_multi_valued_function(self, grad_type):
        # f = x^2 + y^2 + 20
        def cost_func(x):
            return x[0]**2 + x[1]**2 + 20
        x0 = np.array([5, -6.5])
        x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type, rand_seed=TestGradientDescent.RAND_SEED)
        assert x == approx(np.array([0, 0]), abs=1e-3)

    @pytest.mark.parametrize("grad_type", list(GRAD_TYPE))
    def test_minimize_multi_valued_function_with_offsets(self, grad_type):
        # f = (x+5)^2 + (y-4)^2 + 20
        def cost_func(x):
            return (x[0] + 5) ** 2 + (x[1] - 4) ** 2 + 20
        x0 = np.array([0, 0])
        x, cost = minimize(cost_func, x0, max_iters=500, grad_type=grad_type, rand_seed=TestGradientDescent.RAND_SEED)
        assert x == approx(np.array([-5, 4]), abs=1e-3)
