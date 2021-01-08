import unittest
from gd_procgen.gradient_descent import minimize, maximize
import numpy as np


class GradientDescentTest(unittest.TestCase):
    def test_minimize_simple_func(self):
        cost_func = lambda x: x**2
        x0 = np.array([5])
        x, cost = minimize(cost_func, x0)
        self.assertAlmostEqual(0, x, delta=1e-3)

    def test_maximize_simple_func(self):
        cost_func = lambda x: -x**2
        x0 = np.array([-3.4])
        x, cost = maximize(cost_func, x0)
        self.assertAlmostEqual(0, x, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
