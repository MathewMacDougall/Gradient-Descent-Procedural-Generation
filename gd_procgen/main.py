import numpy as np
from gd_procgen.optimization_problem import OptimizationProblem
import matplotlib.pyplot as plt

class SinglePoint(OptimizationProblem):
    def x0(self):
        return np.random.default_rng().uniform(np.full(2, -5), np.full(2, 5))

    def cost(self, x):
        return np.linalg.norm(x)

    def draw(self, data):
        x = data[0]
        y = data[1]

        return [
            self._ax.plot(x, y, "ro"),
            self._ax.plot(x+1, y+1, "bo"),
            self._ax.add_artist(plt.Circle((x-1, y-1), 2, color='b', alpha=0.5))
        ]

    def xlim(self):
        return (-5, 5)

    def ylim(self):
        return (-5, 5)


def main():
    problem = SinglePoint()
    problem.optimize()
    problem.play_animation()


if __name__ == "__main__":
    main()
