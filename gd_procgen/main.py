import numpy as np
from gd_procgen.gradient_descent import minimize, GRAD_TYPE
from pathlib import Path
import argparse
from gd_procgen.animation import play_animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class OptimizationProblem:
    def __init__(self, x0, cost_func):
        self._data = []
        self._x0 = x0
        self._cost_func = cost_func

    def optimize(self):
        def callback(x):
            self._data.append(x)

        minimize(self._cost_func, self._x0, grad_type=GRAD_TYPE.BATCH, callback=callback)

    def play_animation(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid()

        def animate(i):
            x = self._data[i][0]
            y = self._data[i][1]
            return ax.plot(x, y, "ro")

        ani = animation.FuncAnimation(
            fig, animate, len(self._data), interval=0.1 * 1000, blit=True
        )
        plt.show()


def main():
    x0 = np.array([2, 5])

    def cost_func(x):
        return np.linalg.norm(x, 2)

    problem = OptimizationProblem(x0, cost_func)
    problem.optimize()
    problem.play_animation()


if __name__ == "__main__":
    main()
