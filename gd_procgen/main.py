import numpy as np
from gd_procgen.gradient_descent import minimize, GRAD_TYPE
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OptimizationProblem:
    def __init__(self, x0, cost_func, draw_func):
        self._data = []
        self._x0 = x0
        self._cost_func = cost_func
        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax
        self._set_animation_function(draw_func)

    def optimize(self):
        def callback(x):
            self._data.append(x)

        minimize(
            self._cost_func, self._x0, grad_type=GRAD_TYPE.BATCH, callback=callback
        )

    def _set_animation_function(self, func):
        def animate(i):
            data = self._data[i]
            return func(data, self._ax)

        self._animation_func = animate

    def play_animation(self, xlim, ylim):
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._ax.set_aspect("equal")
        self._ax.grid()

        animation.FuncAnimation(
            self._fig,
            self._animation_func,
            len(self._data),
            interval=0.1 * 1000,
            blit=True,
        )
        plt.show()


def main():
    x0 = np.array([2, 5])

    def cost_func(x):
        return np.linalg.norm(x, 2)

    def draw(data, ax):
        x = data[0]
        y = data[1]
        return ax.plot(x, y, "ro")

    problem = OptimizationProblem(x0, cost_func, draw)
    problem.optimize()
    problem.play_animation(xlim=(-5, 5), ylim=(-5, 5))


if __name__ == "__main__":
    main()
