from gd_procgen.gradient_descent import minimize, GRAD_TYPE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod
import itertools


class OptimizationProblem(ABC):
    def __init__(self):
        self._data = []
        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax
        super().__init__()

    @abstractmethod
    def xlim(self):
        pass

    @abstractmethod
    def ylim(self):
        pass

    @abstractmethod
    def cost(self, x):
        pass

    @abstractmethod
    def draw(self, x):
        pass

    @abstractmethod
    def x0(self):
        pass

    def optimize(self):
        def callback(x):
            self._data.append(x)

        minimize(
            self.cost, self.x0(), grad_type=GRAD_TYPE.BATCH, callback=callback
        )


    def play_animation(self):
        def _create_animation_function(func):
            def animate(i):
                data = self._data[i]
                result = func(data)
                assert isinstance(result, list)

                def create_iterable(x):
                    _result = [k if isinstance(k, list) else [k] for k in x]
                    return itertools.chain(*_result)

                return create_iterable(result)

            return animate

        assert data

        self._ax.set_xlim(self.xlim())
        self._ax.set_ylim(self.ylim())
        self._ax.set_aspect("equal")
        self._ax.grid()

        animation_func = _create_animation_function(self.draw)

        animation.FuncAnimation(
            self._fig,
            animation_func,
            len(self._data),
            interval=0.1 * 1000,
            blit=True
        )
        plt.show()
