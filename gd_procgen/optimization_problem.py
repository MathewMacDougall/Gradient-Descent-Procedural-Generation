from gd_procgen.gradient_descent import minimize, GRAD_TYPE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from celluloid import Camera
from abc import ABC, abstractmethod
import itertools


class OptimizationProblem(ABC):
    """
    Any subclass must use self._rng for any random operations
    """

    def __init__(self, rand_seed=None):
        self._data = []
        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax
        self._rand_seed = rand_seed
        self._rng = np.random.RandomState(rand_seed)
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
            print(self.cost(x))

        # TODO: Warn about crazy high cost values. May indicate extreme functions used in the cost
        # function

        minimize(
            self.cost,
            self.x0(),
            grad_type=GRAD_TYPE.RANDOM_STOCHASTIC,
            callback=callback,
            callback_every=1,
            rand_seed=self._rand_seed,
        )

    def _create_animation_function(self, func):
        def animate(i):
            data = self._data[i]
            result = func(data)
            assert isinstance(result, list)

            def create_iterable(x):
                _result = [k if isinstance(k, list) else [k] for k in x]
                return itertools.chain(*_result)

            return create_iterable(result)

        return animate

    def _setup_animation(self):
        assert self._data

        self._ax.set_xlim(self.xlim())
        self._ax.set_ylim(self.ylim())
        self._ax.set_aspect("equal")
        self._ax.grid()

    def save_animation(self, filepath):
        self._setup_animation()
        animation_func = self._create_animation_function(self.draw)
        camera = Camera(self._fig)

        INTERVAL_MS = 15
        REPEAT_DELAY_S = 3
        END_DELAY_S = 3
        anim = camera.animate(
            interval=INTERVAL_MS, blit=False, repeat_delay=int(REPEAT_DELAY_S * 1000)
        )
        # A bit of a hack to show the final state for a few seconds. The repeat_delay doesn't seem to apply
        # to the output file
        for i in list(range(len(self._data))) + int(
            END_DELAY_S * 1000 / INTERVAL_MS
        ) * [len(self._data) - 1]:
            animation_func(i)
            camera.snap()
        anim.save(str(filepath))

    def play_animation(self):
        self._setup_animation()
        animation_func = self._create_animation_function(self.draw)

        animation.FuncAnimation(
            self._fig,
            animation_func,
            len(self._data),
            interval=15,
            blit=True,
            repeat=False,
        )

        plt.show()
