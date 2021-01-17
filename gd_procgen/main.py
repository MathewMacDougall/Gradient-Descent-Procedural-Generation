import numpy as np
from gd_procgen.optimization_problem import OptimizationProblem
import matplotlib.pyplot as plt
from gd_procgen.data_enum import *
from pathlib import Path
import math


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
            self._ax.plot(x + 1, y + 1, "bo"),
            self._ax.add_artist(plt.Circle((x - 1, y - 1), 2, color="b", alpha=0.5)),
        ]

    def xlim(self):
        return (-5, 5)

    def ylim(self):
        return (-5, 5)


class LakeAndTowns(OptimizationProblem):
    @unique
    class Data(DataEnumValueGenerator):
        LAKE_X = auto()
        LAKE_Y = auto()
        LAKE_RADIUS = auto()
        CITY_1_X = auto()
        CITY_1_Y = auto()
        CITY_2_X = auto()
        CITY_2_Y = auto()
        CITY_3_X = auto()
        CITY_3_Y = auto()

    def x0(self):
        return self._rng.uniform(
            np.full(len(list(self.Data)), -5), np.full(len(list(self.Data)), 5)
        )

    def cost(self, x):
        water_pos = [x[self.Data.LAKE_X], x[self.Data.LAKE_Y]]
        water_radius = x[self.Data.LAKE_RADIUS]
        city_1_pos = [x[self.Data.CITY_1_X], x[self.Data.CITY_1_Y]]
        city_2_pos = [x[self.Data.CITY_2_X], x[self.Data.CITY_2_Y]]
        city_3_pos = [x[self.Data.CITY_3_X], x[self.Data.CITY_3_Y]]

        def dist_from_water(xx):
            return math.hypot(xx[0] - water_pos[0], xx[1] - water_pos[1]) - water_radius

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        cities = [city_1_pos, city_2_pos, city_3_pos]

        def dist_to_other_cities(c):
            ret = 0
            for other in cities:
                if other == c:
                    continue
                ret += dist(c, other)
            return ret

        values = [
            math.fabs(water_radius - 2),
            math.fabs(dist_from_water(city_1_pos) - 1),
            math.fabs(dist_from_water(city_2_pos) - 1),
            math.fabs(dist_from_water(city_3_pos) - 1),
            math.exp(-dist_to_other_cities(city_1_pos)),
            math.exp(-dist_to_other_cities(city_2_pos)),
            math.exp(-dist_to_other_cities(city_3_pos)),
        ]
        return np.sum(values)

    def draw(self, data):
        return [
            self._ax.plot(data[self.Data.CITY_1_X], data[self.Data.CITY_1_Y], "ro"),
            self._ax.plot(data[self.Data.CITY_2_X], data[self.Data.CITY_2_Y], "ro"),
            self._ax.plot(data[self.Data.CITY_3_X], data[self.Data.CITY_3_Y], "ro"),
            self._ax.add_artist(
                plt.Circle(
                    (data[self.Data.LAKE_X], data[self.Data.LAKE_Y]),
                    data[self.Data.LAKE_RADIUS],
                    color="b",
                    alpha=0.5,
                )
            ),
        ]

    def xlim(self):
        return (-5, 5)

    def ylim(self):
        return (-5, 5)


def main():
    problem = LakeAndTowns(rand_seed=1)
    problem.optimize()
    problem.play_animation()
    # problem.save_animation(Path(__file__).parent / "lake_and_towns.mp4")


if __name__ == "__main__":
    main()
