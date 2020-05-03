import numpy as np
import math
import matplotlib.pyplot as plt
import projgrad
from enum import IntEnum, unique, auto
from celluloid import Camera
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


WIDTH = 10
HEIGHT = 10

# A custom value-generator for enums that returns increasing integers
# starting from 0
class DataEnumValueGenerator(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return start+count-1

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


def sigmoid(x, offset, width):
    """
    A sigmoid function with a given offset from 0 and rate of change

    By default this function increases as the value of x increases.
    ie. y = 1 / (1+e^(-x)
    To flip the sigmoid so the function decreases as the value of x increases,
    subtract it from 1

    :param x: The value to evaluate over the sigmoid
    :param offset: The offset of the center of the sigmoid from 0
    :param width: The total amount of change (centered around the offset) x must undergo
    to cause the value of the sigmoid to go from 0.018 to 0.982
    :return: A value in [0, 1] that is the value of the sigmoid at x
    """
    # This is the factor that changes how quickly the sigmoid goes from 0 to 1.
    # We divide 8 by it because that is the distance a sigmoid function centered
    # about 0 takes to go from 0.018 to 0.982 (and that is what the 'width' parameter is)
    change_factor = 8 / width
    # Help prevent math range errors
    # https://stackoverflow.com/a/36269186
    if x < 0:
        return 1 - 1 / (1 + math.exp(change_factor * (offset + x)))
    else:
        return 1 / (1 + math.exp(change_factor * (offset - x)))


def project(x, mask=len(list(Data))*[False]):
    """
    Projects the data into the feasible set
    :param x: A list of data
    :return: A list of data with values projected / constrained to various ranges
    """
    for index, m in enumerate(mask):
        if m:
            if index == Data.LAKE_RADIUS:
                x[index] = np.clip(x[index], 1, 2)
            else:
                x[index] = np.clip(x[index], -WIDTH/2, WIDTH/2)
    return x

def score(x):
    """
    A scoring function for the given data. Higher scores are more optimal
    :param x: data
    :return: the score
    """
    water_pos = [x[Data.LAKE_X], x[Data.LAKE_Y]]
    water_radius = x[Data.LAKE_RADIUS]
    city_1_pos = [x[Data.CITY_1_X], x[Data.CITY_1_Y]]
    city_2_pos = [x[Data.CITY_2_X], x[Data.CITY_2_Y]]
    city_3_pos = [x[Data.CITY_3_X], x[Data.CITY_3_Y]]

    # lakes are around 0.5-3m in radius
    water_score = sigmoid(water_radius, 0.5, 0.1) - sigmoid(water_radius, 2, 1)

    def dist_from_water(xx):
        return math.hypot(xx[0] - water_pos[0], xx[1] - water_pos[1]) - water_radius

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # Cities prefer to be close to the water (in about [0.1, 1])
    city_1_water_score = sigmoid(dist_from_water(city_1_pos), 0.1, 0.05) - sigmoid(dist_from_water(city_1_pos), 0.3, 0.05) - 0.01*abs(dist_from_water(city_1_pos)-0.2)
    city_2_water_score = sigmoid(dist_from_water(city_2_pos), 0.1, 0.05) - sigmoid(dist_from_water(city_2_pos), 0.3, 0.05) - 0.01*abs(dist_from_water(city_2_pos)-0.2)
    city_3_water_score = sigmoid(dist_from_water(city_3_pos), 0.1, 0.05) - sigmoid(dist_from_water(city_3_pos), 0.3, 0.05) - 0.01*abs(dist_from_water(city_3_pos)-0.2)
    city_1_water_score = - abs(dist_from_water(city_1_pos)-0.2)
    city_2_water_score = - abs(dist_from_water(city_2_pos)-0.2)
    city_3_water_score = - abs(dist_from_water(city_3_pos)-0.2)

    city_1_score = city_1_water_score
    city_2_score = city_2_water_score
    city_3_score = city_3_water_score

    water_weight = 1
    city_weight = 10

    score = water_weight * (water_score) + city_weight * (city_1_score + city_2_score + city_3_score)
    return score

# Cost is opposite of score
def cost(x):
    return -score(x)

def grad(fun, x):
    ret = []
    dx = 0.01
    for val in range(len(x)):
        xpos = x.copy()
        xpos[val] += dx
        xneg = x.copy()
        xneg[val] -= dx
        diff = (fun(xpos) - fun(xneg))
        g = diff / (2*dx)
        ret.append(g)
    # Need asarray so can multiply by scalars (needed by projgrad)
    return np.asarray(ret)

def init_x0():
    np.random.seed(0)
    num_elements = len(list(Data))
    x0 = np.random.rand(num_elements)
    spread = max(WIDTH, HEIGHT)
    x0 = x0 * spread
    x0 = x0 - (spread / 2)
    x0[2] = max(x0[2], 0)
    return x0


def plot_data(ax, x0, x):
    def plot_cities(ax, x, color):
        ax.scatter(x[Data.CITY_1_X], x[Data.CITY_1_Y], c=color)
        ax.scatter(x[Data.CITY_2_X], x[Data.CITY_2_Y], c=color)
        ax.scatter(x[Data.CITY_3_X], x[Data.CITY_3_Y], c=color)

    def plot_water(ax, x, color):
        water_pos = (x[Data.LAKE_X], x[Data.LAKE_Y])
        water_radius = x[Data.LAKE_RADIUS]

        water_circle = plt.Circle(water_pos, water_radius, color=color, alpha=0.5)
        ax.add_artist(water_circle)

    # fig, ax = plt.subplots()
    # ax.axis('square')
    # ax.axis([-WIDTH/2, WIDTH/2, -HEIGHT/2, HEIGHT/2])

    # plot_water(ax, x0, 'r')
    plot_water(ax, x, 'b')

    # plot_cities(ax, x0, 'r')
    plot_cities(ax, x, 'g')

    # plt.show(block=True)


def main():
    print("Testing Procedural Generation with constrainted optimization")

    print("Generating x0")
    x0 = init_x0()
    print("x0: {}".format(x0))
    print("grad: {}".format(grad(cost, x0)))

    print("Projecting initial x0")
    x0 = project(x0)
    print("projected x0: {}".format(x0))


    print("Optimizing...")
    def minimization_func(x):
        return cost(x), grad(cost, x)
    mask = len(list(Data)) * [False]
    # mask[Data.LAKE_X] = True
    # mask[Data.LAKE_Y] = True
    mask[Data.LAKE_RADIUS] = True
    data = []
    def cb(f, p):
        print("current score: {}".format(f))
        data.append(p)
    res = projgrad.minimize(minimization_func, x0, project=project, mask=mask, maxiters=1500, disp=False, callback=cb, nboundupdate=1)
    optimized_x = res.x
    print("done optimizing")
    print("optimized: {}".format(optimized_x))
    print("\nStarting cost: {}".format(cost(x0)))
    print("Optimized cost: {}".format(cost(optimized_x)))

    fig, ax = plt.subplots()
    ax.axis('square')
    ax.axis([-WIDTH/2, WIDTH/2, -HEIGHT/2, HEIGHT/2])
    camera = Camera(fig)
    # A hack to display the final frame for a bit longer
    for d in data + 10*[data[-1]]:
        plot_data(ax, x0, d)
        camera.snap()
    anim = camera.animate(interval=200, blit=True, repeat_delay=500)
    plt.show()


if __name__ == '__main__':
    main()
