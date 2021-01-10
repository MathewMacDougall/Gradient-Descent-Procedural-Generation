import numpy as np
from gd_procgen.gradient_descent import minimize, GRAD_TYPE
from pathlib import Path
import argparse
from gd_procgen.animation import play_animation


def main(output_file):
    x0 = np.array([2, 5])

    def cost_func(x):
        return np.linalg.norm(x, 2)

    data = []

    def callback(x):
        data.append(x)

    minimize(cost_func, x0, grad_type=GRAD_TYPE.BATCH, callback=callback)
    np.savetxt(str(output_file), np.asarray(data), delimiter=",")

    play_animation(output_file, None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=Path)

    args = parser.parse_args()

    output_file = args.output_file
    output_file = output_file.resolve()
    assert output_file.suffix == ".csv"

    main(output_file)
