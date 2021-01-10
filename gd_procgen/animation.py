import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def play_animation(filepath, draw_function):
    assert filepath.is_file()
    data =  np.genfromtxt(str(filepath), delimiter=",")

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid()

    def animate(i):
        x = data[i][0]
        y = data[i][1]
        return ax.plot(x, y, 'ro')

    ani = animation.FuncAnimation(
        fig, animate, len(data), interval=0.1*1000, blit=True)
    plt.show()
