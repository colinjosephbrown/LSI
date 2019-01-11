import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from lsi import lsi


def lsi_demo_3d(n=32, params=None, animate=False, output_path=None):
    """
    This script shows a visual demo of the local synthetic instances (LSI) method.
    In particular, it first generates N random points in 2D, representing an input dataset, and then generates synthetic instances using LSI.
    The LSI method is run for different values of 'p' a parameter which determines how much to trust any given real sample.
    """
    if params is None:
        params = {}

    fig = plt.figure()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    x, y, m = random_data(n)

    if animate:
        params['num_synthetic_instances'] = params.get('num_synthetic_instances', 15)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], 'go', c=[0.9, 0.1, 0.1])

        p_vals = np.arange(1.6, 3.0, 0.01)
        intro_len = 75
        outro_len = 100

        def update(i):
            if i >= intro_len and i < intro_len + len(p_vals):
                p = p_vals[i - intro_len]
                params['p'] = p
                print('p = ' + str(p))
                new_x, new_y = lsi(x, y, params)
                ax.scatter(new_x[:, 0], new_x[:, 1], new_x[:, 2], c=[0.1, 0.1, 0.9])

            ax.view_init(30, i)
            plt.draw()
            plt.waitforbuttonpress(0.01)

        frames = np.arange(intro_len + len(p_vals) + outro_len)

        anim = FuncAnimation(fig, update, frames=frames, interval=30)
        if output_path is not None:
            anim.save(output_path, dpi=80, writer='imagemagick')
        else:
            plt.show()

    else:
        params['num_synthetic_instances'] = params.get('num_synthetic_instances', 1000)
        params['prange'] = params.get('prange', [1.2, 3.0])

        synth_x, synth_y = lsi(x, y, params)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=[0.9, 0.1, 0.1])
        ax.scatter(synth_x[:, 0], synth_x[:, 1], synth_x[:, 2], c=[0.1, 0.1, 0.9])
        ax.legend(['Real Data', 'Synthetic Data'])
        plt.draw()


def random_data(n):
    m = n
    x = np.random.rand(n, m)
    y = np.random.rand(n, 1)
    return x, y, m


def wave_data(n):
    m = 3

    x = np.zeros((n, m))
    n_i = int(math.sqrt(n))
    i = j = 0
    for k in range(n):
        t_i = math.pi * float(i) / n_i
        t_j = (2 * math.pi) * float(j) / n_i

        x[k, 0] = t_i
        x[k, 1] = t_j
        x[k, 2] = math.sin(t_i) + math.cos(t_j)

        i += 1
        if i >= n_i:
            i = 0
            j += 1

    y = np.random.rand(n, 1)
    return x, y, m


def helix_data(n):
    m = 3

    x = np.zeros((n, m))
    for n_i in range(n):
        t = (math.pi * 4) * float(n_i) / n
        x[n_i, 0] = math.cos(t)
        x[n_i, 1] = math.sin(t)
        x[n_i, 2] = 2 * t

    y = np.random.rand(n, 1)
    return x, y, m

if __name__ == '__main__':
    lsi_demo_3d()
