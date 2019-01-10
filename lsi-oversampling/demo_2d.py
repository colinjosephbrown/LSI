import numpy as np
import matplotlib.pyplot as plt

from lsi import lsi


def lsi_demo(n=10, params=None):
    """
    This script shows a visual demo of the local synthetic instances (LSI) method.
    In particular, it first generates N random points in 2D, representing an input dataset, and then generates synthetic instances using LSI.
    The LSI method is run for different values of 'p' a parameter which determines how much to trust any given real sample.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    m = 2
    x = np.random.rand(n, m)
    y = np.random.rand(n, 1)
    params = {'num_synthetic_instances': 100}

    synth_x = None
    synth_y = None

    for p in np.arange(1.2, 3.0, 0.2):
        params['p'] = p
        print('p = ' + str(p))

        for pt_id in range(n):
            plt.plot(x[pt_id,0], x[pt_id,1], 'go')

        new_x, new_y = lsi(x, y, params)
        
        synth_x = new_x if synth_x is None else np.concatenate((synth_x, new_x), axis=0)
        synth_y = new_y if synth_y is None else np.concatenate((synth_y, new_y), axis=0)

        for pt_id in range(synth_x.shape[0]):
            plt.plot(synth_x[pt_id,0], synth_x[pt_id,1], 'b+')

        plt.draw()
        plt.waitforbuttonpress(0.01)

if __name__ == '__main__':
    lsi_demo()
