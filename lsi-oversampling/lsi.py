# Colin Joseph Brown, 2018
import numpy as np
import random


def lsi(x, y, params=None):
    """
    An implementation of the local synthetic instances (LSI) method for over-sampling a dataset.
    Arguments:
    x - An N by M numpy array containing N samples of M dimensions (i.e. features)
    y - An N by D numpy array containing N labels of D dimensions
    params - A dictionary containing method parameters including:
        num_synthetic_instances - The number of synthetic samples to generate (default=N)	
        max_num_weighted_samples - The maximum number of real samples to interpolate between (default=inf)
        p_range - The range of the exponent p, (p_min, p_max), defining how much to trust any one sample - see [1] for details (default=[1.2, 3.0])

    Tips:
    -A good range for p is roughly [1.2, 3.0]
    -In the case that num_synthetic_instances >> N, you likely want to vary p in order to mitigate 'banding'.
    -LSI is designed for the use-case of sparsely sampled, high-dimensional data. If you do not have this case, you may want to also try SMOTE or other comparable synthetic oversampling methods to see what works best.  
    

    [1] Brown, Colin J., et al. "Prediction of motor function in very preterm infants using connectome features and local synthetic instances." MICCAI, 2015.
    """
    assert isinstance(x, np.ndarray), 'Input x must be a Numpy array'
    assert isinstance(y, np.ndarray), 'Input y must by a Numpy array'
    
    n = x.shape[0]

    max_num_weighted_samples = params.get('max_num_weighted_samples', np.inf)
    num_synthetic_instances = params.get('num_synthetic_instances', n)
    num_weighted_samples = min(n, max_num_weighted_samples)

    if params is None:
        params = {}

    if 'p_range' in params:
        p_range = params['p_range']
        if not hasattr(p_range, '__getitem__'):
            p_range = [p_range, p_range]
    elif 'p' in params:  # Backwards compatibility with prev version:
        p = params.get('p', 2.0)
        p_range = [p, p]
    else:
        p_range = (1.2, 3.0)

    assert len(p_range) == 2, 'p_range must have exactly two elements: (p_min, p_max)'
    p_min = p_range[0]
    p_max = p_range[1]

    b = np.array(range(num_weighted_samples)) + 1.0

    synth_x = np.zeros([num_synthetic_instances] + list(x.shape[1:]))
    synth_y = np.zeros([num_synthetic_instances] + list(y.shape[1:]))

    for i in range(num_synthetic_instances):
        t = float(i) / num_synthetic_instances
        p_val = p_min + t * (p_max - p_min)
        w = 1.0 / np.power(b, p_val)
        w /= sum(w)

        inds = random.sample(range(n), num_weighted_samples)

        synth_x_components = np.array([w[j] * x[inds[j], ...]
                                      for j in range(num_weighted_samples)])
        synth_x[i, ...] = np.sum(synth_x_components, axis=0)

        synth_y_components = np.array([w[j] * y[inds[j], ...]
                                      for j in range(num_weighted_samples)])
        synth_y[i, ...] = np.sum(synth_y_components, axis=0)

    return synth_x, synth_y
