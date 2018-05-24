# Colin Joseph Brown, 2018
import numpy as np
import random

def lsi(x, y, params={}):
"""
An implementation of the local synthetic instances (LSI) method for over-sampling a dataset.

Arguments:
x - An N by M numpy array containing N samples of M dimensions (i.e. features)
y - An N by D numpy array containing N labels of D dimensions
params - A dictionary containing method parameters including:
	num_synthetic instances - The number of synthetic samples to generate (default=N)	
	max_num_weighted_samples - The maximum number of real samples to interpolate between (default=inf)
	p - The exponent defining how much to trust any one sample - see [1] for details (default=2)

[1] Brown, Colin J., et al. "Prediction of motor function in very preterm infants using connectome features and local synthetic instances." MICCAI, 2015.
"""
    assert isinstance(x, np.ndarray), 'Input x must be a Numpy array'
    assert isinstance(y, np.ndarray), 'Input y must by a Numpy array'
    
    n = x.shape[0]
    
    p = params.get('p', 2.0)
    max_num_weighted_samples = params.get('max_num_weighted_samples', np.inf)
    num_synthetic_instances = params.get('num_synthetic_instances', n)
    
    num_weighted_samples = min(n, max_num_weighted_samples)
    
    w = np.array([1 / np.power(j+1, p) for j in range(num_weighted_samples)])
    w = w * 1.0 / float(sum(w))
    
    synth_x = np.zeros([num_synthetic_instances] + list(x.shape[1:]))
    synth_y = np.zeros([num_synthetic_instances] + list(y.shape[1:]))
    
    for i in range(num_synthetic_instances):
        inds = random.sample(range(n), num_weighted_samples)
        
        synth_x_components = np.array([ w[j] * x[inds[j], ...] 
                                        for j in range(num_weighted_samples)])
        synth_x[i, ...] = np.sum(synth_x_components, axis=0)
    
        synth_y_components = np.array([ w[j] * y[inds[j], ...] 
                                        for j in range(num_weighted_samples)])
        synth_y[i, ...] = np.sum(synth_y_components, axis=0) 
    
    return synth_x, synth_y
