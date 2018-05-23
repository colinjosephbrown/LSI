import numpy as np
import random

def lsi(x, y, params={}):
	assert isinstance(x, np.ndarray), 'Input x must be a Numpy array'
	assert isinstance(y, np.ndarray), 'Input y must by a Numpy array'

	n = x.shape[0]

	p = params.get('p', 2.0)
	max_num_weighted_samples = params.get('max_num_weighted_samples', np.inf)
	num_synthetic_instances = params.get('num_synthetic_instances', n)

	num_weighted_samples = min(n, max_num_weighted_samples)
	w = np.array([1 / np.power(j, p) for j in range(num_weighted_samples)])
	w = w / float(sum(w));

	synth_x = np.zeros([num_synthetic_instances] + list(x.shape[1:]))
	synth_y = np.zeros([num_synthetic_instances] + list(y.shape[1:]))

	for i in range(num_synthetic_instances):
		inds = random.sample(range(n), num_weighted_samples)

		synth_x[i, ...] = sum([ w(j) * x[inds[j], ...] for j in range(num_weighted_samples) ], axis=0)
		synth_y[i, ...] = sum([ w(j) * y[inds[j], ...] for j in range(num_weighted_samples) ], axis=0) 

	return synth_x, synth_y
