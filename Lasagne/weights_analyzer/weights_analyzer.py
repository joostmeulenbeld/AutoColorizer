import os

import numpy as np
import theano
import theano.tensor as T

import pickle

#get all numpy files and pick the first one
# numpy_files = [filename for filename in os.listdir() if filename.endswith('.npy')]
# numpy_file = numpy_files[0]
with open("vgg16.pkl", 'rb') as f:
    a = pickle.load(f, fix_imports=True, encoding='latin-1')

val = a['param values']
total = 0
for (i, layer) in enumerate(val):
    print("layer {} (index {}): {} kernels, shape: {}, number of weights: {}".format(i//2+1, i, len(layer), np.shape(layer), np.size(layer)))
    if i <= 25:
        total += np.size(layer)
print("Total amount of weights (fully connected excluded): {}".format(total))

# for (i, kernel) in enumerate(val[0]):
#     print("kernel: {} (index: {}), sum stddev: {}".format(i//2, i, np.mean(np.std(kernel, 0))/np.mean(kernel)))
#     print()
    # print(kernel)

    # break
