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
val[26:-1] = []
print(len(val))
total = 0
for (i, layer) in enumerate(val):
    print("layer {} (index {}): {} kernels, shape: {}, number of weights: {}".format(i//2+1, i, len(layer), np.shape(layer), np.size(layer)))
    if i <= 25:
        total += np.size(layer)
print("Total amount of weights (fully connected excluded): {}".format(total))


val[0] = np.mean(val[0], axis=1, keepdims=True)


# val_float64 = [layer.astype('float64') for layer in val]


# print(val_float64[0].dtype)

with open("vgg16_only_conv.pkl", 'wb') as f:
    pickle.dump(val, f)

# for (i, kernel) in enumerate(val[0]):
#     print("kernel: {} (index: {}), sum stddev: {}".format(i//2, i, np.mean(np.std(kernel, 0))/np.mean(kernel)))
#     print()
    # print(kernel)

    # break
