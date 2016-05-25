import os

import numpy as np
import theano
import theano.tensor as T

#get all numpy files and pick the first one
numpy_files = [filename for filename in os.listdir() if filename.endswith('.npy')]
numpy_file = numpy_files[0]
