from NNPreprocessor import NNPreprocessor
from time import time
import numpy as np
#from Colorizer import Colorizer


# "Make sure that the main module can be safely imported by a new Python interpreter without causing unintended side effects (such a starting a new process)."
# Needed to run on windows.....
if __name__ == '__main__': 
    train_data = NNPreprocessor(1000, "fruit_training", blur=True)
    batch = train_data.get_batch()

    print("Minimum L = {}".format(np.amin(np.amin(batch[:,0,:,:],(1,2))) ))
    print("Maximum L = {}".format(np.amax(np.amax(batch[:,0,:,:],(1,2))) ))
    print("Minimum a = {}".format(np.amin(np.amin(batch[:,1,:,:],(1,2))) ))
    print("Maximum a = {}".format(np.amax(np.amax(batch[:,1,:,:],(1,2))) ))
    print("Minimum b = {}".format(np.amin(np.amin(batch[:,2,:,:],(1,2))) ))
    print("Maximum b = {}".format(np.amax(np.amax(batch[:,2,:,:],(1,2))) ))

#validation_data = NNPreprocessor(10,'validation')

## Load network + add param
#param_file = None #'params_landscape.npy'
#NNColorizer = Colorizer(param_file=param_file)

#for i in range(100):
#    t = time()
#    _, train_error = NNColorizer.train_NN(train_data.get_batch())
#    print(train_error)
#    print("training one batch took: {} [s]".format(time()-t))   

    # TODO: get #epoch, get progress through data
    #           Visualize output!
    #           save parameters
    #           save errors






