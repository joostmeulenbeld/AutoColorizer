from NNPreprocessor import NNPreprocessor
from time import time
import numpy as np
from Colorizer import Colorizer

# Number of epochs to train the network
n_epoch = 2

# Load data
train_data = NNPreprocessor(batch_size=10, folder='flower_training', random_superbatches=True, blur=True, randomize=True)
validation_data = NNPreprocessor(batch_size=10, folder='flower_validation', random_superbatches=True, blur=False, randomize=True)

# Create network object
param_file = None #'params_landscape.npy'
NNColorizer = Colorizer(param_file=param_file)

prev_epoch = 0

# keep track of time
start_time = time()

# Keep track of the training error
train_error = 0

while train_data.get_epoch < n_epoch:
    
    _, error = NNColorizer.train_NN(train_data.get_batch)
    train_error += error

    print("Progress of the training: {:3.1f}%            \r".format(train_data.get_epochProgress),end="")

    if train_data.get_epoch_done:

        # Keep track of the validation error
        validation_error = 0

        # Determine the validation error
        while not(validation_data.get_epoch_done):
            # validate the network
            _, error = NNColorizer.validate_NN(validation_data.get_batch)
            validation_error += error
            print("Progress of the validation: {:3.1f}%            \r".format(validation_data.get_epochProgress),end="")

        # print the results for this epoch:
        print("---------------------------------------")
        print("Epoch {} of {} took {:.3f}s".format(train_data.get_epoch, n_epoch, time() - start_time))
        print("The last train error is: {!s:}".format(train_error / train_data.get_n_batches ))
        print("Validation error: {!s:}".format(validation_error / validation_data.get_n_batches ))

        # reset time 
        start_time = time()
        
































# "Make sure that the main module can be safely imported by a new Python interpreter without causing unintended side effects (such a starting a new process)."
# Needed to run on windows.....
#if __name__ == '__main__': 
#    train_data = NNPreprocessor(1000, "fruit_training", blur=True)
#    batch = train_data.get_batch()

#    print("Minimum L = {}".format(np.amin(np.amin(batch[:,0,:,:],(1,2))) ))
#    print("Maximum L = {}".format(np.amax(np.amax(batch[:,0,:,:],(1,2))) ))
#    print("Minimum a = {}".format(np.amin(np.amin(batch[:,1,:,:],(1,2))) ))
#    print("Maximum a = {}".format(np.amax(np.amax(batch[:,1,:,:],(1,2))) ))
#    print("Minimum b = {}".format(np.amin(np.amin(batch[:,2,:,:],(1,2))) ))
#    print("Maximum b = {}".format(np.amax(np.amax(batch[:,2,:,:],(1,2))) ))

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






