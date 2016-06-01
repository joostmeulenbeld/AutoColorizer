from NNPreprocessor import NNPreprocessor
from time import time
#from Colorizer import Colorizer


# "Make sure that the main module can be safely imported by a new Python interpreter without causing unintended side effects (such a starting a new process)."
# Needed to run on windows.....
if __name__ == '__main__': 
    train_data = NNPreprocessor(5, "testing_files", blur=True)

    #validation_data = NNPreprocessor(10,'validation')

    # Load network + add param
    param_file = None #'params_landscape.npy'
    #NNColorizer = Colorizer(param_file=param_file)

    for i in range(10):
        #_, train_error = NNColorizer.train_NN(train_data.get_batch())

        #print(train_error)
        t = time()
        print(train_data.get_batch().shape)
        print("get_batch in: {} [s]".format(time()-t))





