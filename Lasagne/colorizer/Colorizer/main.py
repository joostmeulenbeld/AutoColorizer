from NNPreprocessor import NNPreprocessor
from Colorizer import Colorizer

train_data = NNPreprocessor(10,'testing_files')
#validation_data = NNPreprocessor(10,'validation')

# Load network + add param
param_file = None #'params_landscape.npy'
NNColorizer = Colorizer(param_file=param_file)

for i in range(2):
    _, train_error = NNColorizer.train_NN(train_data.get_batch())

    print(train_error)







