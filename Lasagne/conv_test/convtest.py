import numpy as np
import theano
import theano.tensor as T
import lasagne



def fruityfly(input_var=None):

    """ SETTINGS: """
    filter_size = (3, 3) # The convolutional filter size (EVEN FILTER SIZE IS NOT SUPPORTED!)
    pool_size = 2 # The pool size between layers

    # First make the input layer, batch size=none so any batch size will be accepted!
    # image size is 128x128
    L_input = lasagne.layers.InputLayer(shape=(None, 1, 128, 128), input_var=input_var)

    # Define the first layer
    # Pad = same for keeping the dimensions equal to the input!
    L_1 = lasagne.layers.Conv2DLayer(L_input, num_filters=3, filter_size=filter_size, pad='same')

    # Define the second layer
    L_2 = lasagne.layers.Conv2DLayer(L_1, num_filters=12, filter_size=filter_size, pad='same')
    # Max pool on second layer
    L_2 = lasagne.layers.MaxPool2DLayer(L_2, pool_size=pool_size)

    # Define the third layer
    L_3 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')
    # Max pool on the third layer
    L_3 = lasagne.layers.MaxPool2DLayer(L_3, pool_size=pool_size)

    # Get the batch norm and reduce feature maps to fit previous layer
    L_3 = lasagne.layers.batch_norm(L_3)
    L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=12, filter_size=(1,1), pad='same')
    # Upscale layer 3 to fit L2 size
    L_3 = lasagne.layers.Upscale2DLayer(L_3, scale_factor=pool_size)

    # Get the batch norm of L_2 and concate with L_3
    L_2 = lasagne.layers.batch_norm(L_2)
    L_2 = lasagne.layers.concat([L_2, L_3])
    # Convolve L_2 to fit feature maps to L1
    L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=3, filter_size=filter_size, pad='same')
    # Upscale L_2 to fit L_1 size
    L_2 = lasagne.layers.Upscale2DLayer(L_2, scale_factor=pool_size)
    
    # Do the same for layer 1
    L_1 = lasagne.layers.batch_norm(L_1)
    L_1 = lasagne.layers.concat([L_1, L_2])
    # Convolve L_1 to fit feature maps to L1
    L_1 = lasagne.layers.Conv2DLayer(L_1, num_filters=3, filter_size=filter_size, pad='same')
    

    # Convolve L_1 to fit the desired output
    L_out = lasagne.layers.Conv2DLayer(L_1, num_filters=2, filter_size=filter_size, pad='same')

    return L_out
