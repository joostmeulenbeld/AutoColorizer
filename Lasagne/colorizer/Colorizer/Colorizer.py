﻿"""
The class file that defines the NN and creates the Theano functions to run it.

author: Dawud Hage, written for the NN course IN4015 of the TUDelft

"""
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

from Customsoftmax import logSoftmax as log_softmax

from NNPreprocessor import assert_colorspace


class Colorizer(object):
    """This class defines the neural network and functions to colorize images"""

    def __init__(self, colorspace, param_file=None, architecture='NN', classification=False, numbins=4, batchsize=10, xpixel = 128):
        """ 
        INPUT:
                colorspace: the colorspace that the NN will use;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
                param_file: the location of the file where all the trained parameters of the network are stored
                            i.e.: 'parameters.npy' or None for random initialization
                architecture: The network architecture to use;
                                'VGG16': use the weights and layers of the VGG16 network as initialization
                                'NN': use with random initialisation as the first network design
        """

        # Create the network here and all the theano functions
        print("Initializing the network")

        # Check if colorspace is properly defined
        assert_colorspace(colorspace)
        self._colorspace = colorspace
        self._classification = classification
        self._numbins = numbins
        self._batch_size = batchsize
        self._x_pixel = xpixel

        self._input = T.tensor4('input')
        self._target = T.tensor4('target') # shape=(batch_size,2,image_x,image_y)
        self._histogram = T.row('histogram')

        # Create the neural network
        print("---Create the neural network")

        if architecture=='VGG16':
            self._network = self._vgg16NN(self._input, reconstruct=1)
        elif architecture=='NN':
            self._network = self._NN(self._input,reconstruct=1)
        elif architecture=='NN_more_end_fmaps':
            self._network = self._NN(self._input,reconstruct=2)
        elif architecture=='zhangNN' :
            self._network = self._zhangNN(self._input)
        elif architecture == 'VGG16_concat_class':
            self._network = self._vgg16NN(self._input, reconstruct=2)
        elif architecture == 'VGG16_dilated_class':
            self._network = self._vgg16NN(self._input, reconstruct=3)
        else:
            raise AttributeError("The provided architecture is unknown.")


        # Set params if given
        if param_file is not None:
            param_load = np.load(param_file)
            lasagne.layers.set_all_param_values(self._network['out'], param_load)
            print("---Loaded param file: {}".format(param_file))

        # Get the output of the network
        print("---Get the output of the network")
        output = lasagne.layers.get_output(self._network['out'])

        # Get the sum squared error per image
        print("---Define loss function")
        if (self._colorspace == 'CIEL*a*b*'):
            # OLD Loss function:
            # Get the sum squared error per image
            loss = lasagne.objectives.squared_error(output,self._target) # shape = (batch_size, 2, image_x, image_y)
            loss = loss.sum(axis=[1,2,3]) # shape (batch_size, 1)
            # And take the mean over the batch
            loss = loss.mean()

            #loss_output = T.sgn(output-0.5)* 2**(abs(output-0.5)) 
            #loss_target = T.sgn(self._target-0.5)* 2**(abs(self._target-0.5)) 
            #loss = lasagne.objectives.squared_error(loss_output, loss_target) # shape = (batch_size, 2, image_x, image_y)
            #loss = loss.sum(axis=[1,2,3]) # shape (batch_size, 1)
            ## And take the mean over the batch
            #loss = loss.mean()
        elif (self._colorspace == 'HSV'):
            ## CHANGE THIS, THE ERROR IS NOW SUMMED OVER THE BATCHES
            # Only on the first layer, the H layer compute the distance.
            # The coordinates are circular so 0 == 1 
            Hx = output[:,0,:,:]
            Hy = self._target[:,0,:,:]

            # The minimum distance on a circle can be one of three things:
            # First if both points closest to eachother rotating from 0/1 CCW on a unit circle
            # Second if point Hx is closer to 0/1 CCW, and point Hy CW
            # Third if point Hy is closer to 0/1 CCW, and point Hx CW
            Hdist = ( T.minimum( abs(Hx - Hy), 1 - T.maximum(Hx,Hy) + T.minimum(Hx,Hy)) )**2

            # On the saturation layer penalize large saturation error! 
            # the 2 can be changes if not saturated enough
            Sx = output[:,1,:,:]
            Sy = self._target[:,1,:,:]
            Sdist = ( 2**(Sx) -  2**(Sy) )**2

            # summaraze to define the loss
            loss = T.sum(Sdist) + T.sum(Hdist)
        elif (self._colorspace == 'YCbCr'):
            # OLD Loss function:
            # Get the sum squared error per image
            loss = lasagne.objectives.squared_error(output,self._target) # shape = (batch_size, 2, image_x, image_y)
            loss = loss.sum(axis=[1,2,3]) # shape (batch_size, 1)
            # And take the mean over the batch
            loss = loss.mean()

            ## New loss function
            #loss_output = T.sgn(output-0.5)* 2**(abs(output-0.5))
            #loss_target = T.sgn(self._target-0.5)* 2**(abs(self._target-0.5))
            #loss = lasagne.objectives.squared_error(loss_output, loss_target) # shape = (batch_size, 2, image_x, image_y)
            #loss = loss.sum(axis=[1,2,3]) # shape (batch_size, 1)
            ## And take the mean over the batch
            #loss = loss.mean()
        elif (self._classification == True):
            #cross entropy loss function: output of the network is [batch,classes,x,y]
            target_dimshuffled = self._target.dimshuffle((0,2,3,1))
            target_reshaped = target_dimshuffled.reshape((target_dimshuffled.shape[0]*self._x_pixel*self._x_pixel,self._numbins))
            value_function = T.sum((target_reshaped * 1/self._histogram),axis = 1)
            
            loss = self.categorical_crossentropy_logdomain(output,target_reshaped)
            # Apply class rebalancing; take the mean
            loss = T.mean(loss * value_function)
            
        else:
            raise ValueError("Cannot handle this colorspace, can only process 'CIEL*a*b*', 'HSV' and 'YCbCr'")

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with adadelta and nesterov momentum.
        print("---Get all trainable parameters")
        params = lasagne.layers.get_all_params(self._network['out'], trainable=True)
        print("--- --- # of parameters: {} ".format(lasagne.layers.count_params(self._network['out'])))
        print("--- --- # of trainable parameters: {} ".format(lasagne.layers.count_params(self._network['out'], trainable=True)))
        print("--- --- # of colorbins: {} ".format(self._numbins))
        print("---Define update function")
        updates = lasagne.updates.adadelta(loss, params, learning_rate=1, rho=0.9, epsilon=1e-06)
        # Add nesterov momentum
        updates = lasagne.updates.apply_nesterov_momentum(updates,params,momentum=0.9)

        # Create theano functions to be used in the functions
        print("---Create the theano functions")
        print("--- ---Create eval_fn")
        self._eval_fn = theano.function([self._input],output)
        print("--- ---Create val_fn")
        if self._classification:
            self._val_fn = theano.function([self._input, self._target, self._histogram],[output, loss])
            print("--- ---Create train_fn")
            self._train_fn = theano.function([self._input, self._target, self._histogram],[output, loss], updates=updates)
        else:
            self._val_fn = theano.function([self._input, self._target],[output, loss])
            print("--- ---Create train_fn")
            self._train_fn = theano.function([self._input, self._target],[output, loss], updates=updates)
            

        # Create an empty dict to store the layer functions in created by 
        self._layer_function = {}

        print("Initialized the network")
        
    def categorical_crossentropy_logdomain(self,log_predictions, targets):
        
        return -T.sum(targets * log_predictions, axis=1)

    def evaluate_NN(self, batch):
        """ 
        INPUT:
                batch: The batch to be evaluated by the NN, batch has shape=(batch_size, image_x, image_y)

        OUTPUT:
                The output of the NN
        """
        if not(len(batch.shape) is 3):
            raise Exception('The input batch does not have the correct shape.')

        # evaluate the network
        return self._eval_fn(batch)

    def validate_NN(self, batch, histogram=np.array([])):
        """ 
        INPUT:
                batch: The batch used to train the network, batch has shape=(batch_size, 3, image_x, image_y)

        OUTPUT:
                A list containing the output of the NN and the validation error        
                i.e. [output, validation_error]
        """
        # split the batch
        batch_input, batch_target = self._split_batch(batch)

        # Train the network
        if self._classification:
            return self._val_fn(batch_input, batch_target, histogram.astype('float32'))
        else:
            return self._val_fn(batch_input, batch_target)

    def train_NN(self, batch, histogram=np.array([])):
        """ 
        INPUT:
                batch: The batch used to train the network, batch has shape=(batch_size, 3, image_x, image_y)

        OUTPUT:
                A list containing the output of the NN and the train error        
                i.e. [output, train_error]
        """
        # split the batch
        batch_input, batch_target = self._split_batch(batch)

        # Train the network
        if self._classification:
            return self._train_fn(batch_input, batch_target, histogram.astype('float32'))
        else:
            return self._train_fn(batch_input, batch_target)

    def save_parameters(self, parameter_file):
        """
        INPUT:
                parameter_file: the filelocation to store the (current) parameters of the NN
                                i.e. 'parameters.npy' 
        """

        np.save(parameter_file,lasagne.layers.get_all_param_values(self._network['out']))
        print("Stored the parameters to file: {}".format(parameter_file))

    def get_layer_output(self, batch, layer_name, classification=False):
        """ Create a theano function that outputs the output of a specific layer specified by layer_name 
        INPUT:
                batch: a batch to forward through the network
                        can be either size=(batch_size,3,image_x,image_y), then the first layer will be used
                        or size=(batch_size,1,image_x,image_y).
                        In both cases only the first image is analysed.
                layer_name: The name of the layer, the key in the dict as specified in the architecture definition 
        OUTPUT: 
                The output of that layer (whatever size it may be..)
               
        """

        # Check if the layer function already exists otherways create it
        if not(layer_name in self._layer_function.keys()):
            print("--- Creating the theano function")
            # Function does not exist, create it
            self._create_layer_output_function(layer_name)
        
        print("--- Evaluate the NN until layer {}".format(layer_name))
        # Split the batch if needed
        if batch.shape[1] == 3:
            if classification == True:
                batch_input = batch[:,0,:,:].reshape((batch.shape[0],1,batch.shape[2],batch.shape[3]))
            else:
                batch_input, _ = self._split_batch(batch)
        elif batch.shape[1] == 1:
            batch_input = batch
        else:
            raise IndexError("The batch size is not correct!")

        # Evaluate the function
        return self._layer_function[layer_name](batch_input)


    @property
    def get_layer_names(self):
        """ Obtain all the layer names as specified in the architecture
        OUTPUT:
                A list of the layer names, sorted alphabetically
        """

        return sorted(self._network.keys())

    ########## Private functions ##########
    def _split_batch(self, batch):
        """ 
        INPUT:
                batch: batch to split into input and target

        OUTPUT:
                a list containing the batch_input and the batch_target: [batch_input, batch_target]
        """
        (batch_size,_,image_x,image_y) = batch.shape
        
        if not self._classification:

            if (self._colorspace == 'CIEL*a*b*') or (self._colorspace == 'YCbCr'):
                # target is the a*b* layers
                batch_target = batch[:,[1,2],:,:]
            elif (self._colorspace == 'HSV'):
                # target is the HS layers
                batch_target = batch[:,[0,1],:,:]
    
            batch_target = batch_target.reshape(batch_size,2,image_x,image_y)
    
            if (self._colorspace == 'CIEL*a*b*') or (self._colorspace == 'YCbCr'):
                # input is the L* layer
                batch_input = batch[:,0,:,:]
            elif (self._colorspace == 'HSV'):
                # input is the V layer
                batch_input = batch[:,2,:,:]
    
            batch_input = batch_input.reshape(batch_size,1,image_x,image_y)
        
        else:
            batch_target = batch[:,1:,:,:]
            batch_target = batch_target.reshape(batch_size,self._numbins,image_x,image_y)
            batch_input = batch[:,0,:,:]
            batch_input = batch_input.reshape(batch_size,1,image_x,image_y)

        return [batch_input, batch_target]

    def _create_layer_output_function(self, layer_name):
        """ Create a theano function that outputs the output of a specific layer specified by layer_name 
            This function will be added to the dict self._layer_function, that is callible by the function get_layer_output
        INPUT:
                layer_name: The name of the layer, the key in the dict as specified in the architecture definition 
               
        """

        # Check if the key exists
        assert layer_name in self._network, "No such key in the NN architecture, given key: {}".format(layer_name)


        output = lasagne.layers.get_output(self._network[layer_name])

        self._layer_function[layer_name] = theano.function([self._input],output)

    def _NN(self, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2, reconstruct=1):
        """ 
        This function defines the architecture of the Fruit colorizer network 
   
        INPUT:
                input_var: a theano.tensor.tensor4 (and after theano function creation the data of size(batch_size, 1, image_size[1], image_size[2])
                image_size: the size of the images, a 2D tuple 
                filter_size: The convolutional filter size, a 2D tuple (EVEN FILTER SIZE IS NOT SUPPORTED!)
                pool_size: the max_pool filter size between layers (also the upscale factor!)
        OUTPUT:
                Last lasagne layer of the network
        """

        # First make the input layer, batch size=none so any batch size will be accepted!
        # image size is 128x128
        network = {}
        network['input'] = lasagne.layers.InputLayer(shape=(None, 1, 128, 128), input_var=input_var)

        # Define the first layer
        # Pad = same for keeping the dimensions equal to the input!
        network['conv1']        = lasagne.layers.Conv2DLayer(network['input'] , num_filters=3, filter_size=filter_size, pad='same')
        network['conv2']        = lasagne.layers.Conv2DLayer(network['conv1'], num_filters=12, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm1']  = lasagne.layers.batch_norm(network['conv2'])

        # Define the second layer
        # Max pool on first layer
        network['max_pool1']    = lasagne.layers.MaxPool2DLayer(network['batch_norm1'], pool_size=pool_size)
        network['conv3']        = lasagne.layers.Conv2DLayer(network['max_pool1'], num_filters=48, filter_size=filter_size, pad='same')
        network['conv4']        = lasagne.layers.Conv2DLayer(network['conv3'], num_filters=48, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm2']  = lasagne.layers.batch_norm(network['conv4'])

        # Define the third layer
        # Max pool on the second layer
        network['max_pool2']    = lasagne.layers.MaxPool2DLayer(network['batch_norm2'], pool_size=pool_size)
        network['conv5']        = lasagne.layers.Conv2DLayer(network['max_pool2'], num_filters=96, filter_size=filter_size, pad='same')
        network['conv6']        = lasagne.layers.Conv2DLayer(network['conv5'], num_filters=96, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm3']  = lasagne.layers.batch_norm(network['conv6'])

        # Define the fourth layer
        # Max pool on the second layer
        network['max_pool3']    = lasagne.layers.MaxPool2DLayer(network['batch_norm3'], pool_size=pool_size)
        network['conv7']        = lasagne.layers.Conv2DLayer(network['max_pool3'], num_filters=192, filter_size=filter_size, pad='same')
        network['conv8']   = lasagne.layers.Conv2DLayer(network['conv7'], num_filters=192, filter_size=filter_size, pad='same')
        network['conv_final']        = lasagne.layers.Conv2DLayer(network['conv8'], num_filters=96, filter_size=(1,1), pad='same')

        if (reconstruct == 1):
            return self._reconstructNN(network, input_var=input_var, image_size=image_size, filter_size=filter_size, pool_size=pool_size)
        elif(reconstruct == 2):
            return self._reconstructNN2(network, input_var=input_var, image_size=image_size, filter_size=filter_size, pool_size=pool_size)
        else:
            raise AttributeError("Unknown reconstruct id")
            
            
    def _zhangNN(self, input_var=None, image_size=(128, 128), filter_size = (3, 3), stride = (2, 2)):
        """ 
        This function defines the architecture of the Fruit colorizer network as defined by Zhang et al
   
        INPUT:
                input_var: a theano.tensor.tensor4 (and after theano function creation the data of size(batch_size, 1, image_size[1], image_size[2])
                image_size: the size of the images, a 2D tuple 
                filter_size: The convolutional filter size, a 2D tuple (EVEN FILTER SIZE IS NOT SUPPORTED!)
                stride: the stride between the layers, also the upscale factor
        OUTPUT:
                Last lasagne layer of the network
        """
        
        # First make the input layer, batch size=none so any batch size will be accepted!
        # image size is 128x128
        network = {}
        network['input'] = lasagne.layers.InputLayer(shape=(None, 1, 128, 128), input_var=input_var)

        # Define the first layer
        # Pad = same for keeping the dimensions equal to the input!
        network['conv1']        = lasagne.layers.Conv2DLayer(network['input'] , num_filters=3, filter_size=filter_size, pad='same')
        network['conv2']        = lasagne.layers.Conv2DLayer(network['conv1'], num_filters=12, filter_size=filter_size, pad='same')
        network['conv3']        = lasagne.layers.Conv2DLayer(network['conv2'], num_filters=48, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm1']  = lasagne.layers.batch_norm(network['conv3'])

        # Define the second layer. input resolution=128*128, output resolution=64*64
        network['conv4']        = lasagne.layers.Conv2DLayer(network['batch_norm1'], num_filters=48, filter_size=filter_size, stride=stride, pad='same')
        network['conv5']        = lasagne.layers.Conv2DLayer(network['conv4'], num_filters=48, filter_size=filter_size, pad='same')
        network['conv6']        = lasagne.layers.Conv2DLayer(network['conv5'], num_filters=48, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm2']  = lasagne.layers.batch_norm(network['conv6'])

        # Define the third layer. input resolution=64*64, output resolution=32*32
        network['conv7']    = lasagne.layers.Conv2DLayer(network['batch_norm2'], num_filters=96, filter_size=filter_size, stride=stride, pad='same')
        network['conv8']        = lasagne.layers.Conv2DLayer(network['conv7'], num_filters=96, filter_size=filter_size, pad='same')
        # Take the batch norm
        network['batch_norm3']  = lasagne.layers.batch_norm(network['conv8'])

        # Define the fourth layer, first dilated layer. input resolution=32*32, output resolution=64*64
        network['conv9']        = lasagne.layers.Conv2DLayer(network['batch_norm3'], num_filters=128, filter_size=filter_size, pad='same')
        network['pad1']     = lasagne.layers.PadLayer(network['conv9'], width=(3,3))
        network['conv10']   = lasagne.layers.DilatedConv2DLayer(network['pad1'], num_filters=128, filter_size=filter_size, dilation=(3,3))
        network['pad2']     = lasagne.layers.PadLayer(network['conv10'], width=(7,7))
        network['conv11']   = lasagne.layers.DilatedConv2DLayer(network['pad2'], num_filters=128, filter_size=filter_size, dilation=(7,7))
        # Take the batch norm
        network['batch_norm4']  = lasagne.layers.batch_norm(network['conv11'])
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['batch_norm4'], scale_factor=2)
        
        # Define the fifth layer, first upscale layer, second dilated layer. input resolution=64*64, output resolution=128*128
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale1']])
        network['pad3']     = lasagne.layers.PadLayer(network['re_concat1'],width=(3,3))
        network['conv12']   = lasagne.layers.DilatedConv2DLayer(network['pad3'], num_filters=96, filter_size=filter_size, dilation=(3,3))
        network['pad4']     = lasagne.layers.PadLayer(network['conv12'],width=(7,7))
        network['conv13']   = lasagne.layers.DilatedConv2DLayer(network['pad4'], num_filters=96, filter_size=filter_size, dilation=(7,7))
        network['pad5']     = lasagne.layers.PadLayer(network['conv13'],width=(15,15))
        network['conv14']   = lasagne.layers.DilatedConv2DLayer(network['pad5'], num_filters=96, filter_size=filter_size, dilation=(15,15))
        # Take the batch norm
        network['batch_norm5']  = lasagne.layers.batch_norm(network['conv14'])
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['batch_norm5'], scale_factor=2)
        
        # Define the sixth and final layer,
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale2']])    
        network['out1'] = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=self._numbins, filter_size=(1,1), pad='same', nonlinearity = None)
        network['outdimshuffled'] = lasagne.layers.DimshuffleLayer(network['out1'],(0,2,3,1))
        network['outreshaped'] = lasagne.layers.ReshapeLayer(network['outdimshuffled'],((input_var.shape[0]*self._x_pixel*self._x_pixel,self._numbins)))
        network['out'] = log_softmax(network['outreshaped'])
        
        return network
        
        
        

    def _vgg16NN(self, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2, trainable=False, reconstruct=1):
        # Build the convolutional layers according to the VGG16 structure
        # after every convolutional block, before maxpool, the batch norm is taken but not further used in the VGG16 network
        # This is because the reconstruct network requires normalized layers
        #
        # adapted from https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
        # Only implement the first 4 convolutional blocks since our image is smaller
        try:
            f = open("vgg16_only_conv.pkl", 'rb')
            a = pickle.load(f)
        except:
            print("Couldn't load vgg16 weights pickle file")
            raise
            
        a = [layer.astype(theano.config.floatX) for layer in a]
        network = {}
        
        network['input'] = lasagne.layers.InputLayer((None, 1, 128, 128), input_var=input_var)
        
        network['conv1_1'] = lasagne.layers.Conv2DLayer(network['input'], 64, 3, pad='same', flip_filters=False, W=a[0], b=a[1])
        network['conv1_2'] = lasagne.layers.Conv2DLayer(network['conv1_1'], 64, 3, pad='same', flip_filters=False, W=a[2], b=a[3])
        network['conv1_2_copy'] = lasagne.layers.Conv2DLayer(network['conv1_1'], 64, 3, pad='same', flip_filters=False, W=a[2], b=a[3])
        network['batch_norm1'] = lasagne.layers.batch_norm(network['conv1_2_copy']) #lasagne.layers.batch_norm(network['conv1_2'])
        network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1_2'], 2)
        
        network['conv2_1'] = lasagne.layers.Conv2DLayer(network['pool1'], 128, 3, pad='same', flip_filters=False, W=a[4], b=a[5])
        network['conv2_2'] = lasagne.layers.Conv2DLayer(network['conv2_1'], 128, 3, pad='same', flip_filters=False, W=a[6], b=a[7])
        network['conv2_2_copy'] = lasagne.layers.Conv2DLayer(network['conv2_1'], 128, 3, pad='same', flip_filters=False, W=a[6], b=a[7])
        network['batch_norm2'] = lasagne.layers.batch_norm(network['conv2_2_copy']) #lasagne.layers.batch_norm(network['conv2_2'])
        network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2_2'], 2)
        
        network['conv3_1'] = lasagne.layers.Conv2DLayer(network['pool2'], 256, 3, pad='same', flip_filters=False, W=a[8], b=a[9])
        network['conv3_2'] = lasagne.layers.Conv2DLayer(network['conv3_1'], 256, 3, pad='same', flip_filters=False, W=a[10], b=a[11])
        network['conv3_3'] = lasagne.layers.Conv2DLayer(network['conv3_2'], 256, 3, pad='same', flip_filters=False, W=a[12], b=a[13])
        network['conv3_3_copy'] = lasagne.layers.Conv2DLayer(network['conv3_2'], 256, 3, pad='same', flip_filters=False, W=a[12], b=a[13])
        network['batch_norm3'] = lasagne.layers.batch_norm(network['conv3_3_copy']) #lasagne.layers.batch_norm(network['conv3_3'])
        network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3_3'], 2)
        
        network['conv4_1'] = lasagne.layers.Conv2DLayer(network['pool3'], 512, 3, pad='same', flip_filters=False, W=a[14], b=a[15])
        network['conv4_2'] = lasagne.layers.Conv2DLayer(network['conv4_1'], 512, 3, pad='same', flip_filters=False, W=a[16], b=a[17])
        network['conv4_3'] = lasagne.layers.Conv2DLayer(network['conv4_2'], 512, 3, pad='same', flip_filters=False, W=a[18], b=a[19])
        network['conv_final'] = lasagne.layers.Conv2DLayer(network['conv4_3'], num_filters=256, filter_size=(1,1), pad='same') #in the reconstruct network, a batch norm is taken so it doesn't have to happen here
        
        #Make all trainable parameters non-trainable
        if not trainable:
            for key in sorted(network):
                if "batch_norm" not in key:
                    for param in network[key].get_params(trainable=True):
                        network[key].params[param].remove('trainable')
                
        # network['pool4'] = lasagne.layers.PoolLayer(network['conv4_3'], 2)
        # 
        # network['conv5_1'] = lasagne.layers.ConvLayer(network['pool4'], 512, 3, pad='same', flip_filters=False)
        # network['conv5_2'] = lasagne.layers.ConvLayer(network['conv5_1'], 512, 3, pad='same', flip_filters=False)
        # network['conv5_3'] = lasagne.layers.ConvLayer(network['conv5_2'], 512, 3, pad='same', flip_filters=False)
        # network['pool5'] = lasagne.layers.PoolLayer(network['conv5_3'], 2)
        if (reconstruct == 1):
            return self._reconstructVGG16NN(network, input_var=input_var, image_size=image_size, filter_size=filter_size, pool_size=pool_size)
        elif(reconstruct == 2):
            return self._reconstructVGG16NN_concat_class(network, input_var=input_var, image_size=image_size, filter_size=filter_size, pool_size=pool_size)
        elif(reconstruct == 3):
            return self._reconstructVGG16NN_dilated_class(network, input_var=input_var, image_size=image_size, filter_size=filter_size, pool_size=pool_size)
        else:
            raise AttributeError("Unknown reconstruct id")
        
    def _reconstructVGG16NN_concat_class(self, network, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        ## Now Construct the image again!
        # Get the batch norm and reduce feature maps to fit previous layer
        network['re_batch_norm1']  = lasagne.layers.batch_norm(network['conv_final'])
        # Upscale layer 3 to fit L2 size
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm1'], scale_factor=pool_size)

        # Concate with L_3
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm3'], network['re_Upscale1']])
        network['re_conv2']       = lasagne.layers.Conv2DLayer(network['re_concat1'], num_filters=256, filter_size=filter_size, pad='same')
        network['re_conv3']       = lasagne.layers.Conv2DLayer(network['re_conv2'], num_filters=128, filter_size=filter_size, pad='same')
        network['re_batch_norm2']  = lasagne.layers.batch_norm(network['re_conv3'])
        # Upscale L_3 to fit L_2 size
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm2'], scale_factor=pool_size)

        # Concate with L_2
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale2']])
        # Convolve L_2 to fit feature maps to L1
        network['re_conv4']       = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=128, filter_size=filter_size, pad='same')
        network['re_conv5']       = lasagne.layers.Conv2DLayer(network['re_conv4'], num_filters=64, filter_size=filter_size, pad='same')
        network['re_batch_norm3']  = lasagne.layers.batch_norm(network['re_conv5'])
        # Upscale L_2 to fit L_1 size
        network['re_Upscale3']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm3'], scale_factor=pool_size)
    
        # Do the same for layer 1
        network['re_concat3']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale3']])
        # Convolve L_1 to fit feature maps to L1
        network['re_conv6']       = lasagne.layers.Conv2DLayer(network['re_concat3'], num_filters=64, filter_size=filter_size, pad='same')
        network['re_conv7']       = lasagne.layers.Conv2DLayer(network['re_conv6'], num_filters=64, filter_size=filter_size, pad='same')
    

        # Convolve L_1 to fit the desired output
        network['out1'] = lasagne.layers.Conv2DLayer(network['re_conv7'], num_filters=self._numbins, filter_size=(1,1), pad='same', nonlinearity = None)
        network['outdimshuffled'] = lasagne.layers.DimshuffleLayer(network['out1'],(0,2,3,1))
        network['outreshaped'] = lasagne.layers.ReshapeLayer(network['outdimshuffled'],((input_var.shape[0]*self._x_pixel*self._x_pixel,self._numbins)))
        network['out'] = log_softmax(network['outreshaped'])
        
        return network
        
    def _reconstructVGG16NN_dilated_class(self, network, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        
             
        # Define the fourth layer, first dilated layer. input resolution=32*32, output resolution=64*64
        network['conv9']        = lasagne.layers.Conv2DLayer(network['batch_norm3'], num_filters=128, filter_size=filter_size, pad='same')
        network['pad1']     = lasagne.layers.PadLayer(network['conv9'], width=(3,3))
        network['conv10']   = lasagne.layers.DilatedConv2DLayer(network['pad1'], num_filters=128, filter_size=filter_size, dilation=(3,3))
        network['pad2']     = lasagne.layers.PadLayer(network['conv10'], width=(7,7))
        network['conv11']   = lasagne.layers.DilatedConv2DLayer(network['pad2'], num_filters=128, filter_size=filter_size, dilation=(7,7))
        # Take the batch norm
        network['batch_norm4']  = lasagne.layers.batch_norm(network['conv11'])
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['batch_norm4'], scale_factor=2)
        
        # Define the fifth layer, first upscale layer, second dilated layer. input resolution=64*64, output resolution=128*128
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale1']])
        network['pad3']     = lasagne.layers.PadLayer(network['re_concat1'],width=(3,3))
        network['conv12']   = lasagne.layers.DilatedConv2DLayer(network['pad3'], num_filters=96, filter_size=filter_size, dilation=(3,3))
        network['pad4']     = lasagne.layers.PadLayer(network['conv12'],width=(7,7))
        network['conv13']   = lasagne.layers.DilatedConv2DLayer(network['pad4'], num_filters=96, filter_size=filter_size, dilation=(7,7))
        network['pad5']     = lasagne.layers.PadLayer(network['conv13'],width=(15,15))
        network['conv14']   = lasagne.layers.DilatedConv2DLayer(network['pad5'], num_filters=96, filter_size=filter_size, dilation=(15,15))
        # Take the batch norm
        network['batch_norm5']  = lasagne.layers.batch_norm(network['conv14'])
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['batch_norm5'], scale_factor=2)
        
        # Define the sixth and final layer,
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale2']])    
        network['out1'] = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=self._numbins, filter_size=(1,1), pad='same', nonlinearity = None)
        network['outdimshuffled'] = lasagne.layers.DimshuffleLayer(network['out1'],(0,2,3,1))
        network['outreshaped'] = lasagne.layers.ReshapeLayer(network['outdimshuffled'],((input_var.shape[0]*self._x_pixel*self._x_pixel,self._numbins)))
        network['out'] = log_softmax(network['outreshaped'])

        return network
   
    def _reconstructNN(self, network, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        
        ## Now Construct the image again!
        # Get the batch norm and reduce feature maps to fit previous layer
        network['re_batch_norm1']  = lasagne.layers.batch_norm(network['conv_final'])
        # Upscale layer 3 to fit L2 size
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm1'], scale_factor=pool_size)

        # Concate with L_3
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm3'], network['re_Upscale1']])
        network['re_conv2']       = lasagne.layers.Conv2DLayer(network['re_concat1'], num_filters=96, filter_size=filter_size, pad='same')
        network['re_conv3']       = lasagne.layers.Conv2DLayer(network['re_conv2'], num_filters=48, filter_size=filter_size, pad='same')
        network['re_batch_norm2']  = lasagne.layers.batch_norm(network['re_conv3'])
        # Upscale L_3 to fit L_2 size
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm2'], scale_factor=pool_size)

        # Concate with L_2
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale2']])
        # Convolve L_2 to fit feature maps to L1
        network['re_conv4']       = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=48, filter_size=filter_size, pad='same')
        network['re_conv5']       = lasagne.layers.Conv2DLayer(network['re_conv4'], num_filters=12, filter_size=filter_size, pad='same')
        network['re_batch_norm3']  = lasagne.layers.batch_norm(network['re_conv5'])
        # Upscale L_2 to fit L_1 size
        network['re_Upscale3']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm3'], scale_factor=pool_size)
    
        # Do the same for layer 1
        network['re_concat3']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale3']])
        # Convolve L_1 to fit feature maps to L1
        network['re_conv6']       = lasagne.layers.Conv2DLayer(network['re_concat3'], num_filters=12, filter_size=filter_size, pad='same')
        network['re_conv7']       = lasagne.layers.Conv2DLayer(network['re_conv6'], num_filters=6, filter_size=filter_size, pad='same')
        network['re_conv8']       = lasagne.layers.Conv2DLayer(network['re_conv7'], num_filters=3, filter_size=filter_size, pad='same')
    

        # Convolve L_1 to fit the desired output
        network['out'] = lasagne.layers.Conv2DLayer(network['re_conv8'], num_filters=2, filter_size=filter_size, pad='same', 
                                           nonlinearity=lasagne.nonlinearities.linear)
            
        return network

    def _reconstructNN2(self, network, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        
        ## Now Construct the image again!
        # Get the batch norm and reduce feature maps to fit previous layer
        network['re_batch_norm1']  = lasagne.layers.batch_norm(network['conv_final'])
        # Upscale layer 3 to fit L2 size
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm1'], scale_factor=pool_size)

        # Concate with L_3
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm3'], network['re_Upscale1']])
        network['re_conv2']       = lasagne.layers.Conv2DLayer(network['re_concat1'], num_filters=96, filter_size=filter_size, pad='same')
        network['re_conv3']       = lasagne.layers.Conv2DLayer(network['re_conv2'], num_filters=48, filter_size=filter_size, pad='same')
        network['re_batch_norm2']  = lasagne.layers.batch_norm(network['re_conv3'])
        # Upscale L_3 to fit L_2 size
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm2'], scale_factor=pool_size)

        # Concate with L_2
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale2']])
        # Convolve L_2 to fit feature maps to L1
        network['re_conv4']       = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=48, filter_size=filter_size, pad='same')
        network['re_conv5']       = lasagne.layers.Conv2DLayer(network['re_conv4'], num_filters=12, filter_size=filter_size, pad='same')
        network['re_batch_norm3']  = lasagne.layers.batch_norm(network['re_conv5'])
        # Upscale L_2 to fit L_1 size
        network['re_Upscale3']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm3'], scale_factor=pool_size)
    
        # Do the same for layer 1
        network['re_concat3']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale3']])
        # Convolve L_1 to fit feature maps to L1
        network['re_conv6']       = lasagne.layers.Conv2DLayer(network['re_concat3'], num_filters=24, filter_size=filter_size, pad='same')
        network['re_conv7']       = lasagne.layers.Conv2DLayer(network['re_conv6'], num_filters=24, filter_size=filter_size, pad='same')    

        # Convolve L_1 to fit the desired output
        network['out'] = lasagne.layers.Conv2DLayer(network['re_conv7'], num_filters=2, filter_size=filter_size, pad='same', 
                                           nonlinearity=lasagne.nonlinearities.linear)
        return network

    def _reconstructVGG16NN(self, network, input_var=None, image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        
        ## Now Construct the image again!
        # Get the batch norm and reduce feature maps to fit previous layer
        network['re_batch_norm1']  = lasagne.layers.batch_norm(network['conv_final'])
        # Upscale layer 3 to fit L2 size
        network['re_Upscale1']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm1'], scale_factor=pool_size)

        # Concate with L_3
        network['re_concat1']      = lasagne.layers.concat([network['batch_norm3'], network['re_Upscale1']])
        network['re_conv2']       = lasagne.layers.Conv2DLayer(network['re_concat1'], num_filters=256, filter_size=filter_size, pad='same')
        network['re_conv3']       = lasagne.layers.Conv2DLayer(network['re_conv2'], num_filters=128, filter_size=filter_size, pad='same')
        network['re_batch_norm2']  = lasagne.layers.batch_norm(network['re_conv3'])
        # Upscale L_3 to fit L_2 size
        network['re_Upscale2']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm2'], scale_factor=pool_size)

        # Concate with L_2
        network['re_concat2']      = lasagne.layers.concat([network['batch_norm2'], network['re_Upscale2']])
        # Convolve L_2 to fit feature maps to L1
        network['re_conv4']       = lasagne.layers.Conv2DLayer(network['re_concat2'], num_filters=128, filter_size=filter_size, pad='same')
        network['re_conv5']       = lasagne.layers.Conv2DLayer(network['re_conv4'], num_filters=64, filter_size=filter_size, pad='same')
        network['re_batch_norm3']  = lasagne.layers.batch_norm(network['re_conv5'])
        # Upscale L_2 to fit L_1 size
        network['re_Upscale3']     = lasagne.layers.Upscale2DLayer(network['re_batch_norm3'], scale_factor=pool_size)
    
        # Do the same for layer 1
        network['re_concat3']      = lasagne.layers.concat([network['batch_norm1'], network['re_Upscale3']])
        # Convolve L_1 to fit feature maps to L1
        network['re_conv6']       = lasagne.layers.Conv2DLayer(network['re_concat3'], num_filters=64, filter_size=filter_size, pad='same')
        network['re_conv7']       = lasagne.layers.Conv2DLayer(network['re_conv6'], num_filters=64, filter_size=filter_size, pad='same')
    

        # Convolve L_1 to fit the desired output
        network['out'] = lasagne.layers.Conv2DLayer(network['re_conv7'], num_filters=2, filter_size=filter_size, pad='same', 
                                           nonlinearity=lasagne.nonlinearities.linear)
        return network
        

