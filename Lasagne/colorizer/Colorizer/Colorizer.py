import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image

import matplotlib.pyplot as plot

import sys
import os
import time
import logging

class colorizer(object):

    def __init__(self, param_file=None, logging_file='log.log'):

        # Create the network here and all the theano functions
        print("Initializing the network")

        input = T.tensor4('input')
        target = T.tensor4('target') # shape=(batch_size,2,image_x,image_y)

        # Create the neural network
        self.network = self.fruityfly(input)

        # Set weights if given
        if not(param_file is None):
            param_load = np.load(param_file)
            lasagne.layers.set_all_param_values(self.network, param_load)
            print("Loaded param file: {}".format(param_file))

        # Get the output of the network
        output = lasagne.layers.get_output(self.network)

        # Get the sum squared error per image
        loss = lasagne.objectives.squared_error(output,target)
        loss = loss.sum(axis=[1,2,3])
        # And take the mean over the batch
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with adadelta.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adadelta(loss, params, learning_rate=1, rho=0.9, epsilon=1e-06)#, momentum=0.9)

        # Create theano functions to be used in the functions
        self.eval_fn = theano.function([input],output)
        self.val_fn = theano.function([input, target],[output, loss])
        self.train_fn = theano.function([input, target],[output, loss], updates=updates)

        print("Initialized the network")


    def train_network(self,n_epoch = 20 , n_batches = 100, n_validation_batches = 2, param_save_file=None):

        print("Start training! \n")

        for epoch in range(n_epoch):
            # Set errors to zero for this epoch
            train_error = 0
            validation_error = 0

            # Keep track of time
            start_time = time.time()

            ######### Train the network #########
            # Loop over the training batches
            for batch_id in range(n_batches):
                # Get the batch
                batch_input, batch_target = self.get_batch(batch_id,folder='training')
                # Train the network
                UV_out, loss = self.train_fn(batch_input, batch_target)
                # update the error
                train_error += loss

            ######### Validate the network ##########
            # Loop over the validating batches
            for batch_id in range(n_validation_batches):
                # Get the batch
                batch_input, batch_target = self.get_batch(batch_id,folder='validation')
                # Get the error
                _, loss_val = self.val_fn(batch_input, batch_target)
                validation_error += loss_val

            ######### Done now lets print! #########
            # print the results for this epoch:
            print("---------------------------------------")
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epoch, time.time() - start_time))
            print("Train error: {!s:}".format(train_error/n_batches))
            print("Validation error: {!s:}".format(validation_error/n_validation_batches))

            if not(param_save_file is None) and ((epoch % 5) == 0):
                # Save parameters every 10 epochs
                np.save(param_save_file,lasagne.layers.get_all_param_values(self.network))
                print("Stored the parameters to file: {}".format(param_save_file))
        
        # Store the final parameters!
        if not(param_save_file is None):
                # Save parameters every 10 epochs
                np.save(param_save_file,lasagne.layers.get_all_param_values(self.network))
                print("Stored the final parameters to file: {}".format(param_save_file))

        print("Done training the network")


    def fruityfly(self,input_var=None,image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
        """ This function defines the architecture of the Fruit colorizer network 
   
        Input:
        input_var: a theano.tensor.tensor4 (and after theano function creation the data of size(batch_size, 1, image_size[1], image_size[2])
        image_size: the size of the images, a 2D tuple 
        filter_size: The convolutional filter size, a 2D tuple (EVEN FILTER SIZE IS NOT SUPPORTED!)
        pool_size: the max_pool filter size between layers (also the upscale factor!)
        """

        # First make the input layer, batch size=none so any batch size will be accepted!
        # image size is 128x128
        L_input = lasagne.layers.InputLayer(shape=(None, 1, 128, 128), input_var=input_var)

        # Define the first layer
        # Pad = same for keeping the dimensions equal to the input!
        L_1 = lasagne.layers.Conv2DLayer(L_input, num_filters=3, filter_size=filter_size, pad='same')
        L_1 = lasagne.layers.Conv2DLayer(L_1, num_filters=12, filter_size=filter_size, pad='same')

        # Define the second layer
        # Max pool on first layer
        L_2 = lasagne.layers.MaxPool2DLayer(L_1, pool_size=pool_size)
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')

        # Define the third layer
        # Max pool on the second layer
        L_3 = lasagne.layers.MaxPool2DLayer(L_2, pool_size=pool_size)
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=96, filter_size=filter_size, pad='same')
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=96, filter_size=filter_size, pad='same')
    
        ## Now Construct the image again!
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
        L_out = lasagne.layers.Conv2DLayer(L_1, num_filters=2, filter_size=filter_size, pad='same', 
                                           nonlinearity=lasagne.nonlinearities.linear)

        return L_out


    def get_batch(self,batch_number, folder='training'):
        """This function opens a batch file and processes it for the NN"""
        # generate the path to the batch file
        # Since Joost does not know how to count, we need to obtain the nth filename:
        filename = os.listdir(folder)[batch_number]
        batch_file_path = os.path.join(folder, filename)
    
        #open batch file and create input and target
        batch = np.load(batch_file_path)/np.float32(256)
        # Get dimensions
        (batch_size, _, image_x, image_y) = batch.shape
        # target is the UV layers
        batch_target = batch[:,[1,2],:,:]
        batch_target = batch_target.reshape(batch_size,2,image_x,image_y)
        # input is the Y layer
        batch_input = batch[:,0,:,:]
        batch_input = batch_input.reshape(batch_size,1,image_x,image_y)

        return [batch_input, batch_target]

    def NNout2img(self,NN_input,NN_output):
        """This function generates an image from the output of the NN (and combining it with the grayscale input)"""
        # Convert it to show the image!
        U_out = NN_output[0,0,:,:]*256
        V_out = NN_output[0,1,:,:]*256
        img_out = np.zeros((128,128,3), 'uint8')
        img_out[..., 0] = NN_input*256
        img_out[..., 1] = U_out
        img_out[..., 2] = V_out
        return Image.fromarray(img_out,'YCbCr')

    def batch2img(self,batch_number,img_number,folder='training'):
        # Get the image
        batch_input, batch_target = self.get_batch(batch_number, folder)
        (_, _, image_x, image_y) = batch_input.shape
        # Convert to one image
        img_input = batch_input[img_number,:,:,:].reshape(1,1,image_x,image_y)
        img_target = batch_target[img_number,:,:,:].reshape(1,2,image_x,image_y)
        # Get the original image:
        ORG_img = self.NNout2img(img_input,img_target)
        # Eval the NN
        NN_output = self.eval_fn(img_input)
        # Get NN image
        NN_img = self.NNout2img(img_input,NN_output)

        return [ORG_img, NN_img]
     
    def show_random_images(self,n_images, folder='validation'):
        print("Getting random images from the folder: {}".format(folder))
        # Create figure
        f, ax = plot.subplots(n_images,2)
        
        # Get list of batches
        batch_list = os.listdir(folder)
        # Get batch size
        batch_size, _, _, _, = np.load(os.path.join(folder,batch_list[0])).shape

        # Generate an nx2 array of random numbers where the first row depicts the batch number
        # and the second row the image number 
        batch_ids = np.random.randint(0,len(batch_list),(n_images,1))
        img_ids = np.random.randint(0,batch_size,(n_images,1))

        # and loop over them
        for index in range(n_images):
            # Get the image
            (ORG_img, NN_img) = self.batch2img(batch_ids[index],img_ids[index],folder=folder)
            # Show original image
            ax[index,0].axis('off')
            ax[index,0].imshow(ORG_img)
            # Show the NN image
            ax[index,1].axis('off')
            ax[index,1].imshow(NN_img)

        # Show the figures
        plot.show()

    def show_random_images_with_UV_channels(self,n_images, folder='validation'):
        print("Getting random images from the folder: {}".format(folder))
        # Create figure
        f, ax = plot.subplots(n_images*2,3)
        
        # Get list of batches
        batch_list = os.listdir(folder)
        # Get batch size
        batch_size, _, _, _, = np.load(os.path.join(folder,batch_list[0])).shape

        # Generate an nx2 array of random numbers where the first row depicts the batch number
        # and the second row the image number 
        batch_ids = np.random.randint(0,len(batch_list),(n_images,1))
        img_ids = np.random.randint(0,batch_size,(n_images,1))

        # and loop over them
        for index in range(0,n_images*2,2):
            # Get the image
            (ORG_img, NN_img) = self.batch2img(batch_ids[index/2],img_ids[index/2],folder=folder)
            # Get the U and V layers
            _, ORG_U, ORG_V = ORG_img.split()
            _, NN_U, NN_V = NN_img.split()

            # Show original image
            ax[index,0].axis('off')
            ax[index,0].imshow(ORG_img)
            # show the U layer
            ax[index,1].axis('off')
            ax[index,1].imshow(ORG_U,cmap='gray')
            # Show the V layer
            ax[index,2].axis('off')
            ax[index,2].imshow(ORG_V,cmap='gray')

            # Show the NN image
            ax[index+1,0].axis('off')
            ax[index+1,0].imshow(NN_img)
            # show the U layer
            ax[index+1,1].axis('off')
            ax[index+1,1].imshow(NN_U,cmap='gray')
            # Show the V layer
            ax[index+1,2].axis('off')
            ax[index+1,2].imshow(NN_V,cmap='gray')

        # Show the figures
        plot.show()

