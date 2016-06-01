import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plot

import sys
import os
from glob import glob
import time
import logging

class colorizer(object):

    def __init__(self, param_file=None, error_filename=None):

        # Create the network here and all the theano functions
        print("Initializing the network")

        input = T.tensor4('input')
        target = T.tensor4('target') # shape=(batch_size,2,image_x,image_y)

        # Create the neural network
        self.network = self.fruityfly(input)

        # Set params if given
        if not(param_file is None):
            param_load = np.load(param_file)
            lasagne.layers.set_all_param_values(self.network, param_load)
            print("Loaded param file: {}".format(param_file))

        # Get the output of the network
        output = lasagne.layers.get_output(self.network)

        # Get the sum squared error per image
        loss = lasagne.objectives.squared_error(output,target) # shape = (batch_size, 2, image_x, image_y)
        loss = loss.sum(axis=[1,2,3]) # shape (batch_size, 1)
        # And take the mean over the batch
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with adadelta.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adadelta(loss, params, learning_rate=1, rho=0.9, epsilon=1e-06)
        # Add neserov momentum
        updates = lasagne.updates.apply_nesterov_momentum(updates,params,momentum=0.9)

        # Create theano functions to be used in the functions
        self.eval_fn = theano.function([input],output)
        self.val_fn = theano.function([input, target],[output, loss])
        self.train_fn = theano.function([input, target],[output, loss], updates=updates)

        # Check if the training errors need to be stored in a file
        if not(error_filename is None):
            # Check if there is a file already with the training errors
            if glob(error_filename + '*'):
                # Yes it exists so open it!
                self.error_log = np.load(error_filename + '.npy')
            else:
                # No so create it.
                self.error_log = np.empty((0,3))
        else:
            # Now just put the self.error log to none
            self.error_log = None
        self.error_filename = error_filename



        print("Initialized the network")


    def train_network(self,n_epoch = 20 , n_batches = 'all', n_validation_batches = 'all', param_save_file=None):

        print("Start training! \n")

        # Start plot for errors
        #plot.axis([0,n_epoch,0,1])
        #plot.ion()

        if not isinstance(n_batches,int):
            # Now just take all the batches
            # get number of batches
            n_batches = len(os.listdir('training'))

        if not isinstance(n_validation_batches,int):
            # Now just take all the batches
            # get number of batches
            n_validation_batches = len(os.listdir('validation'))

        for epoch in range(n_epoch):
            # Set errors to zero for this epoch
            train_error = 0
            validation_error = 0

            # Keep track of time
            start_time = time.time()

            ######### Train the network #########
            # Loop over the training batches
            # Do this random so make a shuffled index list
            batch_ids = np.arange(n_batches)
            np.random.shuffle(batch_ids)
            counter = 0
            for batch_id in batch_ids:
                # Get the batch
                batch_input, batch_target = self.get_batch(batch_id,folder='training')
                # Blur the batch target:
                batch_target = self.blur_batch_target(batch_target)
                # Train the network
                UV_out, loss = self.train_fn(batch_input, batch_target)
                # update the error
                train_error += loss
                # Print (update) progress
                print(" Progress: {:3.1f}%            \r".format(counter/(n_batches+ n_validation_batches)*100),end="")
                counter += 1

            ######### Validate the network ##########
            # Loop over the validating batches
            for batch_id in range(n_validation_batches):
                # Get the batch
                batch_input, batch_target = self.get_batch(batch_id,folder='validation')
                # Get the error
                _, loss_val = self.val_fn(batch_input, batch_target)
                validation_error += loss_val
                # Print (update) progress
                print(" Progress: {:3.1f}%            \r".format(counter/(n_batches + n_validation_batches)*100),end="")
                counter += 1


            ######### Done now lets print! #########
            # Store the errors in the error log
            if not(self.error_log is None):
                self.error_log = np.append(self.error_log, np.array([[epoch, train_error/(n_batches), validation_error/(n_validation_batches)]]), axis=0)

            ## plot the errors:
            #plot.cla()
            #plot.plot(self.error_log[:,0],self.error_log[:,1], label='Train error')
            #plot.plot(self.error_log[:,0],self.error_log[:,2], label='Validation error')
            #plot.legend()
            #plot.draw()


            # print the results for this epoch:
            print("---------------------------------------")
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epoch, time.time() - start_time))
            print("Train error: {!s:}".format(train_error/n_batches))
            print("Validation error: {!s:}".format(validation_error/n_validation_batches))

            if not(param_save_file is None) and ((epoch % 5) == 0):
                # Save parameters every 10 epochs
                np.save(param_save_file[:-4] + str(epoch) + '.npy',lasagne.layers.get_all_param_values(self.network))
                print("Stored the parameters to file: {}".format(param_save_file))
        
        # Store the final parameters!
        if not(param_save_file is None):
            # Save parameters
            np.save(param_save_file,lasagne.layers.get_all_param_values(self.network))
            print("Stored the final parameters to file: {}".format(param_save_file))

        # Store the error values
        if not(self.error_log is None):
            # Save the errors
            np.save(self.error_filename + '.npy',self.error_log)
            print("Stored the error values to the file: {}".format(self.error_filename + '.npy'))

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
        # Take the batch norm
        L_1 = lasagne.layers.batch_norm(L_1)

        # Define the second layer
        # Max pool on first layer
        L_2 = lasagne.layers.MaxPool2DLayer(L_1, pool_size=pool_size)
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')
        # Take the batch norm
        L_2 = lasagne.layers.batch_norm(L_2)

        # Define the third layer
        # Max pool on the second layer
        L_3 = lasagne.layers.MaxPool2DLayer(L_2, pool_size=pool_size)
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=96, filter_size=filter_size, pad='same')
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=96, filter_size=filter_size, pad='same')
        # Take the batch norm
        L_3 = lasagne.layers.batch_norm(L_3)

        # Define the fourth layer
        # Max pool on the second layer
        L_4 = lasagne.layers.MaxPool2DLayer(L_3, pool_size=pool_size)
        L_4 = lasagne.layers.Conv2DLayer(L_4, num_filters=192, filter_size=filter_size, pad='same')
        L_4 = lasagne.layers.Conv2DLayer(L_4, num_filters=192, filter_size=filter_size, pad='same')
    
        ## Now Construct the image again!
        # Get the batch norm and reduce feature maps to fit previous layer
        L_4 = lasagne.layers.Conv2DLayer(L_4, num_filters=96, filter_size=(1,1), pad='same')
        L_4 = lasagne.layers.batch_norm(L_4)
        # Upscale layer 3 to fit L2 size
        L_4 = lasagne.layers.Upscale2DLayer(L_4, scale_factor=pool_size)

        # Concate with L_3
        L_3 = lasagne.layers.concat([L_3, L_4])
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=96, filter_size=filter_size, pad='same')
        L_3 = lasagne.layers.Conv2DLayer(L_3, num_filters=48, filter_size=filter_size, pad='same')
        L_3 = lasagne.layers.batch_norm(L_3)
        # Upscale L_3 to fit L_2 size
        L_3 = lasagne.layers.Upscale2DLayer(L_3, scale_factor=pool_size)

        # Concate with L_2
        L_2 = lasagne.layers.concat([L_2, L_3])
        # Convolve L_2 to fit feature maps to L1
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=48, filter_size=filter_size, pad='same')
        L_2 = lasagne.layers.Conv2DLayer(L_2, num_filters=12, filter_size=filter_size, pad='same')
        L_2 = lasagne.layers.batch_norm(L_2)
        # Upscale L_2 to fit L_1 size
        L_2 = lasagne.layers.Upscale2DLayer(L_2, scale_factor=pool_size)
    
        # Do the same for layer 1
        L_1 = lasagne.layers.concat([L_1, L_2])
        # Convolve L_1 to fit feature maps to L1
        L_1 = lasagne.layers.Conv2DLayer(L_1, num_filters=12, filter_size=filter_size, pad='same')
        L_1 = lasagne.layers.Conv2DLayer(L_1, num_filters=6, filter_size=filter_size, pad='same')
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

    def blur_target(self,target, sigma=5):
        """This function blurs the target of shape = (1,2,image_x,image_y)"""
        img_shape = target.shape
        target[0,:,:] = gaussian_filter(target[0,:,:],sigma).reshape(1,img_shape[1],img_shape[2])
        target[1,:,:] = gaussian_filter(target[1,:,:],sigma).reshape(1,img_shape[1],img_shape[2])
        return target

    def blur_batch_target(self,batch_target,sigma=5):
        """This function blurs the batch target"""

        # Get the shape of the batch (batch_size, 2, image_x, image_y)
        batch_shape = batch_target.shape

        # Loop over the batch
        for index in range(batch_shape[0]):
            # Blur each target
            batch_target[index,:,:,:] = self.blur_target(batch_target[index,:,:,:]).reshape(1,2,batch_shape[2],batch_shape[3])

        return batch_target


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

