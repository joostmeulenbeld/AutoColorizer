import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image
import pylab

import sys
import os
import time



def fruityfly(input_var=None,image_size=(128, 128), filter_size = (3, 3), pool_size = 2):
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
    L_out = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(L_1, num_filters=2, filter_size=filter_size, pad='same', 
                                       nonlinearity=lasagne.nonlinearities.linear))

    return L_out
print('created network!')

"""Test the function"""
input = T.tensor4('input')
target = T.tensor4('target') # shape=(batch_size,2,image_x,image_y)

network = fruityfly(input)

# Get the output of the network
output = lasagne.layers.get_output(network)

# Get the mean squared error
loss = lasagne.objectives.squared_error(output,target)
loss = lasagne.objectives.aggregate(loss, mode='sum')
loss = T.sqrt(loss)

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adagrad(loss, params, learning_rate=1.05, epsilon=1e-06)#, momentum=0.9)


eval_fn = theano.function([input],output)
eval_fn2 = theano.function([input, target],[output, loss], updates=updates)

print('created functions!')
######## 1 image test ############
# open an image file, this will be in the form of (150, 150, 3)
img = Image.open('test_image.jpg')
# Convert to YUV colorspace
img = img.convert('YCbCr')
# resize to (128, 128, 3)
img = img.resize((128,128),Image.BICUBIC)
img_org = np.asarray(img, dtype='float32') 

# convert to np.array
img_np = np.asarray(img, dtype='float32') / 256.
# put image in 4D tensor of shape (1, 3, height, width)
img_np = img_np.transpose(2, 0, 1)
img_np_target = img_np[[1,2],:,:]
img_np = img_np[1,:,:]
img_np = img_np.reshape(1, 1, 128, 128)
img_np_target = img_np_target.reshape(1, 2, 128, 128)

########### Batch file test ################
#open batch file and create input and target
batch = np.load("batch_45.npy")/np.float32(256)
batch_target = batch[:,[1,2],:,:]
batch_target = batch_target.reshape(50,2,128,128)
batch_input = batch[:,0,:,:]
batch_input = batch_input.reshape(50,1,128,128)
#create 1 image to feed trough from batch
batch_image=batch[0,:,:,:]
batch_image_input = batch[0,0,:,:]
batch_image_input = batch_image_input.reshape(1,1,128,128)

print("Image loaded!")


for epoch in range(3000):
    error = 0
    start_time = time.time()
    # Now evaluate the image:
    UV_out, loss_out = eval_fn2(batch_input, batch_target)
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, 150, time.time() - start_time))
    print(loss_out)

    
#Feed 1 image through net 
UV_out_image = eval_fn(img_np)
#save parameters
W_save = lasagne.layers.get_all_param_values(network)
np.save("W_save_3000.npy",W_save)

#create image from batch
Y_out = batch_image_input*256.
U_out = UV_out_image[0,0,:,:]*256
V_out = UV_out_image[0,1,:,:]*256
img_out_1 = np.zeros((128,128,3), 'uint8')
img_out_1[..., 0] = Y_out
img_out_1[..., 1] = U_out
img_out_1[..., 2] = V_out
img_out_1_plaatje=Image.fromarray(img_out_1,'YCbCr')
#org img from batch
Y_org = batch_image[0,:,:]*256.
U_org = batch_image[1,:,:]*256
V_org = batch_image[2,:,:]*256
img_org_1 = np.zeros((128,128,3), 'uint8')
img_org_1[..., 0] = Y_org
img_org_1[..., 1] = U_org
img_org_1[..., 2] = V_org
img_org_1_plaatje=Image.fromarray(img_org_1,'YCbCr')
#Show Images
img_out_1_plaatje.show()
img_org_1_plaatje.show()
print("succes")

## create image from output
Y = img_org[:,:,0] #29
U = UV_out[0,0,:,:]*256.
V = UV_out[0,1,:,:]*256.
plaatje = np.zeros((128,128,3), 'uint8')
plaatje[..., 0] = Y
plaatje[..., 1] = U
plaatje[..., 2] = V
img_test=Image.fromarray(plaatje,'YCbCr')
img_test.show()
img.show()
#
# Nice now we have the UV values! 
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out_1[:,:,1]);pylab.gray()
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out_1[:,:,2]);pylab.gray()
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_org_1[:,:,1]); pylab.gray()
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_org_1[:,:,2]); pylab.gray()
pylab.show()