import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image
import pylab




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
    L_out = lasagne.layers.Conv2DLayer(L_1, num_filters=2, filter_size=filter_size, pad='same')

    return L_out


"""Test the function"""
# open an image file, this will be in the form of (150, 150, 3)
img = Image.open('test_image.jpg')
# Convert to YUV colorspace
img = img.convert('YCbCr')
# resize to (128, 128, 3)
img = img.resize((128,128),Image.BICUBIC)
# show image

# convert to np.array
img_np = np.asarray(img, dtype='float32') / 256.
# put image in 4D tensor of shape (1, 3, height, width)
img_np = img_np.transpose(2, 0, 1)
img_np = img_np[1,:,:]
img_np = img_np.reshape(1, 1, 128, 128)

print("Image loaded!")

input = T.tensor4('input')
network = fruityfly(input)

output = lasagne.layers.get_output(network)
eval_fn = theano.function([input],output)

# Now evaluate the image:
UV_out = eval_fn(img_np)

print(UV_out.shape)

# Nice now we have the UV values! 
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(UV_out[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(UV_out[0, 1, :, :])
pylab.show()