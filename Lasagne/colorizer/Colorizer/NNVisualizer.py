"""
The file that defines functions to visualize the output of the NN.

author: Dawud Hage, written for the NN course IN4015 of the TUDelft

"""

import numpy as np
from NNPreprocessor import unmap_CIELab, assert_colorspace
from skimage import color
from PIL import Image
import matplotlib.pyplot as plot
import os

def array2img(array, colorspace):
    """ 
    INPUT:
            array: Lab layers in an array of shape=(3, image_x, image_y)
            colorspace: the colorspace that the array is in;
                        ''CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
    OUTPUT:
            Image
    """
    # Check if colorspace is properly defined
    assert_colorspace(colorspace)

    # Convert the image to shape=(image_x, image_y, 3)
    image = np.transpose(array,[1,2,0]).astype('float64')

    if (colorspace == 'CIEL*a*b*'):
        # Convert to CIELab:
        image = unmap_CIELab(image)

    if ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') ):
        # Convert to rgb:
        image = color.lab2rgb(image)
    
    if (colorspace == 'HSV'):
        image = color.hsv2rgb(image)

    # YCbCr is supported by the PIL Image pkg. so just change the mode that is passed
    if not(colorspace == 'YCbCr'):
        colorspace = 'RGB'

    # Now the image is definitely in a supported colorspace
    return Image.fromarray(np.uint8(image*255.),mode=colorspace)



def show_images_with_ab_channels(ORGbatch, NNbatch, colorspace):
    """ 
    INPUT:
            ORGbatch: batch of (original) images with shape=(batch_size, 3, image_x, image_y)
                        Must be in CIEL*a*b* colorspace, see NNPreprocessor function remap
            NNbatch : batch of (NN output) images with shape=(batch_size, 3, image_x, image_y)
                        Must be in CIEL*a*b* colorspace, see NNPreprocessor function remap
            colorspace: the colorspace that the input batches are in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
    """
    assert (ORGbatch.shape == NNbatch.shape), "ORGbatch and NNbatch do not have the same shape"

    n_images,_,_,_ = ORGbatch.shape

    # Create figure
    f, ax = plot.subplots(n_images*2,4)

    # and loop over the images
    for index in range(0,n_images*2,2):

        # Get the image
        ORG_img = array2img(ORGbatch[int(index/2),:,:,:],colorspace)
        NN_img  = array2img(NNbatch[int(index/2),:,:,:],colorspace)

        # Define the different channel ids to show
        # (gray, layer_1, layer_2)
        if (colorspace == 'CIEL*a*b*') or (colorspace == 'YCbCr'):
            channels = (0,1,2)
        elif (colorspace == 'HSV'):
            channels = (2,0,1)
        else:
            raise ValueError("Cannot handle this colorspace, can only process 'CIEL*a*b*' and 'HSV'")

        # Get the grayscale input
        ORG_gray = ORGbatch[int(index/2),channels[0],:,:]
        # Get the 1 and 2 layers
        ORG_1 = ORGbatch[int(index/2),channels[1],:,:]
        ORG_2 = ORGbatch[int(index/2),channels[2],:,:]
        
        NN_1 = NNbatch[int(index/2),channels[1],:,:]
        NN_2 = NNbatch[int(index/2),channels[2],:,:]

        # Show original image
        # grayscale input
        ax[index,0].axis('off')
        ax[index,0].imshow(ORG_gray,cmap='gray')
        # color image
        ax[index,1].axis('off')
        ax[index,1].imshow(ORG_img)
        # show the a layer
        ax[index,2].axis('off')
        ax[index,2].imshow(ORG_1,cmap='gray')
        # Show the b layer
        ax[index,3].axis('off')
        ax[index,3].imshow(ORG_2,cmap='gray')

        # Show the NN image
        # grayscale input
        ax[index+1,0].axis('off')
        ax[index+1,0].imshow(ORG_gray,cmap='gray')
        # The colored image
        ax[index+1,1].axis('off')
        ax[index+1,1].imshow(NN_img)
        # show the a layer
        ax[index+1,2].axis('off')
        ax[index+1,2].imshow(NN_1,cmap='gray')
        # Show the b layer
        ax[index+1,3].axis('off')
        ax[index+1,3].imshow(NN_2,cmap='gray')

    # Show the figures
    plot.show()

def print_progress(string, elapsed_time, progress):
    """
    Print a fancy progress bar in the console
    INPUT:
            string: A string printed before the :
            elapsed_time: The elapsed time in seconds
            progress: The progress in percentage
    """
    n_bars = np.floor(progress/100*20).astype('int')
    remaining_time = {'total_seconds': np.floor( (elapsed_time / progress * 100) - elapsed_time).astype('int')}
    remaining_time['hour'] = np.floor(remaining_time['total_seconds'] / 3600).astype('int')
    remaining_time['minutes'] = np.floor( (remaining_time['total_seconds'] - 3600*remaining_time['hour']) / 60).astype('int')
    remaining_time['seconds'] = np.floor( (remaining_time['total_seconds'] - 3600*remaining_time['hour'] - 60*remaining_time['minutes'])).astype('int')
    print("{}: {:3.1f}% |{}| Remaining time: {:0>2}:{:0>2}:{:0>2}                                \r".format(    string,
                                                                                                                progress,
                                                                                                                "#"*n_bars + " "*(20 - n_bars),
                                                                                                                remaining_time['hour'],
                                                                                                                remaining_time['minutes'],
                                                                                                                remaining_time['seconds']),end="")


def plot_errors(error):
    """This function plots the validation error and train error
    INPUT:
            error: the error log as saved in the error files
    """

    plot.plot(error[:,0],error[:,1],label='Train error')
    plot.plot(error[:,0],error[:,2],label='Validation error')
    plot.xlabel('epoch')
    plot.ylabel('error')
    plot.legend()
    plot.show()

##### MENU FUNCTIONS #####

def cls():
    """ Clear the console """
    os.system('cls' if os.name=='nt' else 'clear')


def get_int(validation_list=None):
    """
    Get an integer from the user
    INPUT:
            List of integers to check for existance
    """
    while True:
        try:
            user_int = int(input("\t"))       
        except ValueError:
            print("It should be an integer, please try again!")
            continue
        else:
            if not(validation_list is None):
                # Check if it complies
                if user_int in validation_list:
                    break
                else:
                    print("The provided number is not allowed.")
            else:
                break

    return user_int

def gen_menu(menu_options):
    """ This function prints a menu with options to evaluate/train the NN 
    INPUT:
            a list of options to be printed
    OUTPUT:
            The choice of the user
    """

    cls() # clear the console
    print('-'*50)
    print('Choose one of the following:')
    for i, option in enumerate(menu_options):
        print("{}. {}".format(i, option))
    print('Your choice: \t')

    return get_int()
