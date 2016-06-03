import numpy as np
from NNPreprocessor import unmap_CIELab
from skimage import color
from PIL import Image
import matplotlib.pyplot as plot

def array2img(array, colorspace):
    """ 
    INPUT:
            array: Lab layers in an array of shape=(3, image_x, image_y)
            colorspace: the colorspace that the image is in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
    OUTPUT:
            Image
    """
    assert ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') or (colorspace == 'RGB') ), \
        "the colorspace must be 'CIELab' or 'CIEL*a*b*' or 'RGB'"

    # Convert the image to shape=(image_x, image_y, 3)
    image = np.transpose(array,[1,2,0]).astype('float64')

    if (colorspace is 'CIEL*a*b*'):
        # Convert to CIELab:
        image = unmap_CIELab(image)

    if ( (colorspace is 'CIELab') or (colorspace is 'CIEL*a*b*') ):
        # Convert to rgb:
        image = color.lab2rgb(image)

    # Now the image is definitely in RGB colorspace
    return Image.fromarray(np.uint8(image*255),"RGB")



def show_images_with_ab_channels(ORGbatch, NNbatch):
    """ 
    INPUT:
            ORGbatch: batch of (original) images with shape=(batch_size, 3, image_x, image_y)
                        Must be in CIEL*a*b* colorspace, see NNPreprocessor function remap
            NNbatch : batch of (NN output) images with shape=(batch_size, 3, image_x, image_y)
                        Must be in CIEL*a*b* colorspace, see NNPreprocessor function remap
    """
    assert (ORGbatch.shape == NNbatch.shape), "ORGbatch and NNbatch do not have the same shape"

    n_images,_,_,_ = ORGbatch.shape

    # Create figure
    f, ax = plot.subplots(n_images*2,3)

    # and loop over the images
    for index in range(0,n_images*2,2):

        # Get the image
        ORG_img = array2img(ORGbatch[int(index/2),:,:,:],'CIEL*a*b*')
        NN_img  = array2img(NNbatch[int(index/2),:,:,:],'CIEL*a*b*')

        # Get the a and b layers
        ORG_a = ORGbatch[int(index/2),1,:,:]
        ORG_b = ORGbatch[int(index/2),2,:,:]
        
        NN_a = NNbatch[int(index/2),1,:,:]
        NN_b = NNbatch[int(index/2),2,:,:]

        # Show original image
        ax[index,0].axis('off')
        ax[index,0].imshow(ORG_img)
        # show the a layer
        ax[index,1].axis('off')
        ax[index,1].imshow(ORG_a,cmap='gray')
        # Show the b layer
        ax[index,2].axis('off')
        ax[index,2].imshow(ORG_b,cmap='gray')

        # Show the NN image
        ax[index+1,0].axis('off')
        ax[index+1,0].imshow(NN_img)
        # show the a layer
        ax[index+1,1].axis('off')
        ax[index+1,1].imshow(NN_a,cmap='gray')
        # Show the b layer
        ax[index+1,2].axis('off')
        ax[index+1,2].imshow(NN_b,cmap='gray')

    # Show the figures
    plot.show()

