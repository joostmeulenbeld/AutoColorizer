import numpy as np
from NNPreprocessor import unmap_CIELab, assert_colorspace
from skimage import color
from PIL import Image
import matplotlib.pyplot as plot

def array2img(array, colorspace):
    """ 
    INPUT:
            array: Lab layers in an array of shape=(3, image_x, image_y)
            colorspace: the colorspace that the array is in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
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

    # Now the image is definitely in RGB colorspace
    return Image.fromarray(np.uint8(image*255),"RGB")



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
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
    """
    assert (ORGbatch.shape == NNbatch.shape), "ORGbatch and NNbatch do not have the same shape"

    n_images,_,_,_ = ORGbatch.shape

    # Create figure
    f, ax = plot.subplots(n_images*2,3)

    # and loop over the images
    for index in range(0,n_images*2,2):

        # Get the image
        ORG_img = array2img(ORGbatch[int(index/2),:,:,:],colorspace)
        NN_img  = array2img(NNbatch[int(index/2),:,:,:],colorspace)

        if (colorspace == 'CIEL*a*b*'):
            # Get the a and b layers
            ORG_1 = ORGbatch[int(index/2),1,:,:]
            ORG_2 = ORGbatch[int(index/2),2,:,:]
        
            NN_1 = NNbatch[int(index/2),1,:,:]
            NN_2 = NNbatch[int(index/2),2,:,:]
        elif (colorspace == 'HSV'):
            # Get the H and S layers
            ORG_1 = ORGbatch[int(index/2),0,:,:]
            ORG_2 = ORGbatch[int(index/2),1,:,:]
        
            NN_1 = NNbatch[int(index/2),0,:,:]
            NN_2 = NNbatch[int(index/2),1,:,:]
        else:
            raise ValueError("Cannot handle this colorspace, can only process 'CIEL*a*b*' and 'HSV'")

        # Show original image
        ax[index,0].axis('off')
        ax[index,0].imshow(ORG_img)
        # show the a layer
        ax[index,1].axis('off')
        ax[index,1].imshow(ORG_1,cmap='gray')
        # Show the b layer
        ax[index,2].axis('off')
        ax[index,2].imshow(ORG_2,cmap='gray')

        # Show the NN image
        ax[index+1,0].axis('off')
        ax[index+1,0].imshow(NN_img)
        # show the a layer
        ax[index+1,1].axis('off')
        ax[index+1,1].imshow(NN_1,cmap='gray')
        # Show the b layer
        ax[index+1,2].axis('off')
        ax[index+1,2].imshow(NN_2,cmap='gray')

    # Show the figures
    plot.show()

