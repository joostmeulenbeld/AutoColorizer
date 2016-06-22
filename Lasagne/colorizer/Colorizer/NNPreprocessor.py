"""
The file that processes the already preprocessed images (put in batches) to the format rquired by the NN

author: Dawud Hage & Joost Meulenbeld, written for the NN course IN4015 of the TUDelft

"""

import numpy as np
import os
from skimage import color
from skimage.filters import gaussian

from multiprocessing import Pool
from queue import Queue
from time import time
import random
#from labmeshtest import Colorbins
from itertools import starmap

from PIL import Image


class NNPreprocessor(object):
    """This class preprocesses the batches previously generated"""    

    def __init__(self, batch_size, folder, colorspace, random_superbatches=True, blur=False, randomize=True, workers=4, sigma=3, classification=False):
        """ 
        INPUT:
                batch_size: amount of images returned by get batch, (needed for preloading)
                folder: the folder where the batches are stored (as .npy files)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
                random_superbatches: Open the superbatch files in a random order if True
                blur: Blur the target output of the images if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
                randomize: randomize the image order inside the superbatch if True, (needed for preloading)
                workers: number of workers to do the batch processing
                sigma: The gaussian blur standard deviation, applied to the a and b layers
        """

        self._batch_size = batch_size
        self._folder = folder
        self._colorspace = colorspace
        self._random_superbatches = random_superbatches
        self._blur = blur
        self._randomize = randomize
        self._sigma = sigma
        self._epoch = -1
        self._epoch_done = False
        self._colorbins = Colorbins(k=3,T=1, sigma=3, nbins=12)
        self._classification = classification
        

        # Create queue for the batches
        self._batch_queue = Queue()

        # Create queue for the superbatches
        self._superbatch_queue = Queue()

        # self.filenames
        try:
            self._filenames = os.listdir(folder)
        except:
            print("Folder '{}' does not exist".format(folder))
            raise

        self._n_superbatches = len(self._filenames)
        self._superbatch_shape = np.load(os.path.join(self._folder, self._filenames[0]) , mmap_mode='r').shape

    @property
    def get_epoch(self):
        """
        OUTPUT:
                number of epochs 
        """

        return self._epoch

    @property
    def get_epochProgress(self):
        """
        OUTPUT:
                percentage of progress in epoch
        """

        total_batches = self.get_n_batches
        batches_left_in_superbatch_queue = self._superbatch_queue.qsize() * self.get_batches_per_superbatch
        batches_left_in_batch_queue = self._batch_queue.qsize() 
        total_batches_to_go = batches_left_in_superbatch_queue + batches_left_in_batch_queue

        return ( 1 - total_batches_to_go / (total_batches+1) ) * 100.

    @property
    def get_n_batches(self):
        """
        OUTPUT:
                number of batches used in one epoch
        """

        return self._n_superbatches * self.get_batches_per_superbatch

    @property
    def get_batches_per_superbatch(self):
        """
        OUTPUT:
                number of batches extracted from one superbatch
        """

        return np.floor(self._superbatch_shape[0] / self._batch_size)

    @property
    def get_epoch_done(self):
        """
        OUTPUT:
                bool, true if looped over all data, after this function has been called the bool is set to false!
        """
        done = self._epoch_done
        self._epoch_done = False

        return done

    @property
    def  get_batch(self):
        """
        OUTPUT:
                a batch of shape=(batch_size, 3, image_x, image_y)
        """
        # if batch array is empty, call _proces_next_superbatch, save result in self.batches
        if self._batch_queue.empty():
            self._process_next_superbatch()

        batch =  self._batch_queue.get()

        # Check if this was the last batch of the last superbatch
        if self._batch_queue.empty() and self._superbatch_queue.empty():
            # done with the epoch!
            self._epoch_done = True
        else: 
            self._epoch_done = False

        # return next item in array
        return batch


    def get_image(self, superbatch_id, image_id, blur=False, colorspace='CIEL*a*b*'):
        """ 
        INPUT:
                superbatch_id: the filenumber of the superbatch in the folder
                image_id: The image number in the superbatch
                blur: Blur the target output of the image if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
                colorspace: the colorspace that the image will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
        OUTPUT:
                An image in the specified colorspace of shape=(1, 3,image_x,image_y)
        """

        # Get list of superbatch_filenames
        superbatchlist = sorted(os.listdir(self._folder))
        assert superbatch_id < len(superbatchlist) and superbatch_id >= 0, \
            print("Requested superbatch {}, but only {} superbatches present in folder {}".format(superbatch_id, len(superbatchlist), self._folder))

        # Get the required superbatch read only (eficient memory usage)
        superbatch = np.load(os.path.join(self._folder, sorted(os.listdir(self._folder))[superbatch_id]), mmap_mode='r')

        # Check if image number exists
        assert image_id < superbatch.shape[0] and image_id >= 0, \
            print("Requested image {}, but only {} images present in the requrested batch".format(image_id, len(superbatch.shape[0]), self._folder))

        # Extract image and rearange to shape=(image_x,image_y,3)
        image = np.transpose(superbatch[image_id,:,:,:], [1,2,0]) / 255.

        # Convert colorspace and blur if needed
        image = self._convert_colorspace(image,colorspace,blur,self._sigma)
        
        image = np.transpose(image, [2,0,1]).reshape(1,3,self._superbatch_shape[2],self._superbatch_shape[3]).astype('float32')

        # Transpose back to format required for the neural network
        return image

    def get_random_image(self, colorspace, blur=False):
        """ 
        INPUT:
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
                blur: Blur the target output of the image if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
                
        OUTPUT:
                A randomly selected image in the specified colorspace of shape=(1, 3,image_x,image_y)
        """

        # get sorted list of superbatches
        superbatchlist = sorted(os.listdir(self._folder))

        # pick a random superbatch
        superbatchnr = random.randint(0, len(superbatchlist)-1)

        # load the random superbatch with mmap_mode='r' such that it doesn't actually load it to memory
        superbatch = np.load(os.path.join(self._folder, superbatchlist[superbatchnr]), mmap_mode='r')

        # pick a random image from the superbatch
        imagenr = random.randint(0, superbatch.shape[0]-1)

        # return the image using the found indices
        return self.get_image(superbatchnr, imagenr, blur, colorspace)


    def get_random_images(self,n_images, colorspace, blur=False):
        """ 
        INPUT:
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
                n_images:  number of images to return
                blur: Blur the target output of the image if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
                
        OUTPUT:
                A randomly selected image batch in the specified colorspace of shape=(n_images, 3,image_x,image_y)
        """

        batch = np.empty((0,3,self._superbatch_shape[2],self._superbatch_shape[3])).astype('float32')
        # Get random images andstack them
        for i in range(n_images):

            image = self.get_random_image(colorspace, blur)
            batch = np.append(batch, image, axis=0)

        return batch



    ########## Private functions ##########
    def _process_next_superbatch(self):
        """
            Pupulate the self.batch_queue
        """
        # if the superbatch queue has no items, repopulate superbatch queue (in random order if needed)
        if self._superbatch_queue.empty():

            if  self._random_superbatches:
                # randomize file list
                np.random.shuffle(self._filenames)

            # Add to queue
            [self._superbatch_queue.put(filename) for filename in self._filenames]

            # add 1 to epoch counter
            self._epoch += 1


        # load superbatch numpy array by taking superbatch_queue.get()
        filename = self._superbatch_queue.get()
        superbatch = np.load(os.path.join(self._folder,filename)) / 255.
           

        # process next superbatch
        self._process_superbatch(superbatch)


    def _process_superbatch(self, superbatch):
        """ 
        INPUT:
                superbatch: numpy array containing the superbatch to be processed
        OUTPUT:
                processed superbatch (gets settings as provided by the constructor)
        """

        # get superbatch size
        (superbatch_size,_,_,_) = superbatch.shape

        # randomize the images inside the superbatch if needed
        if self._randomize:
            np.random.shuffle(superbatch)

        # split superbatch
        # create a list of numpy arrays of shape=(batch_size, 3, image_x, image_y)
        pool_list = [superbatch[index:index+self._batch_size, :, :, :] for index in range(0, superbatch_size, self._batch_size)]
        
        # Check if the last batch still has the same size, otherways delete it from the list!
        if not(pool_list[-1].shape is pool_list[0].shape):
            del pool_list[-1]

        processed_batches = map(self._process_batch, pool_list)
        
        #if self._classification:
        #    assert ( (self._colorspace == 'CIELab') ), \
        #    "to use classification the colorspace must be CIELab"
            
        #    processed_batches = map(self._to_classification, processed_batches)

        for batch in processed_batches:
            self._batch_queue.put(batch)

        
    def _process_batch(self, batch):
        """ 
        OUTPUT
                processed batch (gets settings as provided by the constructor)
        """

        # Get the shape of the batch (batch_size, 3, image_x, image_y)
        batch_shape = batch.shape


        # Loop over the batch
        for index in range(batch_shape[0]):
            # Get an image from the batch and change axis directions to match normal specification (x, y, n_channels)
            image = np.transpose(batch[index,:,:,:], [1,2,0])

            # Convert the image to the correct colorspace, and blur if needed
            image = self._convert_colorspace(image,self._colorspace,self._classification,self._blur, self._sigma)

            # Transpose back to format required for the neural network and save it in the original batch
            batch[index,:,:,:] = np.transpose(image, [2,0,1])

        # Cast to float32 
        batch = batch.astype('float32')
        return batch
        
    #def _to_classification(self, batch):
    #    """
    #    OUTPUT
    #        processed batch to classification. Final shape is [batch size, 1+classes, x, y]
    #        the final shape has as first element on 2nd dimension the L values
    #    """
        
    #    # Get the shape of the batch (batch_size, 2, image_x, image_y)
    #    new_batch_shape=np.array([batch.shape[0],self._colorbins.numbins+1,batch.shape[2],batch.shape[3]])
    #    batch_new = np.zeros(new_batch_shape)
        
        # Loop over the batch
        for image_index in range(batch.shape[0]):
            batch_new[image_index,0,:,:] = batch[image_index,0,:,:]
            # Loop over the pixels
            for x in range(batch.shape[2]):
                #loop over the x pixels
                batch_new[image_index,1:,x,:]=self._colorbins.k_means(np.transpose(batch[image_index,1:3,x,:],(1,0)))
        print('one batch done')
        return batch_new.astype('float32')

                    
                    
            

    @staticmethod
    def _convert_colorspace(image,colorspace,classification=False,blur=False, sigma=3):
        """ 
        INPUT:
                image: The image to be converted to the specified colorspace, should have shape=(image_x,image_y,3)
                        the input colorspace should be RGB mapped between [0 and 1], (it will return the same image if colorspace is set to RGB)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
                blur: Blur the target output of the image if True.
                        Supported colorspaces:
                            'CIELab'
                            'CIEL*a*b*'
                            'HSV'
                            'YUV'
        OUTPUT:
                The image converted to the specified colorspace of shape=(image_x,image_y,3)
        """
        # Check if colorspace is properly defined
        assert_colorspace(colorspace)

        # Convert to CIELab
        if ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') ):
            # This converts the rgb to XYZ where:
            # X is in [0, 95.05]
            # Y is in [0, 100]
            # Z is in [0, 108.9]
            # Then from XYZ to CIELab where: (DOES DEPEND ON THE WHITEPOINT!, here for default)
            # L is in [0, 100]
            # a is in [-431.034,  431.034] --> [-500*(1-16/116), 500*(1-16/116)]
            # b is in [-172.41379, 172.41379] --> [-200*(1-16/116), 200*(1-16/116)]
            image = color.rgb2lab(image)
            
        if classification == True:
            # Normalise the L layer so it is between 0 and 1
            image[:,:,0] /= 100
            
            

            

        # convert to CIEL*a*b
        if (colorspace == 'CIEL*a*b*'):
            image = remap_CIELab(image)

        # convert to YCbCr
        if (colorspace == 'YCbCr'):
            # Convert to a PIL Image (Should be replaced by the scikit-image equivalent color.rgb2YCbCr in the future)
            im = Image.fromarray( np.uint8(image*255.), mode='RGB' )
            im = im.convert('YCbCr')
            # Put back into a numpy array
            image = np.array(im)/255.

        # convert to HSV
        if (colorspace == 'HSV'):
            image = color.rgb2hsv(image)

        # blur the target if needed
        if (blur and ( (colorspace == 'CIELab') or
                      (colorspace == 'CIEL*a*b*') or 
                      (colorspace == 'YCbCr') ) ):
            image[:,:,1] = gaussian(image[:,:,1], sigma)
            image[:,:,2] = gaussian(image[:,:,2], sigma) 
        elif (blur and (colorspace == 'HSV') ):
            image[:,:,0] = gaussian(image[:,:,1], sigma)
            image[:,:,1] = gaussian(image[:,:,2], sigma) 
            

        return image




def remap_CIELab(image):
    """
    INPUT:
            image: the image to be remapped. The input colorspace should be CIELab
                    image.shape=(image_x, image_y, 3)
    OUTPUT: 
            an image where the L a and b layers are scaled to be between 0 and 1 for most frequently used colors.
            they could however be beyond  the scale! (this colorspace is called CIEL*a*b* from now on)
    """

    # Normalize image so that all layers are between 0 - 1 (this is to fit the hole rgb part of the LAB space in a unit-cube)
    # Max(L) = 99.6549222328
    # Max(a) = 97.9408293554
    # Max(b) = 94.1970679706

    # Min(L) = 0.0
    # Min(a) = -85.9266517489
    # Min(b) = -107.536445411
    
    # White:
    #  L = 1.00000000e+02 
    #  a = -2.45493786e-03 ~ 0
    #  b = 4.65342115e-03 ~ 0

    # Black:
    #  L = 0
    #  a = 0
    #  b = 0

    # Keep aspectratio constant between a and b layers, so devide both a and b by 108, and recenter around 0.5:
    image[:,:,0] /= 100.
    image[:,:,1] = (image[:,:,1] + 108.) / (2.*108) # take the same subset as for the b values, so the colors wont be skewed
    image[:,:,2] = (image[:,:,2] + 108.) / (2.*108) # Normalize all b values

    return image

def unmap_CIELab(image):
    """
    Inverse of remap_CIELab
    INPUT:
            image: the image to be remapped from CIEL*a*b* to normal CIELab values
                    image.shape=(image_x, image_y, 3)
    OUTPUT: 
            an image where the L a and b layers have their original values
    """

    # Inverse of function remap_CIELab
    image[:,:,0] *= 100.
    image[:,:,1] = image[:,:,1] * (2.*108.)  - 108. 
    image[:,:,2] = image[:,:,2] * (2.*108.)  - 108.

    return image


def assert_colorspace(colorspace):
    """ Raise an error if the colorspace is not an allowed one 
    INPUT:
            colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YCbCr' for YCbCr
                        'HSV' for HSV
    """
    # Check if colorspace is properly defined
    assert ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') or (colorspace == 'RGB') or (colorspace == 'YCbCr') or (colorspace == 'HSV') ), \
    "the colorspace must be 'CIELab' or 'CIEL*a*b*' or 'RGB' or YCbCr or 'HSV'" 
   
if __name__ == "__main__":
    traindata=NNPreprocessor(batch_size=10,folder='fruit_training',colorspace='CIELab',random_superbatches=True,blur=False,randomize=True,workers=4,sigma=3,classification=True)
    traindata.get_batch

