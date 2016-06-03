import numpy as np
import os
from skimage import color
from skimage.filters import gaussian

from multiprocessing import Pool
from queue import Queue
from time import time
import random
from itertools import starmap

from PIL import Image


class NNPreprocessor(object):
    """This class preprocesses the batches previously generated"""    

    def __init__(self, batch_size, folder, colorspace, random_superbatches=True, blur=False, randomize=True, workers=4, sigma=3):

        """ 
        INPUT:
                batch_size: amount of images returned by get batch, (needed for preloading)
                folder: the folder where the batches are stored (as .npy files)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
                random_superbatches: Open the superbatch files in a random order if True
                blur: Blur the target output of every batch (obtained by get_batch) if True, (needed for preloading)
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
        self._workers = workers
        self._sigma = sigma
        self._epoch = -1
        self._epoch_done = False
        

        # Create queue for the batches
        self._batch_queue = Queue()

        # Create queue for the superbatches
        self._superbatch_queue = Queue()

        # self.filenames
        try:
            self._filenames = os.listdir(folder)
        except:
            print("Folder does not exist")

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
                blur: Blur the target output of the image if True (only performed on the CIELab or CIEL*a*b* colorspace output)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
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
        image = self._convert_colorspace(image,colorspace,blur)

        # Transpose back to format required for the neural network
        return np.transpose(image, [2,0,1]).reshape(1,3,self._superbatch_shape[2],self._superbatch_shape[3]).astype('float32')

    def get_random_image(self, blur=False, colorspace='CIEL*a*b*'):
        """ 
        INPUT:
                blur: Blur the target output of the image if True (only performed on the CIELab colorspace output)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
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


    def get_random_images(self,n_images, blur=False, colorspace='CIEL*a*b*'):
        """ 
        INPUT:
                n_images:  number of images to return
                blur: Blur the target output of the image if True (only performed on the CIELab colorspace output)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
        OUTPUT:
                A randomly selected image batch in the specified colorspace of shape=(n_images, 3,image_x,image_y)
        """

        batch = np.empty((0,3,self._superbatch_shape[2],self._superbatch_shape[3])).astype('float32')
        # Get random images andstack them
        for i in range(n_images):

            image = self.get_random_image(blur,colorspace)
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

        ## Pool: process batches
        ## Create pool for processing images in superbatch
        #p = Pool(self._workers)
        #processed_batches = p.starmap(_process_batch, [(batch, self._blur) for batch in pool_list])
        ## Close the pool
        #p.close()

        processed_batches = map(self._process_batch, pool_list)

        for batch in processed_batches:
            self._batch_queue.put(batch)

        
    def _process_batch(self, batch):
        """ 
        OUTPUT
                processed batch (gets settings as provided by the constructor)
        """

        # Get the shape of the batch (batch_size, 2, image_x, image_y)
        batch_shape = batch.shape
        

        # Loop over the batch
        for index in range(batch_shape[0]):
            # Get an image from the batch and change axis directions to match normal specification (x, y, n_channels)
            image = np.transpose(batch[index,:,:,:], [1,2,0])

            # Convert the image to the correct colorspace, and blur if needed
            image = self._convert_colorspace(image,self._colorspace,self._blur)

            # Transpose back to format required for the neural network and save it in the original batch
            batch[index,:,:,:] = np.transpose(image, [2,0,1])

        # Cast to float32 
        batch = batch.astype('float32')
        return batch

    def _convert_colorspace(self,image,colorspace,blur=False):
        """ 
        INPUT:
                image: The image to be converted to the specified colorspace, should have shape=(image_x,image_y,3)
                        the input colorspace should be RGB mapped between [0 and 1], (it will return the same image if colorspace is set to RGB)
                colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
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

            if blur:
                image[:,:,1] = gaussian(image[:,:,1], self._sigma)
                image[:,:,2] = gaussian(image[:,:,2], self._sigma)

        # convert to CIEL*a*b
        if (colorspace == 'CIEL*a*b*'):
            image = remap_CIELab(image)

        # convert to YCbCr
        if (colorspace == 'YUV'):
            # image = color.rgb2yuv(image)
            pass
            # remap?

        if (colorspace == 'HSV'):
            image = color.rgb2hsv(image)

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
    image[:,:,0] /= 100.
    image[:,:,1] = (image[:,:,1] + 500.*(1-16./116.)) / (1000.*(1-16./116.))
    image[:,:,2] = (image[:,:,2] + 200*(1-16./116.)) / (400.*(1-16./116.))

    # L max ~ 100, L min ~ 0
    # a max ~ 100, a min ~ -65
    # b max ~ 95, b min ~ -105
    #image[:,:,0] /= 100.
    #image[:,:,1] = (image[:,:,1] + 125.) / 250.
    #image[:,:,2] = (image[:,:,2] + 150.) / 300.

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
    #image[:,:,0] *= 100.
    #image[:,:,1] = image[:,:,1]*250. - 125.
    #image[:,:,2] = image[:,:,2]*300. - 150.

    image[:,:,0] *= 100.
    image[:,:,1] = image[:,:,1] * (1000.*(1-16./116.)) - 500.*(1-16./116.)
    image[:,:,2] = image[:,:,2] * (400.*(1-16./116.))  - 200.*(1-16./116.) 

    return image


def assert_colorspace(colorspace):
    """ Raise an error if the colorspace is not an allowed one 
    INPUT:
            colorspace: the colorspace that the images will be put in;
                        'CIELab' for CIELab colorspace
                        'CIEL*a*b*' for the mapped CIELab colorspace (by function remap_CIELab in NNPreprocessor)
                        'RGB' for rgb mapped between [0 and 1]
                        'YUV' for YUV (NOT FUNCTIONAL YET)
                        'HSV' for HSV
    """
    # Check if colorspace is properly defined
    assert ( (colorspace == 'CIELab') or (colorspace == 'CIEL*a*b*') or (colorspace == 'RGB') or (colorspace == 'YCbCr') or (colorspace == 'HSV') ), \
    "the colorspace must be 'CIELab' or 'CIEL*a*b*' or 'RGB' or YCbCr or 'HSV'" 