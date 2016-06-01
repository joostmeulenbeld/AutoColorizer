import numpy as np
import os
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
from queue import Queue
from time import time
import random


class NNPreprocessor(object):
    """This class preprocesses the batches previously generated"""
    def __init__(self, batch_size, folder, random_superbatches=True, blur=False, randomize=True, workers=16):

        """ INPUT:
                    batch_size: amount of images returned by get batch, (needed for preloading)
                    folder: the folder where the batches are stored (as .npy files)
                    random_superbatches: Open the superbatch files in a random order if True
                    blur: Blur the target output of every batch (obtained by get_batch) if True, (needed for preloading)
                    randomize: randomize the image order inside the superbatch if True, (needed for preloading)
                    workers: number of workers to do the batch processing
        """

        self._batch_size = batch_size
        self._folder = folder
        self._random_superbatches = random_superbatches
        self._blur = blur
        self._randomize = randomize

        # Create queue for the batches
        self._batch_queue = Queue()

        # Create queue for the superbatches
        self._superbatch_queue = Queue()

        # self.filenames
        try:
            self._filenames = os.listdir(folder)
        except:
            print("Folder does not exist")

        # Create pool for processing images in superbatch
        self._p = Pool(workers)



    def  get_batch(self):
        """ OUTPUT:
                    a batch of shape=(batch_size, 3, image_x, image_y)
        """
        # if batch array is empty, call _proces_next_superbatch, save result in self.batches
        if self._batch_queue.empty:
            self._process_next_superbatch()

        # return next item in array (pop)
        return self._batch_queue.get()


    def get_image(self, superbatch_id, image_id, blur=False):
        """ INPUT:
                    superbatch_id: the filenumber of the superbatch in the folder
                    image_id: The image number in the superbatch
                    blur: Blur the target output of the image if True (only performed on the CIELab colorspace output)
            OUTPUT:
                    a image in two colorspaces (CIELab and RGB) of shape=(2, 3, image_x, image_y)
                    where the first axis is: CIELab for 0 and RGB for 1. RGB has values 0-255
        """
        superbatchlist = sorted(os.listdir(self._folder))
        assert superbatch_id < len(superbatchlist) and superbatch_id >= 0, \
            print("Requested superbatch {}, but only {} superbatches present in folder {}".format(superbatch_id, len(superbatchlist), self._folder))

        # Get list of superbatch_filenames and get the required one
        superbatch = np.load(os.path.join(self._folder, sorted(os.listdir(self._folder))[superbatch_id]), mmap_mode='r')

        assert image_id < superbatch.shape[0] and image_id >= 0, \
            print("Requested image {}, but only {} images present in the requrested batch".format(image_id, len(superbatch.shape[0]), self._folder))

        image_rgb = np.transpose(superbatch[image_id,:,:,:], [1,2,0]) / 256

        image_lab = color.rgb2lab(image_rgb)

        if blur:
            image_lab[:,:,1] = gaussian_filter(image_lab[:,:,1], sigma)
            image_lab[:,:,2] = gaussian_filter(image_lab[:,:,2], sigma)

        # Transpose back to format required for the neural network and save it in the original batch
        return np.stack([np.transpose(image_lab, [2,0,1]), np.transpose(image_rgb, [2,0,1])], axis=0)

    def get_random_image(self, blur=False):
        """ INPUT:
                    blur: Blur the target output of the image if True (only performed on the CIELab colorspace output)
            OUTPUT:
                    a randomly selected image in two colorspaces (CIELab and RGB) of shape=(2, 3, image_x, image_y)
                    where the first axis is: CIELab for 0 and RGB for 1
        """
        # get sorted list of superbatches
        superbatchlist = sorted(os.listdir(self._folder))
        # pick a random superbatch
        superbatchnr = random.randint(0, len(superbatchlist)-1)
        # load the random superbatch with mmap_mode='r' such that it doesn't actually load it to memory
        superbatch = np.load(os.path.join(self._folder, superbatchlist[superbatchnr]), mmap_mode='r')
        # pick a random image from the superbatch
        imagenr = random.randint(0, superbatch.shape[0]-1)
        #return the image using the found indices
        return self.get_image(superbatchnr, imagenr, blur)



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


        # load superbatch numpy array by taking superbatch_queue.get()
        filename = self._superbatch_queue.get()
        superbatch = np.load(os.path.join(self._folder,filename))/256

        # process next superbatch
        self._process_superbatch(superbatch)


    def _process_superbatch(self, superbatch):
        """ INPUT:
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
        # Pool: process batches
        processed_batches = self._p.starmap(_process_batch, [(batch, self._blur) for batch in pool_list])
        for batch in processed_batches:
            self._batch_queue.put(batch)


def _process_batch(batch, blur):
    """ INPUT:
                batch: numpy array containing the batch to be processed

        OUTPUT
                processed batch (gets settings as provided by the constructor)
    """

    # Get the shape of the batch (batch_size, 2, image_x, image_y)
    batch_shape = batch.shape
    # Define gaussian blur standard deviation
    sigma = 3

    # Loop over the batch
    for index in range(batch_shape[0]):
        # Get an image from the batch and change axis directions to match normal specification (x, y, n_channels)
        image = np.transpose(batch[index,:,:,:], [1,2,0])

        image = color.rgb2lab(image)

        if blur:
            image[:,:,1] = gaussian_filter(image[:,:,1], sigma)
            image[:,:,2] = gaussian_filter(image[:,:,2], sigma)

        # Transpose back to format required for the neural network and save it in the original batch
        batch[index,:,:,:] = np.transpose(image, [2,0,1])

    return batch


if __name__ == "__main__":
    nnp = NNPreprocessor(5, "testing_files", blur=True)

    print(nnp.get_batch().shape)
    # print(nnp.get_random_image().shape)
    # Test processing of a single batch
    # testbatch = np.load(os.path.join("testing_files", "testbatch.npy"))
    # nnp._process_batch(testbatch)
